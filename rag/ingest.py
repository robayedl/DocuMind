from __future__ import annotations

import base64
import os
import re
import unicodedata
from pathlib import Path
from typing import List, Tuple

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from unstructured.documents.elements import Image as UnstructuredImage
from unstructured.documents.elements import Table
from unstructured.partition.pdf import partition_pdf

from rag.chains.retrieval import save_bm25
from rag.contextualize import contextualize_chunk
from rag.store import DEFAULT_COLLECTION, add_documents, clear_document


_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
    separators=["\n\n", "\n", ". ", " ", ""],
)

_MAX_FIGURES = 30

_FIGURE_PROMPT = (
    "Describe this figure from a document in 2-3 sentences for retrieval purposes. "
    "Include visible numbers, labels, or trends."
)

_IMAGE_MIME = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


def get_storage_dir() -> Path:
    return Path(os.getenv("STORAGE_DIR", "storage"))


def get_pdf_path(doc_id: str) -> Path:
    return get_storage_dir() / "pdfs" / f"{doc_id}.pdf"


def _clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r"[^\x20-\x7E\n]", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_figures_enabled() -> bool:
    return os.getenv("EXTRACT_FIGURES", "").lower() in ("1", "true", "yes")


def _use_contextual_retrieval() -> bool:
    return os.getenv("CONTEXTUAL_RETRIEVAL", "").lower() in ("1", "true", "yes")


def _html_to_markdown(html: str) -> str:
    try:
        import markdownify
        return markdownify.markdownify(html, heading_style="ATX").strip()
    except Exception:
        return re.sub(r"<[^>]+>", " ", html).strip()


def _caption_figure(image_path: str) -> str:
    from rag.llm import get_llm

    path = Path(image_path)
    if not path.exists():
        return ""

    mime_type = _IMAGE_MIME.get(path.suffix.lower(), "image/png")
    with open(path, "rb") as fh:
        image_b64 = base64.b64encode(fh.read()).decode("utf-8")

    message = HumanMessage(
        content=[
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{image_b64}"},
            },
            {"type": "text", "text": _FIGURE_PROMPT},
        ]
    )
    try:
        return get_llm().invoke([message]).content.strip()
    except Exception:
        return ""


def extract_elements(pdf_path: Path, doc_id: str) -> list:
    kwargs: dict = {
        "filename": str(pdf_path),
        "infer_table_structure": True,
        "strategy": "hi_res",
    }
    if _extract_figures_enabled():
        figures_dir = Path(f"data/figures/{doc_id}/")
        figures_dir.mkdir(parents=True, exist_ok=True)
        kwargs["extract_images_in_pdf"] = True
        kwargs["extract_image_block_output_dir"] = str(figures_dir)

    return partition_pdf(**kwargs)


def _flush_text_buffer(
    buf: List[Tuple[int, str]],
    doc_id: str,
    source_name: str,
    contextual: bool,
    full_doc_text: str,
    chunk_id: int,
) -> Tuple[List[Document], int]:
    if not buf:
        return [], chunk_id

    page = buf[0][0]
    combined = "\n\n".join(text for _, text in buf)
    docs: List[Document] = []

    for c in _SPLITTER.split_text(combined):
        if not c.strip():
            continue
        ref = f"{doc_id}_p{page}_c{chunk_id}"
        embed_text = (
            f"{contextualize_chunk(full_doc_text, c)} {c}" if contextual else c
        )
        docs.append(
            Document(
                page_content=embed_text,
                metadata={
                    "doc_id": doc_id,
                    "ref": ref,
                    "page": page,
                    "chunk_id": chunk_id,
                    "source": source_name,
                    "element_type": "text",
                    "original_content": c,
                },
            )
        )
        chunk_id += 1

    return docs, chunk_id


def _build_docs_from_elements(
    elements: list,
    doc_id: str,
    source_name: str,
    contextual: bool,
    full_doc_text: str,
) -> List[Document]:
    all_docs: List[Document] = []
    chunk_id = 0
    figure_count = 0
    # Accumulate consecutive text elements per page before chunking so the
    # splitter sees dense prose blocks instead of isolated title/sentence stubs.
    text_buf: List[Tuple[int, str]] = []
    current_page: int = -1

    def flush() -> None:
        nonlocal chunk_id
        new_docs, chunk_id = _flush_text_buffer(
            text_buf, doc_id, source_name, contextual, full_doc_text, chunk_id
        )
        all_docs.extend(new_docs)
        text_buf.clear()

    for el in elements:
        page = el.metadata.page_number or 0

        if isinstance(el, Table):
            # Flush pending text before emitting table
            flush()
            current_page = page
            html = getattr(el.metadata, "text_as_html", None) or ""
            md_text = _html_to_markdown(html) if html else _clean_text(el.text or "")
            if not md_text.strip():
                continue
            ref = f"{doc_id}_p{page}_c{chunk_id}"
            embed_text = (
                f"{contextualize_chunk(full_doc_text, md_text)} {md_text}"
                if contextual
                else md_text
            )
            all_docs.append(
                Document(
                    page_content=embed_text,
                    metadata={
                        "doc_id": doc_id,
                        "ref": ref,
                        "page": page,
                        "chunk_id": chunk_id,
                        "source": source_name,
                        "element_type": "table",
                        "original_content": md_text,
                    },
                )
            )
            chunk_id += 1

        elif isinstance(el, UnstructuredImage):
            flush()
            current_page = page
            if not _extract_figures_enabled() or figure_count >= _MAX_FIGURES:
                continue
            image_path = (
                getattr(el.metadata, "image_path", None)
                or getattr(el.metadata, "filename", None)
            )
            if not image_path:
                continue
            caption = _caption_figure(str(image_path))
            if not caption:
                continue
            ref = f"{doc_id}_p{page}_fig{figure_count}"
            all_docs.append(
                Document(
                    page_content=caption,
                    metadata={
                        "doc_id": doc_id,
                        "ref": ref,
                        "page": page,
                        "chunk_id": chunk_id,
                        "source": source_name,
                        "element_type": "figure",
                        "image_path": str(image_path),
                        "original_content": caption,
                    },
                )
            )
            chunk_id += 1
            figure_count += 1

        else:
            text = _clean_text(el.text or "")
            if not text:
                continue
            # Flush on page change so page metadata stays accurate
            if page != current_page and text_buf:
                flush()
            current_page = page
            text_buf.append((page, text))

    flush()
    return all_docs


def index_document(doc_id: str) -> Tuple[int, str]:
    pdf_path = get_pdf_path(doc_id)
    if not pdf_path.exists():
        raise FileNotFoundError(str(pdf_path))

    # Purge any previous chunks so stale vectors from prior indexing runs
    # don't pollute retrieval when the chunk structure changes.
    clear_document(doc_id)

    source_name = pdf_path.name
    contextual = _use_contextual_retrieval()

    elements = extract_elements(pdf_path, doc_id)

    full_doc_text = (
        "\n\n".join(
            el.text
            for el in elements
            if not isinstance(el, UnstructuredImage) and el.text
        )
        if contextual
        else ""
    )
    all_docs = _build_docs_from_elements(
        elements, doc_id, source_name, contextual, full_doc_text
    )

    if not all_docs:
        return 0, DEFAULT_COLLECTION

    add_documents(doc_id, all_docs)
    save_bm25(doc_id, all_docs)
    return len(all_docs), DEFAULT_COLLECTION
