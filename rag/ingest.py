from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

from pypdf import PdfReader

from rag.embed import embed_texts
from rag.store import DEFAULT_COLLECTION, get_collection
from rag.text_clean import normalize_keep_lines


def get_storage_dir() -> Path:
    return Path(os.getenv("STORAGE_DIR", "storage"))


def get_pdf_path(doc_id: str) -> Path:
    return get_storage_dir() / "pdfs" / f"{doc_id}.pdf"


def extract_pages(pdf_path: Path) -> List[str]:
    reader = PdfReader(str(pdf_path))
    pages: List[str] = []
    for p in reader.pages:
        txt = p.extract_text() or ""
        pages.append(txt)
    return pages


def chunk_text_keep_lines(text: str, max_chars: int = 1200, overlap_lines: int = 3) -> List[str]:
    """
    Chunk by lines (keeps headings and section structure).
    """
    t = normalize_keep_lines(text)
    if not t:
        return []

    lines = t.split("\n")
    chunks: List[str] = []

    cur: List[str] = []
    cur_len = 0

    def flush():
        nonlocal cur, cur_len
        if cur:
            chunks.append("\n".join(cur).strip())
        cur = []
        cur_len = 0

    for line in lines:
        add_len = len(line) + 1
        if cur_len + add_len > max_chars and cur:
            flush()
            # overlap last few lines to keep continuity
            if overlap_lines > 0 and len(chunks) > 0:
                tail = chunks[-1].split("\n")[-overlap_lines:]
                cur = tail[:]  # start new chunk with overlap
                cur_len = sum(len(x) + 1 for x in cur)

        cur.append(line)
        cur_len += add_len

    flush()
    return [c for c in chunks if c]


def index_document(doc_id: str) -> Tuple[int, str]:
    pdf_path = get_pdf_path(doc_id)
    if not pdf_path.exists():
        raise FileNotFoundError(str(pdf_path))

    pages = extract_pages(pdf_path)

    all_docs: List[str] = []
    all_ids: List[str] = []
    all_metas: List[dict] = []

    chunk_id = 0
    source_name = pdf_path.name

    for page_idx, page_text in enumerate(pages, start=1):
        chunks = chunk_text_keep_lines(page_text)

        for c in chunks:
            ref = f"{doc_id}_p{page_idx}_c{chunk_id}"
            all_ids.append(ref)
            all_docs.append(c)  # IMPORTANT: keep lines
            all_metas.append(
                {
                    "doc_id": doc_id,
                    "ref": ref,
                    "page": page_idx,
                    "chunk_id": chunk_id,
                    "source": source_name,
                }
            )
            chunk_id += 1

    if not all_docs:
        return 0, DEFAULT_COLLECTION

    embeddings = embed_texts(all_docs)

    collection = get_collection(DEFAULT_COLLECTION)
    collection.upsert(
        ids=all_ids,
        documents=all_docs,
        metadatas=all_metas,
        embeddings=embeddings,
    )

    return len(all_docs), DEFAULT_COLLECTION