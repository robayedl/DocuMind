from __future__ import annotations

import os
import re
import unicodedata
from pathlib import Path
from typing import List, Tuple

import pdfplumber
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.chains.retrieval import save_bm25
from rag.store import DEFAULT_COLLECTION, add_documents

_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def get_storage_dir() -> Path:
    return Path(os.getenv("STORAGE_DIR", "storage"))


def get_pdf_path(doc_id: str) -> Path:
    return get_storage_dir() / "pdfs" / f"{doc_id}.pdf"


def _clean_text(text: str) -> str:
    """Normalise PDF-extracted text: fix ligatures, strip junk, normalise whitespace."""
    # Decompose Unicode ligatures (ﬁ→fi, ﬂ→fl, ﬀ→ff, etc.)
    text = unicodedata.normalize("NFKD", text)
    # Remove non-printable / control characters (keep printable ASCII + newlines)
    text = re.sub(r"[^\x20-\x7E\n]", " ", text)
    # Collapse runs of spaces/tabs to a single space
    text = re.sub(r"[ \t]+", " ", text)
    # Collapse 3+ consecutive newlines to a paragraph break
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pages(pdf_path: Path) -> List[Tuple[int, str]]:
    """Return (page_number, cleaned_text) pairs via pdfplumber."""
    pages: List[Tuple[int, str]] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            raw = page.extract_text(x_tolerance=2, y_tolerance=2) or ""
            cleaned = _clean_text(raw)
            if cleaned:
                pages.append((i, cleaned))
    return pages


def index_document(doc_id: str) -> Tuple[int, str]:
    pdf_path = get_pdf_path(doc_id)
    if not pdf_path.exists():
        raise FileNotFoundError(str(pdf_path))

    pages = extract_pages(pdf_path)

    all_docs: List[Document] = []
    source_name = pdf_path.name
    chunk_id = 0

    for page_idx, page_text in pages:
        chunks = _SPLITTER.split_text(page_text)
        for c in chunks:
            if not c.strip():
                continue
            ref = f"{doc_id}_p{page_idx}_c{chunk_id}"
            all_docs.append(
                Document(
                    page_content=c,
                    metadata={
                        "doc_id": doc_id,
                        "ref": ref,
                        "page": page_idx,
                        "chunk_id": chunk_id,
                        "source": source_name,
                    },
                )
            )
            chunk_id += 1

    if not all_docs:
        return 0, DEFAULT_COLLECTION

    add_documents(doc_id, all_docs)
    save_bm25(doc_id, all_docs)
    return len(all_docs), DEFAULT_COLLECTION
