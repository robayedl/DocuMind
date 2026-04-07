from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from rag.store import DEFAULT_COLLECTION, add_documents


def get_storage_dir() -> Path:
    return Path(os.getenv("STORAGE_DIR", "storage"))


def get_pdf_path(doc_id: str) -> Path:
    return get_storage_dir() / "pdfs" / f"{doc_id}.pdf"


def extract_pages(pdf_path: Path) -> List[str]:
    reader = PdfReader(str(pdf_path))
    return [p.extract_text() or "" for p in reader.pages]


def index_document(doc_id: str) -> Tuple[int, str]:
    pdf_path = get_pdf_path(doc_id)
    if not pdf_path.exists():
        raise FileNotFoundError(str(pdf_path))

    pages = extract_pages(pdf_path)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    all_docs: List[Document] = []
    source_name = pdf_path.name
    chunk_id = 0

    for page_idx, page_text in enumerate(pages, start=1):
        chunks = splitter.split_text(page_text)
        for c in chunks:
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
    return len(all_docs), DEFAULT_COLLECTION
