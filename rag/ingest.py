from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

from pypdf import PdfReader


@dataclass
class Chunk:
    text: str
    page: int
    chunk_id: int


def normalize_text(s: str) -> str:
    """
    Clean PDF-extracted text to avoid broken spacing/newlines.
    """
    s = s.replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def extract_pages(pdf_path: str) -> List[str]:
    """
    Extract text from each page of a PDF as a list[str].
    """
    reader = PdfReader(pdf_path)
    pages: List[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(normalize_text(text))
    return pages


def chunk_text(
    pages: List[str],
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
) -> List[Chunk]:
    """
    Safe character-based chunking that always makes progress.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks: List[Chunk] = []
    step = chunk_size - chunk_overlap

    for page_idx, page_text in enumerate(pages, start=1):
        text = (page_text or "").strip()
        if not text:
            continue

        chunk_id = 0
        for start in range(0, len(text), step):
            end = min(start + chunk_size, len(text))
            piece = text[start:end].strip()
            if piece:
                chunks.append(Chunk(text=piece, page=page_idx, chunk_id=chunk_id))
                chunk_id += 1

            if end >= len(text):
                break

    return chunks