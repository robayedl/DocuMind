from __future__ import annotations

from typing import List

from langchain_core.documents import Document

from rag.store import DEFAULT_COLLECTION, similarity_search


def retrieve_top_k(
    doc_id: str,
    question: str,
    top_k: int = 5,
    collection_name: str = DEFAULT_COLLECTION,
) -> List[Document]:
    return similarity_search(doc_id, question, k=top_k)
