from __future__ import annotations

from typing import List

from langchain_core.documents import Document

from rag.retrieve import RetrievedChunk
from rag.chains.generation import get_rag_chain


def make_answer(question: str, hits: List[RetrievedChunk]) -> str:
    """Generate a grounded LLM answer from retrieved chunks."""
    docs = [
        Document(
            page_content=h.text,
            metadata={
                "ref": h.ref,
                "page": h.page,
                "chunk_id": h.chunk_id,
                "source": h.source,
            },
        )
        for h in hits
    ]

    chain = get_rag_chain()
    return chain.invoke({"input": question, "context": docs})
