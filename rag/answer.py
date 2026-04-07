from __future__ import annotations

from typing import List

from langchain_core.documents import Document

from rag.chains.generation import get_rag_chain


def make_answer(question: str, hits: List[Document]) -> str:
    """Generate a grounded LLM answer from retrieved documents."""
    chain = get_rag_chain()
    return chain.invoke({"input": question, "context": hits})
