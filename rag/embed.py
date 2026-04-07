from __future__ import annotations

from typing import List

from rag.llm import get_embeddings


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed a list of texts using LangChain HuggingFace embeddings."""
    embeddings = get_embeddings()
    return embeddings.embed_documents(texts)
