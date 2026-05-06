from __future__ import annotations

import os
from pathlib import Path
from typing import List

from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from rag.llm import get_embeddings

# Bump this when the embedding model changes to force a clean rebuild
EMBEDDING_VERSION = "v2"
DEFAULT_COLLECTION = f"pdf_chunks_{EMBEDDING_VERSION}"


def get_chroma_dir() -> str:
    return os.getenv("CHROMA_DIR", "chroma_db")


def get_vectorstore(doc_id: str | None = None) -> Chroma:
    chroma_dir = get_chroma_dir()
    Path(chroma_dir).mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=DEFAULT_COLLECTION,
        embedding_function=get_embeddings(),
        persist_directory=chroma_dir,
        collection_metadata={"hnsw:space": "cosine"},
        client_settings=Settings(anonymized_telemetry=False),
    )


def add_documents(doc_id: str, documents: List[Document]) -> None:
    vs = get_vectorstore()
    ids = [doc.metadata["ref"] for doc in documents]
    vs.add_documents(documents, ids=ids)


def similarity_search(doc_id: str, query: str, k: int = 5) -> List[Document]:
    vs = get_vectorstore()
    return vs.similarity_search(query, k=k, filter={"doc_id": doc_id})


def similarity_search_by_vector(doc_id: str, embedding: List[float], k: int = 5) -> List[Document]:
    vs = get_vectorstore()
    return vs.similarity_search_by_vector(embedding, k=k, filter={"doc_id": doc_id})
