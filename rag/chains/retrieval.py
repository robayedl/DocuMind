from __future__ import annotations

import pickle
from pathlib import Path
from typing import List, Optional, Tuple

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from rag.store import get_chroma_dir, similarity_search

RRF_K = 60  # standard RRF constant


def _bm25_path(doc_id: str) -> Path:
    return Path(get_chroma_dir()) / f"bm25_{doc_id}.pkl"


def load_bm25(doc_id: str) -> Optional[Tuple[BM25Okapi, List[Document]]]:
    """Load a persisted BM25 index for a document. Returns None if not found."""
    path = _bm25_path(doc_id)
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def save_bm25(doc_id: str, documents: List[Document]) -> None:
    """Build and persist a BM25 index for a list of documents."""
    corpus = [doc.page_content.lower().split() for doc in documents]
    bm25 = BM25Okapi(corpus)
    path = _bm25_path(doc_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump((bm25, documents), f)


def _rrf_score(rank: int) -> float:
    return 1.0 / (RRF_K + rank + 1)


def hybrid_search(doc_id: str, query: str, k: int = 10) -> List[Document]:
    """
    Combine vector search and BM25 keyword search using Reciprocal Rank Fusion.

    RRF score = sum of 1/(k + rank) across both ranked lists.
    Falls back to pure vector search if no BM25 index exists.
    """
    # ── Vector search ──────────────────────────────────────────────
    vector_results: List[Document] = similarity_search(doc_id, query, k=k)

    # ── BM25 search ────────────────────────────────────────────────
    bm25_data = load_bm25(doc_id)
    if bm25_data is None:
        return vector_results

    bm25, corpus_docs = bm25_data
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    bm25_results: List[Document] = [corpus_docs[i] for i in top_indices]

    # ── Reciprocal Rank Fusion ──────────────────────────────────────
    rrf_scores: dict[str, float] = {}

    for rank, doc in enumerate(vector_results):
        ref = doc.metadata["ref"]
        rrf_scores[ref] = rrf_scores.get(ref, 0.0) + _rrf_score(rank)

    for rank, doc in enumerate(bm25_results):
        ref = doc.metadata["ref"]
        rrf_scores[ref] = rrf_scores.get(ref, 0.0) + _rrf_score(rank)

    # Collect all unique docs and sort by combined RRF score
    all_docs: dict[str, Document] = {
        doc.metadata["ref"]: doc
        for doc in vector_results + bm25_results
    }
    sorted_refs = sorted(rrf_scores, key=lambda r: rrf_scores[r], reverse=True)

    return [all_docs[ref] for ref in sorted_refs[:k]]
