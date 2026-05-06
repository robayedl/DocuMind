from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from rag.store import get_chroma_dir, similarity_search, similarity_search_by_vector

logger = logging.getLogger(__name__)

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


def _rrf_merge(
    list_a: List[Document], list_b: List[Document], k: int
) -> List[Document]:
    """Merge two ranked document lists using Reciprocal Rank Fusion."""
    rrf_scores: dict[str, float] = {}
    for rank, doc in enumerate(list_a):
        ref = doc.metadata["ref"]
        rrf_scores[ref] = rrf_scores.get(ref, 0.0) + _rrf_score(rank)
    for rank, doc in enumerate(list_b):
        ref = doc.metadata["ref"]
        rrf_scores[ref] = rrf_scores.get(ref, 0.0) + _rrf_score(rank)

    all_docs: dict[str, Document] = {
        doc.metadata["ref"]: doc for doc in list_a + list_b
    }
    sorted_refs = sorted(rrf_scores, key=lambda r: rrf_scores[r], reverse=True)
    return [all_docs[ref] for ref in sorted_refs[:k]]


def hybrid_search(doc_id: str, query: str, k: int = 10) -> List[Document]:
    """BM25 + vector search fused with RRF. Falls back to pure vector if no BM25 index."""
    vector_results: List[Document] = similarity_search(doc_id, query, k=k)

    bm25_data = load_bm25(doc_id)
    if bm25_data is None:
        return vector_results

    bm25, corpus_docs = bm25_data
    scores = bm25.get_scores(query.lower().split())
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    bm25_results: List[Document] = [corpus_docs[i] for i in top_indices]

    return _rrf_merge(vector_results, bm25_results, k=k)


def _hyde_dense_search(doc_id: str, query: str, k: int) -> List[Document]:
    from rag.llm import get_embeddings, get_llm

    hypothetical = get_llm().invoke(
        f"Write a hypothetical 3-sentence passage that directly answers: {query}"
    ).content.strip()
    logger.info("[HyDE] hypothetical: %r", hypothetical[:120])
    return similarity_search_by_vector(doc_id, get_embeddings().embed_query(hypothetical), k=k)


def retrieve_with_hyde(doc_id: str, query: str, top_k: int = 5) -> tuple[List[Document], bool]:
    """Hybrid search + reranking. Triggers HyDE when top reranker score < HYDE_THRESHOLD.

    Returns (docs, hyde_triggered).
    """
    from rag.chains.rerank import rerank_with_score

    hyde_threshold = float(os.getenv("HYDE_THRESHOLD", "0.3"))
    candidate_k = top_k * 2

    candidates = hybrid_search(doc_id, query, k=candidate_k)
    if not candidates:
        return [], False

    docs, top_score = rerank_with_score(query, candidates, top_k=top_k)

    if top_score < hyde_threshold:
        logger.info("[HyDE] score=%.3f < %.3f, triggering for: %r", top_score, hyde_threshold, query)
        hyde_candidates = _hyde_dense_search(doc_id, query, k=candidate_k)
        merged = _rrf_merge(candidates, hyde_candidates, k=candidate_k)
        docs, _ = rerank_with_score(query, merged, top_k=top_k)
        return docs, True

    return docs, False
