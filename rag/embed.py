from __future__ import annotations

from functools import lru_cache
from typing import List


@lru_cache(maxsize=1)
def _get_embedder():
    # 1) Try fastembed
    try:
        from fastembed import TextEmbedding

        return ("fastembed", TextEmbedding(model_name="BAAI/bge-small-en-v1.5"))
    except Exception:
        pass

    # 2) Try sentence-transformers
    try:
        from sentence_transformers import SentenceTransformer

        return ("st", SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2"))
    except Exception:
        pass

    raise RuntimeError(
        "No embedder available. Install one of: fastembed OR sentence-transformers."
    )


def embed_texts(texts: List[str]) -> List[List[float]]:
    kind, emb = _get_embedder()

    if kind == "fastembed":
        # fastembed returns generator of numpy arrays
        vectors = list(emb.embed(texts))
        return [v.tolist() for v in vectors]

    # sentence-transformers
    vectors = emb.encode(texts, normalize_embeddings=True)
    return [v.tolist() for v in vectors]