from __future__ import annotations

from dataclasses import dataclass
from typing import List

from rag.embed import embed_texts
from rag.store import DEFAULT_COLLECTION, get_collection
from rag.text_clean import normalize_one_line


@dataclass(frozen=True)
class RetrievedChunk:
    ref: str
    page: int
    chunk_id: int
    source: str
    text: str          # original (keeps \n)
    distance: float


def retrieve_top_k(doc_id: str, question: str, top_k: int = 5, collection_name: str = DEFAULT_COLLECTION) -> List[RetrievedChunk]:
    collection = get_collection(collection_name)

    q_emb = embed_texts([question])[0]

    res = collection.query(
        query_embeddings=[q_emb],
        n_results=top_k,
        where={"doc_id": doc_id},
        include=["documents", "metadatas", "distances"],
    )

    docs = res.get("documents", [[]])[0] or []
    metas = res.get("metadatas", [[]])[0] or []
    dists = res.get("distances", [[]])[0] or []

    hits: List[RetrievedChunk] = []
    for i in range(min(len(docs), len(metas), len(dists))):
        md = metas[i] or {}
        raw_text = docs[i] or ""

        hits.append(
            RetrievedChunk(
                ref=str(md.get("ref", f"{doc_id}_unknown_{i}")),
                page=int(md.get("page", -1)),
                chunk_id=int(md.get("chunk_id", i)),
                source=str(md.get("source", f"{doc_id}.pdf")),
                text=raw_text,  # keep original lines
                distance=float(dists[i]),
            )
        )

    return hits