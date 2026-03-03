from __future__ import annotations

from typing import Any, Dict, List, Tuple

from rag.store import get_collection


def retrieve_top_k(doc_id: str, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Retrieve top-k chunks for a given doc_id using a query embedding.
    Returns list of: {"text": ..., "metadata": {...}, "distance": ...}
    """
    col = get_collection()

    res = col.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where={"doc_id": doc_id},
        include=["documents", "metadatas", "distances"],
    )

    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]

    out: List[Dict[str, Any]] = []
    for i in range(min(len(docs), len(metas), len(dists))):
        out.append({"text": docs[i], "metadata": metas[i], "distance": dists[i]})
    return out


def build_cited_answer(question: str, retrieved: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Deterministic answer builder (non-LLM).
    - Answer: returns a clean extractive answer (optionally formatted)
    - Citations: list of {ref, page, chunk_id, source}
    """
    if not retrieved:
        return "I could not find relevant information in the document.", []

    top = retrieved[0]
    text = (top.get("text") or "").strip()

    q = question.lower()

    # If user asks about "skills", try to present a neat bullet list
    if "skill" in q:
        # Heuristic: split by common separators; resumes often have commas or bullets
        candidates: List[str] = []
        if "•" in text:
            candidates = [p.strip(" •-\t") for p in text.split("•") if p.strip()]
        else:
            # fallback: try commas
            parts = [p.strip() for p in text.split(",") if p.strip()]
            # only use comma split if it looks reasonable
            if 4 <= len(parts) <= 25:
                candidates = parts

        if len(candidates) >= 3:
            bullets = candidates[:10]
            answer = "Main skills (from the document):\n- " + "\n- ".join(bullets)
        else:
            answer = text
    else:
        answer = text

    # Trim very long answers
    if len(answer) > 900:
        answer = answer[:900].rstrip() + "..."

    citations: List[Dict[str, Any]] = []
    for r in retrieved:
        m = r.get("metadata") or {}
        doc_id = m.get("doc_id")
        page = m.get("page")
        chunk_id = m.get("chunk_id")

        citations.append(
            {
                "ref": f"{doc_id}_p{page}_c{chunk_id}",
                "page": page,
                "chunk_id": chunk_id,
                "source": m.get("source"),
            }
        )

    return answer, citations