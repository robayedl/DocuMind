from __future__ import annotations

import time
from statistics import median

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.config import settings
from app.storage import new_doc_id, pdf_path

from rag.ingest import extract_pages, chunk_text
from rag.embed import embed_texts
from rag.store import get_collection
from rag.retrieve import retrieve_top_k, build_cited_answer

app = FastAPI(title=settings.app_name)


@app.get("/health")
def health():
    return {"status": "ok", "environment": settings.environment}


@app.post("/documents")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    if file.content_type not in {"application/pdf", "application/x-pdf"}:
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    head = await file.read(5)
    if head != b"%PDF-":
        raise HTTPException(status_code=400, detail="Invalid PDF file")

    rest = await file.read()
    content = head + rest

    doc_id = new_doc_id()
    path = pdf_path(doc_id)
    path.write_bytes(content)

    return {"doc_id": doc_id, "filename": file.filename, "stored_path": str(path)}


@app.post("/documents/{doc_id}/index")
def index_document(doc_id: str):
    path = pdf_path(doc_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Document not found")

    t0 = time.time()
    print("INDEX: start", doc_id, flush=True)

    pages = extract_pages(str(path))
    print("INDEX: extracted pages", len(pages), "elapsed", round(time.time() - t0, 2), "s", flush=True)

    chunks = chunk_text(pages)
    print("INDEX: chunks", len(chunks), "elapsed", round(time.time() - t0, 2), "s", flush=True)

    if not chunks:
        raise HTTPException(status_code=400, detail="No extractable text found in PDF")

    texts = [c.text for c in chunks]
    print("INDEX: embedding...", len(texts), "chunks", flush=True)

    embeddings = embed_texts(texts)
    print("INDEX: embedded", len(embeddings), "elapsed", round(time.time() - t0, 2), "s", flush=True)

    ids = [f"{doc_id}_p{c.page}_c{c.chunk_id}" for c in chunks]
    metadatas = [{"doc_id": doc_id, "page": c.page, "chunk_id": c.chunk_id, "source": path.name} for c in chunks]

    col = get_collection()
    print("INDEX: upserting to chroma...", flush=True)

    col.upsert(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)
    print("INDEX: done", "elapsed", round(time.time() - t0, 2), "s", flush=True)

    return {"doc_id": doc_id, "chunks_indexed": len(chunks), "collection": col.name}


class QueryRequest(BaseModel):
    doc_id: str
    question: str = Field(min_length=3)
    top_k: int = Field(default=5, ge=1, le=20)


class QueryResponse(BaseModel):
    doc_id: str
    question: str
    answer: str
    citations: list[dict]
    retrieved: int
    latency_ms: float


@app.post("/query", response_model=QueryResponse)
def query_rag(payload: QueryRequest):
    path = pdf_path(payload.doc_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Document not found")

    t0 = time.time()

    q_emb = embed_texts([payload.question])[0]
    retrieved = retrieve_top_k(payload.doc_id, q_emb, top_k=payload.top_k)
    answer, citations = build_cited_answer(payload.question, retrieved)

    latency_ms = (time.time() - t0) * 1000.0

    return QueryResponse(
        doc_id=payload.doc_id,
        question=payload.question,
        answer=answer,
        citations=citations,
        retrieved=len(retrieved),
        latency_ms=round(latency_ms, 2),
    )


class BenchRequest(BaseModel):
    doc_id: str
    question: str = Field(min_length=3)
    top_k: int = Field(default=5, ge=1, le=20)
    runs: int = Field(default=20, ge=5, le=200)
    warmup: int = Field(default=2, ge=0, le=10)


class BenchResponse(BaseModel):
    doc_id: str
    question: str
    top_k: int
    runs: int
    warmup: int
    p50_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float


def _percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * p
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)


@app.post("/bench", response_model=BenchResponse)
def bench(payload: BenchRequest):
    path = pdf_path(payload.doc_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Document not found")

    # Warmup
    for _ in range(payload.warmup):
        q_emb = embed_texts([payload.question])[0]
        _ = retrieve_top_k(payload.doc_id, q_emb, top_k=payload.top_k)

    times: list[float] = []
    for _ in range(payload.runs):
        t0 = time.time()
        q_emb = embed_texts([payload.question])[0]
        _ = retrieve_top_k(payload.doc_id, q_emb, top_k=payload.top_k)
        times.append((time.time() - t0) * 1000.0)

    times_sorted = sorted(times)

    return BenchResponse(
        doc_id=payload.doc_id,
        question=payload.question,
        top_k=payload.top_k,
        runs=payload.runs,
        warmup=payload.warmup,
        p50_ms=round(_percentile(times_sorted, 0.50), 2),
        p95_ms=round(_percentile(times_sorted, 0.95), 2),
        min_ms=round(times_sorted[0], 2),
        max_ms=round(times_sorted[-1], 2),
    )