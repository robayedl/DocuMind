from __future__ import annotations

import os
import time
from pathlib import Path
from typing import List

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from rag.answer import make_answer
from rag.ingest import index_document
from rag.llm import get_embeddings, get_llm
from rag.retrieve import retrieve_top_k, RetrievedChunk
from app.storage import new_doc_id, pdf_path

APP_ENV = os.getenv("APP_ENV", "local")

app = FastAPI(title="rag-pdf-assistant")


@app.on_event("startup")
def _warm_up():
    """Pre-load the embedding model and LLM client at startup to avoid cold-start latency."""
    get_embeddings()
    get_llm()


# ==============================
# Response / Request Models
# ==============================

class HealthResponse(BaseModel):
    status: str = "ok"
    environment: str = APP_ENV


class UploadResponse(BaseModel):
    doc_id: str
    filename: str
    stored_path: str


class IndexResponse(BaseModel):
    doc_id: str
    chunks_indexed: int
    collection: str


class QueryRequest(BaseModel):
    doc_id: str = Field(..., min_length=1)
    question: str = Field(..., min_length=2)  # IMPORTANT (test requires 422 for short question)
    top_k: int = Field(5, ge=1, le=20)


class Citation(BaseModel):
    ref: str
    page: int
    chunk_id: int
    source: str


class QueryResponse(BaseModel):
    doc_id: str
    question: str
    answer: str
    citations: List[Citation]
    retrieved: int
    latency_ms: float


# ==============================
# Routes
# ==============================

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse()


@app.post("/documents", response_model=UploadResponse)
def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    doc_id = new_doc_id()
    out_path = pdf_path(doc_id)

    content = file.file.read()
    out_path.write_bytes(content)

    return UploadResponse(
        doc_id=doc_id,
        filename=file.filename,
        stored_path=str(out_path),
    )


@app.post("/documents/{doc_id}/index", response_model=IndexResponse)
def index(doc_id: str) -> IndexResponse:
    # 404 if document does not exist
    path = pdf_path(doc_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Document not found.")

    chunks_indexed, collection_name = index_document(doc_id)

    return IndexResponse(
        doc_id=doc_id,
        chunks_indexed=chunks_indexed,
        collection=collection_name,
    )


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    t0 = time.perf_counter()

    # 404 if document does not exist
    path = pdf_path(req.doc_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Document not found.")

    hits: List[RetrievedChunk] = retrieve_top_k(
        doc_id=req.doc_id,
        question=req.question,
        top_k=req.top_k,
    )

    # If document exists but not indexed → treat as 404 (tests expect this)
    if not hits:
        raise HTTPException(status_code=404, detail="Document not indexed.")

    answer = make_answer(req.question, hits)

    if not answer:
        raise HTTPException(status_code=404, detail="No relevant content found.")

    citations = [
        Citation(
            ref=h.ref,
            page=h.page,
            chunk_id=h.chunk_id,
            source=h.source,
        )
        for h in hits
    ]

    latency_ms = (time.perf_counter() - t0) * 1000.0

    return QueryResponse(
        doc_id=req.doc_id,
        question=req.question,
        answer=answer,
        citations=citations,
        retrieved=len(hits),
        latency_ms=round(latency_ms, 2),
    )