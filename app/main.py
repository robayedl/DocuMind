from __future__ import annotations

import json
import os
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator, List, Optional

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from langchain_core.documents import Document

from rag.agents.graph import run_agent
from rag import cache as semantic_cache
from rag.ingest import index_document
from rag.llm import get_embeddings, get_llm
from app.storage import new_doc_id, pdf_path

APP_ENV = os.getenv("APP_ENV", "local")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load the embedding model and LLM client at startup to avoid cold-start latency."""
    get_embeddings()
    try:
        get_llm()
    except RuntimeError:
        pass  # GOOGLE_API_KEY not set; LLM will fail at query time
    yield


app = FastAPI(title="DocuMind", lifespan=lifespan)

_cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:8501").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
    session_id: Optional[str] = Field(None, description="Optional session ID for conversation memory")


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
    retries: int
    latency_ms: float
    from_cache: bool = False
    hyde_triggered: bool = False


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


@app.get("/documents/{doc_id}/file")
def get_document_file(doc_id: str) -> FileResponse:
    path = pdf_path(doc_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Document not found.")
    return FileResponse(path, media_type="application/pdf", filename=path.name)


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    t0 = time.perf_counter()

    # 404 if document does not exist
    path = pdf_path(req.doc_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Document not found.")

    cached = semantic_cache.lookup(req.question)
    if cached:
        latency_ms = (time.perf_counter() - t0) * 1000.0
        raw_citations = cached.get("citations", [])
        citations = [
            Citation(
                ref=c.get("ref", ""),
                page=c.get("page", -1),
                chunk_id=c.get("chunk_id", -1),
                source=c.get("source", ""),
            )
            for c in raw_citations
        ]
        return QueryResponse(
            doc_id=req.doc_id,
            question=req.question,
            answer=cached["answer"],
            citations=citations,
            retrieved=len(citations),
            retries=0,
            latency_ms=round(latency_ms, 2),
            from_cache=True,
        )

    state = run_agent(
        question=req.question,
        doc_id=req.doc_id,
        session_id=req.session_id or "",
    )

    answer = state.get("generation", "")
    docs: List[Document] = state.get("documents", [])

    # If no answer was generated and no documents were found, doc likely not indexed
    if not answer and not docs:
        raise HTTPException(status_code=404, detail="Document not indexed.")

    citations = [
        Citation(
            ref=doc.metadata.get("ref", ""),
            page=doc.metadata.get("page", -1),
            chunk_id=doc.metadata.get("chunk_id", -1),
            source=doc.metadata.get("source", ""),
        )
        for doc in docs
    ]

    latency_ms = (time.perf_counter() - t0) * 1000.0

    if answer:
        semantic_cache.store(
            req.question,
            answer,
            [c.model_dump() for c in citations],
        )

    return QueryResponse(
        doc_id=req.doc_id,
        question=req.question,
        answer=answer,
        citations=citations,
        retrieved=len(docs),
        retries=state.get("retry_count", 0),
        latency_ms=round(latency_ms, 2),
        hyde_triggered=state.get("hyde_triggered", False),
    )


class StreamQueryRequest(BaseModel):
    doc_id: str = Field(..., min_length=1)
    question: str = Field(..., min_length=2)
    session_id: Optional[str] = Field(None)


@app.post("/query/stream")
async def query_stream(req: StreamQueryRequest) -> StreamingResponse:
    path = pdf_path(req.doc_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Document not found.")

    async def event_stream() -> AsyncIterator[str]:
        try:
            import asyncio

            cached = semantic_cache.lookup(req.question)
            if cached:
                yield _sse("status", "Cache hit — returning cached answer…")
                answer: str = cached["answer"]
                words = answer.split(" ")
                for i, word in enumerate(words):
                    token = word if i == 0 else " " + word
                    yield _sse("token", token)
                    await asyncio.sleep(0.008)
                yield _sse("citations", json.dumps(cached.get("citations", [])))
                yield _sse("done", "")
                return

            _STATUS_STEPS = [
                "Routing your question…",
                "Searching the document…",
                "Reading relevant sections…",
                "Grading document quality…",
                "Generating answer…",
                "Checking for accuracy…",
            ]

            loop = asyncio.get_event_loop()
            future = loop.run_in_executor(
                None,
                lambda: run_agent(
                    question=req.question,
                    doc_id=req.doc_id,
                    session_id=req.session_id or "",
                ),
            )

            # Emit status messages every 2.5 s while the agent runs
            step = 0
            yield _sse("status", _STATUS_STEPS[step])
            step += 1
            while not future.done():
                await asyncio.sleep(2.5)
                if not future.done() and step < len(_STATUS_STEPS):
                    yield _sse("status", _STATUS_STEPS[step])
                    step += 1

            state = await future

            answer = state.get("generation", "")
            docs: List[Document] = state.get("documents", [])

            if not answer and not docs:
                yield _sse("error", "Document not indexed.")
                return

            if not answer:
                yield _sse("error", "Could not generate an answer. Try rephrasing your question.")
                return

            # Stream the answer word by word so tokens appear progressively
            words = answer.split(" ")
            for i, word in enumerate(words):
                token = word if i == 0 else " " + word
                yield _sse("token", token)
                await asyncio.sleep(0.018)

            citations_data = [
                {
                    "ref": d.metadata.get("ref", ""),
                    "page": d.metadata.get("page", -1),
                    "chunk_id": d.metadata.get("chunk_id", -1),
                    "source": d.metadata.get("source", ""),
                    "text": d.page_content[:200],
                }
                for d in docs
            ]
            yield _sse("citations", json.dumps(citations_data))
            yield _sse("meta", json.dumps({"hyde_triggered": state.get("hyde_triggered", False)}))
            yield _sse("done", "")

            if answer:
                semantic_cache.store(req.question, answer, citations_data)

        except Exception as e:
            yield _sse("error", str(e))

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def _sse(event: str, data: str) -> str:
    return f"event: {event}\ndata: {data}\n\n"
