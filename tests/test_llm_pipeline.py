from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def test_llm_init(monkeypatch):
    """LLM initialises correctly when GOOGLE_API_KEY is present."""
    monkeypatch.setenv("GOOGLE_API_KEY", "test-api-key")
    from rag.llm import get_llm
    get_llm.cache_clear()
    try:
        llm = get_llm()
        assert llm.model == "gemini-2.5-flash"
    finally:
        get_llm.cache_clear()


def test_embeddings_dimension():
    """HuggingFace all-MiniLM-L6-v2 produces 384-dimensional vectors."""
    from rag.llm import get_embeddings
    embeddings = get_embeddings()
    vector = embeddings.embed_query("hello world")
    assert len(vector) == 384


def test_ingest_and_retrieve(tmp_path, monkeypatch):
    """Ingest a document and retrieve relevant chunks from ChromaDB."""
    monkeypatch.setenv("STORAGE_DIR", str(tmp_path))
    monkeypatch.setenv("CHROMA_DIR", str(tmp_path / "chroma"))

    doc_id = "test_doc_001"
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir(parents=True)
    (pdf_dir / f"{doc_id}.pdf").write_bytes(b"placeholder")

    fake_text = (
        "Artificial intelligence is the simulation of human intelligence in machines. "
        "Machine learning is a subset of AI that enables systems to learn automatically from data."
    )

    with patch("rag.ingest.extract_pages", return_value=[fake_text]):
        from rag.ingest import index_document
        n, collection = index_document(doc_id)

    assert n > 0
    assert collection == "pdf_chunks"

    from rag.retrieve import retrieve_top_k
    docs = retrieve_top_k(doc_id, "What is machine learning?", top_k=2)

    assert len(docs) > 0
    assert all(hasattr(d, "page_content") for d in docs)
    assert all(d.metadata.get("doc_id") == doc_id for d in docs)


def test_full_pipeline_with_mocked_llm(tmp_path, monkeypatch):
    """Full pipeline: ingest → retrieve → LLM answer (LLM mocked)."""
    monkeypatch.setenv("STORAGE_DIR", str(tmp_path))
    monkeypatch.setenv("CHROMA_DIR", str(tmp_path / "chroma2"))
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

    doc_id = "pipeline_test_doc"
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir(parents=True)
    (pdf_dir / f"{doc_id}.pdf").write_bytes(b"placeholder")

    fake_text = (
        "Deep learning uses neural networks with many layers to learn "
        "complex representations directly from raw data."
    )

    with patch("rag.ingest.extract_pages", return_value=[fake_text]):
        from rag.ingest import index_document
        index_document(doc_id)

    from rag.retrieve import retrieve_top_k
    docs = retrieve_top_k(doc_id, "What is deep learning?", top_k=2)
    assert len(docs) > 0

    mock_chain = MagicMock()
    mock_chain.invoke.return_value = "Deep learning uses neural networks to learn from data."

    with patch("rag.answer.get_rag_chain", return_value=mock_chain):
        from rag.answer import make_answer
        answer = make_answer("What is deep learning?", docs)

    assert isinstance(answer, str)
    assert len(answer) > 0
    assert "deep learning" in answer.lower()
