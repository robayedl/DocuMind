from __future__ import annotations


def test_llm_init(monkeypatch):
    """LLM initialises correctly when GOOGLE_API_KEY is present."""
    monkeypatch.setenv("GOOGLE_API_KEY", "test-api-key")
    from rag.llm import get_llm
    get_llm.cache_clear()
    try:
        llm = get_llm()
        assert "gemini-2.5-flash" in llm.model
    finally:
        get_llm.cache_clear()


def test_embeddings_dimension():
    """HuggingFace all-mpnet-base-v2 produces 768-dimensional vectors."""
    from rag.llm import get_embeddings
    embeddings = get_embeddings()
    vector = embeddings.embed_query("hello world")
    assert len(vector) == 768
