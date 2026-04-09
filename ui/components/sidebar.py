from __future__ import annotations

import requests
import streamlit as st

API_BASE = "http://localhost:8000"


def render_sidebar() -> None:
    st.sidebar.title("RAG PDF Assistant")

    # ── Upload ────────────────────────────────────────────────────────────────
    uploaded_file = st.sidebar.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded_file is not None:
        already_uploaded = any(
            d["filename"] == uploaded_file.name
            for d in st.session_state.documents
        )
        if not already_uploaded:
            _upload_and_index(uploaded_file)

    # ── Document list ─────────────────────────────────────────────────────────
    if st.session_state.documents:
        st.sidebar.divider()
        st.sidebar.subheader("Your Documents")

        doc_names = [d["filename"] for d in st.session_state.documents]
        current_name = next(
            (d["filename"] for d in st.session_state.documents
             if d["doc_id"] == st.session_state.current_doc_id),
            doc_names[0],
        ) if st.session_state.current_doc_id else doc_names[0]

        selected_name = st.sidebar.radio(
            "Select document to query",
            doc_names,
            index=doc_names.index(current_name),
            label_visibility="collapsed",
        )

        selected_doc = next(d for d in st.session_state.documents if d["filename"] == selected_name)
        if selected_doc["doc_id"] != st.session_state.current_doc_id:
            st.session_state.current_doc_id = selected_doc["doc_id"]
            st.session_state.chat_history = []
            st.rerun()

        st.sidebar.caption(
            f"**File:** {selected_doc['filename']}  \n"
            f"**Chunks indexed:** {selected_doc['chunks_indexed']}"
        )

        # ── Clear chat ────────────────────────────────────────────────────────
        st.sidebar.divider()
        if st.sidebar.button("Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    # ── About ─────────────────────────────────────────────────────────────────
    st.sidebar.divider()
    st.sidebar.caption(
        "**About this app**  \n"
        "Upload any PDF and ask questions about it in plain English. "
        "The assistant reads only your document — it won't guess or make things up.  \n\n"
        "*Powered by Gemini 2.5 Flash · LangGraph · Hybrid Search*  \n\n"
        "© 2026 RAG PDF Assistant"
    )


def _upload_and_index(uploaded_file) -> None:
    progress = st.sidebar.progress(0, text="Uploading…")
    try:
        upload_resp = requests.post(
            f"{API_BASE}/documents",
            files={"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")},
            timeout=30,
        )
        upload_resp.raise_for_status()
        doc_id = upload_resp.json()["doc_id"]
    except requests.ConnectionError:
        progress.empty()
        st.sidebar.error("Cannot connect to backend. Is the FastAPI server running on port 8000?")
        return
    except Exception as e:
        progress.empty()
        st.sidebar.error(f"Upload failed: {e}")
        return

    progress.progress(40, text="Indexing…")
    try:
        index_resp = requests.post(
            f"{API_BASE}/documents/{doc_id}/index",
            timeout=120,
        )
        index_resp.raise_for_status()
        chunks_indexed = index_resp.json()["chunks_indexed"]
    except Exception as e:
        progress.empty()
        st.sidebar.error(f"Indexing failed: {e}")
        return

    progress.progress(100, text="Done!")
    progress.empty()

    st.session_state.documents.append({
        "doc_id": doc_id,
        "filename": uploaded_file.name,
        "chunks_indexed": chunks_indexed,
    })
    st.session_state.current_doc_id = doc_id
    st.session_state.chat_history = []
    st.sidebar.success(f"Indexed {chunks_indexed} chunks.")
    st.rerun()
