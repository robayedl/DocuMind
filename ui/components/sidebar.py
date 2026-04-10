from __future__ import annotations

import os

import requests
import streamlit as st

API_BASE = os.getenv("BACKEND_URL", "http://localhost:8000")


def render_sidebar() -> None:
    st.sidebar.markdown("## 📂 Documents")

    # ── Upload ────────────────────────────────────────────────────────────────
    uploaded_file = st.sidebar.file_uploader(
        "Upload a PDF to get started",
        type=["pdf"],
        label_visibility="collapsed",
        help="Only PDF files are supported.",
    )

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

        doc_names = [d["filename"] for d in st.session_state.documents]
        current_name = next(
            (d["filename"] for d in st.session_state.documents
             if d["doc_id"] == st.session_state.current_doc_id),
            doc_names[0],
        ) if st.session_state.current_doc_id else doc_names[0]

        st.sidebar.markdown("**Active document**")
        selected_name = st.sidebar.radio(
            "Select document",
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
            f"🗂 {selected_doc['chunks_indexed']} chunks indexed"
        )

        st.sidebar.divider()
        if st.sidebar.button("🗑 Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    # ── About ─────────────────────────────────────────────────────────────────
    st.sidebar.divider()
    st.sidebar.markdown(
        "<div style='font-size:0.78rem; color:#8892a4; line-height:1.7'>"
        "Upload any PDF and ask questions in plain English.<br>"
        "Answers are grounded in your document only — no guessing.<br><br>"
        "<b>Powered by</b><br>"
        "Gemini 2.5 Flash &nbsp;·&nbsp; LangGraph<br>"
        "Hybrid Search &nbsp;·&nbsp; Cross-Encoder Reranking<br><br>"
        "© 2026 DocuMind"
        "</div>",
        unsafe_allow_html=True,
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

    progress.progress(40, text="Indexing chunks…")
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

    progress.progress(100, text="Ready!")
    progress.empty()

    st.session_state.documents.append({
        "doc_id": doc_id,
        "filename": uploaded_file.name,
        "chunks_indexed": chunks_indexed,
    })
    st.session_state.current_doc_id = doc_id
    st.session_state.chat_history = []
    st.sidebar.success(f"✓ Indexed {chunks_indexed} chunks")
    st.rerun()
