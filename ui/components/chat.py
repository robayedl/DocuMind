from __future__ import annotations

import json
import os

import requests
import streamlit as st

API_BASE = os.getenv("BACKEND_URL", "http://localhost:8000")

_WELCOME = """
**Getting started**

1. Upload a PDF using the sidebar on the left
2. Wait a moment while it indexes
3. Type your question below — the assistant will answer using only your document
4. Ask follow-up questions; the assistant remembers the conversation
5. Use **Clear Chat** in the sidebar to start fresh
"""


def render_chat() -> None:
    if not st.session_state.current_doc_id:
        # ── Welcome screen ────────────────────────────────────────────────────
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #1a2035 0%, #1e2d40 100%);
            border: 1px solid #2d4a6e;
            border-radius: 14px;
            padding: 2rem 2.5rem;
            max-width: 600px;
            margin: 2rem auto;
        ">
            <h3 style="color:#7eb8f7; margin-top:0;">Welcome to DocuMind 🧠</h3>
            <p style="color:#c0cad8; font-size:0.95rem; line-height:1.7">
                Upload a PDF from the sidebar to start a conversation with your document.<br><br>
                This assistant uses <b>hybrid search</b> (semantic + keyword),
                <b>cross-encoder reranking</b>, and <b>Gemini 2.5 Flash</b> to give you
                precise, document-grounded answers.
            </p>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("How to use", expanded=False):
            st.markdown(_WELCOME)
        return

    selected_doc = next(
        (d for d in st.session_state.documents if d["doc_id"] == st.session_state.current_doc_id),
        None,
    )
    if selected_doc:
        st.markdown(
            f"<div style='font-size:0.82rem; color:#8892a4; margin-bottom:0.5rem'>"
            f"📄 Chatting about <b style='color:#a0aec0'>{selected_doc['filename']}</b>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # ── Chat history ──────────────────────────────────────────────────────────
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                _render_sources(msg["sources"])

    # ── Input ─────────────────────────────────────────────────────────────────
    user_input = st.chat_input("Ask anything about your document…")
    if not user_input:
        return

    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        try:
            answer, sources = _stream_query(user_input)
        except requests.ConnectionError:
            st.error("Cannot connect to backend. Is the FastAPI server running on port 8000?")
            st.session_state.chat_history.pop()
            return
        except Exception as e:
            error_msg = str(e)
            if "GOOGLE_API_KEY" in error_msg or "api key" in error_msg.lower():
                st.error(
                    "LLM API key is not configured. "
                    "Set GOOGLE_API_KEY in your .env file and restart the backend."
                )
            else:
                st.error(f"Error: {error_msg}")
            st.session_state.chat_history.pop()
            return

        if sources:
            _render_sources(sources)

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })


def _stream_query(question: str):
    sources = []
    status_placeholder = st.empty()
    answer_placeholder = st.empty()
    full_answer = ""
    first_token = True

    with requests.post(
        f"{API_BASE}/query/stream",
        json={
            "doc_id": st.session_state.current_doc_id,
            "question": question,
            "session_id": st.session_state.session_id,
        },
        stream=True,
        timeout=120,
    ) as resp:
        if resp.status_code != 200:
            detail = resp.json().get("detail", resp.text)
            raise RuntimeError(detail)

        event = ""
        for line in resp.iter_lines():
            if not line:
                continue
            line = line.decode("utf-8") if isinstance(line, bytes) else line

            if line.startswith("event:"):
                event = line[len("event:"):].strip()
            elif line.startswith("data:"):
                raw = line[len("data:"):]
                data = raw[1:] if raw.startswith(" ") else raw

                if event == "status":
                    status_placeholder.markdown(
                        f"<div style='color:#7eb8f7; font-size:0.88rem; "
                        f"padding:4px 0; display:flex; align-items:center; gap:8px'>"
                        f"<span style='display:inline-block; width:8px; height:8px; "
                        f"background:#7eb8f7; border-radius:50%; "
                        f"animation:pulse 1s infinite'></span>{data}</div>",
                        unsafe_allow_html=True,
                    )
                elif event == "token":
                    if first_token:
                        status_placeholder.empty()
                        first_token = False
                    full_answer += data
                    answer_placeholder.markdown(full_answer + "▌")
                elif event == "citations":
                    sources = json.loads(data)
                elif event == "error":
                    status_placeholder.empty()
                    raise RuntimeError(data)
                elif event == "done":
                    answer_placeholder.markdown(full_answer)
                event = ""

    return full_answer, sources


def _render_sources(sources: list) -> None:
    with st.expander(f"📚 Sources ({len(sources)})", expanded=False):
        for i, src in enumerate(sources, start=1):
            page = src.get("page", "?")
            text = src.get("text", "")
            ref = src.get("ref", "")

            st.markdown(
                f"<div style='font-size:0.82rem; color:#8892a4; margin-bottom:2px'>"
                f"Source {i} &nbsp;·&nbsp; Page {page} &nbsp;·&nbsp; "
                f"<code style='font-size:0.75rem'>{ref}</code>"
                f"</div>",
                unsafe_allow_html=True,
            )
            if text:
                st.markdown(
                    f"<div style='"
                    f"background:#1a2035; border-left:3px solid #2d4a6e; "
                    f"border-radius:4px; padding:8px 12px; margin:4px 0 12px 0; "
                    f"font-size:0.82rem; color:#c0cad8; line-height:1.5"
                    f"'>{text[:200]}{'…' if len(text) >= 200 else ''}</div>",
                    unsafe_allow_html=True,
                )
