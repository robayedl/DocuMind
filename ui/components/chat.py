from __future__ import annotations

import json

import requests
import streamlit as st

API_BASE = "http://localhost:8000"

_INSTRUCTIONS = """
**How to use:**
1. Upload a PDF using the sidebar on the left
2. Wait for indexing to complete
3. Type your question in the box below
4. Ask follow-up questions — the assistant remembers the conversation
5. Use **Clear Chat** in the sidebar to start a new conversation
"""


def render_chat() -> None:
    if not st.session_state.current_doc_id:
        st.info("Upload a PDF from the sidebar to get started.")
        with st.expander("How to use", expanded=True):
            st.markdown(_INSTRUCTIONS)
        return

    selected_doc = next(
        (d for d in st.session_state.documents if d["doc_id"] == st.session_state.current_doc_id),
        None,
    )
    if selected_doc:
        st.caption(f"Chatting about: **{selected_doc['filename']}**")

    # ── Render history ─────────────────────────────────────────────────────────
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                _render_sources(msg["sources"])

    # ── Chat input ─────────────────────────────────────────────────────────────
    user_input = st.chat_input("Ask a question about the document…")
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
            if "GOOGLE_API_KEY" in error_msg or "API key" in error_msg.lower():
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
    """Stream tokens from /query/stream and return (full_answer, citations)."""
    sources = []
    answer_placeholder = st.empty()
    answer_placeholder.status("Thinking…", state="running")
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
                # Strip only the single SSE protocol space, not all whitespace
                raw = line[len("data:"):]
                data = raw[1:] if raw.startswith(" ") else raw

                if event == "token":
                    if first_token:
                        answer_placeholder.empty()
                        first_token = False
                    full_answer += data
                    answer_placeholder.markdown(full_answer + "▌")
                elif event == "citations":
                    sources = json.loads(data)
                elif event == "error":
                    raise RuntimeError(data)
                elif event == "done":
                    answer_placeholder.markdown(full_answer)
                event = ""

    return full_answer, sources


def _render_sources(sources: list) -> None:
    with st.expander("Sources"):
        for i, src in enumerate(sources, start=1):
            page = src.get("page", "?")
            text = src.get("text", "")
            ref = src.get("ref", "")
            st.markdown(f"**Source {i}** — Page {page}  \n`{ref}`")
            if text:
                st.caption(text[:200] + ("…" if len(text) >= 200 else ""))
            if i < len(sources):
                st.divider()
