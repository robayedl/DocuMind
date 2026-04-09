from __future__ import annotations

import requests
import streamlit as st

API_BASE = "http://localhost:8000"


def render_chat() -> None:
    if not st.session_state.current_doc_id:
        st.info("Upload and select a PDF from the sidebar to start chatting.")
        return

    selected_doc = next(
        (d for d in st.session_state.documents if d["doc_id"] == st.session_state.current_doc_id),
        None,
    )
    if selected_doc:
        st.caption(f"Chatting about: **{selected_doc['filename']}**")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("Sources"):
                    for src in msg["sources"]:
                        st.markdown(
                            f"- **Page {src['page']}** — chunk `{src['ref']}` "
                            f"*(score rank {src['chunk_id']})*"
                        )

    user_input = st.chat_input("Ask a question about the document…")
    if not user_input:
        return

    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            answer, sources = _query(user_input)
        st.markdown(answer)
        if sources:
            with st.expander("Sources"):
                for src in sources:
                    st.markdown(
                        f"- **Page {src['page']}** — chunk `{src['ref']}` "
                        f"*(score rank {src['chunk_id']})*"
                    )

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })


def _query(question: str):
    try:
        resp = requests.post(
            f"{API_BASE}/query",
            json={
                "doc_id": st.session_state.current_doc_id,
                "question": question,
                "session_id": st.session_state.session_id,
            },
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["answer"], data.get("citations", [])
    except requests.HTTPError as e:
        detail = e.response.json().get("detail", str(e)) if e.response else str(e)
        return f"Error: {detail}", []
    except Exception as e:
        return f"Error: {e}", []
