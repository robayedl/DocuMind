from __future__ import annotations

import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

from ui.components.sidebar import render_sidebar
from ui.components.chat import render_chat
from ui.components.pdf_viewer import render_pdf_viewer

st.set_page_config(
    page_title="DocuMind",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.stApp { background-color: #0f1117; }

[data-testid="stSidebar"] {
    background-color: #161b27;
    border-right: 1px solid #2a2f3e;
}
[data-testid="stSidebar"] h1 {
    font-size: 1.1rem !important;
    font-weight: 700;
    letter-spacing: 0.04em;
    color: #e2e8f0;
}
[data-testid="stFileUploader"] {
    background: #1e2433;
    border: 1px dashed #3a4155;
    border-radius: 8px;
    padding: 4px;
}
[data-testid="stChatInput"] textarea {
    background-color: #1e2433 !important;
    border: 1px solid #3a4155 !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
    font-size: 0.95rem;
}
[data-testid="stChatMessage"] {
    border-radius: 12px;
    padding: 4px 8px;
    margin-bottom: 4px;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background-color: #1a2035;
}
.stButton > button {
    background-color: #1e2d40;
    border: 1px solid #2d4a6e;
    color: #7eb8f7;
    border-radius: 8px;
    font-weight: 500;
    transition: all 0.2s;
}
.stButton > button:hover {
    background-color: #2d4a6e;
    border-color: #7eb8f7;
    color: #ffffff;
}
[data-testid="stExpander"] {
    background-color: #161b27;
    border: 1px solid #2a2f3e;
    border-radius: 8px;
}
hr { border-color: #2a2f3e !important; }
[data-testid="stRadio"] label { font-size: 0.85rem; }
[data-testid="stStatus"] { border-radius: 8px; }
h1 { font-weight: 700 !important; letter-spacing: -0.02em; }
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
for key, default in [
    ("documents", []),
    ("current_doc_id", None),
    ("chat_history", []),
    ("session_id", str(uuid.uuid4())),
    ("show_pdf", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Sidebar ────────────────────────────────────────────────────────────────────
render_sidebar()

# ── Header ─────────────────────────────────────────────────────────────────────
header_left, header_right = st.columns([6, 1])
with header_left:
    st.markdown("""
    <div style="display:flex; align-items:center; gap:16px; padding: 0.4rem 0 0.6rem 0;">
        <div style="
            background: linear-gradient(135deg, #1e3a5f 0%, #0f2540 100%);
            border: 1px solid #2d5a8e;
            border-radius: 14px;
            width: 52px; height: 52px;
            display: flex; align-items: center; justify-content: center;
            font-size: 1.7rem; flex-shrink: 0;
            box-shadow: 0 0 20px rgba(126,184,247,0.25), 0 0 40px rgba(167,139,250,0.1);
        ">🧠</div>
        <div>
            <div style="
                font-size: 2.1rem;
                font-weight: 900;
                letter-spacing: -0.04em;
                line-height: 1.05;
                background: linear-gradient(90deg, #7eb8f7 0%, #a78bfa 55%, #f0abfc 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            ">DocuMind</div>
            <div style="font-size:0.92rem; color:#7a8aa8; letter-spacing:0.01em; margin-top:1px;">
                Intelligent answers &nbsp;·&nbsp; grounded in your document
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
with header_right:
    if st.session_state.current_doc_id:
        label = "Hide PDF 📕" if st.session_state.show_pdf else "View PDF 📖"
        if st.button(label, use_container_width=True):
            st.session_state.show_pdf = not st.session_state.show_pdf
            st.rerun()

st.divider()

# ── Main layout ────────────────────────────────────────────────────────────────
if st.session_state.show_pdf and st.session_state.current_doc_id:
    chat_col, pdf_col = st.columns([1, 1], gap="large")
    with chat_col:
        render_chat()
    with pdf_col:
        render_pdf_viewer()
else:
    render_chat()
