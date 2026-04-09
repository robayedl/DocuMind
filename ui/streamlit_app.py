from __future__ import annotations

import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

from ui.components.sidebar import render_sidebar
from ui.components.chat import render_chat

st.set_page_config(
    page_title="RAG PDF Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Main background */
.stApp { background-color: #0f1117; }

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #161b27;
    border-right: 1px solid #2a2f3e;
}

/* Sidebar title */
[data-testid="stSidebar"] h1 {
    font-size: 1.1rem !important;
    font-weight: 700;
    letter-spacing: 0.04em;
    color: #e2e8f0;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: #1e2433;
    border: 1px dashed #3a4155;
    border-radius: 8px;
    padding: 4px;
}

/* Chat input */
[data-testid="stChatInput"] textarea {
    background-color: #1e2433 !important;
    border: 1px solid #3a4155 !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
    font-size: 0.95rem;
}

/* Chat messages */
[data-testid="stChatMessage"] {
    border-radius: 12px;
    padding: 4px 8px;
    margin-bottom: 4px;
}

/* User message bubble */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background-color: #1a2035;
}

/* Buttons */
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

/* Expander (Sources) */
[data-testid="stExpander"] {
    background-color: #161b27;
    border: 1px solid #2a2f3e;
    border-radius: 8px;
}

/* Divider */
hr { border-color: #2a2f3e !important; }

/* Radio buttons */
[data-testid="stRadio"] label { font-size: 0.85rem; }

/* Status widget */
[data-testid="stStatus"] { border-radius: 8px; }

/* Page title */
h1 { font-weight: 700 !important; letter-spacing: -0.02em; }
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
for key, default in [
    ("documents", []),
    ("current_doc_id", None),
    ("chat_history", []),
    ("session_id", str(uuid.uuid4())),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Layout ─────────────────────────────────────────────────────────────────────
render_sidebar()

st.markdown("## 📄 RAG PDF Assistant")
st.caption("Ask questions about your PDF — answers are grounded strictly in your document.")
st.divider()

render_chat()
