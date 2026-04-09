from __future__ import annotations

import sys
import uuid
from pathlib import Path

# Ensure the project root is on sys.path regardless of where Streamlit is launched from
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

from ui.components.sidebar import render_sidebar
from ui.components.chat import render_chat

st.set_page_config(
    page_title="RAG PDF Assistant",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state initialisation ──────────────────────────────────────────────
if "documents" not in st.session_state:
    st.session_state.documents = []

if "current_doc_id" not in st.session_state:
    st.session_state.current_doc_id = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# ── Layout ─────────────────────────────────────────────────────────────────────
render_sidebar()

st.title("RAG PDF Assistant")
render_chat()
