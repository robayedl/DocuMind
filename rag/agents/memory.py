from __future__ import annotations

from langchain_core.chat_history import InMemoryChatMessageHistory

_sessions: dict[str, InMemoryChatMessageHistory] = {}


def get_memory(session_id: str) -> InMemoryChatMessageHistory:
    """Return (or create) the message history for a session."""
    if session_id not in _sessions:
        _sessions[session_id] = InMemoryChatMessageHistory()
    return _sessions[session_id]


def clear_memory(session_id: str) -> None:
    """Remove all history for a session."""
    _sessions.pop(session_id, None)
