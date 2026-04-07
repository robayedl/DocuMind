from __future__ import annotations

from typing import List
from typing_extensions import TypedDict

from langchain_core.documents import Document


class GraphState(TypedDict):
    question: str        # the user's question
    generation: str      # the LLM's generated answer
    documents: List[Document]  # retrieved documents
    doc_id: str          # which PDF to query
    retry_count: int     # prevents infinite loops (max 3)
    route: str           # "retrieve" or "direct"
