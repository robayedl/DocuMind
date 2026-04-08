from __future__ import annotations

import json
from typing import List

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from rag.agents.state import GraphState
from rag.llm import get_llm

_GRADER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a relevance grader. Given a user question and a numbered list of document chunks, "
            "return a JSON array of 'yes' or 'no' for each chunk indicating whether it is relevant "
            "to answering the question.\n"
            "Example output for 3 chunks: [\"yes\", \"no\", \"yes\"]\n"
            "Return only the JSON array, nothing else.",
        ),
        (
            "human",
            "Question: {question}\n\nDocuments:\n{documents}",
        ),
    ]
)


def grade_documents(state: GraphState) -> GraphState:
    """Filter retrieved documents to only those relevant to the question (single LLM call)."""
    llm = get_llm()
    chain = _GRADER_PROMPT | llm | StrOutputParser()

    documents: List[Document] = state.get("documents", [])
    if not documents:
        return {"documents": []}

    numbered = "\n\n".join(
        f"[{i+1}] {doc.page_content}" for i, doc in enumerate(documents)
    )

    raw = chain.invoke({"question": state["question"], "documents": numbered})

    # Parse the JSON array response
    try:
        scores: List[str] = json.loads(raw.strip())
    except (json.JSONDecodeError, ValueError):
        # If parsing fails, keep all documents
        scores = ["yes"] * len(documents)

    relevant_docs = [
        doc for doc, score in zip(documents, scores)
        if str(score).strip().lower() == "yes"
    ]

    return {"documents": relevant_docs}
