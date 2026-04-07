from __future__ import annotations

from typing import List, Literal

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from rag.agents.state import GraphState
from rag.llm import get_llm

_GRADER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a relevance grader. Given a user question and a document chunk, "
            "decide if the document contains information useful for answering the question.\n"
            "Answer 'yes' if the document is relevant, 'no' if it is not.",
        ),
        (
            "human",
            "Question: {question}\n\nDocument:\n{document}",
        ),
    ]
)


class RelevanceScore(BaseModel):
    """Structured output for document relevance grading."""
    score: Literal["yes", "no"] = Field(
        description="'yes' if the document is relevant to the question, 'no' otherwise."
    )


def grade_documents(state: GraphState) -> GraphState:
    """Filter retrieved documents to only those relevant to the question."""
    llm = get_llm()
    structured_llm = llm.with_structured_output(RelevanceScore)
    chain = _GRADER_PROMPT | structured_llm

    question = state["question"]
    documents: List[Document] = state.get("documents", [])

    relevant_docs: List[Document] = []
    for doc in documents:
        result: RelevanceScore = chain.invoke(
            {"question": question, "document": doc.page_content}
        )
        if result.score == "yes":
            relevant_docs.append(doc)

    # If no documents pass grading, signal a retry by returning empty list
    return {"documents": relevant_docs}
