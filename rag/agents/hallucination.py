from __future__ import annotations

import logging
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from rag.agents.state import GraphState
from rag.llm import get_llm

logger = logging.getLogger(__name__)

_HALLUCINATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a factual grounding checker. Given source documents and a generated answer, "
            "decide if every claim in the answer is fully supported by the documents.\n"
            "Answer 'yes' if the answer is grounded in the documents.\n"
            "Answer 'no' if any part of the answer contains information not present in the documents.",
        ),
        (
            "human",
            "Source documents:\n{context}\n\nGenerated answer:\n{generation}",
        ),
    ]
)


class GroundednessScore(BaseModel):
    """Structured output for hallucination grading."""
    grounded: Literal["yes", "no"] = Field(
        description="'yes' if the answer is fully supported by the documents, 'no' otherwise."
    )


def check_hallucination(state: GraphState) -> GraphState:
    """Verify the generated answer is supported by the retrieved documents."""
    try:
        llm = get_llm()
        structured_llm = llm.with_structured_output(GroundednessScore)
        chain = _HALLUCINATION_PROMPT | structured_llm

        context = "\n\n".join(doc.page_content for doc in state["documents"])
        result: GroundednessScore = chain.invoke(
            {"context": context, "generation": state["generation"]}
        )

        if result.grounded == "yes":
            return {"grounded": True}

        return {
            "grounded": False,
            "generation": "",
            "retry_count": state["retry_count"] + 1,
        }

    except Exception as e:
        logger.error(f"Hallucination check failed, assuming grounded: {e}")
        return {"grounded": True}
