from __future__ import annotations

import logging
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from rag.agents.state import GraphState
from rag.llm import get_llm

logger = logging.getLogger(__name__)

_ROUTER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a query router. Given a user question, decide whether it requires "
            "looking up information from documents or is general conversation.\n"
            "Return 'retrieve' if the question asks about document content, facts, or specific information.\n"
            "Return 'direct' if the question is a greeting, small talk, or does not need document lookup.",
        ),
        ("human", "{question}"),
    ]
)


class RouteDecision(BaseModel):
    """Structured output for the routing decision."""
    route: Literal["retrieve", "direct"] = Field(
        description="'retrieve' if the question needs document lookup, 'direct' for general conversation."
    )


def route_query(state: GraphState) -> GraphState:
    """Route the question to retrieval or a direct response."""
    try:
        llm = get_llm()
        structured_llm = llm.with_structured_output(RouteDecision)
        chain = _ROUTER_PROMPT | structured_llm
        decision: RouteDecision = chain.invoke({"question": state["question"]})
        return {"route": decision.route}
    except Exception as e:
        logger.error(f"Router failed, defaulting to retrieve: {e}")
        return {"route": "retrieve"}
