from __future__ import annotations

from rag.agents.state import GraphState

_GREETINGS = {"hi", "hello", "hey", "hiya", "howdy", "sup", "what's up", "whats up"}


def route_query(state: GraphState) -> GraphState:
    """Route to direct_response for greetings, retrieve for everything else."""
    question = state["question"].strip().lower()
    is_greeting = any(question == g or question.startswith(g + " ") for g in _GREETINGS)
    return {"route": "direct" if is_greeting else "retrieve"}
