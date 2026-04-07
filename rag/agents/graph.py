from __future__ import annotations

from functools import lru_cache
from typing import Literal

from langgraph.graph import END, StateGraph

from rag.agents.grader import grade_documents
from rag.agents.router import route_query
from rag.agents.state import GraphState
from rag.retrieve import retrieve_top_k


# ──────────────────────────────────────────────
# Node functions
# ──────────────────────────────────────────────

def retrieve(state: GraphState) -> GraphState:
    """Retrieve documents from the vector store for the given question."""
    docs = retrieve_top_k(
        doc_id=state["doc_id"],
        question=state["question"],
        top_k=5,
    )
    return {"documents": docs}


def direct_response(state: GraphState) -> GraphState:
    """Return a canned response for general conversation (no retrieval needed)."""
    return {
        "generation": (
            "Hello! I'm a PDF assistant. Please ask me a question about your document "
            "and I'll look it up for you."
        )
    }


# ──────────────────────────────────────────────
# Conditional edge
# ──────────────────────────────────────────────

def decide_after_routing(state: GraphState) -> Literal["retrieve", "direct_response"]:
    """Route to retrieval or a direct response based on the router's decision."""
    return "retrieve" if state["route"] == "retrieve" else "direct_response"


# ──────────────────────────────────────────────
# Graph builder
# ──────────────────────────────────────────────

@lru_cache(maxsize=1)
def build_graph() -> StateGraph:
    graph = StateGraph(GraphState)

    # Nodes
    graph.add_node("router", route_query)
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade_documents", grade_documents)
    graph.add_node("direct_response", direct_response)

    # Entry point
    graph.set_entry_point("router")

    # Conditional edge from router
    graph.add_conditional_edges(
        "router",
        decide_after_routing,
        {
            "retrieve": "retrieve",
            "direct_response": "direct_response",
        },
    )

    # After retrieval → grade documents
    graph.add_edge("retrieve", "grade_documents")

    # Terminal edges (generator and hallucination checker added in Stage 4)
    graph.add_edge("grade_documents", END)
    graph.add_edge("direct_response", END)

    return graph.compile()
