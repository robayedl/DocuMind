from __future__ import annotations

from functools import lru_cache
from typing import Literal

from langgraph.graph import END, StateGraph

from rag.agents.generator import generate
from rag.agents.grader import grade_documents
from rag.agents.hallucination import check_hallucination
from rag.agents.router import route_query
from rag.agents.rewriter import rewrite_query
from rag.agents.state import GraphState
from rag.retrieve import retrieve_top_k

MAX_RETRIES = 3


# ──────────────────────────────────────────────
# Node functions
# ──────────────────────────────────────────────

def retrieve(state: GraphState) -> GraphState:
    """Retrieve documents from the vector store."""
    docs = retrieve_top_k(
        doc_id=state["doc_id"],
        question=state["question"],
        top_k=5,
    )
    return {"documents": docs}


def direct_response(state: GraphState) -> GraphState:
    """Return a response for general conversation (no retrieval needed)."""
    return {
        "generation": (
            "Hello! I'm a PDF assistant. Please ask me a question about your "
            "document and I'll look it up for you."
        )
    }


def fallback(state: GraphState) -> GraphState:
    """Return a fallback message when max retries are exceeded."""
    return {
        "generation": (
            "I could not find a reliable answer in the document after multiple attempts. "
            "Please try rephrasing your question or check that the document has been indexed."
        )
    }


# ──────────────────────────────────────────────
# Conditional edge functions
# ──────────────────────────────────────────────

def decide_after_routing(state: GraphState) -> Literal["retrieve", "direct_response"]:
    return "retrieve" if state["route"] == "retrieve" else "direct_response"


def decide_after_grading(
    state: GraphState,
) -> Literal["generate", "rewrite_query", "fallback"]:
    if state["documents"]:
        return "generate"
    if state["retry_count"] < MAX_RETRIES:
        return "rewrite_query"
    return "fallback"


def decide_after_hallucination(
    state: GraphState,
) -> Literal["__end__", "generate", "fallback"]:
    if state["grounded"]:
        return END
    if state["retry_count"] < MAX_RETRIES:
        return "generate"
    return "fallback"


# ──────────────────────────────────────────────
# Graph builder
# ──────────────────────────────────────────────

@lru_cache(maxsize=1)
def build_graph():
    graph = StateGraph(GraphState)

    # Nodes
    graph.add_node("router", route_query)
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade_documents", grade_documents)
    graph.add_node("generate", generate)
    graph.add_node("check_hallucination", check_hallucination)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("direct_response", direct_response)
    graph.add_node("fallback", fallback)

    # Entry point
    graph.set_entry_point("router")

    # router → retrieve or direct_response
    graph.add_conditional_edges(
        "router",
        decide_after_routing,
        {"retrieve": "retrieve", "direct_response": "direct_response"},
    )

    # retrieve → grade_documents
    graph.add_edge("retrieve", "grade_documents")

    # grade_documents → generate | rewrite_query | fallback
    graph.add_conditional_edges(
        "grade_documents",
        decide_after_grading,
        {
            "generate": "generate",
            "rewrite_query": "rewrite_query",
            "fallback": "fallback",
        },
    )

    # rewrite_query → retrieve (loop back)
    graph.add_edge("rewrite_query", "retrieve")

    # generate → check_hallucination
    graph.add_edge("generate", "check_hallucination")

    # check_hallucination → END | generate | fallback
    graph.add_conditional_edges(
        "check_hallucination",
        decide_after_hallucination,
        {
            END: END,
            "generate": "generate",
            "fallback": "fallback",
        },
    )

    # Terminal edges
    graph.add_edge("direct_response", END)
    graph.add_edge("fallback", END)

    return graph.compile()


# ──────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────

def run_agent(question: str, doc_id: str) -> GraphState:
    """Compile and invoke the full agentic RAG graph."""
    graph = build_graph()
    return graph.invoke(
        {
            "question": question,
            "generation": "",
            "documents": [],
            "doc_id": doc_id,
            "retry_count": 0,
            "route": "",
            "grounded": False,
        }
    )
