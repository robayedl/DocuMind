from __future__ import annotations

from rag.agents.state import GraphState
from rag.chains.generation import get_rag_chain


def generate(state: GraphState) -> GraphState:
    """Generate an answer from graded documents using the RAG chain."""
    chain = get_rag_chain()
    answer = chain.invoke({
        "input": state["question"],
        "context": state["documents"],
    })
    return {"generation": answer}
