from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
from typing import Any

import requests

# Suppress third-party deprecation noise
warnings.filterwarnings("ignore")

# Ensure project root is on sys.path when run directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)

API_BASE = os.getenv("EVAL_API_BASE", "http://localhost:8000")


def _make_llm() -> LangchainLLMWrapper:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set.")
    return LangchainLLMWrapper(
        ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key,
            temperature=0,
        )
    )


def _make_embeddings() -> LangchainEmbeddingsWrapper:
    return LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    )


def _query_backend(question: str, doc_id: str) -> dict[str, Any]:
    """Call the /query endpoint and return answer + citation dicts."""
    resp = requests.post(
        f"{API_BASE}/query",
        json={"doc_id": doc_id, "question": question},
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    return {
        "answer": data.get("answer", ""),
        "citations": data.get("citations", []),
    }


def evaluate_rag(
    test_data: list[dict[str, str]],
    doc_id: str | None = None,
    verbose: bool = True,
) -> dict[str, float]:
    """
    Run RAGAS evaluation over a list of test samples.

    Each item in test_data must have:
        question     – the user question
        ground_truth – the expected answer
        doc_id       – which document to query (overridden by the doc_id arg)
    """
    llm = _make_llm()
    emb = _make_embeddings()

    metrics = [
        Faithfulness(llm=llm),
        AnswerRelevancy(llm=llm, embeddings=emb),
        ContextPrecision(llm=llm),
        ContextRecall(llm=llm),
    ]

    samples: list[SingleTurnSample] = []

    for i, item in enumerate(test_data, start=1):
        q = item["question"]
        gt = item["ground_truth"]
        did = doc_id or item.get("doc_id", "")

        if verbose:
            print(f"  [{i}/{len(test_data)}] {q[:70]}…")

        try:
            result = _query_backend(q, did)
            answer = result["answer"]
            contexts = [
                c.get("text") or c.get("ref", "")
                for c in result["citations"]
            ] or ["No context retrieved."]
        except Exception as e:
            print(f"    ! Query failed: {e}")
            answer = ""
            contexts = ["Query failed."]

        samples.append(
            SingleTurnSample(
                user_input=q,
                response=answer,
                retrieved_contexts=contexts,
                reference=gt,
            )
        )

    dataset = EvaluationDataset(samples=samples)

    if verbose:
        print("\nRunning RAGAS evaluation…")

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        raise_exceptions=False,
        show_progress=verbose,
    )

    scores: dict[str, float] = {}
    for m in metrics:
        val = result[m.name]
        if val is None:
            scores[m.name] = 0.0
        elif isinstance(val, list):
            valid = [v for v in val if v is not None]
            scores[m.name] = round(sum(valid) / len(valid), 4) if valid else 0.0
        else:
            scores[m.name] = round(float(val), 4)

    return scores
