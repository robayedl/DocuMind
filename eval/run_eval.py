"""
Run RAGAS evaluation against the live RAG backend.

Usage:
    python eval/run_eval.py [--doc-id DOC_ID]

    --doc-id   Override the doc_id in test_queries.json for all questions.
               Required if test_queries.json still has the placeholder value.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is on sys.path when run directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.ragas_eval import evaluate_rag

EVAL_DIR = Path(__file__).parent
QUERIES_FILE = EVAL_DIR / "test_queries.json"
RESULTS_FILE = EVAL_DIR / "results.json"

PASS_THRESHOLD = 0.7
TRACKED_METRICS = ["faithfulness", "answer_relevancy"]


def _load_queries(doc_id_override: str | None) -> list[dict]:
    with open(QUERIES_FILE) as f:
        queries = json.load(f)

    if doc_id_override:
        for q in queries:
            q["doc_id"] = doc_id_override

    missing = [i + 1 for i, q in enumerate(queries) if q.get("doc_id") == "REPLACE_WITH_YOUR_DOC_ID"]
    if missing:
        print(f"ERROR: doc_id not set for queries {missing}.")
        print("Run with --doc-id <your_doc_id> or edit eval/test_queries.json.")
        sys.exit(1)

    return queries


def _print_table(scores: dict[str, float]) -> None:
    print("\n" + "─" * 44)
    print(f"  {'Metric':<26}  {'Score':>6}")
    print("─" * 44)
    for metric, score in scores.items():
        bar = "█" * int(score * 20)
        print(f"  {metric:<26}  {score:>5.3f}  {bar}")
    print("─" * 44)


def _save_results(scores: dict[str, float], queries: list[dict]) -> None:
    history: list[dict] = []
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            history = json.load(f)

    history.append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_queries": len(queries),
        "scores": scores,
    })

    with open(RESULTS_FILE, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nResults saved → {RESULTS_FILE}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation")
    parser.add_argument("--doc-id", default=None, help="Override doc_id for all test queries")
    args = parser.parse_args()

    queries = _load_queries(args.doc_id)

    print(f"Evaluating {len(queries)} queries…\n")
    scores = evaluate_rag(queries, verbose=True)

    _print_table(scores)

    passed = all(scores.get(m, 0) >= PASS_THRESHOLD for m in TRACKED_METRICS)
    verdict = "PASS ✓" if passed else "NEEDS IMPROVEMENT ✗"
    print(f"\nOverall verdict: {verdict}")
    print(
        f"  (faithfulness={scores.get('faithfulness', 0):.3f}, "
        f"answer_relevancy={scores.get('answer_relevancy', 0):.3f}, "
        f"threshold={PASS_THRESHOLD})"
    )

    _save_results(scores, queries)


if __name__ == "__main__":
    main()
