from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings  # noqa: E402
from app.runtime import ApplicationContainer  # noqa: E402
from app.schemas import SearchResult  # noqa: E402


def matches_expected(result: SearchResult, record: dict[str, Any]) -> bool:
    if result.chunk.filename != record["expected_filename"]:
        return False
    expected_page = record.get("expected_page")
    if expected_page is not None and result.chunk.page_number != expected_page:
        return False
    expected_text = record.get("expected_text")
    normalized_expected = " ".join(expected_text.casefold().split()) if expected_text else None
    normalized_text = " ".join(result.chunk.text.casefold().split())
    if normalized_expected and normalized_expected not in normalized_text:
        return False
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate retrieval without an LLM")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "data" / "eval" / "retrieval_eval.json",
    )
    parser.add_argument(
        "--reranker",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override ENABLE_RERANKER",
    )
    parser.add_argument(
        "--rerank-candidates",
        type=int,
        help="Override RERANK_CANDIDATES",
    )
    return parser.parse_args()


def percentile(values: list[float], percentile_value: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(percentile_value * len(ordered)) - 1)
    return ordered[index]


def main() -> None:
    args = parse_args()
    settings = get_settings()
    if args.reranker is not None:
        settings = settings.model_copy(update={"enable_reranker": args.reranker})
    if args.rerank_candidates is not None:
        settings = settings.model_copy(update={"rerank_candidates": args.rerank_candidates})
    records = json.loads(args.dataset.read_text(encoding="utf-8"))
    container = ApplicationContainer(settings)
    positive_ranks: list[int | None] = []
    negative_decisions: list[bool] = []
    latencies_ms: list[float] = []
    base_latencies_ms: list[float] = []
    reranker_latencies_ms: list[float] = []

    try:
        retriever = container.retriever()
        for record in records:
            started = time.perf_counter()
            decision = retriever.retrieve_with_decision(record["query"])
            latency_ms = (time.perf_counter() - started) * 1000
            reranker_latency_ms = (
                container.reranker.last_latency_ms if container.reranker else 0.0
            )
            latencies_ms.append(latency_ms)
            reranker_latencies_ms.append(reranker_latency_ms)
            base_latencies_ms.append(max(0.0, latency_ms - reranker_latency_ms))

            if record["answerable"]:
                rank = next(
                    (
                        rank
                        for rank, result in enumerate(decision.results, start=1)
                        if matches_expected(result, record)
                    ),
                    None,
                )
                positive_ranks.append(rank)
                expected_result = next(
                    (result for result in decision.results if matches_expected(result, record)),
                    None,
                )
                print(
                    json.dumps(
                        {
                            "query": record["query"],
                            "expected": record["expected_filename"],
                            "rrf_rank": expected_result.rrf_rank if expected_result else None,
                            "final_rank": rank,
                            "reranker_score": (
                                expected_result.reranker_score if expected_result else None
                            ),
                            "is_answerable": decision.is_answerable,
                            "reason": decision.reason,
                            "latency_ms": round(latency_ms, 2),
                            "base_latency_ms": round(base_latencies_ms[-1], 2),
                            "reranker_latency_ms": round(reranker_latency_ms, 2),
                        },
                        ensure_ascii=False,
                    )
                )
            else:
                rejected = not decision.is_answerable
                negative_decisions.append(rejected)
                top = decision.results[0] if decision.results else None
                print(
                    json.dumps(
                        {
                            "query": record["query"],
                            "expected": "reject",
                            "rejected": rejected,
                            "confidence": decision.confidence,
                            "reason": decision.reason,
                            "top_dense_score": top.dense_score if top else None,
                            "top_reranker_score": top.reranker_score if top else None,
                            "latency_ms": round(latency_ms, 2),
                            "base_latency_ms": round(base_latencies_ms[-1], 2),
                            "reranker_latency_ms": round(reranker_latency_ms, 2),
                        },
                        ensure_ascii=False,
                    )
                )
    finally:
        container.close()

    positives = len(positive_ranks)
    metrics = {
        "positive_queries": positives,
        "negative_queries": len(negative_decisions),
        "recall_at_1": sum(rank == 1 for rank in positive_ranks) / positives,
        "recall_at_3": sum(rank is not None and rank <= 3 for rank in positive_ranks) / positives,
        "recall_at_5": sum(rank is not None and rank <= 5 for rank in positive_ranks) / positives,
        "mrr": sum(1.0 / rank for rank in positive_ranks if rank is not None) / positives,
        "negative_rejection_accuracy": (
            sum(negative_decisions) / len(negative_decisions) if negative_decisions else 0.0
        ),
        "mean_query_latency_ms": statistics.mean(latencies_ms),
        "median_query_latency_ms": statistics.median(latencies_ms),
        "p95_query_latency_ms": percentile(latencies_ms, 0.95),
        "mean_base_latency_ms": statistics.mean(base_latencies_ms),
        "median_base_latency_ms": statistics.median(base_latencies_ms),
        "p95_base_latency_ms": percentile(base_latencies_ms, 0.95),
        "mean_reranker_latency_ms": statistics.mean(reranker_latencies_ms),
        "median_reranker_latency_ms": statistics.median(reranker_latencies_ms),
        "p95_reranker_latency_ms": percentile(reranker_latencies_ms, 0.95),
        "reranker_enabled": settings.enable_reranker,
        "rerank_candidates": settings.rerank_candidates,
    }
    print("SUMMARY " + json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
