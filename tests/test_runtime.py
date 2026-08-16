from __future__ import annotations

from pathlib import Path
from typing import Any

import app.runtime as runtime_module
from app.config import get_settings
from app.runtime import ApplicationContainer


class FakeReranker:
    def __init__(self, **kwargs: Any) -> None:
        self.last_latency_ms = 0.0

    def __call__(self, query: str, results: list[Any]) -> list[Any]:
        return results[:5]


def test_fast_and_quality_modes_control_reranker(
    tmp_path: Path, monkeypatch: Any
) -> None:
    settings = get_settings().model_copy(
        update={"corpus_path": tmp_path / "corpus.jsonl", "rerank_candidates": 15}
    )
    container = ApplicationContainer(settings)
    container.embeddings = object()  # type: ignore[assignment]
    container.vector_store = object()  # type: ignore[assignment]
    monkeypatch.setattr(runtime_module, "CrossEncoderReranker", FakeReranker)

    fast = container.retriever("fast")
    quality = container.retriever("quality")

    assert fast.rerank is None
    assert quality.rerank is container.reranker
    assert quality.rerank_candidates == 15
