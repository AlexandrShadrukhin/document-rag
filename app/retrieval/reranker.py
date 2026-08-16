from __future__ import annotations

import time

from app.lab.devices import resolve_torch_device
from app.schemas import SearchResult


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str,
        top_k: int = 5,
        batch_size: int = 8,
        device: str | None = None,
    ) -> None:
        from sentence_transformers import CrossEncoder
        from torch import nn

        self.device_selection = resolve_torch_device(device)
        self.device = self.device_selection.selected
        self.model = CrossEncoder(model_name, device=self.device)
        self.top_k = top_k
        self.batch_size = batch_size
        self.activation = nn.Sigmoid()
        self.last_latency_ms = 0.0

    def __call__(self, query: str, results: list[SearchResult]) -> list[SearchResult]:
        if not results:
            self.last_latency_ms = 0.0
            return []
        started = time.perf_counter()
        scores = self.model.predict(
            [(query, result.chunk.text) for result in results],
            batch_size=self.batch_size,
            activation_fn=self.activation,
            show_progress_bar=False,
        )
        self.last_latency_ms = (time.perf_counter() - started) * 1000
        reranked = [
            result.model_copy(update={"reranker_score": float(score)})
            for result, score in zip(results, scores, strict=True)
        ]
        reranked.sort(key=lambda result: result.reranker_score or 0.0, reverse=True)
        return [
            result.model_copy(update={"reranker_rank": rank})
            for rank, result in enumerate(reranked[: self.top_k], start=1)
        ]
