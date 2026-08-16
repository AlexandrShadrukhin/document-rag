from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from app.api.main import app, container
from app.schemas import QueryResponse, Source, Timings


class FakeRAGService:
    def answer(self, query: str) -> QueryResponse:
        return QueryResponse(
            answer="Вильнюс.",
            is_answerable=True,
            sources=[
                Source(
                    filename="wiki_000.pdf",
                    page=1,
                    chunk_id="chunk-1",
                    retrieval_score=0.99,
                    reranker_score=0.99,
                )
            ],
            timings=Timings(
                base_retrieval_ms=30.0,
                reranker_ms=2500.0,
                retrieval_total_ms=2530.0,
                generation_ms=400.0,
                total_ms=2930.0,
                context_chars=900,
            ),
        )


def test_query_api_accepts_mode_and_returns_schema(monkeypatch: Any) -> None:
    modes: list[str] = []

    def rag_service(mode: str) -> FakeRAGService:
        modes.append(mode)
        return FakeRAGService()

    monkeypatch.setattr(container, "rag_service", rag_service)
    with TestClient(app) as client:
        health = client.get("/health")
        response = client.post(
            "/query", json={"query": "Какая столица Литвы?", "mode": "quality"}
        )

    assert health.json() == {"status": "ok"}
    assert response.status_code == 200
    payload = response.json()
    assert modes == ["quality"]
    assert payload["answer"] == "Вильнюс."
    assert payload["sources"][0]["filename"] == "wiki_000.pdf"
    assert payload["timings"]["base_retrieval_ms"] == 30.0
    assert payload["timings"]["reranker_ms"] == 2500.0
    assert payload["timings"]["retrieval_total_ms"] == 2530.0
