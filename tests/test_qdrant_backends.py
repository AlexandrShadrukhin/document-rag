from __future__ import annotations

from pathlib import Path
from typing import Any

from app.config import Settings
from app.lab.experiments import ExperimentConfig, ExperimentRunner
from app.retrieval.vector_store import qdrant_backend_status, qdrant_collection_exists


class FakeResponse:
    def __init__(self, status_code: int, payload: dict[str, object]) -> None:
        self.status_code = status_code
        self._payload = payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError("HTTP error")

    def json(self) -> dict[str, object]:
        return self._payload


def test_local_and_server_backend_status(tmp_path: Path, monkeypatch: Any) -> None:
    local_path = tmp_path / "qdrant"
    local_path.mkdir()
    local = qdrant_backend_status("local", local_path, "http://localhost:6333")
    assert local.mode == "local"
    assert local.available is True
    assert local.endpoint == str(local_path)

    monkeypatch.setattr(
        "app.retrieval.vector_store.httpx.get",
        lambda *args, **kwargs: FakeResponse(200, {"version": "1.13.6"}),
    )
    server = qdrant_backend_status("server", local_path, "http://qdrant:6333")
    assert server.available is True
    assert server.server_version == "1.13.6"
    assert qdrant_collection_exists("http://qdrant:6333", "documents") is True


def test_server_experiments_use_isolated_collections_without_mutating_baseline(
    tmp_path: Path,
) -> None:
    settings = Settings(
        qdrant_mode="server",
        qdrant_url="http://localhost:6333",
        qdrant_collection="documents",
        qdrant_path=tmp_path / "baseline/qdrant",
        corpus_path=tmp_path / "baseline/corpus.jsonl",
        manifest_path=tmp_path / "baseline/manifest.json",
        wiki_dump_path=tmp_path / "source/wiki.bz2",
        wiki_corpus_path=tmp_path / "corpus",
        benchmarks_path=tmp_path / "benchmarks",
        experiments_path=tmp_path / "experiments",
    )
    first = ExperimentConfig(
        name="server-a",
        corpus=tmp_path / "corpus",
        index_config={"chunk_size": 300},
    )
    second = ExperimentConfig(
        name="server-b",
        corpus=tmp_path / "corpus",
        index_config={"chunk_size": 500},
    )

    first_settings, first_root, _ = ExperimentRunner(settings).prepare(first)
    second_settings, second_root, _ = ExperimentRunner(settings).prepare(second)

    assert first_settings.qdrant_mode == "server"
    assert first_settings.qdrant_url == settings.qdrant_url
    assert first_settings.qdrant_collection.startswith("experiment_")
    assert first_settings.qdrant_collection != settings.qdrant_collection
    assert first_settings.qdrant_collection != second_settings.qdrant_collection
    assert first_settings.corpus_path.is_relative_to(first_root)
    assert second_settings.corpus_path.is_relative_to(second_root)
    assert settings.qdrant_collection == "documents"
    assert not settings.qdrant_path.exists()
