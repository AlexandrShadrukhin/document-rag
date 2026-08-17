from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from app.config import Settings
from app.desktop.controller import (
    DEFAULT_INDEX_TARGET,
    BaselineModificationNotConfirmed,
    LabController,
    NoActiveRAGIndex,
)
from app.indexing import IndexingStats
from app.lab.experiments import ExperimentRunner
from app.lab.progress import ProgressEvent
from app.retrieval.vector_store import QdrantBackendStatus
from app.schemas import QueryResponse, Timings


class FakeBenchmarkRun:
    seen_settings: list[Settings] = []

    def __init__(self, kind: str, settings: Settings, *args: object, **kwargs: object) -> None:
        self.path = Path("benchmark")
        self.summary: dict[str, object] = {}
        self.seen_settings.append(settings)

    def start(self) -> None:
        pass

    def progress_callback(self, event: object) -> None:
        pass

    def set_stage(self, stage: str) -> None:
        pass

    def append_query(self, payload: object) -> None:
        pass

    def finish(self, status: str, *args: object, **kwargs: object) -> Path:
        if args and isinstance(args[0], dict):
            self.summary = args[0]
        return self.path


class FakeRAGService:
    def answer(self, query: str) -> QueryResponse:
        return QueryResponse(
            answer="test",
            is_answerable=True,
            sources=[],
            timings=Timings(),
        )


class FakeApplicationContainer:
    seen_settings: list[Settings] = []

    def __init__(self, settings: Settings, load_bm25: bool = True) -> None:
        self.settings = settings
        self.seen_settings.append(settings)
        self.embeddings = type("Embeddings", (), {"device": "cpu"})()
        self.reranker = None

    def index(self, *args: object, **kwargs: object) -> IndexingStats:
        return IndexingStats(files_discovered=1, files_indexed=1, total_seconds=1.0)

    def rag_service(self, mode: str) -> FakeRAGService:
        return FakeRAGService()

    def close(self) -> None:
        pass


def settings_for_test(tmp_path: Path) -> Settings:
    return Settings(
        qdrant_path=tmp_path / "baseline/qdrant",
        corpus_path=tmp_path / "baseline/corpus.jsonl",
        manifest_path=tmp_path / "baseline/manifest.json",
        wiki_dump_path=tmp_path / "source/wiki.bz2",
        wiki_corpus_path=tmp_path / "corpus",
        benchmarks_path=tmp_path / "benchmarks",
        experiments_path=tmp_path / "experiments",
    )


def test_controller_index_summary_uses_real_stage_stats() -> None:
    stats = IndexingStats(
        pages_parsed=20,
        chunks_created=50,
        bytes_processed=10 * 1024**2,
        parsing_seconds=1.0,
        cleaning_seconds=0.2,
        chunking_seconds=0.3,
        embedding_seconds=5.0,
        qdrant_upsert_seconds=1.0,
        total_seconds=10.0,
    )
    summary = LabController.index_summary(stats)
    assert summary["pages_per_second"] == 2.0
    assert summary["chunks_per_second"] == 5.0
    assert summary["megabytes_per_second"] == 1.0
    assert summary["other_seconds"] == 2.5


def test_controller_forwards_structured_progress_events() -> None:
    class Recorder:
        def __init__(self) -> None:
            self.events: list[ProgressEvent] = []

        def progress_callback(self, event: ProgressEvent) -> None:
            self.events.append(event)

    benchmark = Recorder()
    received: list[ProgressEvent] = []
    callback = LabController._combined_callback(benchmark, received.append)  # type: ignore[arg-type]
    event = ProgressEvent(
        "embedding_inference",
        "progress",
        current=2,
        total=5,
        details={"embedding_batches_completed": 12},
    )
    callback(event)
    assert benchmark.events == [event]
    assert received == [event]


def test_isolated_index_is_the_default_and_uses_experiment_runner(
    tmp_path: Path, monkeypatch: Any
) -> None:
    settings = settings_for_test(tmp_path)
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    calls = []
    original = ExperimentRunner.prepare

    def tracked_prepare(self: ExperimentRunner, config: object) -> object:
        calls.append(config)
        return original(self, config)  # type: ignore[arg-type]

    monkeypatch.setattr(ExperimentRunner, "prepare", tracked_prepare)
    target = LabController(settings).plan_index_target(corpus)

    assert DEFAULT_INDEX_TARGET == "isolated"
    assert target.mode == "isolated"
    assert calls
    assert target.experiment_root is not None
    assert target.settings.qdrant_path != settings.qdrant_path
    assert target.settings.corpus_path != settings.corpus_path
    assert target.settings.manifest_path != settings.manifest_path
    assert target.settings.qdrant_path.is_relative_to(target.experiment_root)
    assert target.settings.benchmarks_path.is_relative_to(target.experiment_root)


def test_baseline_requires_explicit_confirmation(tmp_path: Path) -> None:
    settings = settings_for_test(tmp_path)
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    controller = LabController(settings)
    target = controller.plan_index_target(corpus, "baseline")

    assert target.settings.qdrant_path == settings.qdrant_path
    assert target.settings.corpus_path == settings.corpus_path
    with pytest.raises(BaselineModificationNotConfirmed):
        controller.build_index(corpus, target_mode="baseline")
    assert not settings.qdrant_path.exists()
    assert not settings.corpus_path.exists()


def test_isolated_build_becomes_active_and_query_uses_its_settings(
    tmp_path: Path, monkeypatch: Any
) -> None:
    settings = settings_for_test(tmp_path)
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    FakeApplicationContainer.seen_settings.clear()
    FakeBenchmarkRun.seen_settings.clear()
    monkeypatch.setattr("app.desktop.controller.ApplicationContainer", FakeApplicationContainer)
    monkeypatch.setattr("app.desktop.controller.BenchmarkRun", FakeBenchmarkRun)
    controller = LabController(settings)

    controller.build_index(corpus)
    info = controller.active_index_info()

    assert info is not None
    assert info.target_mode == "isolated"
    assert info.qdrant_path != settings.qdrant_path
    assert controller.active_settings is FakeApplicationContainer.seen_settings[0]
    controller.ask("Вопрос", "fast")
    assert FakeBenchmarkRun.seen_settings[-1] is controller.active_settings
    assert controller.container is not None
    assert controller.container.settings is controller.active_settings


def test_query_never_falls_back_to_baseline(tmp_path: Path) -> None:
    controller = LabController(settings_for_test(tmp_path))
    with pytest.raises(NoActiveRAGIndex, match="No active RAG index selected"):
        controller.ask("Вопрос", "fast")


def test_existing_isolated_index_can_be_activated(tmp_path: Path) -> None:
    settings = settings_for_test(tmp_path)
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    controller = LabController(settings)
    target = controller.plan_index_target(corpus)
    target.settings.qdrant_path.mkdir(parents=True)
    target.settings.corpus_path.write_text("", encoding="utf-8")
    target.settings.manifest_path.write_text("{}", encoding="utf-8")

    info = controller.activate_index(corpus)

    assert info.target_mode == "isolated"
    assert info.index_path == target.settings.corpus_path.parent
    assert info.lexical_corpus_path == target.settings.corpus_path
    assert info.manifest_path == target.settings.manifest_path


def test_baseline_is_activated_only_by_explicit_selection(tmp_path: Path) -> None:
    settings = settings_for_test(tmp_path)
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    settings.qdrant_path.mkdir(parents=True)
    settings.corpus_path.write_text("", encoding="utf-8")
    settings.manifest_path.write_text("{}", encoding="utf-8")
    controller = LabController(settings)

    with pytest.raises(FileNotFoundError):
        controller.activate_index(corpus)
    info = controller.activate_index(corpus, "baseline")

    assert info.target_mode == "baseline"
    assert info.qdrant_path == settings.qdrant_path
    assert controller.active_settings is settings


def test_gui_runtime_backend_selection_changes_experiment_identity(tmp_path: Path) -> None:
    settings = settings_for_test(tmp_path)
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    controller = LabController(settings)

    local = controller.plan_index_target(corpus, qdrant_backend="local")
    server = controller.plan_index_target(
        corpus,
        qdrant_backend="server",
        qdrant_url="http://localhost:6333",
    )

    assert local.settings.qdrant_mode == "local"
    assert server.settings.qdrant_mode == "server"
    assert server.settings.qdrant_url == "http://localhost:6333"
    assert local.experiment_root != server.experiment_root
    assert local.settings.qdrant_collection != server.settings.qdrant_collection
    assert server.settings.qdrant_collection.startswith("experiment_")
    assert server.settings.corpus_path.is_relative_to(server.experiment_root)  # type: ignore[arg-type]
    assert server.settings.manifest_path.is_relative_to(server.experiment_root)  # type: ignore[arg-type]


def test_embedding_batch_size_changes_experiment_identity(tmp_path: Path) -> None:
    settings = settings_for_test(tmp_path)
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    controller = LabController(settings)

    batch_64 = controller.plan_index_target(corpus, embedding_batch_size=64)
    batch_128 = controller.plan_index_target(corpus, embedding_batch_size=128)

    assert batch_64.settings.embedding_batch_size == 64
    assert batch_128.settings.embedding_batch_size == 128
    assert batch_64.experiment_root != batch_128.experiment_root
    assert batch_64.settings.qdrant_collection != batch_128.settings.qdrant_collection


def test_unavailable_server_does_not_fall_back_to_local(
    tmp_path: Path, monkeypatch: Any
) -> None:
    settings = settings_for_test(tmp_path)
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    controller = LabController(settings)
    monkeypatch.setattr(
        "app.desktop.controller.qdrant_backend_status",
        lambda *args, **kwargs: QdrantBackendStatus(
            mode="server",
            available=False,
            endpoint="http://localhost:6333",
            error="connection refused",
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="Qdrant Server is unavailable at http://localhost:6333",
    ):
        controller.build_index(
            corpus,
            qdrant_backend="server",
            qdrant_url="http://localhost:6333",
        )
    with pytest.raises(
        RuntimeError,
        match="Qdrant Server is unavailable at http://localhost:6333",
    ):
        controller.activate_index(
            corpus,
            qdrant_backend="server",
            qdrant_url="http://localhost:6333",
        )

    assert controller.active_settings is None
    assert controller.active_index_info() is None
    assert not settings.qdrant_path.exists()


def test_server_activation_and_query_use_server_settings(
    tmp_path: Path, monkeypatch: Any
) -> None:
    settings = settings_for_test(tmp_path)
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    controller = LabController(settings)
    target = controller.plan_index_target(
        corpus,
        qdrant_backend="server",
        qdrant_url="http://localhost:6333",
    )
    target.settings.corpus_path.parent.mkdir(parents=True)
    target.settings.corpus_path.write_text("", encoding="utf-8")
    target.settings.manifest_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "app.desktop.controller.qdrant_backend_status",
        lambda *args, **kwargs: QdrantBackendStatus(
            mode="server",
            available=True,
            endpoint="http://localhost:6333",
            server_version="1.13.6",
        ),
    )
    monkeypatch.setattr(
        "app.desktop.controller.qdrant_collection_exists",
        lambda *args, **kwargs: True,
    )
    FakeApplicationContainer.seen_settings.clear()
    FakeBenchmarkRun.seen_settings.clear()
    monkeypatch.setattr("app.desktop.controller.ApplicationContainer", FakeApplicationContainer)
    monkeypatch.setattr("app.desktop.controller.BenchmarkRun", FakeBenchmarkRun)

    info = controller.activate_index(
        corpus,
        qdrant_backend="server",
        qdrant_url="http://localhost:6333",
    )
    controller.ask("Вопрос", "fast")

    assert info.qdrant_backend == "server"
    assert info.qdrant_endpoint == "http://localhost:6333"
    assert info.qdrant_collection == target.settings.qdrant_collection
    assert controller.active_settings is not None
    assert controller.active_settings.qdrant_mode == "server"
    assert controller.active_settings.qdrant_collection == target.settings.qdrant_collection
    assert FakeApplicationContainer.seen_settings[-1].qdrant_mode == "server"
