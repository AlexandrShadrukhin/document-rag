from __future__ import annotations

import logging
import os
import platform
import shutil
import subprocess
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from app.config import PROJECT_ROOT, Settings
from app.indexing import IndexingStats
from app.lab.benchmark import BenchmarkRun, corpus_snapshot
from app.lab.devices import DeviceSelection, resolve_torch_device
from app.lab.environment import EnvironmentSnapshot, collect_environment
from app.lab.experiments import ExperimentConfig, ExperimentRunner
from app.lab.progress import ProgressCallback, ProgressEvent
from app.lab.resources import ResourceSample, directory_size
from app.retrieval.vector_store import (
    QdrantBackendStatus,
    qdrant_backend_status,
    qdrant_collection_exists,
)
from app.runtime import ApplicationContainer
from app.schemas import RetrievalMode
from app.wiki.corpus import WikiCorpusBuilder, WikiCorpusConfig
from app.wiki.download import download_dump

logger = logging.getLogger(__name__)

IndexTargetMode = Literal["isolated", "baseline"]
QdrantBackendMode = Literal["local", "server"]
DEFAULT_INDEX_TARGET: IndexTargetMode = "isolated"


class BaselineModificationNotConfirmed(RuntimeError):
    pass


class NoActiveRAGIndex(RuntimeError):
    pass


@dataclass(frozen=True)
class IndexTarget:
    mode: IndexTargetMode
    settings: Settings
    experiment_root: Path | None = None

    @property
    def label(self) -> str:
        return "Isolated benchmark / experiment" if self.mode == "isolated" else "Baseline index"


@dataclass(frozen=True)
class ActiveIndexInfo:
    corpus_path: Path
    index_path: Path
    qdrant_path: Path
    qdrant_backend: str
    qdrant_endpoint: str
    qdrant_collection: str
    indexed_chunks: int | None
    lexical_corpus_path: Path
    manifest_path: Path
    target_mode: IndexTargetMode
    target_label: str
    embedding_model: str
    embedding_device: str
    reranker_model: str
    rerank_candidates: int
    ollama_model: str
    qdrant_advisory: str | None


@dataclass(frozen=True)
class PreflightReport:
    environment: EnvironmentSnapshot
    embedding_device: DeviceSelection
    reranker_device: DeviceSelection
    configured_ollama_model: str
    ollama_model_available: bool
    corpus: dict[str, object]
    index_exists: bool
    index_size_bytes: int
    qdrant: QdrantBackendStatus
    warnings: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "environment": self.environment.as_dict(),
            "embedding_device": asdict(self.embedding_device),
            "reranker_device": asdict(self.reranker_device),
            "configured_ollama_model": self.configured_ollama_model,
            "ollama_model_available": self.ollama_model_available,
            "corpus": self.corpus,
            "index_exists": self.index_exists,
            "index_size_bytes": self.index_size_bytes,
            "qdrant": self.qdrant.as_dict(),
            "warnings": list(self.warnings),
        }


@dataclass
class OperationResult:
    value: object
    benchmark_path: Path
    summary: dict[str, object]


class LabController:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.container: ApplicationContainer | None = None
        self.last_benchmark_path: Path | None = None
        self._last_preparation: dict[str, object] | None = None
        self.current_corpus_path: Path | None = None
        self.active_settings: Settings | None = None
        self.current_index_target: IndexTarget | None = None

    def preflight(self, corpus_path: Path | None = None) -> PreflightReport:
        corpus = (corpus_path or self.settings.wiki_corpus_path).expanduser().resolve()
        environment = collect_environment(self.settings.ollama_base_url, PROJECT_ROOT)
        embedding_device = resolve_torch_device(self.settings.embedding_device)
        reranker_device = resolve_torch_device(self.settings.reranker_device)
        model_names = set(environment.ollama_models)
        configured = self.settings.ollama_model
        model_available = configured in model_names or f"{configured}:latest" in model_names
        warnings: list[str] = []
        qdrant = qdrant_backend_status(
            self.settings.qdrant_mode,
            self.settings.qdrant_path,
            self.settings.qdrant_url,
            self.settings.qdrant_api_key,
        )
        if not environment.ollama_available:
            warnings.append("Ollama unavailable; corpus generation and indexing remain available")
        elif not model_available:
            warnings.append(f"Configured Ollama model is missing: {configured}")
        if environment.disk_free_bytes < 10 * 1024**3:
            warnings.append("Less than 10 GB free on the project filesystem")
        qdrant_index_exists = (
            self.settings.qdrant_path.exists()
            if self.settings.qdrant_mode == "local"
            else qdrant_collection_exists(
                self.settings.qdrant_url,
                self.settings.qdrant_collection,
                self.settings.qdrant_api_key,
            )
        )
        if not qdrant_index_exists or not self.settings.corpus_path.exists():
            warnings.append("RAG index is missing or incomplete")
        if self.settings.qdrant_mode == "local":
            warnings.append(
                "Local Qdrant is intended for small development datasets; "
                "use Qdrant Server for larger experiments"
            )
        elif not qdrant.available:
            warnings.append(f"Qdrant Server unavailable at {self.settings.qdrant_url}")
        return PreflightReport(
            environment=environment,
            embedding_device=embedding_device,
            reranker_device=reranker_device,
            configured_ollama_model=configured,
            ollama_model_available=model_available,
            corpus=corpus_snapshot(corpus),
            index_exists=qdrant_index_exists and self.settings.corpus_path.exists(),
            index_size_bytes=directory_size(self.settings.qdrant_path),
            qdrant=qdrant,
            warnings=tuple(warnings),
        )

    def disk_advisory(self, target_path: Path, target_corpus_bytes: int) -> str | None:
        usage = shutil.disk_usage(target_path.expanduser().resolve().parent)
        if usage.free < target_corpus_bytes * 2:
            return (
                f"Free disk is {usage.free / 1024**3:.1f} GB. The requested corpus is "
                f"{target_corpus_bytes / 1024**3:.1f} GB and indexes/cache may require "
                "substantially more space."
            )
        return None

    @staticmethod
    def _combined_callback(
        benchmark: BenchmarkRun,
        callback: ProgressCallback | None,
    ) -> ProgressCallback:
        def update(event: ProgressEvent) -> None:
            benchmark.progress_callback(event)
            if callback:
                callback(event)

        return update

    def download_wiki_dump(
        self,
        url: str | None = None,
        destination: Path | None = None,
        callback: ProgressCallback | None = None,
    ) -> Path:
        return download_dump(
            url or self.settings.wiki_dump_url,
            destination or self.settings.wiki_dump_path,
            callback,
        )

    def build_wiki_corpus(
        self,
        config: WikiCorpusConfig,
        callback: ProgressCallback | None = None,
        on_resource: Callable[[ResourceSample], None] | None = None,
    ) -> OperationResult:
        benchmark = BenchmarkRun(
            "dataset-generation",
            self.settings,
            config.output,
            config={
                "generator": {
                    **asdict(config),
                    "source": str(config.source),
                    "output": str(config.output),
                }
            },
            on_resource_sample=on_resource,
        )
        benchmark.start()
        combined = self._combined_callback(benchmark, callback)
        try:
            stats = WikiCorpusBuilder(config).build(combined)
            summary = {
                **asdict(stats),
                "corpus_gb": stats.corpus_gb,
                "pdf_corpus_ready": datetime.now(UTC).isoformat(),
                "download_excluded": True,
            }
            path = benchmark.finish("completed", summary)
        except Exception as error:
            benchmark.finish("failed", error=str(error))
            raise
        self.last_benchmark_path = path
        return OperationResult(stats, path, benchmark.summary)

    def _reset_container(self) -> None:
        if self.container is not None:
            self.container.close()
        self.container = None

    def plan_index_target(
        self,
        corpus_path: Path,
        mode: IndexTargetMode = DEFAULT_INDEX_TARGET,
        qdrant_backend: QdrantBackendMode | None = None,
        qdrant_url: str | None = None,
    ) -> IndexTarget:
        corpus = corpus_path.expanduser().resolve()
        backend = qdrant_backend or self.settings.qdrant_mode
        if backend not in {"local", "server"}:
            raise ValueError(f"Unsupported Qdrant backend: {backend}")
        server_url = (qdrant_url or self.settings.qdrant_url).strip()
        if backend == "server" and not server_url:
            raise ValueError("Qdrant Server URL is required")
        runtime_settings = (
            self.settings
            if backend == self.settings.qdrant_mode and server_url == self.settings.qdrant_url
            else self.settings.model_copy(
                update={"qdrant_mode": backend, "qdrant_url": server_url}
            )
        )
        if mode == "baseline":
            return IndexTarget(mode, runtime_settings)
        if mode != "isolated":
            raise ValueError(f"Unsupported index target: {mode}")
        index_config: dict[str, object] = {
            "chunk_size": self.settings.chunk_size,
            "chunk_overlap": self.settings.chunk_overlap,
            "embedding_model": self.settings.embedding_model,
        }
        if backend == "server":
            index_config.update(
                {
                    "qdrant_mode": "server",
                    "qdrant_url": server_url,
                }
            )
        experiment = ExperimentConfig(
            name=f"{platform.system().lower()}-{corpus.name}-baseline-config",
            corpus=corpus,
            index_config=index_config,
            force_reindex=True,
        )
        isolated, root, _ = ExperimentRunner(runtime_settings).prepare(experiment)
        isolated = isolated.model_copy(update={"benchmarks_path": root / "benchmarks"})
        return IndexTarget(mode, isolated, root)

    def check_qdrant_server(self, url: str | None = None) -> QdrantBackendStatus:
        return qdrant_backend_status(
            "server",
            self.settings.qdrant_path,
            (url or self.settings.qdrant_url).strip(),
            self.settings.qdrant_api_key,
        )

    @staticmethod
    def _require_qdrant_server(settings: Settings) -> QdrantBackendStatus:
        status = qdrant_backend_status(
            "server",
            settings.qdrant_path,
            settings.qdrant_url,
            settings.qdrant_api_key,
        )
        if not status.available:
            raise RuntimeError(f"Qdrant Server is unavailable at {settings.qdrant_url}")
        return status

    def activate_index(
        self,
        corpus_path: Path,
        mode: IndexTargetMode = DEFAULT_INDEX_TARGET,
        qdrant_backend: QdrantBackendMode | None = None,
        qdrant_url: str | None = None,
    ) -> ActiveIndexInfo:
        corpus = corpus_path.expanduser().resolve()
        target = self.plan_index_target(corpus, mode, qdrant_backend, qdrant_url)
        settings = target.settings
        if settings.qdrant_mode == "server":
            self._require_qdrant_server(settings)
        missing = []
        if not settings.corpus_path.is_file():
            missing.append(str(settings.corpus_path))
        if not settings.manifest_path.is_file():
            missing.append(str(settings.manifest_path))
        if settings.qdrant_mode == "local" and not settings.qdrant_path.is_dir():
            missing.append(str(settings.qdrant_path))
        if settings.qdrant_mode == "server" and not qdrant_collection_exists(
            settings.qdrant_url,
            settings.qdrant_collection,
            settings.qdrant_api_key,
        ):
            missing.append(f"{settings.qdrant_url}/collections/{settings.qdrant_collection}")
        if missing:
            details = "\n- ".join(missing)
            raise FileNotFoundError(f"RAG index is incomplete or missing:\n- {details}")
        self._reset_container()
        self.current_corpus_path = corpus
        self.active_settings = settings
        self.current_index_target = target
        self._last_preparation = None
        info = self.active_index_info()
        assert info is not None
        return info

    def active_index_info(self) -> ActiveIndexInfo | None:
        if (
            self.active_settings is None
            or self.current_index_target is None
            or self.current_corpus_path is None
        ):
            return None
        settings = self.active_settings
        embedding_device = (
            self.container.embeddings.device
            if self.container is not None and self.container.embeddings is not None
            else resolve_torch_device(settings.embedding_device).selected
        )
        indexed_chunks = self._count_lexical_chunks(settings.corpus_path)
        return ActiveIndexInfo(
            corpus_path=self.current_corpus_path,
            index_path=settings.corpus_path.parent,
            qdrant_path=settings.qdrant_path,
            qdrant_backend=settings.qdrant_mode,
            qdrant_endpoint=(
                str(settings.qdrant_path)
                if settings.qdrant_mode == "local"
                else settings.qdrant_url
            ),
            qdrant_collection=settings.qdrant_collection,
            indexed_chunks=indexed_chunks,
            lexical_corpus_path=settings.corpus_path,
            manifest_path=settings.manifest_path,
            target_mode=self.current_index_target.mode,
            target_label=(
                "Isolated experiment"
                if self.current_index_target.mode == "isolated"
                else "Baseline"
            ),
            embedding_model=settings.embedding_model,
            embedding_device=embedding_device,
            reranker_model=settings.reranker_model,
            rerank_candidates=settings.rerank_candidates,
            ollama_model=settings.ollama_model,
            qdrant_advisory=self.local_qdrant_advisory(settings, indexed_chunks),
        )

    @staticmethod
    def local_qdrant_advisory(
        settings: Settings, indexed_chunks: int | None = None
    ) -> str | None:
        if settings.qdrant_mode != "local":
            return None
        suffix = (
            " This index contains more than 20,000 chunks."
            if indexed_chunks is not None and indexed_chunks > 20_000
            else ""
        )
        return (
            "Local Qdrant is intended for small development datasets. "
            "For larger experiments use Qdrant Server." + suffix
        )

    def build_index(
        self,
        corpus_path: Path,
        callback: ProgressCallback | None = None,
        on_resource: Callable[[ResourceSample], None] | None = None,
        target_mode: IndexTargetMode = DEFAULT_INDEX_TARGET,
        baseline_confirmed: bool = False,
        qdrant_backend: QdrantBackendMode | None = None,
        qdrant_url: str | None = None,
    ) -> OperationResult:
        corpus_path = corpus_path.expanduser().resolve()
        if target_mode == "baseline" and not baseline_confirmed:
            raise BaselineModificationNotConfirmed(
                "Baseline index modification requires explicit confirmation"
            )
        target = self.plan_index_target(
            corpus_path,
            target_mode,
            qdrant_backend,
            qdrant_url,
        )
        target_settings = target.settings
        if target_settings.qdrant_mode == "server":
            self._require_qdrant_server(target_settings)
        benchmark = BenchmarkRun(
            "rag-preparation",
            target_settings,
            corpus_path,
            config={
                "index_target": target.mode,
                "experiment_root": (
                    str(target.experiment_root) if target.experiment_root else None
                ),
            },
            on_resource_sample=on_resource,
        )
        benchmark.start()
        combined = self._combined_callback(benchmark, callback)
        try:
            self._reset_container()
            initialization_started = time.perf_counter()
            self.container = ApplicationContainer(target_settings, load_bm25=False)
            runtime_initialization = time.perf_counter() - initialization_started
            index_started_at = datetime.now(UTC)
            target_settings.qdrant_path.parent.mkdir(parents=True, exist_ok=True)
            disk_before = shutil.disk_usage(target_settings.qdrant_path.parent).free
            stats = self.container.index(
                corpus_path,
                show_progress=False,
                progress_callback=combined,
            )
            stats.runtime_initialization_seconds = runtime_initialization
            stats.total_seconds += runtime_initialization
            disk_after = shutil.disk_usage(target_settings.qdrant_path.parent).free
            index_size = directory_size(target_settings.corpus_path.parent)
            summary = self.index_summary(stats)
            summary.update(
                {
                    "index_build_start": index_started_at.isoformat(),
                    "index_ready": datetime.now(UTC).isoformat(),
                    "rag_prepare_seconds": stats.total_seconds,
                    "free_disk_before_bytes": disk_before,
                    "free_disk_after_bytes": disk_after,
                    "approx_bytes_written": max(0, disk_before - disk_after),
                    "final_index_size_bytes": index_size,
                    "index_target": target.mode,
                    "experiment_root": (
                        str(target.experiment_root) if target.experiment_root else None
                    ),
                    "qdrant_path": str(target_settings.qdrant_path),
                    "lexical_corpus_path": str(target_settings.corpus_path),
                    "manifest_path": str(target_settings.manifest_path),
                    "selected_embedding_device": (
                        self.container.embeddings.device
                        if self.container.embeddings is not None
                        else None
                    ),
                }
            )
            path = benchmark.finish("completed", summary)
        except Exception as error:
            benchmark.finish("failed", error=str(error))
            self._reset_container()
            raise
        self.active_settings = target_settings
        self.current_index_target = target
        self.current_corpus_path = corpus_path
        self._last_preparation = {
            "index_build_start": index_started_at.isoformat(),
            "index_ready": datetime.now(UTC).isoformat(),
            "rag_prepare_seconds": stats.total_seconds,
        }
        self.last_benchmark_path = path
        return OperationResult(stats, path, benchmark.summary)

    @staticmethod
    def _count_lexical_chunks(path: Path, stop_after: int = 20_001) -> int | None:
        if not path.is_file():
            return None
        try:
            with path.open("r", encoding="utf-8") as stream:
                return sum(1 for _, _line in zip(range(stop_after), stream, strict=False))
        except OSError:
            return None

    @staticmethod
    def index_summary(stats: IndexingStats) -> dict[str, object]:
        total = max(stats.total_seconds, 1e-9)
        megabytes = stats.bytes_processed / 1024**2
        accounted = sum(
            (
                stats.discovery_seconds,
                stats.manifest_handling_seconds,
                stats.parsing_seconds,
                stats.cleaning_seconds,
                stats.chunking_seconds,
                stats.embedding_model_loading_seconds,
                stats.embedding_seconds,
                stats.qdrant_initialization_seconds,
                stats.qdrant_upsert_seconds,
                stats.lexical_corpus_seconds,
                stats.bm25_build_seconds,
                stats.runtime_initialization_seconds,
            )
        )
        return {
            **stats.as_dict(),
            "pages_per_second": stats.pages_parsed / total,
            "chunks_per_second": stats.chunks_created / total,
            "megabytes_per_second": megabytes / total,
            "other_seconds": max(0.0, stats.total_seconds - accounted),
        }

    def ask(
        self,
        query: str,
        mode: RetrievalMode,
        callback: ProgressCallback | None = None,
        on_resource: Callable[[ResourceSample], None] | None = None,
    ) -> OperationResult:
        if (
            self.active_settings is None
            or self.current_index_target is None
            or self.current_corpus_path is None
        ):
            raise NoActiveRAGIndex("No active RAG index selected.")
        benchmark = BenchmarkRun(
            "query",
            self.active_settings,
            self.current_corpus_path,
            config={"mode": mode},
            on_resource_sample=on_resource,
        )
        benchmark.start()
        benchmark.set_stage("starting_rag")
        if callback:
            callback(ProgressEvent("starting_rag", "started"))
        wall_started = time.perf_counter()
        first_query_start = datetime.now(UTC)
        try:
            if self.container is None:
                self.container = ApplicationContainer(self.active_settings)
            service_started = time.perf_counter()
            service = self.container.rag_service(mode)
            query_runtime_initialization = time.perf_counter() - service_started
            benchmark.set_stage("query")
            response = service.answer(query)
            wall_seconds = time.perf_counter() - wall_started
            query_payload = {
                "query": query,
                "mode": mode,
                "answer": response.answer,
                "is_answerable": response.is_answerable,
                "sources": [source.model_dump(mode="json") for source in response.sources],
                "timings": response.timings.model_dump(mode="json"),
                "wall_seconds": wall_seconds,
            }
            benchmark.append_query(query_payload)
            summary: dict[str, object] = {
                "mode": mode,
                "answerable": response.is_answerable,
                "top_source": (
                    response.sources[0].model_dump(mode="json") if response.sources else None
                ),
                "base_retrieval_ms": response.timings.base_retrieval_ms,
                "reranker_ms": response.timings.reranker_ms,
                "retrieval_total_ms": response.timings.retrieval_total_ms,
                "generation_ms": response.timings.generation_ms,
                "total_ms": response.timings.total_ms,
                "first_query_start": first_query_start.isoformat(),
                "first_query_end": datetime.now(UTC).isoformat(),
                "first_query_retrieval_seconds": response.timings.retrieval_total_ms / 1000,
                "first_query_reranker_seconds": response.timings.reranker_ms / 1000,
                "first_query_generation_seconds": response.timings.generation_ms / 1000,
                "first_query_total_seconds": wall_seconds,
                "response_total_seconds": response.timings.total_ms / 1000,
                "query_runtime_initialization_seconds": query_runtime_initialization,
                "selected_embedding_device": (
                    self.container.embeddings.device
                    if self.container.embeddings is not None
                    else None
                ),
                "selected_reranker_device": (
                    self.container.reranker.device
                    if self.container.reranker is not None
                    else None
                ),
            }
            if self._last_preparation:
                summary.update(self._last_preparation)
                summary["machine_time_to_first_answer"] = (
                    float(self._last_preparation["rag_prepare_seconds"]) + wall_seconds
                )
            path = benchmark.finish("completed", summary)
        except Exception as error:
            benchmark.finish("failed", error=str(error))
            raise
        if callback:
            callback(ProgressEvent("ready", "completed", message="Answer ready"))
        self.last_benchmark_path = path
        return OperationResult(response, path, benchmark.summary)

    def open_benchmark_folder(self) -> None:
        if self.last_benchmark_path is None:
            raise FileNotFoundError("No benchmark run has been created yet")
        path = str(self.last_benchmark_path)
        if platform.system() == "Windows":
            os.startfile(path)  # type: ignore[attr-defined]
            return
        command = "open" if platform.system() == "Darwin" else "xdg-open"
        executable = shutil.which(command)
        if executable is None:
            raise RuntimeError(f"Cannot open folder automatically: {path}")
        subprocess.Popen([executable, path])

    def close(self) -> None:
        self._reset_container()
