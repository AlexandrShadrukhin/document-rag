from __future__ import annotations

import csv
import hashlib
import json
import re
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from app.config import Settings
from app.lab.benchmark import BenchmarkRun, write_json
from app.lab.resources import directory_size
from app.runtime import ApplicationContainer
from app.schemas import SearchResult

REINDEX_FIELDS = frozenset(
    {"chunk_size", "chunk_overlap", "embedding_model", "embedding_batch_size"}
)
QUERY_TIME_FIELDS = frozenset(
    {
        "dense_top_k",
        "bm25_top_k",
        "fusion_top_k",
        "final_top_k",
        "rrf_k",
        "enable_reranker",
        "reranker_model",
        "rerank_candidates",
        "reranker_batch_size",
        "reranker_device",
        "confidence_dense_threshold",
        "confidence_dense_no_agreement_threshold",
        "confidence_reranker_threshold",
        "ollama_model",
        "ollama_base_url",
        "ollama_timeout_seconds",
        "llm_temperature",
        "max_context_chunks",
    }
)


def normalize_settings(values: dict[str, object]) -> dict[str, object]:
    return {key.lower(): value for key, value in values.items()}


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    corpus: Path
    index_config: dict[str, object] = field(default_factory=dict)
    query_config: dict[str, object] = field(default_factory=dict)
    evaluation_set: Path | None = None
    mode: str = "quality"
    queries: tuple[str, ...] = ()
    force_reindex: bool = False

    @classmethod
    def load(cls, path: Path) -> ExperimentConfig:
        payload = json.loads(path.read_text(encoding="utf-8"))
        base = path.parent

        def resolved(value: str | None) -> Path | None:
            if value is None:
                return None
            candidate = Path(value).expanduser()
            return candidate.resolve() if candidate.is_absolute() else (base / candidate).resolve()

        corpus = resolved(str(payload["corpus"]))
        if corpus is None:
            raise ValueError("Experiment corpus is required")
        return cls(
            name=str(payload["name"]),
            corpus=corpus,
            index_config=normalize_settings(payload.get("index_config", {})),
            query_config=normalize_settings(payload.get("query_config", {})),
            evaluation_set=resolved(payload.get("evaluation_set")),
            mode=str(payload.get("mode", "quality")),
            queries=tuple(str(query) for query in payload.get("queries", [])),
            force_reindex=bool(payload.get("force_reindex", False)),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "corpus": str(self.corpus),
            "evaluation_set": str(self.evaluation_set) if self.evaluation_set else None,
        }


def changed_reindex_fields(settings: Settings, config: ExperimentConfig) -> set[str]:
    changed = set()
    for field_name, value in config.index_config.items():
        if field_name in REINDEX_FIELDS and getattr(settings, field_name) != value:
            changed.add(field_name)
    return changed


def requires_reindex(settings: Settings, config: ExperimentConfig) -> bool:
    return config.force_reindex or bool(changed_reindex_fields(settings, config))


def _safe_name(name: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_-]+", "-", name).strip("-")
    return normalized[:48] or "experiment"


def matches_expected(result: SearchResult, record: dict[str, Any]) -> bool:
    if result.chunk.filename != record["expected_filename"]:
        return False
    page = record.get("expected_page")
    if page is not None and result.chunk.page_number != page:
        return False
    substring = record.get("expected_text")
    if substring:
        expected = " ".join(str(substring).casefold().split())
        actual = " ".join(result.chunk.text.casefold().split())
        return expected in actual
    return True


def evaluate_retriever(retriever: object, dataset: Path) -> dict[str, object]:
    records = json.loads(dataset.read_text(encoding="utf-8"))
    ranks: list[int | None] = []
    negatives: list[bool] = []
    base: list[float] = []
    reranker: list[float] = []
    total: list[float] = []
    for record in records:
        decision = retriever.retrieve_with_decision(record["query"])
        base.append(decision.base_retrieval_ms)
        reranker.append(decision.reranker_ms)
        total.append(decision.retrieval_total_ms)
        if record["answerable"]:
            rank = next(
                (
                    index
                    for index, result in enumerate(decision.results, start=1)
                    if matches_expected(result, record)
                ),
                None,
            )
            ranks.append(rank)
        else:
            negatives.append(not decision.is_answerable)
    positives = len(ranks)
    return {
        "positive_queries": positives,
        "negative_queries": len(negatives),
        "recall_at_1": sum(rank == 1 for rank in ranks) / positives if positives else 0.0,
        "recall_at_3": (
            sum(rank is not None and rank <= 3 for rank in ranks) / positives
            if positives
            else 0.0
        ),
        "recall_at_5": (
            sum(rank is not None and rank <= 5 for rank in ranks) / positives
            if positives
            else 0.0
        ),
        "mrr": (
            sum(1.0 / rank for rank in ranks if rank is not None) / positives
            if positives
            else 0.0
        ),
        "negative_rejection_accuracy": (
            sum(negatives) / len(negatives) if negatives else 0.0
        ),
        "mean_base_retrieval_ms": statistics.mean(base) if base else 0.0,
        "mean_reranker_ms": statistics.mean(reranker) if reranker else 0.0,
        "mean_retrieval_total_ms": statistics.mean(total) if total else 0.0,
    }


class ExperimentRunner:
    def __init__(self, baseline: Settings) -> None:
        self.baseline = baseline

    def prepare(self, config: ExperimentConfig) -> tuple[Settings, Path, bool]:
        updates = {**config.index_config, **config.query_config}
        unknown = set(updates) - set(Settings.model_fields)
        if unknown:
            raise ValueError(f"Unknown settings in experiment: {sorted(unknown)}")
        reindex = requires_reindex(self.baseline, config)
        fingerprint = hashlib.sha256(
            json.dumps(config.as_dict(), sort_keys=True, default=str).encode()
        ).hexdigest()[:8]
        experiment_id = f"{_safe_name(config.name)}-{fingerprint}"
        root = self.baseline.experiments_path / experiment_id
        if reindex:
            index = root / "index"
            updates.update(
                {
                    "qdrant_path": index / "qdrant",
                    "corpus_path": index / "corpus.jsonl",
                    "manifest_path": index / "manifest.json",
                    "qdrant_collection": f"experiment_{fingerprint}",
                }
            )
        settings = self.baseline.model_copy(update=updates)
        return settings, root, reindex

    def run(self, config: ExperimentConfig) -> dict[str, object]:
        settings, root, reindex = self.prepare(config)
        root.mkdir(parents=True, exist_ok=True)
        write_json(root / "experiment_config.json", config.as_dict())
        benchmark = BenchmarkRun(
            "experiment",
            settings,
            config.corpus,
            config={"experiment_name": config.name, "requires_reindex": reindex},
        )
        benchmark.start()
        container = ApplicationContainer(settings, load_bm25=not reindex)
        try:
            indexing: dict[str, object] | None = None
            if reindex:
                stats = container.index(
                    config.corpus,
                    show_progress=True,
                    progress_callback=benchmark.progress_callback,
                )
                indexing = stats.as_dict()
            mode = config.mode
            if mode not in {"fast", "quality"}:
                raise ValueError("Experiment mode must be fast or quality")
            metrics = None
            if config.evaluation_set:
                metrics = evaluate_retriever(container.retriever(mode), config.evaluation_set)
            query_rows: list[dict[str, object]] = []
            if config.queries:
                service = container.rag_service(mode)
                for query in config.queries:
                    response = service.answer(query)
                    row = {
                        "query": query,
                        "answer": response.answer,
                        "is_answerable": response.is_answerable,
                        "sources": [source.model_dump(mode="json") for source in response.sources],
                        "timings": response.timings.model_dump(mode="json"),
                    }
                    query_rows.append(row)
                    benchmark.append_query(row)
            result = {
                "experiment": config.name,
                "requires_reindex": reindex,
                "changed_reindex_fields": sorted(changed_reindex_fields(self.baseline, config)),
                "settings": {
                    "chunk_size": settings.chunk_size,
                    "chunk_overlap": settings.chunk_overlap,
                    "embedding_model": settings.embedding_model,
                    "embedding_batch_size": settings.embedding_batch_size,
                    "rerank_candidates": settings.rerank_candidates,
                    "final_top_k": settings.final_top_k,
                    "ollama_model": settings.ollama_model,
                    "mode": mode,
                },
                "indexing": indexing,
                "retrieval_metrics": metrics,
                "queries": query_rows,
                "index_size_bytes": directory_size(settings.qdrant_path) if reindex else None,
                "benchmark_path": str(benchmark.path),
            }
            benchmark.finish("completed", result)
            write_json(root / "result.json", result)
            self._append_summary(result)
            return result
        except Exception as error:
            benchmark.finish("failed", error=str(error))
            raise
        finally:
            container.close()

    def _append_summary(self, result: dict[str, object]) -> None:
        path = self.baseline.experiments_path / "results.csv"
        metrics = result.get("retrieval_metrics") or {}
        indexing = result.get("indexing") or {}
        settings = result["settings"]
        row = {
            "experiment": result["experiment"],
            "chunk_size": settings["chunk_size"],
            "chunk_overlap": settings["chunk_overlap"],
            "embedding_model": settings["embedding_model"],
            "chunks": indexing.get("chunks_created"),
            "index_size_bytes": result["index_size_bytes"],
            "build_time_seconds": indexing.get("total_seconds"),
            "recall_at_1": metrics.get("recall_at_1"),
            "recall_at_3": metrics.get("recall_at_3"),
            "recall_at_5": metrics.get("recall_at_5"),
            "mrr": metrics.get("mrr"),
            "negative_rejection": metrics.get("negative_rejection_accuracy"),
            "retrieval_ms": metrics.get("mean_base_retrieval_ms"),
            "reranker_ms": metrics.get("mean_reranker_ms"),
            "generation_ms": (
                statistics.mean(
                    row["timings"]["generation_ms"] for row in result.get("queries", [])
                )
                if result.get("queries")
                else None
            ),
            "peak_process_ram_bytes": result.get("peak_process_rss_bytes"),
            "peak_vram_bytes": result.get("peak_gpu_vram_used_bytes"),
        }
        exists = path.exists()
        with path.open("a", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            if not exists:
                writer.writeheader()
            writer.writerow(row)
