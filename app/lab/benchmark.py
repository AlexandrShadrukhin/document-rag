from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.config import PROJECT_ROOT, Settings
from app.lab.environment import EnvironmentSnapshot, collect_environment
from app.lab.progress import ProgressEvent
from app.lab.resources import ResourceMonitor, ResourceSample, summarize_resources
from app.retrieval.vector_store import qdrant_backend_status


def _json_default(value: object) -> str:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )


def git_state(project_root: Path = PROJECT_ROOT) -> dict[str, object]:
    def run(args: list[str]) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=project_root,
                check=True,
                capture_output=True,
                text=True,
                timeout=3,
            )
            return result.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None

    commit = run(["rev-parse", "HEAD"])
    status = run(["status", "--porcelain"])
    return {"commit_sha": commit, "dirty": bool(status) if status is not None else None}


def settings_snapshot(settings: Settings) -> dict[str, object]:
    payload = settings.model_dump(mode="json")
    payload.pop("qdrant_api_key", None)
    payload["qdrant_backend_status"] = qdrant_backend_status(
        settings.qdrant_mode,
        settings.qdrant_path,
        settings.qdrant_url,
        settings.qdrant_api_key,
    ).as_dict()
    return payload


def corpus_snapshot(path: Path) -> dict[str, object]:
    resolved = path.expanduser().resolve()
    files = [resolved] if resolved.is_file() else sorted(resolved.rglob("*.pdf"))
    total = 0
    portable_fingerprint = hashlib.sha256()
    for file_path in files:
        try:
            size = file_path.stat().st_size
            total += size
            relative = (
                Path(file_path.name) if resolved.is_file() else file_path.relative_to(resolved)
            )
            portable_fingerprint.update(f"{relative.as_posix()}:{size}\n".encode())
        except OSError:
            continue
    manifest_path = resolved / "corpus_manifest.json" if resolved.is_dir() else None
    manifest_fingerprint = None
    if manifest_path and manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest_fingerprint = manifest.get("aggregate_pdf_fingerprint")
        except (OSError, ValueError):
            manifest_fingerprint = None
    return {
        "path": str(resolved),
        "pdf_count": len(files),
        "bytes": total,
        "fingerprint": manifest_fingerprint or portable_fingerprint.hexdigest(),
        "fingerprint_kind": (
            "aggregate_pdf_sha256" if manifest_fingerprint else "relative_path_and_size_sha256"
        ),
        "manifest_path": str(manifest_path) if manifest_path and manifest_path.is_file() else None,
    }


def make_run_id(kind: str, config: dict[str, object]) -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    encoded = json.dumps(config, sort_keys=True, default=_json_default).encode()
    fingerprint = hashlib.sha256(encoded).hexdigest()[:8]
    safe_kind = "".join(character if character.isalnum() else "-" for character in kind)
    return f"{stamp}-{safe_kind}-{fingerprint}"


@dataclass
class StageRecord:
    stage: str
    started_seconds: float
    ended_seconds: float | None = None
    duration_seconds: float | None = None
    status: str = "running"
    message: str = ""


class BenchmarkRun:
    def __init__(
        self,
        kind: str,
        settings: Settings,
        corpus_path: Path,
        config: dict[str, object] | None = None,
        on_resource_sample: Any | None = None,
    ) -> None:
        self.kind = kind
        self.settings = settings
        self.corpus = corpus_snapshot(corpus_path)
        self.config = settings_snapshot(settings)
        if config:
            self.config.update(config)
        self.run_id = make_run_id(kind, self.config)
        self.path = settings.benchmarks_path / self.run_id
        self.path.mkdir(parents=True, exist_ok=False)
        self.hardware: EnvironmentSnapshot = collect_environment(
            settings.ollama_base_url, self.path
        )
        self.started_at = datetime.now(UTC)
        self.started = time.perf_counter()
        self.stages: list[StageRecord] = []
        self._current: StageRecord | None = None
        self._lock = threading.Lock()
        self.summary: dict[str, object] = {}
        self.monitor = ResourceMonitor(
            disk_path=self.path,
            index_path=settings.qdrant_path,
            interval_seconds=settings.resource_sample_interval_seconds,
            on_sample=on_resource_sample,
        )
        write_json(self.path / "config.json", self.config)
        write_json(self.path / "hardware.json", self.hardware.as_dict())
        (self.path / "queries.jsonl").touch()

    def start(self) -> None:
        if self._current is None:
            self.set_stage("initializing")
        self.monitor.start()

    def set_stage(self, stage: str, message: str = "") -> None:
        now = time.perf_counter() - self.started
        with self._lock:
            if self._current is not None and self._current.ended_seconds is None:
                self._current.ended_seconds = now
                self._current.duration_seconds = now - self._current.started_seconds
                self._current.status = "completed"
            self._current = StageRecord(stage=stage, started_seconds=now, message=message)
            self.stages.append(self._current)
        self.monitor.set_stage(stage)

    def progress_callback(self, event: ProgressEvent) -> None:
        if event.kind == "started" or not self._current or self._current.stage != event.stage:
            self.set_stage(event.stage, event.message)

    def append_query(self, payload: dict[str, object]) -> None:
        with (self.path / "queries.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False, default=_json_default) + "\n")

    def finish(
        self,
        status: str,
        summary: dict[str, object] | None = None,
        error: str | None = None,
    ) -> Path:
        elapsed = time.perf_counter() - self.started
        with self._lock:
            if self._current is not None and self._current.ended_seconds is None:
                self._current.ended_seconds = elapsed
                self._current.duration_seconds = elapsed - self._current.started_seconds
                self._current.status = "completed" if status == "completed" else status
        samples = self.monitor.stop()
        self.summary = summary or {}
        self.summary.update(summarize_resources(samples))
        self._write_csv(self.path / "stages.csv", [asdict(stage) for stage in self.stages])
        self._write_csv(
            self.path / "resources.csv", [sample.as_dict() for sample in samples]
        )
        write_json(
            self.path / "run.json",
            {
                "run_id": self.run_id,
                "kind": self.kind,
                "status": status,
                "started_at": self.started_at.isoformat(),
                "ended_at": datetime.now(UTC).isoformat(),
                "elapsed_seconds": elapsed,
                "download_excluded": True,
                "corpus": self.corpus,
                "git": git_state(),
                "summary": self.summary,
                "error": error,
            },
        )
        return self.path

    @staticmethod
    def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
        if not rows:
            path.write_text("", encoding="utf-8")
            return
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def resource_summary(samples: list[ResourceSample]) -> dict[str, object]:
    return summarize_resources(samples)


def load_benchmark_run(path: Path) -> dict[str, object]:
    run_path = path / "run.json" if path.is_dir() else path
    payload = json.loads(run_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Benchmark run must be a JSON object: {run_path}")
    return payload


def _read_json_object(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (OSError, ValueError):
        return {}


def benchmark_comparison(paths: list[Path]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in paths:
        run = load_benchmark_run(path)
        run_path = path / "run.json" if path.is_dir() else path
        config = _read_json_object(run_path.parent / "config.json")
        hardware = _read_json_object(run_path.parent / "hardware.json")
        corpus = run.get("corpus") if isinstance(run.get("corpus"), dict) else {}
        summary = run.get("summary") if isinstance(run.get("summary"), dict) else {}
        git = run.get("git") if isinstance(run.get("git"), dict) else {}
        qdrant = (
            config.get("qdrant_backend_status")
            if isinstance(config.get("qdrant_backend_status"), dict)
            else {}
        )
        rows.append(
            {
                "run_id": run.get("run_id"),
                "kind": run.get("kind"),
                "status": run.get("status"),
                "elapsed_seconds": run.get("elapsed_seconds"),
                "corpus_fingerprint": corpus.get("fingerprint"),
                "corpus_bytes": corpus.get("bytes"),
                "pdf_count": corpus.get("pdf_count"),
                "git_commit_sha": git.get("commit_sha"),
                "git_dirty": git.get("dirty"),
                "os": hardware.get("os"),
                "cpu": hardware.get("cpu_name"),
                "gpu": hardware.get("cuda_device_name"),
                "python": hardware.get("python_version"),
                "chunk_size": config.get("chunk_size"),
                "chunk_overlap": config.get("chunk_overlap"),
                "embedding_model": config.get("embedding_model"),
                "embedding_device": summary.get("selected_embedding_device"),
                "qdrant_backend": config.get("qdrant_mode"),
                "qdrant_server_version": qdrant.get("server_version"),
                "qdrant_collection": config.get("qdrant_collection"),
                "pages": summary.get("pages_parsed"),
                "chunks": summary.get("chunks_created"),
                "rag_prepare_seconds": summary.get("rag_prepare_seconds"),
                "first_query_total_seconds": summary.get("first_query_total_seconds"),
                "peak_process_rss_bytes": summary.get("peak_process_rss_bytes"),
                "peak_gpu_vram_used_bytes": summary.get("peak_gpu_vram_used_bytes"),
                "final_index_size_bytes": summary.get("final_index_size_bytes"),
            }
        )
    return rows
