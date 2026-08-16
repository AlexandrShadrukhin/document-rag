from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.config import Settings
from app.lab.benchmark import BenchmarkRun, benchmark_comparison, corpus_snapshot
from app.lab.environment import EnvironmentSnapshot
from app.lab.progress import ProgressEvent


def fake_environment() -> EnvironmentSnapshot:
    return EnvironmentSnapshot(
        os="TestOS",
        os_version="1",
        architecture="x64",
        hostname="test",
        python_version="3.12",
        cpu_name="CPU",
        physical_cpu_cores=4,
        logical_cpu_cores=8,
        total_ram_bytes=16,
        available_ram_bytes=8,
        disk_filesystem="testfs",
        disk_free_bytes=100,
        disk_total_bytes=200,
        pytorch_version="2",
        cuda_available=False,
        cuda_device_name=None,
        cuda_vram_bytes=None,
        mps_available=False,
        ollama_available=False,
        ollama_models=[],
        aria2c_available=False,
        docker_available=False,
        nvidia_smi_available=False,
    )


def test_benchmark_metadata_is_serialized(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setattr("app.lab.benchmark.collect_environment", lambda *args: fake_environment())
    settings = Settings(
        benchmarks_path=tmp_path / "benchmarks",
        experiments_path=tmp_path / "experiments",
        qdrant_path=tmp_path / "index/qdrant",
        corpus_path=tmp_path / "index/corpus.jsonl",
        manifest_path=tmp_path / "index/manifest.json",
        wiki_dump_path=tmp_path / "source/wiki.bz2",
        wiki_corpus_path=tmp_path / "corpus",
        resource_sample_interval_seconds=0.2,
    )
    run = BenchmarkRun("test", settings, tmp_path / "corpus")
    run.start()
    run.progress_callback(ProgressEvent("stage", "started"))
    path = run.finish("completed", {"value": 1})
    metadata = json.loads((path / "run.json").read_text())
    assert metadata["status"] == "completed"
    assert metadata["download_excluded"] is True
    assert (path / "config.json").is_file()
    assert (path / "hardware.json").is_file()
    assert (path / "stages.csv").is_file()
    assert (path / "resources.csv").is_file()
    assert (path / "queries.jsonl").is_file()


def test_corpus_fingerprint_and_benchmark_comparison(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.pdf").write_bytes(b"pdf-a")
    (corpus / "b.pdf").write_bytes(b"pdf-b")
    snapshot = corpus_snapshot(corpus)
    assert snapshot["pdf_count"] == 2
    assert snapshot["bytes"] == 10
    assert snapshot["fingerprint_kind"] == "relative_path_and_size_sha256"

    run = tmp_path / "run-a"
    run.mkdir()
    (run / "run.json").write_text(
        json.dumps(
            {
                "run_id": "run-a",
                "kind": "rag-preparation",
                "status": "completed",
                "elapsed_seconds": 12.5,
                "corpus": snapshot,
                "git": {"commit_sha": "abc", "dirty": False},
                "summary": {
                    "pages_parsed": 20,
                    "chunks_created": 40,
                    "rag_prepare_seconds": 12.0,
                    "peak_process_rss_bytes": 123,
                    "final_index_size_bytes": 456,
                },
            }
        ),
        encoding="utf-8",
    )
    rows = benchmark_comparison([run])
    assert rows == [
        {
            "run_id": "run-a",
            "kind": "rag-preparation",
            "status": "completed",
            "elapsed_seconds": 12.5,
            "corpus_fingerprint": snapshot["fingerprint"],
            "corpus_bytes": 10,
            "pdf_count": 2,
            "git_commit_sha": "abc",
            "git_dirty": False,
            "os": None,
            "cpu": None,
            "gpu": None,
            "python": None,
            "chunk_size": None,
            "chunk_overlap": None,
            "embedding_model": None,
            "embedding_device": None,
            "qdrant_backend": None,
            "qdrant_server_version": None,
            "qdrant_collection": None,
            "pages": 20,
            "chunks": 40,
            "rag_prepare_seconds": 12.0,
            "first_query_total_seconds": None,
            "peak_process_rss_bytes": 123,
            "peak_gpu_vram_used_bytes": None,
            "final_index_size_bytes": 456,
        }
    ]
