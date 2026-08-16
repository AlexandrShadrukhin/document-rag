from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings  # noqa: E402
from app.lab.benchmark import BenchmarkRun  # noqa: E402
from app.runtime import ApplicationContainer  # noqa: E402


def rate(value: float, seconds: float) -> float:
    return value / seconds if seconds > 0 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the complete ingestion pipeline")
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    base_settings = get_settings()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    experiment_root = base_settings.experiments_path / f"ingestion-{stamp}"
    settings = base_settings.model_copy(
        update={
            "qdrant_path": experiment_root / "index" / "qdrant",
            "corpus_path": experiment_root / "index" / "corpus.jsonl",
            "manifest_path": experiment_root / "index" / "manifest.json",
            "qdrant_collection": "benchmark_documents",
        }
    )
    corpus_path = args.path.expanduser().resolve()
    run = BenchmarkRun(
        "rag-preparation",
        settings,
        corpus_path,
        config={"limit": args.limit, "index_namespace": str(experiment_root)},
    )
    run.start()
    disk_before = shutil.disk_usage(experiment_root.parent).free
    container_started = time.perf_counter()
    container = ApplicationContainer(settings, load_bm25=False)
    runtime_initialization = time.perf_counter() - container_started
    try:
        stats = container.index(
            corpus_path,
            show_progress=True,
            limit=args.limit,
            progress_callback=run.progress_callback,
        )
        stats.runtime_initialization_seconds = runtime_initialization
        stats.total_seconds += runtime_initialization
        megabytes = stats.bytes_processed / (1024 * 1024)
        final_index_size = sum(
            path.stat().st_size
            for path in experiment_root.rglob("*")
            if path.is_file()
        )
        disk_after = shutil.disk_usage(experiment_root.parent).free
        accounted = sum(
            value
            for key, value in stats.timings().items()
            if key not in {"total", "indexing"}
        )
        summary = {
            **stats.as_dict(),
            "final_index_size_bytes": final_index_size,
            "free_disk_before_bytes": disk_before,
            "free_disk_after_bytes": disk_after,
            "approx_bytes_written": max(0, disk_before - disk_after),
            "other_seconds": max(0.0, stats.total_seconds - accounted),
            "pages_per_second": rate(stats.pages_parsed, stats.total_seconds),
            "chunks_per_second": rate(stats.chunks_created, stats.total_seconds),
            "megabytes_per_second": rate(megabytes, stats.total_seconds),
            "selected_embedding_device": (
                container.embeddings.device if container.embeddings is not None else None
            ),
            "download_excluded": True,
        }
        artifacts = run.finish("completed", summary)
    except Exception as error:
        run.finish("failed", error=str(error))
        raise
    finally:
        container.close()
    print("\nIngestion benchmark")
    print(f"Files: {stats.files_indexed}")
    print(f"Pages: {stats.pages_parsed}")
    print(f"Chunks: {stats.chunks_created}")
    print(f"Data: {megabytes:.2f} MB")
    for stage, seconds in stats.timings().items():
        print(f"{stage.capitalize()}: {seconds:.3f} s")
    print(f"Files/sec: {rate(stats.files_indexed, stats.total_seconds):.2f}")
    print(f"Pages/sec: {rate(stats.pages_parsed, stats.total_seconds):.2f}")
    print(f"Chunks/sec: {rate(stats.chunks_created, stats.total_seconds):.2f}")
    print(f"MB/sec: {rate(megabytes, stats.total_seconds):.2f}")
    print(f"Final index: {final_index_size / 1024**2:.2f} MB")
    print(f"Peak process RAM: {summary.get('peak_process_rss_bytes', 'N/A')} bytes")
    print(f"Peak system RAM: {summary.get('peak_system_ram_used_bytes', 'N/A')} bytes")
    print(f"Peak CPU: {summary.get('peak_system_cpu_percent', 'N/A')}%")
    print(f"Peak GPU: {summary.get('peak_gpu_utilization_percent', 'N/A')}%")
    print(f"Peak VRAM: {summary.get('peak_gpu_vram_used_bytes', 'N/A')} bytes")
    print(f"Free disk before: {disk_before / 1024**3:.2f} GB")
    print(f"Free disk after: {disk_after / 1024**3:.2f} GB")
    print(f"Artifacts: {artifacts}")
    print("SUMMARY " + json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
