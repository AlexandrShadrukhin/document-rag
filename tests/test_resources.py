from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import app.lab.resources as resources_module
from app.lab.resources import ResourceMonitor, ResourceSample, summarize_resources


def test_resource_sampler_isolates_gpu_monitor_failure(
    tmp_path: Path, monkeypatch: Any
) -> None:
    monkeypatch.setattr(
        resources_module,
        "sample_nvidia",
        lambda: (_ for _ in ()).throw(RuntimeError("nvidia-smi failed")),
    )
    monitor = ResourceMonitor(tmp_path, interval_seconds=0.05)
    monitor.set_stage("smoke")
    monitor.start()
    time.sleep(0.12)
    samples = monitor.stop()
    assert samples
    assert all(sample.stage == "smoke" for sample in samples)
    assert all(sample.gpu_utilization_percent is None for sample in samples)


def test_resource_summary_separates_peaks_and_disk_deltas() -> None:
    def sample(**updates: object) -> ResourceSample:
        values: dict[str, object] = {
            "timestamp": "2026-01-01T00:00:00+0000",
            "elapsed_seconds": 0.0,
            "stage": "test",
            "system_cpu_percent": 10.0,
            "process_cpu_percent": 20.0,
            "system_ram_used_bytes": 100,
            "system_ram_percent": 25.0,
            "process_rss_bytes": 50,
            "disk_read_bytes": 1_000,
            "disk_write_bytes": 2_000,
            "disk_read_bytes_per_second": 100.0,
            "disk_write_bytes_per_second": 200.0,
            "disk_free_bytes": 10_000,
            "index_directory_size_bytes": 0,
            "gpu_name": "GPU",
            "gpu_utilization_percent": 30.0,
            "gpu_vram_used_bytes": 300,
            "gpu_vram_total_bytes": 1_000,
            "gpu_temperature_c": 40.0,
            "gpu_power_draw_w": 50.0,
            "gpu_power_limit_w": 100.0,
        }
        values.update(updates)
        return ResourceSample(**values)  # type: ignore[arg-type]

    summary = summarize_resources(
        [
            sample(),
            sample(
                system_cpu_percent=90.0,
                process_cpu_percent=80.0,
                system_ram_percent=70.0,
                process_rss_bytes=500,
                disk_read_bytes=4_000,
                disk_write_bytes=8_000,
                disk_read_bytes_per_second=700.0,
                disk_write_bytes_per_second=900.0,
                disk_free_bytes=9_000,
                gpu_utilization_percent=95.0,
                gpu_vram_used_bytes=800,
                gpu_temperature_c=75.0,
                gpu_power_draw_w=120.0,
            ),
        ]
    )
    assert summary["peak_process_rss_bytes"] == 500
    assert summary["peak_system_ram_percent"] == 70.0
    assert summary["peak_system_cpu_percent"] == 90.0
    assert summary["peak_process_cpu_percent"] == 80.0
    assert summary["peak_gpu_utilization_percent"] == 95.0
    assert summary["peak_gpu_vram_used_bytes"] == 800
    assert summary["peak_gpu_temperature_c"] == 75.0
    assert summary["peak_gpu_power_draw_w"] == 120.0
    assert summary["disk_read_delta_bytes"] == 3_000
    assert summary["disk_write_delta_bytes"] == 6_000
    assert summary["disk_free_before_bytes"] == 10_000
    assert summary["disk_free_after_bytes"] == 9_000
