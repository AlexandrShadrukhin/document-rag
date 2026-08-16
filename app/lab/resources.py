from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import psutil


@dataclass(frozen=True)
class NvidiaSample:
    name: str | None = None
    utilization_percent: float | None = None
    vram_used_bytes: int | None = None
    vram_total_bytes: int | None = None
    temperature_c: float | None = None
    power_draw_w: float | None = None
    power_limit_w: float | None = None


@dataclass(frozen=True)
class ResourceSample:
    timestamp: str
    elapsed_seconds: float
    stage: str
    system_cpu_percent: float | None
    process_cpu_percent: float | None
    system_ram_used_bytes: int | None
    system_ram_percent: float | None
    process_rss_bytes: int | None
    disk_read_bytes: int | None
    disk_write_bytes: int | None
    disk_read_bytes_per_second: float | None
    disk_write_bytes_per_second: float | None
    disk_free_bytes: int | None
    index_directory_size_bytes: int | None
    gpu_name: str | None
    gpu_utilization_percent: float | None
    gpu_vram_used_bytes: int | None
    gpu_vram_total_bytes: int | None
    gpu_temperature_c: float | None
    gpu_power_draw_w: float | None
    gpu_power_limit_w: float | None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += (Path(root) / name).stat().st_size
            except OSError:
                continue
    return total


def sample_nvidia() -> NvidiaSample:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return NvidiaSample()
    fields = [
        "name",
        "utilization.gpu",
        "memory.used",
        "memory.total",
        "temperature.gpu",
        "power.draw",
        "power.limit",
    ]
    try:
        completed = subprocess.run(
            [
                executable,
                f"--query-gpu={','.join(fields)}",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
        values = [part.strip() for part in completed.stdout.splitlines()[0].split(",")]
        if len(values) != len(fields):
            return NvidiaSample()

        def number(value: str) -> float | None:
            try:
                return float(value)
            except ValueError:
                return None

        used_mb, total_mb = number(values[2]), number(values[3])
        return NvidiaSample(
            name=values[0] or None,
            utilization_percent=number(values[1]),
            vram_used_bytes=int(used_mb * 1024 * 1024) if used_mb is not None else None,
            vram_total_bytes=int(total_mb * 1024 * 1024) if total_mb is not None else None,
            temperature_c=number(values[4]),
            power_draw_w=number(values[5]),
            power_limit_w=number(values[6]),
        )
    except (OSError, subprocess.SubprocessError, IndexError):
        return NvidiaSample()


class ResourceMonitor:
    def __init__(
        self,
        disk_path: Path,
        index_path: Path | None = None,
        interval_seconds: float = 1.0,
        index_size_interval_seconds: float = 30.0,
        on_sample: Callable[[ResourceSample], None] | None = None,
    ) -> None:
        self.disk_path = disk_path.resolve()
        self.index_path = index_path.resolve() if index_path else None
        self.interval_seconds = interval_seconds
        self.index_size_interval_seconds = index_size_interval_seconds
        self.on_sample = on_sample
        self.samples: list[ResourceSample] = []
        self._process = psutil.Process()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = 0.0
        self._stage = "idle"
        self._stage_lock = threading.Lock()

    def set_stage(self, stage: str) -> None:
        with self._stage_lock:
            self._stage = stage

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self.samples = []
        self._stop.clear()
        self._started = time.perf_counter()
        self._process.cpu_percent(None)
        psutil.cpu_percent(None)
        self._thread = threading.Thread(target=self._run, name="resource-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> list[ResourceSample]:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=max(2.0, self.interval_seconds * 2))
        return list(self.samples)

    def _safe(self, function: Callable[[], object]) -> object | None:
        try:
            return function()
        except (OSError, psutil.Error, RuntimeError):
            return None

    def _run(self) -> None:
        last_disk = self._safe(psutil.disk_io_counters)
        last_elapsed = 0.0
        last_index_size: int | None = None
        last_size_check = -self.index_size_interval_seconds
        while not self._stop.is_set():
            elapsed = time.perf_counter() - self._started
            memory = self._safe(psutil.virtual_memory)
            process_memory = self._safe(self._process.memory_info)
            disk_io = self._safe(psutil.disk_io_counters)
            disk_usage = self._safe(lambda: shutil.disk_usage(self.disk_path))
            interval = max(elapsed - last_elapsed, 1e-9)
            read_rate = write_rate = None
            if disk_io is not None and last_disk is not None:
                read_rate = max(0.0, (disk_io.read_bytes - last_disk.read_bytes) / interval)
                write_rate = max(0.0, (disk_io.write_bytes - last_disk.write_bytes) / interval)
            if self.index_path and elapsed - last_size_check >= self.index_size_interval_seconds:
                last_index_size = self._safe(lambda: directory_size(self.index_path))  # type: ignore[assignment]
                last_size_check = elapsed
            gpu_value = self._safe(sample_nvidia)
            gpu = gpu_value if isinstance(gpu_value, NvidiaSample) else NvidiaSample()
            with self._stage_lock:
                stage = self._stage
            sample = ResourceSample(
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                elapsed_seconds=elapsed,
                stage=stage,
                system_cpu_percent=self._safe(lambda: psutil.cpu_percent(None)),  # type: ignore[arg-type]
                process_cpu_percent=self._safe(lambda: self._process.cpu_percent(None)),  # type: ignore[arg-type]
                system_ram_used_bytes=getattr(memory, "used", None),
                system_ram_percent=getattr(memory, "percent", None),
                process_rss_bytes=getattr(process_memory, "rss", None),
                disk_read_bytes=getattr(disk_io, "read_bytes", None),
                disk_write_bytes=getattr(disk_io, "write_bytes", None),
                disk_read_bytes_per_second=read_rate,
                disk_write_bytes_per_second=write_rate,
                disk_free_bytes=getattr(disk_usage, "free", None),
                index_directory_size_bytes=last_index_size,
                gpu_name=gpu.name,
                gpu_utilization_percent=gpu.utilization_percent,
                gpu_vram_used_bytes=gpu.vram_used_bytes,
                gpu_vram_total_bytes=gpu.vram_total_bytes,
                gpu_temperature_c=gpu.temperature_c,
                gpu_power_draw_w=gpu.power_draw_w,
                gpu_power_limit_w=gpu.power_limit_w,
            )
            self.samples.append(sample)
            if self.on_sample:
                try:
                    self.on_sample(sample)
                except Exception:
                    pass
            last_disk = disk_io
            last_elapsed = elapsed
            self._stop.wait(self.interval_seconds)


def summarize_resources(samples: list[ResourceSample]) -> dict[str, float | int | None]:
    def maximum(name: str) -> float | int | None:
        values = [getattr(sample, name) for sample in samples]
        available = [value for value in values if value is not None]
        return max(available) if available else None

    first = samples[0] if samples else None
    last = samples[-1] if samples else None

    def delta(name: str) -> int | None:
        if first is None or last is None:
            return None
        before = getattr(first, name)
        after = getattr(last, name)
        if before is None or after is None:
            return None
        return max(0, int(after) - int(before))

    return {
        "peak_process_rss_bytes": maximum("process_rss_bytes"),
        "peak_system_ram_used_bytes": maximum("system_ram_used_bytes"),
        "peak_system_ram_percent": maximum("system_ram_percent"),
        "peak_system_cpu_percent": maximum("system_cpu_percent"),
        "peak_process_cpu_percent": maximum("process_cpu_percent"),
        "peak_gpu_utilization_percent": maximum("gpu_utilization_percent"),
        "peak_gpu_vram_used_bytes": maximum("gpu_vram_used_bytes"),
        "peak_gpu_temperature_c": maximum("gpu_temperature_c"),
        "peak_gpu_power_draw_w": maximum("gpu_power_draw_w"),
        "peak_disk_read_bytes_per_second": maximum("disk_read_bytes_per_second"),
        "peak_disk_write_bytes_per_second": maximum("disk_write_bytes_per_second"),
        "disk_read_delta_bytes": delta("disk_read_bytes"),
        "disk_write_delta_bytes": delta("disk_write_bytes"),
        "disk_free_before_bytes": first.disk_free_bytes if first else None,
        "disk_free_after_bytes": last.disk_free_bytes if last else None,
        "minimum_disk_free_bytes": min(
            (sample.disk_free_bytes for sample in samples if sample.disk_free_bytes is not None),
            default=None,
        ),
    }
