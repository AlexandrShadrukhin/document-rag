from __future__ import annotations

import importlib.metadata
import os
import platform
import shutil
import socket
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx
import psutil


@dataclass(frozen=True)
class EnvironmentSnapshot:
    os: str
    os_version: str
    architecture: str
    hostname: str
    python_version: str
    cpu_name: str
    physical_cpu_cores: int | None
    logical_cpu_cores: int | None
    total_ram_bytes: int
    available_ram_bytes: int
    disk_filesystem: str | None
    disk_free_bytes: int
    disk_total_bytes: int
    pytorch_version: str | None
    cuda_available: bool
    cuda_device_name: str | None
    cuda_vram_bytes: int | None
    mps_available: bool
    ollama_available: bool
    ollama_models: list[str]
    aria2c_available: bool
    docker_available: bool
    nvidia_smi_available: bool

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _run_optional(args: list[str], timeout: float = 2.0) -> str | None:
    try:
        completed = subprocess.run(
            args,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return completed.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _cpu_name() -> str:
    if platform.system() == "Darwin":
        name = _run_optional(["sysctl", "-n", "machdep.cpu.brand_string"]) or ""
        if not name:
            hardware = _run_optional(["system_profiler", "SPHardwareDataType"]) or ""
            for line in hardware.splitlines():
                if line.strip().startswith("Chip:"):
                    name = line.partition(":")[2].strip()
                    break
    elif platform.system() == "Windows":
        name = os.environ.get("PROCESSOR_IDENTIFIER", "")
    elif Path("/proc/cpuinfo").is_file():
        for line in Path("/proc/cpuinfo").read_text(errors="ignore").splitlines():
            if line.lower().startswith("model name"):
                name = line.partition(":")[2].strip()
                break
    else:
        name = ""
    name = name or platform.processor().strip()
    return name or "Unknown CPU"


def _filesystem(path: Path) -> str | None:
    resolved = path.resolve()
    best_match: tuple[int, str] | None = None
    for partition in psutil.disk_partitions(all=True):
        mount = Path(partition.mountpoint)
        try:
            resolved.relative_to(mount)
        except ValueError:
            continue
        candidate = (len(str(mount)), partition.fstype or "unknown")
        if best_match is None or candidate[0] > best_match[0]:
            best_match = candidate
    return best_match[1] if best_match else None


def _torch_info() -> tuple[str | None, bool, str | None, int | None, bool]:
    try:
        version = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        return None, False, None, None, False
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        cuda_name = torch.cuda.get_device_name(0) if cuda_available else None
        cuda_vram = (
            int(torch.cuda.get_device_properties(0).total_memory)
            if cuda_available
            else None
        )
        mps_backend = getattr(torch.backends, "mps", None)
        mps_available = bool(mps_backend and mps_backend.is_available())
        return version, cuda_available, cuda_name, cuda_vram, mps_available
    except Exception:
        return version, False, None, None, False


def ollama_status(base_url: str, timeout: float = 1.5) -> tuple[bool, list[str]]:
    try:
        response = httpx.get(f"{base_url.rstrip('/')}/api/tags", timeout=timeout)
        response.raise_for_status()
        models = response.json().get("models", [])
        names = sorted(
            str(item.get("name")) for item in models if isinstance(item, dict) and item.get("name")
        )
        return True, names
    except (httpx.HTTPError, KeyError, TypeError, ValueError):
        return False, []


def collect_environment(base_url: str, disk_path: Path) -> EnvironmentSnapshot:
    memory = psutil.virtual_memory()
    disk = shutil.disk_usage(disk_path.resolve())
    torch_version, cuda, cuda_name, cuda_vram, mps = _torch_info()
    ollama, models = ollama_status(base_url)
    return EnvironmentSnapshot(
        os=platform.system(),
        os_version=platform.version(),
        architecture=platform.machine(),
        hostname=socket.gethostname(),
        python_version=sys.version.split()[0],
        cpu_name=_cpu_name(),
        physical_cpu_cores=psutil.cpu_count(logical=False),
        logical_cpu_cores=psutil.cpu_count(logical=True),
        total_ram_bytes=int(memory.total),
        available_ram_bytes=int(memory.available),
        disk_filesystem=_filesystem(disk_path),
        disk_free_bytes=int(disk.free),
        disk_total_bytes=int(disk.total),
        pytorch_version=torch_version,
        cuda_available=cuda,
        cuda_device_name=cuda_name,
        cuda_vram_bytes=cuda_vram,
        mps_available=mps,
        ollama_available=ollama,
        ollama_models=models,
        aria2c_available=shutil.which("aria2c") is not None,
        docker_available=shutil.which("docker") is not None,
        nvidia_smi_available=shutil.which("nvidia-smi") is not None,
    )
