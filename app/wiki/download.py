from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

import httpx

from app.lab.progress import ProgressCallback, ProgressEvent, emit


def download_dump(
    url: str,
    destination: Path,
    callback: ProgressCallback | None = None,
) -> Path:
    destination = destination.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    aria2 = shutil.which("aria2c")
    if aria2:
        emit(callback, ProgressEvent("download", "started", message="aria2c"))
        subprocess.run(
            [
                aria2,
                "--continue=true",
                "--max-connection-per-server=8",
                "--split=8",
                f"--dir={destination.parent}",
                f"--out={destination.name}",
                url,
            ],
            check=True,
        )
        emit(callback, ProgressEvent("download", "completed", message=str(destination)))
        return destination

    partial = destination.with_suffix(destination.suffix + ".part")
    existing = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={existing}-"} if existing else {}
    started = time.perf_counter()
    with httpx.stream("GET", url, headers=headers, follow_redirects=True, timeout=None) as response:
        response.raise_for_status()
        append = existing > 0 and response.status_code == 206
        if not append:
            existing = 0
        expected = response.headers.get("Content-Length")
        total = existing + int(expected) if expected and expected.isdigit() else None
        mode = "ab" if append else "wb"
        emit(callback, ProgressEvent("download", "started", current=existing, total=total))
        downloaded = existing
        with partial.open(mode) as stream:
            for chunk in response.iter_bytes(1024 * 1024):
                stream.write(chunk)
                downloaded += len(chunk)
                emit(
                    callback,
                    ProgressEvent(
                        "download",
                        current=downloaded,
                        total=total,
                        details={
                            "elapsed_seconds": time.perf_counter() - started,
                            "downloaded_bytes": downloaded,
                        },
                    ),
                )
    os.replace(partial, destination)
    emit(callback, ProgressEvent("download", "completed", current=downloaded, total=total))
    return destination
