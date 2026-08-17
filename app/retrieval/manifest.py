from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

_WINDOWS_REPLACE_RETRY_DELAYS = (0.05, 0.1, 0.2, 0.4, 0.8)
_IS_WINDOWS = os.name == "nt"
logger = logging.getLogger(__name__)


@dataclass
class ManifestPersistTimings:
    serialization_seconds: float = 0.0
    write_seconds: float = 0.0
    atomic_replace_seconds: float = 0.0
    persist_count: int = 0


def _replace(temporary: Path, destination: Path) -> None:
    delays = _WINDOWS_REPLACE_RETRY_DELAYS if _IS_WINDOWS else ()
    for delay in (*delays, None):
        try:
            temporary.replace(destination)
            return
        except OSError:
            if delay is None:
                raise
            time.sleep(delay)


class IndexManifest:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.entries: dict[str, dict[str, str]] = self._load()
        self.dirty = False
        self.timings = ManifestPersistTimings()

    def _load(self) -> dict[str, dict[str, str]]:
        if not self.path.exists():
            return {}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def unchanged(self, source: str, file_hash: str) -> bool:
        return self.entries.get(source, {}).get("file_hash") == file_hash

    def previous_document_id(self, source: str) -> str | None:
        return self.entries.get(source, {}).get("document_id")

    def record(self, source: str, file_hash: str, document_id: str) -> None:
        self.entries[source] = {"file_hash": file_hash, "document_id": document_id}
        self.dirty = True

    def persist(self, *, force: bool = False) -> None:
        if not self.dirty and not force:
            return

        started = time.perf_counter()
        serialized = json.dumps(self.entries, ensure_ascii=False, indent=2)
        serialization_seconds = time.perf_counter() - started

        temporary = self.path.with_suffix(".tmp")
        started = time.perf_counter()
        temporary.write_text(serialized, encoding="utf-8")
        write_seconds = time.perf_counter() - started
        try:
            started = time.perf_counter()
            _replace(temporary, self.path)
            replace_seconds = time.perf_counter() - started
        except OSError:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            raise
        self.dirty = False
        self.timings.serialization_seconds += serialization_seconds
        self.timings.write_seconds += write_seconds
        self.timings.atomic_replace_seconds += replace_seconds
        self.timings.persist_count += 1
        logger.info(
            "Manifest persisted: entries=%d serialization=%.3fs write=%.3fs "
            "atomic_replace=%.3fs",
            len(self.entries),
            serialization_seconds,
            write_seconds,
            replace_seconds,
        )
