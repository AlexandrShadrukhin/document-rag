from __future__ import annotations

import json
import os
import time
from pathlib import Path

_WINDOWS_REPLACE_RETRY_DELAYS = (0.05, 0.1, 0.2, 0.4, 0.8)
_IS_WINDOWS = os.name == "nt"


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
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(self.entries, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        try:
            _replace(temporary, self.path)
        except OSError:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            raise
