from __future__ import annotations

import json
from pathlib import Path


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
        temporary.replace(self.path)
