import json
from pathlib import Path
from typing import Any

from app.retrieval import manifest as manifest_module
from app.retrieval.manifest import IndexManifest


def test_record_retries_windows_replace_permission_error(
    tmp_path: Path, monkeypatch: Any
) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest = IndexManifest(manifest_path)
    original_replace: Any = Path.replace
    replace_attempts = 0
    sleeps: list[float] = []

    def flaky_replace(path: Path, target: Path) -> Path:
        nonlocal replace_attempts
        replace_attempts += 1
        if replace_attempts < 3:
            raise PermissionError(5, "Access is denied")
        return original_replace(path, target)

    monkeypatch.setattr(manifest_module, "_IS_WINDOWS", True)
    monkeypatch.setattr(manifest_module.time, "sleep", sleeps.append)
    monkeypatch.setattr(Path, "replace", flaky_replace)

    manifest.record("document.txt", "hash", "document-id")

    assert replace_attempts == 3
    assert sleeps == [0.05, 0.1]
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == {
        "document.txt": {"file_hash": "hash", "document_id": "document-id"}
    }
    assert not manifest_path.with_suffix(".tmp").exists()
