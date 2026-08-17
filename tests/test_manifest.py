import json
from pathlib import Path
from typing import Any

from app.retrieval import manifest as manifest_module
from app.retrieval.manifest import IndexManifest


def test_loads_existing_manifest_format(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    payload = {"legacy.pdf": {"file_hash": "old-hash", "document_id": "old-id"}}
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    manifest = IndexManifest(manifest_path)

    assert manifest.unchanged("legacy.pdf", "old-hash")
    assert manifest.previous_document_id("legacy.pdf") == "old-id"


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
    manifest.persist()

    assert replace_attempts == 3
    assert sleeps == [0.05, 0.1]
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == {
        "document.txt": {"file_hash": "hash", "document_id": "document-id"}
    }
    assert not manifest_path.with_suffix(".tmp").exists()


def test_record_does_not_serialize_full_manifest_each_time(
    tmp_path: Path, monkeypatch: Any
) -> None:
    manifest = IndexManifest(tmp_path / "manifest.json")
    dumps_calls = 0
    original_dumps = manifest_module.json.dumps

    def tracked_dumps(*args: Any, **kwargs: Any) -> str:
        nonlocal dumps_calls
        dumps_calls += 1
        return original_dumps(*args, **kwargs)

    monkeypatch.setattr(manifest_module.json, "dumps", tracked_dumps)
    for index in range(100):
        manifest.record(f"document-{index}.pdf", f"hash-{index}", f"id-{index}")

    assert dumps_calls == 0
    manifest.persist()
    assert dumps_calls == 1
    assert len(json.loads(manifest.path.read_text(encoding="utf-8"))) == 100
