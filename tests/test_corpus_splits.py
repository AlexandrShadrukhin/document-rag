from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from scripts.create_corpus_splits import (
    MANIFEST_NAME,
    PdfFile,
    SplitSpec,
    corpus_fingerprint,
    create_splits,
    scan_pdfs,
    select_split_files,
    verify_splits,
)


def _master(tmp_path: Path, sizes: list[int]) -> Path:
    source = tmp_path / "master"
    source.mkdir()
    for index, size in enumerate(sizes):
        (source / f"wiki_{index:03d}.pdf").write_bytes(bytes([index]) * size)
    return source


def test_threshold_file_is_included_and_splits_are_nested(tmp_path: Path) -> None:
    files = scan_pdfs(_master(tmp_path, [40, 60, 25, 75, 100]))
    specs = (SplitSpec("small", 100), SplitSpec("medium", 180), SplitSpec("large", 300))

    selected = select_split_files(files, specs)

    assert [file.name for file in selected["small"]] == [
        "wiki_000.pdf",
        "wiki_001.pdf",
    ]
    assert [file.name for file in selected["medium"]] == [
        "wiki_000.pdf",
        "wiki_001.pdf",
        "wiki_002.pdf",
        "wiki_003.pdf",
    ]
    assert set(selected["small"]) < set(selected["medium"]) < set(selected["large"])


def test_fingerprint_is_deterministic_and_uses_order_name_and_size(tmp_path: Path) -> None:
    first = PdfFile(tmp_path / "a.pdf", "a.pdf", 10)
    second = PdfFile(tmp_path / "b.pdf", "b.pdf", 20)

    fingerprint = corpus_fingerprint([first, second])

    assert fingerprint == corpus_fingerprint([first, second])
    assert fingerprint != corpus_fingerprint([second, first])
    assert fingerprint != corpus_fingerprint([first, PdfFile(second.path, second.name, 21)])


def test_create_is_idempotent_and_verifiable(tmp_path: Path) -> None:
    source = _master(tmp_path, [40, 60, 25, 75, 100])
    output = tmp_path / "splits"
    specs = (SplitSpec("small", 100), SplitSpec("large", 300))

    first = create_splits(source, output, specs)
    second = create_splits(source, output, specs)
    verified = verify_splits(source, output, specs)

    assert [result.fingerprint for result in first] == [result.fingerprint for result in second]
    assert [result.fingerprint for result in first] == [
        result.fingerprint for result in verified
    ]
    assert (output / "small" / MANIFEST_NAME).is_file()


def test_hardlink_failure_falls_back_to_copy(tmp_path: Path, monkeypatch: Any) -> None:
    source = _master(tmp_path, [40, 60])
    output = tmp_path / "splits"
    specs = (SplitSpec("small", 100),)

    def fail_link(source_path: Path, target_path: Path) -> None:
        raise OSError("hard links unavailable")

    monkeypatch.setattr(os, "link", fail_link)
    results = create_splits(source, output, specs)

    assert results[0].link_mode == "copy"
    for file in scan_pdfs(source):
        assert (output / "small" / file.name).read_bytes() == file.path.read_bytes()
