from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

MIB = 1024**2
GIB = 1024**3
MANIFEST_NAME = "benchmark_split_manifest.json"


@dataclass(frozen=True)
class SplitSpec:
    name: str
    target_bytes: int


@dataclass(frozen=True)
class PdfFile:
    path: Path
    name: str
    size: int


@dataclass(frozen=True)
class SplitResult:
    spec: SplitSpec
    files: tuple[PdfFile, ...]
    actual_bytes: int
    fingerprint: str
    link_mode: str


DEFAULT_SPLITS = (
    SplitSpec("wiki_100mb", 100 * MIB),
    SplitSpec("wiki_500mb", 500 * MIB),
    SplitSpec("wiki_1gb", GIB),
    SplitSpec("wiki_4gb", 4 * GIB),
    SplitSpec("wiki_8gb", 8 * GIB),
)


class SplitError(RuntimeError):
    """Raised when benchmark splits cannot be created or verified."""


def scan_pdfs(source: Path) -> tuple[PdfFile, ...]:
    source = source.expanduser().resolve()
    if not source.is_dir():
        raise SplitError(f"Master corpus directory does not exist: {source}")
    files = tuple(
        PdfFile(path=path, name=path.name, size=path.stat().st_size)
        for path in sorted(source.iterdir(), key=lambda item: item.name)
        if path.is_file() and path.suffix == ".pdf"
    )
    if not files:
        raise SplitError(f"Master corpus contains no *.pdf files: {source}")
    return files


def select_split_files(
    files: Sequence[PdfFile], specs: Sequence[SplitSpec]
) -> dict[str, tuple[PdfFile, ...]]:
    if any(spec.target_bytes <= 0 for spec in specs):
        raise ValueError("Split target sizes must be positive")
    if any(
        left.target_bytes >= right.target_bytes
        for left, right in zip(specs, specs[1:], strict=False)
    ):
        raise ValueError("Split target sizes must be strictly increasing")

    total_bytes = sum(file.size for file in files)
    largest_target = specs[-1].target_bytes if specs else 0
    if total_bytes < largest_target:
        raise SplitError(
            f"Master corpus is too small: {total_bytes} bytes available, "
            f"{largest_target} bytes required"
        )

    selected: dict[str, tuple[PdfFile, ...]] = {}
    cumulative = 0
    next_spec = 0
    for index, file in enumerate(files):
        cumulative += file.size
        while next_spec < len(specs) and cumulative >= specs[next_spec].target_bytes:
            selected[specs[next_spec].name] = tuple(files[: index + 1])
            next_spec += 1
        if next_spec == len(specs):
            break
    return selected


def corpus_fingerprint(files: Iterable[PdfFile]) -> str:
    digest = hashlib.sha256()
    for file in files:
        encoded_name = file.name.encode("utf-8")
        digest.update(len(encoded_name).to_bytes(8, "big"))
        digest.update(encoded_name)
        digest.update(file.size.to_bytes(8, "big", signed=False))
    return digest.hexdigest()


def human_size(size: int) -> str:
    if size >= GIB:
        return f"{size / GIB:.3f} GiB"
    return f"{size / MIB:.2f} MiB"


def expected_manifest(
    source: Path, spec: SplitSpec, files: Sequence[PdfFile], link_mode: str
) -> dict[str, object]:
    actual_bytes = sum(file.size for file in files)
    return {
        "source_master_path": str(source.expanduser().resolve()),
        "target_bytes": spec.target_bytes,
        "actual_bytes": actual_bytes,
        "actual_size": human_size(actual_bytes),
        "pdf_count": len(files),
        "first_filename": files[0].name,
        "last_filename": files[-1].name,
        "fingerprint": corpus_fingerprint(files),
        "filenames": [file.name for file in files],
        "link_mode": link_mode,
    }


def _read_manifest(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SplitError(f"Cannot read manifest {path}: {error}") from error
    if not isinstance(value, dict):
        raise SplitError(f"Manifest is not a JSON object: {path}")
    return value


def _manifest_matches(
    manifest: dict[str, object], source: Path, spec: SplitSpec, files: Sequence[PdfFile]
) -> bool:
    link_mode = manifest.get("link_mode")
    if not isinstance(link_mode, str) or not link_mode:
        return False
    return manifest == expected_manifest(source, spec, files, link_mode)


def _directory_matches(target: Path, files: Sequence[PdfFile]) -> bool:
    expected_names = {file.name for file in files}
    actual_names = {
        path.name
        for path in target.iterdir()
        if path.is_file() and path.suffix == ".pdf"
    }
    if actual_names != expected_names:
        return False
    return all((target / file.name).stat().st_size == file.size for file in files)


def _existing_split_is_valid(
    source: Path, target: Path, spec: SplitSpec, files: Sequence[PdfFile]
) -> dict[str, object] | None:
    manifest_path = target / MANIFEST_NAME
    if not target.is_dir() or not manifest_path.is_file():
        return None
    try:
        manifest = _read_manifest(manifest_path)
        if _manifest_matches(manifest, source, spec, files) and _directory_matches(target, files):
            return manifest
    except (OSError, SplitError):
        pass
    return None


def _link_or_copy(source: Path, target: Path) -> str:
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        shutil.copy2(source, target)
        return "copy"


def _combined_link_mode(modes: set[str]) -> str:
    ordered = [mode for mode in ("hardlink", "copy", "reused") if mode in modes]
    return "+".join(ordered)


def create_split(
    source: Path, output_root: Path, spec: SplitSpec, files: Sequence[PdfFile]
) -> SplitResult:
    target = output_root / spec.name
    existing_manifest = _existing_split_is_valid(source, target, spec, files)
    if existing_manifest is not None:
        return SplitResult(
            spec=spec,
            files=tuple(files),
            actual_bytes=sum(file.size for file in files),
            fingerprint=corpus_fingerprint(files),
            link_mode=str(existing_manifest["link_mode"]),
        )

    target.mkdir(parents=True, exist_ok=True)
    expected_names = {file.name for file in files}
    unexpected = [
        path.name
        for path in target.iterdir()
        if path.name != MANIFEST_NAME and path.name not in expected_names
    ]
    if unexpected:
        names = ", ".join(sorted(unexpected)[:5])
        raise SplitError(f"Refusing to remove unexpected files from {target}: {names}")

    modes: set[str] = set()
    for file in files:
        destination = target / file.name
        if destination.exists():
            if destination.is_file() and destination.stat().st_size == file.size:
                modes.add("reused")
                continue
            destination.unlink()
        modes.add(_link_or_copy(file.path, destination))

    link_mode = _combined_link_mode(modes)
    manifest = expected_manifest(source, spec, files, link_mode)
    manifest_path = target / MANIFEST_NAME
    temporary_manifest = target / f".{MANIFEST_NAME}.tmp"
    temporary_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary_manifest, manifest_path)
    return SplitResult(
        spec=spec,
        files=tuple(files),
        actual_bytes=int(manifest["actual_bytes"]),
        fingerprint=str(manifest["fingerprint"]),
        link_mode=link_mode,
    )


def create_splits(
    source: Path,
    output_root: Path,
    specs: Sequence[SplitSpec] = DEFAULT_SPLITS,
) -> list[SplitResult]:
    source = source.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    files = scan_pdfs(source)
    selections = select_split_files(files, specs)
    return [
        create_split(source, output_root, spec, selections[spec.name]) for spec in specs
    ]


def _verify_threshold(files: Sequence[PdfFile], target_bytes: int) -> None:
    actual_bytes = sum(file.size for file in files)
    before_last = actual_bytes - files[-1].size
    if actual_bytes < target_bytes or before_last >= target_bytes:
        raise SplitError(
            f"Invalid threshold boundary: {before_last} < {target_bytes} <= {actual_bytes} is false"
        )


def verify_splits(
    source: Path,
    output_root: Path,
    specs: Sequence[SplitSpec] = DEFAULT_SPLITS,
) -> list[SplitResult]:
    source = source.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    master_files = scan_pdfs(source)
    selections = select_split_files(master_files, specs)
    results: list[SplitResult] = []
    previous_names: set[str] = set()

    for spec in specs:
        files = selections[spec.name]
        target = output_root / spec.name
        manifest_path = target / MANIFEST_NAME
        if not target.is_dir():
            raise SplitError(f"Missing split directory: {target}")
        manifest = _read_manifest(manifest_path)
        if not _manifest_matches(manifest, source, spec, files):
            raise SplitError(f"Manifest does not match master corpus or target: {manifest_path}")
        if not _directory_matches(target, files):
            raise SplitError(f"Split files or sizes do not match manifest: {target}")

        names = {file.name for file in files}
        if not previous_names.issubset(names):
            raise SplitError(f"Split is not a superset of the previous split: {target}")
        previous_names = names
        _verify_threshold(files, spec.target_bytes)
        results.append(
            SplitResult(
                spec=spec,
                files=tuple(files),
                actual_bytes=sum(file.size for file in files),
                fingerprint=corpus_fingerprint(files),
                link_mode=str(manifest["link_mode"]),
            )
        )
    return results


def print_summary(results: Sequence[SplitResult]) -> None:
    headers = ("split", "PDFs", "bytes", "MiB/GiB", "fingerprint", "link mode")
    rows = [
        (
            result.spec.name,
            str(len(result.files)),
            str(result.actual_bytes),
            human_size(result.actual_bytes),
            result.fingerprint[:12],
            result.link_mode,
        )
        for result in results
    ]
    widths = [max(len(headers[index]), *(len(row[index]) for row in rows)) for index in range(6)]
    print(" | ".join(value.ljust(widths[index]) for index, value in enumerate(headers)))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create deterministic, nested benchmark splits from a master PDF corpus"
    )
    parser.add_argument("--source", type=Path, default=Path("data/corpus/wiki_pdf"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/corpus/benchmark_splits"),
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify existing splits without modifying them",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.verify:
            results = verify_splits(args.source, args.output_root)
            print("Benchmark corpus splits verified successfully.\n")
        else:
            results = create_splits(args.source, args.output_root)
            print("Benchmark corpus splits are ready.\n")
        print_summary(results)
    except SplitError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
