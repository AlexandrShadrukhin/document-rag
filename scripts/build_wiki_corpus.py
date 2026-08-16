from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings  # noqa: E402
from app.lab.progress import ProgressEvent  # noqa: E402
from app.wiki.corpus import WikiCorpusBuilder, WikiCorpusConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Build a PDF corpus from Russian Wikipedia")
    parser.add_argument("--source", type=Path, default=settings.wiki_dump_path)
    parser.add_argument("--output", type=Path, default=settings.wiki_corpus_path)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--limit", type=int)
    target.add_argument("--target-size-mb", type=float)
    target.add_argument("--target-size-gb", type=float)
    parser.add_argument("--min-article-chars", type=int, default=3000)
    parser.add_argument("--font-path", type=Path, default=settings.pdf_font_path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--hash-source", action="store_true")
    return parser.parse_args()


def target_bytes(args: argparse.Namespace) -> int | None:
    if args.target_size_mb is not None:
        return int(args.target_size_mb * 1024**2)
    if args.target_size_gb is not None:
        return int(args.target_size_gb * 1024**3)
    return None


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    config = WikiCorpusConfig(
        source=args.source,
        output=args.output,
        limit=args.limit,
        target_bytes=target_bytes(args),
        min_article_chars=args.min_article_chars,
        font_path=args.font_path,
        overwrite=args.overwrite,
        resume=not args.no_resume,
        source_sha256=args.hash_source,
    )
    total = config.target_bytes if config.target_bytes else config.limit
    unit = "B" if config.target_bytes else "pdf"
    progress = tqdm(total=total, unit=unit, unit_scale=bool(config.target_bytes))
    previous = 0.0

    def update(event: ProgressEvent) -> None:
        nonlocal previous
        if event.current is not None:
            progress.update(max(0.0, float(event.current) - previous))
            previous = float(event.current)
        details = event.details or {}
        if details:
            progress.set_postfix(
                pdfs=details.get("generated_pdfs", 0),
                inspected=details.get("inspected_pages", 0),
                eta_s=(
                    f"{float(details['eta_seconds']):.0f}"
                    if details.get("eta_seconds") is not None
                    else "N/A"
                ),
            )

    try:
        stats = WikiCorpusBuilder(config).build(update)
    finally:
        progress.close()
    print("\nWikipedia corpus summary")
    print(f"PDFs: {stats.generated_pdfs}")
    print(f"Accepted articles: {stats.accepted_articles}")
    print(f"Inspected pages: {stats.inspected_pages}")
    print(f"Corpus: {stats.corpus_bytes / 1024**2:.2f} MB ({stats.corpus_gb:.3f} GB)")
    print(f"Elapsed: {stats.elapsed_seconds:.2f} s")
    print(f"Manifest: {stats.manifest_path}")


if __name__ == "__main__":
    main()
