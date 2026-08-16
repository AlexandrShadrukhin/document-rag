from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings  # noqa: E402
from app.indexing import IndexingService  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Index PDF/TXT documents")
    parser.add_argument("--path", type=Path, required=True, help="File or directory to index")
    parser.add_argument("--limit", type=int, help="Only process the first N files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    service = IndexingService(settings)
    stats = service.index_path(args.path, show_progress=True, limit=args.limit)
    print("\nIndexing summary")
    print(f"Documents: {stats.files_discovered}")
    print(f"Indexed: {stats.files_indexed}")
    print(f"Skipped: {stats.files_skipped}")
    print(f"Pages: {stats.pages_parsed}")
    print(f"Chunks: {stats.chunks_created}")
    print(f"Embedding batches: {stats.embedding_batches}")
    print(f"Discovery: {stats.discovery_seconds:.2f} s")
    print(f"Manifest handling: {stats.manifest_handling_seconds:.2f} s")
    print(f"Parsing: {stats.parsing_seconds:.2f} s")
    print(f"Cleaning: {stats.cleaning_seconds:.2f} s")
    print(f"Chunking: {stats.chunking_seconds:.2f} s")
    print(f"Embedding model loading: {stats.embedding_model_loading_seconds:.2f} s")
    print(f"Embedding: {stats.embedding_seconds:.2f} s")
    print(f"Qdrant initialization: {stats.qdrant_initialization_seconds:.2f} s")
    print(f"Qdrant upsert: {stats.qdrant_upsert_seconds:.2f} s")
    print(f"Lexical corpus: {stats.lexical_corpus_seconds:.2f} s")
    print(f"Total: {stats.total_seconds:.2f} s")


if __name__ == "__main__":
    main()
