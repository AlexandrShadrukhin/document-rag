from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings  # noqa: E402
from app.generation.llm import LLMUnavailableError  # noqa: E402
from app.runtime import ApplicationContainer  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask a grounded question")
    parser.add_argument("query", help="Question in natural language")
    parser.add_argument("--mode", choices=("fast", "quality"), default="fast")
    args = parser.parse_args()
    settings = get_settings()
    logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))
    container = ApplicationContainer(settings)
    try:
        response = container.rag_service(args.mode).answer(args.query)
    except LLMUnavailableError as error:
        raise SystemExit(str(error)) from error
    finally:
        container.close()

    print(f"Answer:\n{response.answer}\n")
    print("Sources:")
    if not response.sources:
        print("—")
    for index, source in enumerate(response.sources, start=1):
        page = f", page {source.page}" if source.page is not None else ""
        print(
            f"{index}. {source.filename}{page} "
            f"(chunk={source.chunk_id}, score={source.retrieval_score:.4f})"
        )
    print("\nTimings:")
    print(f"Base retrieval: {response.timings.base_retrieval_ms:.1f} ms")
    print(f"Reranker: {response.timings.reranker_ms:.1f} ms")
    print(f"Retrieval total: {response.timings.retrieval_total_ms:.1f} ms")
    print(f"Generation: {response.timings.generation_ms:.1f} ms")
    print(f"Total: {response.timings.total_ms:.1f} ms")
    print(f"Context: {response.timings.context_chars} chars")


if __name__ == "__main__":
    main()
