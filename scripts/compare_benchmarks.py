from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.lab.benchmark import benchmark_comparison  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare saved RAG Lab benchmark runs")
    parser.add_argument("runs", nargs="+", type=Path, help="Run directories or run.json files")
    args = parser.parse_args()
    print(json.dumps(benchmark_comparison(args.runs), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
