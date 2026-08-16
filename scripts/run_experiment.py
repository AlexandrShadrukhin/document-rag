from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings  # noqa: E402
from app.lab.experiments import ExperimentConfig, ExperimentRunner  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an isolated RAG experiment")
    parser.add_argument("config", type=Path)
    parser.add_argument(
        "--plan",
        action="store_true",
        help="Show resolved paths and reindex decision without running models",
    )
    args = parser.parse_args()
    config = ExperimentConfig.load(args.config.resolve())
    runner = ExperimentRunner(get_settings())
    settings, root, reindex = runner.prepare(config)
    if args.plan:
        print(
            json.dumps(
                {
                    "experiment_root": str(root),
                    "requires_reindex": reindex,
                    "qdrant_path": str(settings.qdrant_path),
                    "corpus_path": str(settings.corpus_path),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    print(json.dumps(runner.run(config), ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
