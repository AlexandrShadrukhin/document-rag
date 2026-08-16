from __future__ import annotations

import argparse
import json
import logging

from app.config import get_settings
from app.desktop.controller import LabController


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the cross-platform Document RAG Lab")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Run headless preflight and exit without opening Tkinter",
    )
    args = parser.parse_args()
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.check:
        controller = LabController(settings)
        try:
            print(json.dumps(controller.preflight().as_dict(), ensure_ascii=False, indent=2))
        finally:
            controller.close()
        return
    try:
        from app.desktop.gui import launch
    except ImportError as error:
        raise SystemExit(
            "Tkinter is unavailable in this Python installation. Install a Python build "
            "with Tk support, or continue using scripts/ask.py and FastAPI."
        ) from error
    launch(settings)


if __name__ == "__main__":
    main()
