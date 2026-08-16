from __future__ import annotations

import os
import platform
from collections.abc import Iterable
from pathlib import Path


class FontResolutionError(RuntimeError):
    pass


def font_candidates(
    system_name: str | None = None,
    home: Path | None = None,
    windows_dir: Path | None = None,
) -> list[Path]:
    system = system_name or platform.system()
    user_home = home or Path.home()
    if system == "Windows":
        fonts = (windows_dir or Path(os.environ.get("WINDIR", "C:/Windows"))) / "Fonts"
        return [
            fonts / "arial.ttf",
            fonts / "Arial.ttf",
            fonts / "arialuni.ttf",
            fonts / "DejaVuSans.ttf",
            user_home / "AppData/Local/Microsoft/Windows/Fonts/DejaVuSans.ttf",
        ]
    if system == "Darwin":
        return [
            user_home / "Library/Fonts/DejaVuSans.ttf",
            Path("/Library/Fonts/DejaVuSans.ttf"),
            Path("/Library/Fonts/Arial Unicode.ttf"),
            Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
            Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        ]
    return [
        user_home / ".local/share/fonts/DejaVuSans.ttf",
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/local/share/fonts/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"),
    ]


def resolve_cyrillic_font(
    configured: Path | str | None = None,
    *,
    system_name: str | None = None,
    candidates: Iterable[Path] | None = None,
) -> Path:
    configured_value = configured or os.environ.get("PDF_FONT_PATH")
    if configured_value:
        path = Path(configured_value).expanduser().resolve()
        if not path.is_file():
            raise FontResolutionError(f"PDF_FONT_PATH does not point to a file: {path}")
        if path.suffix.lower() not in {".ttf", ".otf"}:
            raise FontResolutionError(f"PDF font must be a TTF/OTF file: {path}")
        return path

    checked = list(candidates) if candidates is not None else font_candidates(system_name)
    for path in checked:
        resolved = path.expanduser()
        if resolved.is_file() and resolved.suffix.lower() in {".ttf", ".otf"}:
            return resolved.resolve()
    locations = "\n".join(f"- {path}" for path in checked)
    raise FontResolutionError(
        "Cyrillic-capable TTF/OTF font was not found. Install DejaVu Sans or set "
        f"PDF_FONT_PATH explicitly. Checked:\n{locations}"
    )
