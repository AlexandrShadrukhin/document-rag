from __future__ import annotations

from pathlib import Path

import pytest

from app.wiki.fonts import FontResolutionError, font_candidates, resolve_cyrillic_font


def test_windows_font_candidates_use_windows_fonts_directory(tmp_path: Path) -> None:
    candidates = font_candidates("Windows", home=tmp_path / "home", windows_dir=tmp_path)
    assert candidates[0] == tmp_path / "Fonts" / "arial.ttf"
    assert all("/System/" not in str(candidate) for candidate in candidates)


def test_configured_font_path_has_priority(tmp_path: Path) -> None:
    font = tmp_path / "custom.ttf"
    font.write_bytes(b"font fixture")
    assert resolve_cyrillic_font(font, candidates=[]) == font.resolve()


def test_missing_font_has_actionable_error(tmp_path: Path) -> None:
    with pytest.raises(FontResolutionError, match="PDF_FONT_PATH"):
        resolve_cyrillic_font(candidates=[tmp_path / "missing.ttf"])

