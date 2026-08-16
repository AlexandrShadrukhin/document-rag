from __future__ import annotations

import re
import unicodedata

_HORIZONTAL_SPACE = re.compile(r"[^\S\n]+")
_EXCESS_NEWLINES = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    """Normalize layout without deleting meaningful punctuation or identifiers."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [_HORIZONTAL_SPACE.sub(" ", line).strip() for line in text.split("\n")]
    normalized = "\n".join(lines)
    return _EXCESS_NEWLINES.sub("\n\n", normalized).strip()
