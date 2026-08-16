from __future__ import annotations

import re

from app.ingestion.common import deterministic_chunk_id
from app.schemas import Chunk, DocumentPage

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?…])\s+(?=[А-ЯA-ZЁ0-9«\"(])")


class SmartChunker:
    """Paragraph-first, sentence-second chunking with bounded character fallback."""

    def __init__(self, chunk_size: int = 1000, overlap: int = 150) -> None:
        if chunk_size < 100:
            raise ValueError("chunk_size must be at least 100")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must be non-negative and smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def _hard_split(self, text: str) -> list[str]:
        pieces: list[str] = []
        remaining = text.strip()
        while len(remaining) > self.chunk_size:
            cut = remaining.rfind(" ", 0, self.chunk_size + 1)
            if cut < self.chunk_size // 2:
                cut = self.chunk_size
            pieces.append(remaining[:cut].strip())
            remaining = remaining[cut:].strip()
        if remaining:
            pieces.append(remaining)
        return pieces

    def _units(self, text: str) -> list[str]:
        units: list[str] = []
        for paragraph in re.split(r"\n\s*\n", text):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            if len(paragraph) <= self.chunk_size:
                units.append(paragraph)
                continue
            for sentence in _SENTENCE_BOUNDARY.split(paragraph):
                sentence = sentence.strip()
                if len(sentence) <= self.chunk_size:
                    if sentence:
                        units.append(sentence)
                else:
                    units.extend(self._hard_split(sentence))
        return units

    def split_text(self, text: str) -> list[str]:
        units = self._units(text)
        if not units:
            return []

        chunks: list[str] = []
        current = ""
        for unit in units:
            separator = "\n\n" if current else ""
            if current and len(current) + len(separator) + len(unit) > self.chunk_size:
                chunks.append(current.strip())
                overlap_text = current[-self.overlap :].lstrip() if self.overlap else ""
                if overlap_text and " " in overlap_text:
                    overlap_text = overlap_text[overlap_text.find(" ") + 1 :]
                current = f"{overlap_text}\n\n{unit}".strip() if overlap_text else unit
                if len(current) > self.chunk_size:
                    overflow = self._hard_split(current)
                    chunks.extend(overflow[:-1])
                    current = overflow[-1]
            else:
                current = f"{current}{separator}{unit}"
        if current.strip():
            chunks.append(current.strip())
        return chunks

    def chunk_pages(self, pages: list[DocumentPage]) -> list[Chunk]:
        chunks: list[Chunk] = []
        chunk_index = 0
        for page in pages:
            for text in self.split_text(page.text):
                chunks.append(
                    Chunk(
                        chunk_id=deterministic_chunk_id(
                            page.file_hash, page.page_number, chunk_index
                        ),
                        document_id=page.document_id,
                        filename=page.filename,
                        source=page.source,
                        file_hash=page.file_hash,
                        page_number=page.page_number,
                        chunk_index=chunk_index,
                        text=text,
                    )
                )
                chunk_index += 1
        return chunks
