from __future__ import annotations

import math
import os
import re
import shutil
from collections import Counter
from pathlib import Path

from app.schemas import Chunk, SearchResult

_TOKEN = re.compile(r"[\w№.-]+", flags=re.UNICODE)


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN.findall(text)]


class BM25Index:
    def __init__(self, chunks: list[Chunk] | None = None, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.chunks: list[Chunk] = []
        self._frequencies: list[Counter[str]] = []
        self._lengths: list[int] = []
        self._document_frequency: Counter[str] = Counter()
        self._average_length = 0.0
        self.rebuild(chunks or [])

    def rebuild(self, chunks: list[Chunk]) -> None:
        self.chunks = list(chunks)
        self._frequencies = [Counter(tokenize(chunk.text)) for chunk in self.chunks]
        self._lengths = [sum(frequency.values()) for frequency in self._frequencies]
        self._document_frequency = Counter()
        for frequency in self._frequencies:
            self._document_frequency.update(frequency.keys())
        self._average_length = (
            sum(self._lengths) / len(self._lengths) if self._lengths else 0.0
        )

    def _idf(self, term: str) -> float:
        documents = len(self.chunks)
        frequency = self._document_frequency.get(term, 0)
        return math.log(1.0 + (documents - frequency + 0.5) / (frequency + 0.5))

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        if not self.chunks:
            return []
        terms = tokenize(query)
        if not terms:
            return []
        scores: list[tuple[int, float]] = []
        for index, frequency in enumerate(self._frequencies):
            score = 0.0
            length = self._lengths[index]
            for term in terms:
                term_frequency = frequency.get(term, 0)
                if not term_frequency:
                    continue
                denominator = term_frequency + self.k1 * (
                    1.0 - self.b + self.b * length / max(self._average_length, 1.0)
                )
                score += self._idf(term) * term_frequency * (self.k1 + 1.0) / denominator
            if score > 0:
                scores.append((index, score))
        scores.sort(key=lambda item: item[1], reverse=True)
        return [
            SearchResult(
                chunk=self.chunks[index],
                score=score,
                bm25_score=score,
            )
            for index, score in scores[:top_k]
        ]


class CorpusStore:
    """Persistent chunk corpus used to rebuild the local BM25 index."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> list[Chunk]:
        if not self.path.exists():
            return []
        chunks: list[Chunk] = []
        with self.path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    chunks.append(Chunk.model_validate_json(line))
        return chunks

    def save(self, chunks: list[Chunk]) -> None:
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            for chunk in chunks:
                stream.write(chunk.model_dump_json() + "\n")
        temporary.replace(self.path)

    def append_to(self, path: Path, chunks: list[Chunk]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            for chunk in chunks:
                stream.write(chunk.model_dump_json() + "\n")

    def merge_new_file(self, new_chunks_path: Path, replaced_document_ids: set[str]) -> None:
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as target:
            if self.path.exists():
                with self.path.open("r", encoding="utf-8") as source:
                    for line in source:
                        if not line.strip():
                            continue
                        chunk = Chunk.model_validate_json(line)
                        if chunk.document_id not in replaced_document_ids:
                            target.write(line if line.endswith("\n") else line + "\n")
            if new_chunks_path.exists():
                with new_chunks_path.open("r", encoding="utf-8") as source:
                    shutil.copyfileobj(source, target)
        os.replace(temporary, self.path)

    def replace_document(self, document_id: str, chunks: list[Chunk]) -> None:
        retained = [chunk for chunk in self.load() if chunk.document_id != document_id]
        self.save(retained + chunks)

    def delete_document(self, document_id: str) -> None:
        self.save([chunk for chunk in self.load() if chunk.document_id != document_id])
