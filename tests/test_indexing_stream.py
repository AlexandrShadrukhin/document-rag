from __future__ import annotations

from pathlib import Path

import numpy as np

from app.config import Settings
from app.indexing import IndexingService
from app.lab.progress import ProgressEvent


class FakeEmbeddings:
    dimension = 3

    def embed_passages(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        return np.ones((len(texts), 3), dtype=np.float32)


class FakeVectorStore:
    def __init__(self) -> None:
        self.chunks: list[object] = []
        self.deleted: list[str] = []
        self.upsert_sizes: list[int] = []

    def upsert(self, chunks: list[object], vectors: np.ndarray) -> None:
        self.upsert_sizes.append(len(chunks))
        self.chunks.extend(chunks)

    def delete_document(self, document_id: str) -> None:
        self.deleted.append(document_id)


def test_batched_indexing_persists_corpus_and_skips_unchanged(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "a.txt").write_text("Первый абзац. " * 80, encoding="utf-8")
    (source / "b.txt").write_text("Второй абзац. " * 80, encoding="utf-8")
    settings = Settings(
        qdrant_path=tmp_path / "index/qdrant",
        corpus_path=tmp_path / "index/corpus.jsonl",
        manifest_path=tmp_path / "index/manifest.json",
        wiki_dump_path=tmp_path / "wiki.bz2",
        wiki_corpus_path=tmp_path / "wiki",
        benchmarks_path=tmp_path / "benchmarks",
        experiments_path=tmp_path / "experiments",
        index_document_batch_size=1,
    )
    vectors = FakeVectorStore()
    service = IndexingService(settings, FakeEmbeddings(), vectors)  # type: ignore[arg-type]
    first = service.index_path(source, show_progress=False)
    second = service.index_path(source, show_progress=False)
    assert first.files_indexed == 2
    assert first.chunks_created > 0
    assert len(vectors.upsert_sizes) == 2
    assert settings.corpus_path.is_file()
    assert len(settings.corpus_path.read_text().splitlines()) == first.chunks_created
    assert second.files_skipped == 2
    assert second.files_indexed == 0


def test_indexing_emits_live_progress_for_multiple_batches(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    for index in range(3):
        (source / f"{index}.txt").write_text("Тестовый абзац. " * 80, encoding="utf-8")
    settings = Settings(
        qdrant_path=tmp_path / "index/qdrant",
        corpus_path=tmp_path / "index/corpus.jsonl",
        manifest_path=tmp_path / "index/manifest.json",
        wiki_dump_path=tmp_path / "wiki.bz2",
        wiki_corpus_path=tmp_path / "wiki",
        benchmarks_path=tmp_path / "benchmarks",
        experiments_path=tmp_path / "experiments",
        index_document_batch_size=1,
        embedding_batch_size=2,
    )
    events: list[ProgressEvent] = []
    service = IndexingService(settings, FakeEmbeddings(), FakeVectorStore())  # type: ignore[arg-type]

    stats = service.index_path(source, show_progress=False, progress_callback=events.append)

    stages = {event.stage for event in events}
    assert {
        "corpus_discovery",
        "manifest_handling",
        "parsing_cleaning_chunking",
        "embedding_inference",
        "qdrant_upsert",
        "lexical_corpus",
        "finalizing",
        "index_persisted",
    } <= stages
    document_progress = [
        event
        for event in events
        if event.stage == "parsing_cleaning_chunking" and event.kind == "progress"
    ]
    assert [event.current for event in document_progress] == [1, 2, 3]
    assert document_progress[-1].details == {
        "document_batch": 3,
        "document_batches": 3,
        "documents_processed": 3,
        "documents_total": 3,
        "pages_processed": 3,
        "chunks_processed": stats.chunks_created,
    }
    completed_embeddings = [
        event
        for event in events
        if event.stage == "embedding_inference" and event.kind == "completed"
    ]
    assert len(completed_embeddings) == 3
    assert completed_embeddings[-1].details is not None
    assert (
        completed_embeddings[-1].details["embedding_batches_completed"]
        == stats.embedding_batches
    )
