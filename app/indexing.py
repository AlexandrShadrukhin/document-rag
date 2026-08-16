from __future__ import annotations

import math
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from tqdm.auto import tqdm

from app.config import Settings
from app.ingestion.chunker import SmartChunker
from app.ingestion.common import file_sha256, normalized_source
from app.ingestion.pipeline import IngestionPipeline, PreparedDocument
from app.lab.progress import ProgressCallback, ProgressEvent, emit
from app.retrieval.bm25 import CorpusStore
from app.retrieval.embeddings import EmbeddingProvider, SentenceTransformerEmbeddings
from app.retrieval.manifest import IndexManifest
from app.retrieval.vector_store import QdrantVectorStore


@dataclass
class IndexingStats:
    files_discovered: int = 0
    files_indexed: int = 0
    files_skipped: int = 0
    pages_parsed: int = 0
    chunks_created: int = 0
    embedding_batches: int = 0
    bytes_processed: int = 0
    discovery_seconds: float = 0.0
    manifest_handling_seconds: float = 0.0
    parsing_seconds: float = 0.0
    cleaning_seconds: float = 0.0
    chunking_seconds: float = 0.0
    embedding_model_loading_seconds: float = 0.0
    embedding_seconds: float = 0.0
    qdrant_initialization_seconds: float = 0.0
    qdrant_upsert_seconds: float = 0.0
    lexical_corpus_seconds: float = 0.0
    bm25_build_seconds: float = 0.0
    indexing_seconds: float = 0.0
    runtime_initialization_seconds: float = 0.0
    total_seconds: float = 0.0

    def timings(self) -> dict[str, float]:
        return {
            "discovery": self.discovery_seconds,
            "manifest_handling": self.manifest_handling_seconds,
            "parsing": self.parsing_seconds,
            "cleaning": self.cleaning_seconds,
            "chunking": self.chunking_seconds,
            "embedding_model_loading": self.embedding_model_loading_seconds,
            "embedding": self.embedding_seconds,
            "qdrant_initialization": self.qdrant_initialization_seconds,
            "qdrant_upsert": self.qdrant_upsert_seconds,
            "lexical_corpus": self.lexical_corpus_seconds,
            "bm25_build": self.bm25_build_seconds,
            "runtime_initialization": self.runtime_initialization_seconds,
            "indexing": self.indexing_seconds,
            "total": self.total_seconds,
        }

    def as_dict(self) -> dict[str, int | float]:
        return asdict(self)


class IndexingService:
    def __init__(
        self,
        settings: Settings,
        embeddings: EmbeddingProvider | None = None,
        vector_store: QdrantVectorStore | None = None,
    ) -> None:
        self.settings = settings
        self.embeddings = embeddings
        self.vector_store = vector_store
        self.pipeline = IngestionPipeline(
            SmartChunker(settings.chunk_size, settings.chunk_overlap)
        )
        self.corpus = CorpusStore(settings.corpus_path)
        self.manifest = IndexManifest(settings.manifest_path)
        self._backend_timings = {"embedding_model_loading": 0.0, "qdrant_initialization": 0.0}

    def _ensure_backends(
        self, callback: ProgressCallback | None = None
    ) -> tuple[EmbeddingProvider, QdrantVectorStore]:
        if self.embeddings is None:
            emit(callback, ProgressEvent("loading_embedding_model", "started"))
            started = time.perf_counter()
            self.embeddings = SentenceTransformerEmbeddings(
                self.settings.embedding_model,
                self.settings.embedding_batch_size,
                self.settings.embedding_device,
            )
            self._backend_timings["embedding_model_loading"] += time.perf_counter() - started
            emit(callback, ProgressEvent("loading_embedding_model", "completed"))
        if self.vector_store is None:
            emit(callback, ProgressEvent("qdrant_initialization", "started"))
            started = time.perf_counter()
            self.vector_store = QdrantVectorStore(
                collection=self.settings.qdrant_collection,
                vector_size=self.embeddings.dimension,
                mode=self.settings.qdrant_mode,
                path=self.settings.qdrant_path,
                url=self.settings.qdrant_url,
                api_key=self.settings.qdrant_api_key,
            )
            self._backend_timings["qdrant_initialization"] += time.perf_counter() - started
            emit(callback, ProgressEvent("qdrant_initialization", "completed"))
        return self.embeddings, self.vector_store

    def index_path(
        self,
        path: Path,
        show_progress: bool = True,
        limit: int | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> IndexingStats:
        total_started = time.perf_counter()
        emit(progress_callback, ProgressEvent("corpus_discovery", "started"))
        discovery_started = time.perf_counter()
        files = self.pipeline.discover(path)
        if limit is not None:
            files = files[:limit]
        stats = IndexingStats(files_discovered=len(files))
        stats.discovery_seconds = time.perf_counter() - discovery_started
        emit(
            progress_callback,
            ProgressEvent(
                "corpus_discovery", "completed", current=len(files), total=len(files)
            ),
        )
        changed: list[tuple[Path, str, str | None]] = []

        emit(progress_callback, ProgressEvent("manifest_handling", "started"))
        manifest_started = time.perf_counter()
        for file_index, file_path in enumerate(files, start=1):
            source = normalized_source(file_path)
            file_hash = file_sha256(file_path)
            if self.manifest.unchanged(source, file_hash):
                stats.files_skipped += 1
            else:
                old_id = self.manifest.previous_document_id(source)
                changed.append((file_path, file_hash, old_id))
            emit(
                progress_callback,
                ProgressEvent(
                    "manifest_handling",
                    "progress",
                    current=file_index,
                    total=len(files),
                    message=file_path.name,
                    details={
                        "documents_scanned": file_index,
                        "documents_changed": len(changed),
                        "documents_skipped": stats.files_skipped,
                    },
                ),
            )
        stats.manifest_handling_seconds += time.perf_counter() - manifest_started
        emit(
            progress_callback,
            ProgressEvent(
                "manifest_handling",
                "completed",
                current=len(files),
                total=len(files),
                details={
                    "documents_changed": len(changed),
                    "documents_skipped": stats.files_skipped,
                },
            ),
        )

        document_batch_total = math.ceil(
            len(changed) / self.settings.index_document_batch_size
        )
        iterator = tqdm(
            range(0, len(changed), self.settings.index_document_batch_size),
            desc="Indexing document batches",
            unit="batch",
            disable=not show_progress,
        )
        new_file_handle = tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", delete=False, dir=self.corpus.path.parent
        )
        new_chunks_path = Path(new_file_handle.name)
        new_file_handle.close()
        replaced_ids: set[str] = set()
        manifest_records: list[tuple[str, str, str]] = []
        try:
            for batch_index, offset in enumerate(iterator, start=1):
                batch_metadata = changed[
                    offset : offset + self.settings.index_document_batch_size
                ]
                emit(
                    progress_callback,
                    ProgressEvent(
                        "parsing_cleaning_chunking",
                        "started",
                        current=batch_index - 1,
                        total=document_batch_total,
                        message=f"Document batch {batch_index} / {document_batch_total}",
                        details={
                            "document_batch": batch_index,
                            "document_batches": document_batch_total,
                            "documents_processed": stats.files_indexed,
                            "documents_total": len(changed),
                            "pages_processed": stats.pages_parsed,
                            "chunks_processed": stats.chunks_created,
                        },
                    ),
                )
                prepared: list[PreparedDocument] = []
                for file_path, _, old_id in batch_metadata:
                    document = self.pipeline.prepare(file_path)
                    prepared.append(document)
                    stats.files_indexed += 1
                    stats.pages_parsed += len(document.pages)
                    stats.chunks_created += len(document.chunks)
                    stats.bytes_processed += file_path.stat().st_size
                    stats.parsing_seconds += document.timings["parsing"]
                    stats.cleaning_seconds += document.timings["cleaning"]
                    stats.chunking_seconds += document.timings["chunking"]
                    if old_id:
                        replaced_ids.add(old_id)
                    if document.pages:
                        page = document.pages[0]
                        replaced_ids.add(page.document_id)
                        manifest_records.append((page.source, page.file_hash, page.document_id))
                    emit(
                        progress_callback,
                        ProgressEvent(
                            "parsing_cleaning_chunking",
                            "progress",
                            current=stats.files_indexed,
                            total=len(changed),
                            message=file_path.name,
                            details={
                                "document_batch": batch_index,
                                "document_batches": document_batch_total,
                                "documents_processed": stats.files_indexed,
                                "documents_total": len(changed),
                                "pages_processed": stats.pages_parsed,
                                "chunks_processed": stats.chunks_created,
                            },
                        ),
                    )

                chunks = [chunk for document in prepared for chunk in document.chunks]
                batch_old_ids = {
                    metadata[2] for metadata in batch_metadata if metadata[2] is not None
                }
                if batch_old_ids and not chunks:
                    _, vector_store = self._ensure_backends(progress_callback)
                    emit(progress_callback, ProgressEvent("qdrant_upsert", "started"))
                    delete_started = time.perf_counter()
                    for old_id in batch_old_ids:
                        vector_store.delete_document(old_id)
                    stats.qdrant_upsert_seconds += time.perf_counter() - delete_started
                    emit(progress_callback, ProgressEvent("qdrant_upsert", "completed"))
                if not chunks:
                    continue
                embeddings, vector_store = self._ensure_backends(progress_callback)
                current_embedding_batches = math.ceil(
                    len(chunks) / self.settings.embedding_batch_size
                )
                emit(
                    progress_callback,
                    ProgressEvent(
                        "embedding_inference",
                        "started",
                        current=batch_index - 1,
                        total=document_batch_total,
                        message=f"Document batch {batch_index} / {document_batch_total}",
                        details={
                            "chunks_in_batch": len(chunks),
                            "embedding_batches_in_document_batch": current_embedding_batches,
                            "embedding_batches_completed": stats.embedding_batches,
                        },
                    ),
                )
                embedding_started = time.perf_counter()
                vectors = embeddings.embed_passages(
                    [chunk.text for chunk in chunks], show_progress=show_progress
                )
                stats.embedding_seconds += time.perf_counter() - embedding_started
                stats.embedding_batches += current_embedding_batches
                emit(
                    progress_callback,
                    ProgressEvent(
                        "embedding_inference",
                        "completed",
                        current=batch_index,
                        total=document_batch_total,
                        message=f"Embedded {len(chunks)} chunks",
                        details={
                            "chunks_processed": stats.chunks_created,
                            "embedding_batches_completed": stats.embedding_batches,
                        },
                    ),
                )

                emit(
                    progress_callback,
                    ProgressEvent(
                        "qdrant_upsert",
                        "started",
                        current=batch_index - 1,
                        total=document_batch_total,
                        message=f"Uploading document batch {batch_index} / {document_batch_total}",
                        details={"chunks_in_batch": len(chunks)},
                    ),
                )
                upsert_started = time.perf_counter()
                for old_id in batch_old_ids:
                    vector_store.delete_document(old_id)
                vector_store.upsert(chunks, vectors)
                stats.qdrant_upsert_seconds += time.perf_counter() - upsert_started
                emit(
                    progress_callback,
                    ProgressEvent(
                        "qdrant_upsert",
                        "completed",
                        current=batch_index,
                        total=document_batch_total,
                        message=f"Uploaded {len(chunks)} chunks",
                        details={
                            "document_batch": batch_index,
                            "document_batches": document_batch_total,
                            "chunks_processed": stats.chunks_created,
                        },
                    ),
                )
                self.corpus.append_to(new_chunks_path, chunks)

            if changed:
                emit(
                    progress_callback,
                    ProgressEvent(
                        "lexical_corpus",
                        "started",
                        message="Updating lexical corpus",
                        details={"chunks_processed": stats.chunks_created},
                    ),
                )
                lexical_started = time.perf_counter()
                self.corpus.merge_new_file(new_chunks_path, replaced_ids)
                stats.lexical_corpus_seconds = time.perf_counter() - lexical_started
                emit(progress_callback, ProgressEvent("lexical_corpus", "completed"))
                manifest_started = time.perf_counter()
                for source, file_hash, document_id in manifest_records:
                    self.manifest.record(source, file_hash, document_id)
                stats.manifest_handling_seconds += time.perf_counter() - manifest_started
        finally:
            new_chunks_path.unlink(missing_ok=True)

        stats.embedding_model_loading_seconds = self._backend_timings[
            "embedding_model_loading"
        ]
        stats.qdrant_initialization_seconds = self._backend_timings[
            "qdrant_initialization"
        ]
        stats.indexing_seconds = (
            stats.qdrant_upsert_seconds
            + stats.lexical_corpus_seconds
            + stats.manifest_handling_seconds
        )

        stats.total_seconds = time.perf_counter() - total_started
        emit(progress_callback, ProgressEvent("finalizing", "started"))
        emit(
            progress_callback,
            ProgressEvent("index_persisted", "completed", message="Vector and lexical data ready"),
        )
        return stats
