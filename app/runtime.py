from __future__ import annotations

import time
from pathlib import Path

from app.config import Settings, get_settings
from app.generation.llm import OllamaProvider
from app.generation.rag import RAGService
from app.indexing import IndexingService, IndexingStats
from app.lab.progress import ProgressCallback, ProgressEvent, emit
from app.retrieval.bm25 import BM25Index, CorpusStore
from app.retrieval.embeddings import SentenceTransformerEmbeddings
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.vector_store import QdrantVectorStore
from app.schemas import RetrievalMode


class ApplicationContainer:
    """Own heavyweight models and the Qdrant client once per process."""

    def __init__(self, settings: Settings | None = None, load_bm25: bool = True) -> None:
        self.settings = settings or get_settings()
        self.embeddings: SentenceTransformerEmbeddings | None = None
        self.vector_store: QdrantVectorStore | None = None
        self.reranker: CrossEncoderReranker | None = None
        chunks = CorpusStore(self.settings.corpus_path).load() if load_bm25 else []
        self.bm25 = BM25Index(chunks)

    def _ensure_retrieval_backends(self) -> None:
        if self.embeddings is None:
            self.embeddings = SentenceTransformerEmbeddings(
                self.settings.embedding_model,
                self.settings.embedding_batch_size,
                self.settings.embedding_device,
            )
        if self.vector_store is None:
            self.vector_store = QdrantVectorStore(
                collection=self.settings.qdrant_collection,
                vector_size=self.embeddings.dimension,
                mode=self.settings.qdrant_mode,
                path=self.settings.qdrant_path,
                url=self.settings.qdrant_url,
                api_key=self.settings.qdrant_api_key,
            )

    def index(
        self,
        path: Path,
        show_progress: bool = False,
        limit: int | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> IndexingStats:
        service = IndexingService(self.settings, self.embeddings, self.vector_store)
        stats = service.index_path(
            path,
            show_progress=show_progress,
            limit=limit,
            progress_callback=progress_callback,
        )
        self.embeddings = service.embeddings  # type: ignore[assignment]
        self.vector_store = service.vector_store
        emit(progress_callback, ProgressEvent("bm25_build", "started"))
        started = time.perf_counter()
        self.bm25.rebuild(CorpusStore(self.settings.corpus_path).load())
        stats.bm25_build_seconds = time.perf_counter() - started
        stats.total_seconds += stats.bm25_build_seconds
        emit(progress_callback, ProgressEvent("bm25_build", "completed"))
        emit(progress_callback, ProgressEvent("ready", "completed", message="RAG ready"))
        return stats

    def retriever(self, mode: RetrievalMode | None = None) -> HybridRetriever:
        self._ensure_retrieval_backends()
        assert self.embeddings is not None and self.vector_store is not None
        enable_reranker = self.settings.enable_reranker if mode is None else mode == "quality"
        if enable_reranker and self.reranker is None:
            self.reranker = CrossEncoderReranker(
                model_name=self.settings.reranker_model,
                top_k=self.settings.final_top_k,
                batch_size=self.settings.reranker_batch_size,
                device=self.settings.reranker_device,
            )
        return HybridRetriever(
            self.embeddings,
            self.vector_store,
            self.bm25,
            self.settings.dense_top_k,
            self.settings.bm25_top_k,
            self.settings.fusion_top_k,
            self.settings.final_top_k,
            self.settings.rrf_k,
            self.settings.rerank_candidates,
            self.reranker if enable_reranker else None,
            self.settings.confidence_dense_threshold,
            self.settings.confidence_dense_no_agreement_threshold,
            self.settings.confidence_reranker_threshold,
        )

    def rag_service(self, mode: RetrievalMode | None = None) -> RAGService:
        retriever = self.retriever(mode)
        llm = OllamaProvider(
            self.settings.ollama_base_url,
            self.settings.ollama_model,
            self.settings.ollama_timeout_seconds,
            self.settings.llm_temperature,
        )
        return RAGService(
            retriever,
            llm,
            min(self.settings.max_context_chunks, self.settings.final_top_k),
        )

    def close(self) -> None:
        if self.vector_store is not None:
            self.vector_store.close()
