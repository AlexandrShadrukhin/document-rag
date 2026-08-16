from __future__ import annotations

import time
from collections.abc import Callable

from app.retrieval.bm25 import BM25Index
from app.retrieval.embeddings import EmbeddingProvider
from app.retrieval.vector_store import QdrantVectorStore
from app.schemas import RetrievalDecision, SearchResult


def reciprocal_rank_fusion(
    rankings: list[list[SearchResult]], top_k: int, rrf_k: int = 60
) -> list[SearchResult]:
    fused: dict[str, dict[str, object]] = {}
    for ranking in rankings:
        for rank, result in enumerate(ranking, start=1):
            chunk_id = result.chunk.chunk_id
            entry = fused.setdefault(
                chunk_id,
                {
                    "chunk": result.chunk,
                    "score": 0.0,
                    "dense": None,
                    "bm25": None,
                    "dense_rank": None,
                    "bm25_rank": None,
                },
            )
            entry["score"] = float(entry["score"]) + 1.0 / (rrf_k + rank)
            if result.dense_score is not None:
                entry["dense"] = result.dense_score
                entry["dense_rank"] = rank
            if result.bm25_score is not None:
                entry["bm25"] = result.bm25_score
                entry["bm25_rank"] = rank
    ordered = sorted(fused.values(), key=lambda item: float(item["score"]), reverse=True)
    return [
        SearchResult(
            chunk=item["chunk"],
            score=float(item["score"]),
            rrf_score=float(item["score"]),
            dense_score=item["dense"],
            bm25_score=item["bm25"],
            dense_rank=item["dense_rank"],
            bm25_rank=item["bm25_rank"],
            rrf_rank=rank,
        )
        for rank, item in enumerate(ordered[:top_k], start=1)
    ]


class HybridRetriever:
    def __init__(
        self,
        embeddings: EmbeddingProvider,
        vector_store: QdrantVectorStore,
        bm25: BM25Index,
        dense_top_k: int = 30,
        bm25_top_k: int = 30,
        fusion_top_k: int = 30,
        final_top_k: int = 5,
        rrf_k: int = 60,
        rerank_candidates: int = 15,
        rerank: Callable[[str, list[SearchResult]], list[SearchResult]] | None = None,
        dense_confidence_threshold: float = 0.85,
        dense_no_agreement_threshold: float = 0.88,
        reranker_confidence_threshold: float = 0.50,
    ) -> None:
        self.embeddings = embeddings
        self.vector_store = vector_store
        self.bm25 = bm25
        self.dense_top_k = dense_top_k
        self.bm25_top_k = bm25_top_k
        self.fusion_top_k = fusion_top_k
        self.final_top_k = final_top_k
        self.rrf_k = rrf_k
        self.rerank_candidates = rerank_candidates
        self.rerank = rerank
        self.dense_confidence_threshold = dense_confidence_threshold
        self.dense_no_agreement_threshold = dense_no_agreement_threshold
        self.reranker_confidence_threshold = reranker_confidence_threshold

    def _rank(self, query: str) -> list[SearchResult]:
        query_vector = self.embeddings.embed_query(query)
        dense = self.vector_store.search(query_vector, self.dense_top_k)
        lexical = self.bm25.search(query, self.bm25_top_k)
        fused = reciprocal_rank_fusion([dense, lexical], self.fusion_top_k, self.rrf_k)
        if self.rerank:
            return self.rerank(query, fused[: self.rerank_candidates])
        return fused[: self.final_top_k]

    def _decide(self, results: list[SearchResult]) -> tuple[bool, float, str]:
        if not results:
            return False, 0.0, "no_results"
        top = results[0]
        dense_score = top.dense_score or 0.0
        agreement = top.dense_score is not None and top.bm25_score is not None
        dense_threshold = (
            self.dense_confidence_threshold
            if agreement
            else self.dense_no_agreement_threshold
        )
        if dense_score < dense_threshold:
            return False, dense_score, "dense_below_threshold"
        if self.rerank is not None:
            reranker_score = top.reranker_score or 0.0
            if reranker_score < self.reranker_confidence_threshold:
                return False, reranker_score, "reranker_below_threshold"
            reason = "reranker_with_hybrid_agreement" if agreement else "reranker_dense_only"
            return True, reranker_score, reason
        reason = "dense_with_hybrid_agreement" if agreement else "dense_only"
        return True, dense_score, reason

    def retrieve_with_decision(self, query: str) -> RetrievalDecision:
        started = time.perf_counter()
        results = self._rank(query)
        is_answerable, confidence, reason = self._decide(results)
        retrieval_total_ms = (time.perf_counter() - started) * 1000
        reranker_ms = float(getattr(self.rerank, "last_latency_ms", 0.0))
        return RetrievalDecision(
            results=results,
            is_answerable=is_answerable,
            confidence=confidence,
            reason=reason,
            base_retrieval_ms=max(0.0, retrieval_total_ms - reranker_ms),
            reranker_ms=reranker_ms,
            retrieval_total_ms=retrieval_total_ms,
        )

    def retrieve(self, query: str) -> list[SearchResult]:
        return self.retrieve_with_decision(query).results
