from conftest import make_chunk

from app.retrieval.hybrid import HybridRetriever, reciprocal_rank_fusion
from app.schemas import SearchResult


def test_rrf_rewards_results_present_in_both_rankings() -> None:
    a = make_chunk("a", "A")
    b = make_chunk("b", "B")
    c = make_chunk("c", "C")
    dense = [
        SearchResult(chunk=a, score=0.9, dense_score=0.9),
        SearchResult(chunk=b, score=0.8, dense_score=0.8),
    ]
    lexical = [
        SearchResult(chunk=c, score=4.0, bm25_score=4.0),
        SearchResult(chunk=a, score=3.0, bm25_score=3.0),
    ]
    fused = reciprocal_rank_fusion([dense, lexical], top_k=3, rrf_k=60)
    assert fused[0].chunk.chunk_id == "a"
    assert fused[0].dense_score == 0.9
    assert fused[0].bm25_score == 3.0
    assert fused[0].dense_rank == 1
    assert fused[0].bm25_rank == 2
    assert fused[0].rrf_rank == 1


def retriever_with_confidence(reranker: bool = False) -> HybridRetriever:
    rerank = (lambda query, results: results) if reranker else None
    return HybridRetriever(
        embeddings=None,  # type: ignore[arg-type]
        vector_store=None,  # type: ignore[arg-type]
        bm25=None,  # type: ignore[arg-type]
        rerank=rerank,
        dense_confidence_threshold=0.85,
        dense_no_agreement_threshold=0.88,
        reranker_confidence_threshold=0.50,
    )


def test_confidence_uses_hybrid_agreement() -> None:
    result = SearchResult(
        chunk=make_chunk("answer", "answer"),
        score=0.03,
        dense_score=0.86,
        bm25_score=4.0,
    )
    answerable, confidence, reason = retriever_with_confidence()._decide([result])
    assert answerable is True
    assert confidence == 0.86
    assert reason == "dense_with_hybrid_agreement"


def test_confidence_requires_stronger_dense_score_without_agreement() -> None:
    result = SearchResult(
        chunk=make_chunk("weak", "weak"),
        score=0.02,
        dense_score=0.86,
    )
    answerable, _, reason = retriever_with_confidence()._decide([result])
    assert answerable is False
    assert reason == "dense_below_threshold"


def test_confidence_rejects_low_reranker_score() -> None:
    result = SearchResult(
        chunk=make_chunk("negative", "negative"),
        score=0.03,
        dense_score=0.90,
        bm25_score=5.0,
        reranker_score=0.10,
    )
    answerable, confidence, reason = retriever_with_confidence(True)._decide([result])
    assert answerable is False
    assert confidence == 0.10
    assert reason == "reranker_below_threshold"


def test_reranker_receives_configured_candidate_count() -> None:
    dense = [
        SearchResult(
            chunk=make_chunk(str(index), f"chunk {index}"),
            score=1.0 - index / 100,
            dense_score=1.0 - index / 100,
        )
        for index in range(10)
    ]
    received: list[int] = []

    class Embeddings:
        def embed_query(self, query: str) -> None:
            return None

    class VectorStore:
        def search(self, query_vector: None, top_k: int) -> list[SearchResult]:
            return dense[:top_k]

    class BM25:
        def search(self, query: str, top_k: int) -> list[SearchResult]:
            return []

    def rerank(query: str, results: list[SearchResult]) -> list[SearchResult]:
        received.append(len(results))
        return results[:5]

    retriever = HybridRetriever(
        Embeddings(),  # type: ignore[arg-type]
        VectorStore(),  # type: ignore[arg-type]
        BM25(),  # type: ignore[arg-type]
        fusion_top_k=30,
        final_top_k=5,
        rerank_candidates=7,
        rerank=rerank,
    )

    retriever.retrieve("query")

    assert received == [7]
