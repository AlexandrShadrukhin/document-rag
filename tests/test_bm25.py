from conftest import make_chunk

from app.retrieval.bm25 import BM25Index


def test_bm25_finds_exact_identifier() -> None:
    index = BM25Index(
        [
            make_chunk("one", "Общее описание договора"),
            make_chunk("two", "Документ № АБ-2025-17 подписан Ивановым"),
        ]
    )
    results = index.search("АБ-2025-17", top_k=2)
    assert results[0].chunk.chunk_id == "two"
    assert results[0].bm25_score is not None
    assert results[0].bm25_score > 0.0


def test_bm25_empty_corpus() -> None:
    assert BM25Index().search("что-нибудь", 5) == []
