from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class DocumentPage(BaseModel):
    model_config = ConfigDict(frozen=True)

    document_id: str
    filename: str
    source: str
    file_hash: str
    page_number: int | None
    text: str


class Chunk(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunk_id: str
    document_id: str
    filename: str
    source: str
    file_hash: str
    page_number: int | None = None
    chunk_index: int
    text: str

    def payload(self) -> dict[str, object]:
        return self.model_dump()


class SearchResult(BaseModel):
    chunk: Chunk
    score: float
    rrf_score: float | None = None
    reranker_score: float | None = None
    dense_score: float | None = None
    bm25_score: float | None = None
    dense_rank: int | None = None
    bm25_rank: int | None = None
    rrf_rank: int | None = None
    reranker_rank: int | None = None


class RetrievalDecision(BaseModel):
    results: list[SearchResult]
    is_answerable: bool
    confidence: float
    reason: str
    base_retrieval_ms: float = 0.0
    reranker_ms: float = 0.0
    retrieval_total_ms: float = 0.0


class Source(BaseModel):
    filename: str
    page: int | None
    chunk_id: str
    retrieval_score: float
    dense_score: float | None = None
    bm25_score: float | None = None
    rrf_score: float | None = None
    reranker_score: float | None = None


class Timings(BaseModel):
    base_retrieval_ms: float = 0.0
    reranker_ms: float = 0.0
    retrieval_total_ms: float = 0.0
    generation_ms: float = 0.0
    total_ms: float = 0.0
    context_chars: int = 0


RetrievalMode = Literal["fast", "quality"]


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    mode: RetrievalMode = "fast"

    @field_validator("query")
    @classmethod
    def reject_blank_query(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("query must not be blank")
        return value


class QueryResponse(BaseModel):
    answer: str
    is_answerable: bool
    sources: list[Source]
    timings: Timings


class IndexRequest(BaseModel):
    path: Path


class IndexResponse(BaseModel):
    files_discovered: int
    files_indexed: int
    files_skipped: int
    pages_parsed: int
    chunks_created: int
    embedding_batches: int
    timings: dict[str, float]
