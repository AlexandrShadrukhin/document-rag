from __future__ import annotations

import time
from typing import Protocol

from app.generation.llm import BaseLLMProvider
from app.generation.prompt import REFUSAL_ANSWER, SYSTEM_PROMPT, build_user_prompt
from app.schemas import QueryResponse, RetrievalDecision, Source, Timings


class Retriever(Protocol):
    def retrieve_with_decision(self, query: str) -> RetrievalDecision: ...


class RAGService:
    def __init__(
        self,
        retriever: Retriever,
        llm: BaseLLMProvider,
        max_context_chunks: int = 8,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.max_context_chunks = max_context_chunks

    def answer(self, query: str) -> QueryResponse:
        total_started = time.perf_counter()
        decision = self.retriever.retrieve_with_decision(query)

        if not decision.is_answerable:
            total_ms = (time.perf_counter() - total_started) * 1000
            return QueryResponse(
                answer=REFUSAL_ANSWER,
                is_answerable=False,
                sources=[],
                timings=Timings(
                    base_retrieval_ms=decision.base_retrieval_ms,
                    reranker_ms=decision.reranker_ms,
                    retrieval_total_ms=decision.retrieval_total_ms,
                    total_ms=total_ms,
                ),
            )

        relevant = decision.results[: self.max_context_chunks]
        user_prompt = build_user_prompt(query, relevant)
        generation_started = time.perf_counter()
        answer = self.llm.generate(SYSTEM_PROMPT, user_prompt)
        generation_ms = (time.perf_counter() - generation_started) * 1000
        total_ms = (time.perf_counter() - total_started) * 1000
        sources = [
            Source(
                filename=result.chunk.filename,
                page=result.chunk.page_number,
                chunk_id=result.chunk.chunk_id,
                retrieval_score=(
                    result.reranker_score
                    if result.reranker_score is not None
                    else result.dense_score
                    if result.dense_score is not None
                    else result.score
                ),
                dense_score=result.dense_score,
                bm25_score=result.bm25_score,
                rrf_score=result.rrf_score,
                reranker_score=result.reranker_score,
            )
            for result in relevant
        ]
        return QueryResponse(
            answer=answer,
            is_answerable=True,
            sources=sources,
            timings=Timings(
                base_retrieval_ms=decision.base_retrieval_ms,
                reranker_ms=decision.reranker_ms,
                retrieval_total_ms=decision.retrieval_total_ms,
                generation_ms=generation_ms,
                total_ms=total_ms,
                context_chars=sum(len(result.chunk.text) for result in relevant),
            ),
        )
