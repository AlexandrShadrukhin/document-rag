from conftest import make_chunk

from app.generation.prompt import REFUSAL_ANSWER
from app.generation.rag import RAGService
from app.schemas import RetrievalDecision, SearchResult


class DecisionRetriever:
    def __init__(self, decision: RetrievalDecision) -> None:
        self.decision = decision

    def retrieve_with_decision(self, query: str) -> RetrievalDecision:
        return self.decision


class FailIfCalledLLM:
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        raise AssertionError("LLM must not be called for rejected retrieval")


class CapturingLLM:
    def __init__(self) -> None:
        self.system_prompt = ""
        self.user_prompt = ""

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        return "Ответ из источника"


def test_empty_retrieval_returns_refusal_without_llm_call() -> None:
    retriever = DecisionRetriever(
        RetrievalDecision(
            results=[], is_answerable=False, confidence=0.0, reason="no_results"
        )
    )
    response = RAGService(retriever, FailIfCalledLLM()).answer("Вопрос")
    assert response.answer == REFUSAL_ANSWER
    assert response.is_answerable is False
    assert response.sources == []
    assert response.timings.generation_ms == 0.0


def test_low_confidence_returns_refusal_without_llm_call() -> None:
    result = SearchResult(
        chunk=make_chunk("low", "нерелевантно"),
        score=0.01,
        dense_score=0.4,
    )
    retriever = DecisionRetriever(
        RetrievalDecision(
            results=[result],
            is_answerable=False,
            confidence=0.4,
            reason="dense_below_threshold",
        )
    )
    response = RAGService(retriever, FailIfCalledLLM()).answer("Вопрос")
    assert response.answer == REFUSAL_ANSWER
    assert response.is_answerable is False
    assert response.sources == []
    assert response.timings.generation_ms == 0.0


def test_prompt_and_sources_use_only_selected_chunks() -> None:
    first = SearchResult(
        chunk=make_chunk("first", "Байкал имеет глубину 1642 м."),
        score=0.03,
        dense_score=0.91,
        rrf_score=0.03,
        reranker_score=0.90,
    )
    second = SearchResult(
        chunk=make_chunk("second", "Этот текст не должен попасть в prompt."),
        score=0.02,
        dense_score=0.89,
        rrf_score=0.02,
        reranker_score=0.80,
    )
    retriever = DecisionRetriever(
        RetrievalDecision(
            results=[first, second],
            is_answerable=True,
            confidence=0.90,
            reason="reranker_with_hybrid_agreement",
            base_retrieval_ms=30.0,
            reranker_ms=2500.0,
            retrieval_total_ms=2530.0,
        )
    )
    llm = CapturingLLM()

    response = RAGService(retriever, llm, max_context_chunks=1).answer("Глубина?")

    assert "Байкал имеет глубину 1642 м." in llm.user_prompt
    assert "Этот текст не должен попасть" not in llm.user_prompt
    assert "[SOURCE 1]" in llm.user_prompt
    assert "[SOURCE 2]" not in llm.user_prompt
    assert [source.chunk_id for source in response.sources] == ["first"]
    assert response.sources[0].retrieval_score == 0.90
    assert response.sources[0].reranker_score == 0.90
    assert response.timings.base_retrieval_ms == 30.0
    assert response.timings.reranker_ms == 2500.0
    assert response.timings.retrieval_total_ms == 2530.0
    assert response.timings.context_chars == len(first.chunk.text)
