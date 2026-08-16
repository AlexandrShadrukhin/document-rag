from __future__ import annotations

from app.schemas import SearchResult

REFUSAL_ANSWER = "В предоставленных документах нет информации для ответа."

SYSTEM_PROMPT = f"""Ты помощник по базе документов.
Отвечай только на основании предоставленных источников.
Не используй знания, которых нет в источниках, и не придумывай отсутствующие факты.
Не описывай внутренний процесс поиска.
Не добавляй список источников: он формируется приложением.
Если контекста недостаточно, ответь ровно: «{REFUSAL_ANSWER}»
Ответ должен быть кратким и по существу, на русском языке."""


def build_user_prompt(query: str, results: list[SearchResult]) -> str:
    sources: list[str] = []
    for index, result in enumerate(results, start=1):
        page = result.chunk.page_number if result.chunk.page_number is not None else "—"
        sources.append(
            f"[SOURCE {index}]\n"
            f"filename: {result.chunk.filename}\n"
            f"page: {page}\n"
            f"text:\n{result.chunk.text}"
        )
    return "\n\n".join(sources) + f"\n\nQUESTION:\n{query}"
