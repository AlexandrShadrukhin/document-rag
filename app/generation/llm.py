from __future__ import annotations

from abc import ABC, abstractmethod

import httpx


class LLMUnavailableError(RuntimeError):
    pass


class BaseLLMProvider(ABC):
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        raise NotImplementedError


class OllamaProvider(BaseLLMProvider):
    def __init__(
        self,
        base_url: str,
        model: str,
        timeout_seconds: float = 120.0,
        temperature: float = 0.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.temperature = temperature

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        try:
            response = httpx.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "stream": False,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "options": {"temperature": self.temperature},
                },
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
            return str(payload["message"]["content"]).strip()
        except (httpx.HTTPError, KeyError, ValueError) as error:
            raise LLMUnavailableError(
                f"Ollama недоступна или вернула некорректный ответ: {self.base_url}. "
                f"Проверьте, что Ollama запущена и модель '{self.model}' загружена."
            ) from error
