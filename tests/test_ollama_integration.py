from __future__ import annotations

import os

import httpx
import pytest

from app.config import get_settings
from app.generation.llm import OllamaProvider


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("RUN_OLLAMA_INTEGRATION") != "1",
    reason="requires a running Ollama server and the configured model",
)
def test_configured_ollama_provider_returns_text() -> None:
    settings = get_settings()
    response = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=5.0)
    response.raise_for_status()
    models = {model["name"] for model in response.json()["models"]}
    assert settings.ollama_model in models

    provider = OllamaProvider(
        settings.ollama_base_url,
        settings.ollama_model,
        settings.ollama_timeout_seconds,
        settings.llm_temperature,
    )
    answer = provider.generate(
        "Отвечай кратко на русском языке.",
        "Ответь одним словом: столица Литвы?",
    )
    assert answer.strip()
