from __future__ import annotations

import os

import pytest

from app.config import get_settings
from app.runtime import ApplicationContainer


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("RUN_RETRIEVAL_INTEGRATION") != "1",
    reason="requires the persistent Qdrant index and cached embedding model",
)
def test_baikal_depth_chunk_is_in_final_top_k() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    settings = get_settings().model_copy(
        update={"enable_reranker": False, "embedding_device": "cpu"}
    )
    container = ApplicationContainer(settings)
    try:
        results = container.retriever().retrieve(
            "Какова максимальная глубина озера Байкал?"
        )
    finally:
        container.close()
    assert any(
        result.chunk.filename == "wiki_001.pdf"
        and result.chunk.page_number == 7
        and "1642" in result.chunk.text
        for result in results
    )
