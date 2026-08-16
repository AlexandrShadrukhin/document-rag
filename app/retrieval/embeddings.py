from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np

from app.lab.devices import resolve_torch_device


class EmbeddingProvider(Protocol):
    @property
    def dimension(self) -> int: ...

    def embed_passages(self, texts: Sequence[str], show_progress: bool = False) -> np.ndarray: ...

    def embed_query(self, query: str) -> np.ndarray: ...


class SentenceTransformerEmbeddings:
    def __init__(self, model_name: str, batch_size: int = 64, device: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.batch_size = batch_size
        self._uses_e5 = "e5" in model_name.lower()
        self.device_selection = resolve_torch_device(device)
        self.device = self.device_selection.selected
        self._model = SentenceTransformer(model_name, device=self.device)

    @property
    def dimension(self) -> int:
        dimension_getter = getattr(self._model, "get_embedding_dimension", None)
        dimension = (
            dimension_getter()
            if dimension_getter is not None
            else self._model.get_sentence_embedding_dimension()
        )
        if dimension is None:
            raise RuntimeError("Embedding model did not report its vector dimension")
        return int(dimension)

    def _encode(self, texts: Sequence[str], show_progress: bool) -> np.ndarray:
        return np.asarray(
            self._model.encode(
                list(texts),
                batch_size=self.batch_size,
                normalize_embeddings=True,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
            ),
            dtype=np.float32,
        )

    def embed_passages(self, texts: Sequence[str], show_progress: bool = False) -> np.ndarray:
        prepared = [f"passage: {text}" for text in texts] if self._uses_e5 else list(texts)
        if not prepared:
            return np.empty((0, self.dimension), dtype=np.float32)
        return self._encode(prepared, show_progress)

    def embed_query(self, query: str) -> np.ndarray:
        prepared = f"query: {query}" if self._uses_e5 else query
        return self._encode([prepared], False)[0]
