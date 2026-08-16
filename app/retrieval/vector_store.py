from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import httpx
import numpy as np

from app.schemas import Chunk, SearchResult


@dataclass(frozen=True)
class QdrantBackendStatus:
    mode: str
    available: bool
    endpoint: str
    server_version: str | None = None
    client_version: str | None = None
    error: str | None = None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def qdrant_backend_status(
    mode: str,
    path: Path,
    url: str,
    api_key: str | None = None,
    timeout_seconds: float = 2.0,
) -> QdrantBackendStatus:
    try:
        client_version = version("qdrant-client")
    except PackageNotFoundError:
        client_version = None
    if mode == "local":
        return QdrantBackendStatus(
            mode=mode,
            available=path.exists(),
            endpoint=str(path),
            client_version=client_version,
        )
    if mode != "server":
        return QdrantBackendStatus(
            mode=mode,
            available=False,
            endpoint=url,
            client_version=client_version,
            error=f"Unsupported Qdrant mode: {mode}",
        )
    headers = {"api-key": api_key} if api_key else None
    try:
        response = httpx.get(url.rstrip("/") + "/", headers=headers, timeout=timeout_seconds)
        response.raise_for_status()
        payload = response.json()
        return QdrantBackendStatus(
            mode=mode,
            available=True,
            endpoint=url,
            server_version=str(payload.get("version")) if payload.get("version") else None,
            client_version=client_version,
        )
    except (httpx.HTTPError, ValueError) as error:
        return QdrantBackendStatus(
            mode=mode,
            available=False,
            endpoint=url,
            client_version=client_version,
            error=str(error),
        )


def qdrant_collection_exists(
    url: str,
    collection: str,
    api_key: str | None = None,
    timeout_seconds: float = 2.0,
) -> bool:
    headers = {"api-key": api_key} if api_key else None
    try:
        response = httpx.get(
            f"{url.rstrip('/')}/collections/{collection}",
            headers=headers,
            timeout=timeout_seconds,
        )
        return response.status_code == 200
    except httpx.HTTPError:
        return False


class QdrantVectorStore:
    def __init__(
        self,
        collection: str,
        vector_size: int,
        mode: str = "local",
        path: Path | None = None,
        url: str = "http://localhost:6333",
        api_key: str | None = None,
    ) -> None:
        from qdrant_client import QdrantClient, models

        self.collection = collection
        self.mode = mode
        self.path = path
        self.url = url
        self._models = models
        if mode == "local":
            if path is None:
                raise ValueError("Qdrant local mode requires a persistent path")
            path.mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(path))
        elif mode == "server":
            self.client = QdrantClient(url=url, api_key=api_key)
        else:
            raise ValueError(f"Unsupported Qdrant mode: {mode}")

        if not self.client.collection_exists(collection):
            self.client.create_collection(
                collection_name=collection,
                vectors_config=models.VectorParams(
                    size=vector_size, distance=models.Distance.COSINE
                ),
            )

    def upsert(self, chunks: Sequence[Chunk], vectors: np.ndarray, batch_size: int = 256) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("Chunks and vectors must have equal lengths")
        for start in range(0, len(chunks), batch_size):
            points = [
                self._models.PointStruct(
                    id=chunk.chunk_id,
                    vector=vector.tolist(),
                    payload=chunk.payload(),
                )
                for chunk, vector in zip(
                    chunks[start : start + batch_size],
                    vectors[start : start + batch_size],
                    strict=True,
                )
            ]
            self.client.upsert(collection_name=self.collection, points=points, wait=True)

    def delete_document(self, document_id: str) -> None:
        selector = self._models.FilterSelector(
            filter=self._models.Filter(
                must=[
                    self._models.FieldCondition(
                        key="document_id", match=self._models.MatchValue(value=document_id)
                    )
                ]
            )
        )
        self.client.delete(collection_name=self.collection, points_selector=selector, wait=True)

    def search(self, query_vector: np.ndarray, top_k: int) -> list[SearchResult]:
        response = self.client.query_points(
            collection_name=self.collection,
            query=query_vector.tolist(),
            limit=top_k,
            with_payload=True,
        )
        results: list[SearchResult] = []
        for point in response.points:
            payload = dict(point.payload or {})
            chunk = Chunk.model_validate(payload)
            cosine = float(point.score)
            results.append(
                SearchResult(
                    chunk=chunk,
                    score=cosine,
                    dense_score=cosine,
                )
            )
        return results

    def close(self) -> None:
        self.client.close()
