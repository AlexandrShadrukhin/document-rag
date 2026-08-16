from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    """Application configuration loaded from environment variables or .env."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    chunk_size: int = Field(default=1000, ge=100)
    chunk_overlap: int = Field(default=150, ge=0)

    embedding_model: str = "intfloat/multilingual-e5-small"
    embedding_batch_size: int = Field(default=64, ge=1)
    embedding_device: str | None = None
    index_document_batch_size: int = Field(default=16, ge=1)

    qdrant_mode: Literal["local", "server"] = "local"
    qdrant_path: Path = PROJECT_ROOT / "data" / "index" / "qdrant"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "documents"
    corpus_path: Path = PROJECT_ROOT / "data" / "index" / "corpus.jsonl"
    manifest_path: Path = PROJECT_ROOT / "data" / "index" / "manifest.json"

    dense_top_k: int = Field(default=30, ge=1)
    bm25_top_k: int = Field(default=30, ge=1)
    fusion_top_k: int = Field(default=30, ge=1)
    final_top_k: int = Field(default=5, ge=1)
    rrf_k: int = Field(default=60, ge=1)

    enable_reranker: bool = False
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_candidates: int = Field(default=15, ge=1)
    reranker_batch_size: int = Field(default=8, ge=1)
    reranker_device: str | None = None
    confidence_dense_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    confidence_dense_no_agreement_threshold: float = Field(
        default=0.88, ge=0.0, le=1.0
    )
    confidence_reranker_threshold: float = Field(default=0.50, ge=0.0, le=1.0)

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3:4b-instruct"
    ollama_timeout_seconds: float = Field(default=120.0, gt=0)
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_context_chunks: int = Field(default=8, ge=1)

    wiki_dump_url: str = (
        "https://dumps.wikimedia.org/ruwiki/latest/"
        "ruwiki-latest-pages-articles.xml.bz2"
    )
    wiki_dump_path: Path = PROJECT_ROOT / "data" / "source" / "ruwiki-pages-articles.xml.bz2"
    wiki_corpus_path: Path = PROJECT_ROOT / "data" / "corpus" / "wiki_pdf"
    benchmarks_path: Path = PROJECT_ROOT / "data" / "benchmarks"
    experiments_path: Path = PROJECT_ROOT / "data" / "experiments"
    pdf_font_path: Path | None = None
    resource_sample_interval_seconds: float = Field(default=1.0, ge=0.2)

    log_level: str = "INFO"

    @field_validator(
        "qdrant_path",
        "corpus_path",
        "manifest_path",
        "wiki_dump_path",
        "wiki_corpus_path",
        "benchmarks_path",
        "experiments_path",
        "pdf_font_path",
        mode="before",
    )
    @classmethod
    def resolve_project_path(cls, value: str | Path | None) -> Path | None:
        if value is None or value == "":
            return None
        path = Path(value).expanduser()
        return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()

    @field_validator("embedding_device", "reranker_device")
    @classmethod
    def validate_device(cls, value: str | None) -> str | None:
        if value is None or not value.strip() or value.lower() == "auto":
            return None
        normalized = value.lower()
        if normalized not in {"cpu", "cuda", "mps"}:
            raise ValueError("device must be one of: auto, cpu, cuda, mps")
        return normalized

    @field_validator("chunk_overlap")
    @classmethod
    def validate_overlap(cls, value: int, info: object) -> int:
        data = getattr(info, "data", {})
        if "chunk_size" in data and value >= data["chunk_size"]:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        return value

    def ensure_runtime_dirs(self) -> None:
        self.qdrant_path.parent.mkdir(parents=True, exist_ok=True)
        self.corpus_path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.wiki_dump_path.parent.mkdir(parents=True, exist_ok=True)
        self.wiki_corpus_path.mkdir(parents=True, exist_ok=True)
        self.benchmarks_path.mkdir(parents=True, exist_ok=True)
        self.experiments_path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_runtime_dirs()
    return settings
