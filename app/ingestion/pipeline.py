from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from app.ingestion.chunker import SmartChunker
from app.ingestion.cleaner import clean_text
from app.ingestion.pdf_loader import PDFLoader
from app.ingestion.txt_loader import TXTLoader
from app.schemas import Chunk, DocumentPage

SUPPORTED_SUFFIXES = {".pdf", ".txt"}


@dataclass
class PreparedDocument:
    path: Path
    pages: list[DocumentPage]
    chunks: list[Chunk]
    timings: dict[str, float] = field(default_factory=dict)


class IngestionPipeline:
    def __init__(self, chunker: SmartChunker) -> None:
        self.chunker = chunker
        self.pdf_loader = PDFLoader()
        self.txt_loader = TXTLoader()

    @staticmethod
    def discover(path: Path) -> list[Path]:
        path = path.expanduser().resolve()
        if path.is_file():
            return [path] if path.suffix.lower() in SUPPORTED_SUFFIXES else []
        if not path.is_dir():
            raise FileNotFoundError(f"Input path does not exist: {path}")
        return sorted(
            item for item in path.rglob("*")
            if item.is_file() and item.suffix.lower() in SUPPORTED_SUFFIXES
        )

    def prepare(self, path: Path) -> PreparedDocument:
        started = time.perf_counter()
        if path.suffix.lower() == ".pdf":
            pages = self.pdf_loader.load(path)
        elif path.suffix.lower() == ".txt":
            pages = self.txt_loader.load(path)
        else:
            raise ValueError(f"Unsupported document type: {path.suffix}")
        parsed = time.perf_counter()

        cleaned_pages = [page.model_copy(update={"text": clean_text(page.text)}) for page in pages]
        cleaned = time.perf_counter()
        chunks = self.chunker.chunk_pages(cleaned_pages)
        chunked = time.perf_counter()
        return PreparedDocument(
            path=path,
            pages=cleaned_pages,
            chunks=chunks,
            timings={
                "parsing": parsed - started,
                "cleaning": cleaned - parsed,
                "chunking": chunked - cleaned,
            },
        )
