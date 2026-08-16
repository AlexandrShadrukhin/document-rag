from __future__ import annotations

import logging
from pathlib import Path

import pymupdf

from app.ingestion.common import document_id_for, file_sha256, normalized_source
from app.schemas import DocumentPage

logger = logging.getLogger(__name__)


class PDFLoader:
    """Extract one record per PDF page. OCR is intentionally not performed."""

    def load(self, path: Path) -> list[DocumentPage]:
        path = path.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"PDF not found: {path}")
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a PDF file: {path}")

        file_hash = file_sha256(path)
        document_id = document_id_for(file_hash)
        pages: list[DocumentPage] = []
        with pymupdf.open(path) as document:
            for index, page in enumerate(document):
                pages.append(
                    DocumentPage(
                        document_id=document_id,
                        filename=path.name,
                        source=normalized_source(path),
                        file_hash=file_hash,
                        page_number=index + 1,
                        text=page.get_text("text"),
                    )
                )

        if not any(page.text.strip() for page in pages):
            logger.warning("PDF appears to be scanned / no text extracted: %s", path)
        return pages
