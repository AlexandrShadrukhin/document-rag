from __future__ import annotations

from pathlib import Path

from app.ingestion.common import document_id_for, file_sha256, normalized_source
from app.schemas import DocumentPage


class TXTLoader:
    def load(self, path: Path) -> list[DocumentPage]:
        path = path.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"TXT file not found: {path}")
        if path.suffix.lower() != ".txt":
            raise ValueError(f"Expected a TXT file: {path}")

        raw = path.read_bytes()
        text: str | None = None
        for encoding in ("utf-8-sig", "utf-8", "cp1251"):
            try:
                text = raw.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
        if text is None:
            raise UnicodeError(f"Cannot decode text file as UTF-8 or CP1251: {path}")

        file_hash = file_sha256(path)
        return [
            DocumentPage(
                document_id=document_id_for(file_hash),
                filename=path.name,
                source=normalized_source(path),
                file_hash=file_hash,
                page_number=None,
                text=text,
            )
        ]
