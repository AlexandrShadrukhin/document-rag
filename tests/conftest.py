from __future__ import annotations

from app.schemas import Chunk


def make_chunk(
    chunk_id: str,
    text: str,
    *,
    document_id: str = "doc",
    page_number: int | None = 1,
    chunk_index: int = 0,
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id=document_id,
        filename="sample.pdf",
        source="sample.pdf",
        file_hash="a" * 64,
        page_number=page_number,
        chunk_index=chunk_index,
        text=text,
    )
