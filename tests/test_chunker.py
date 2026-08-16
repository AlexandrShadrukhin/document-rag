from app.ingestion.chunker import SmartChunker
from app.schemas import DocumentPage


def page(text: str) -> DocumentPage:
    return DocumentPage(
        document_id="doc-id",
        filename="wiki.pdf",
        source="/data/wiki.pdf",
        file_hash="f" * 64,
        page_number=7,
        text=text,
    )


def test_chunker_prefers_paragraphs_and_bounds_length() -> None:
    text = ("Первый абзац. " * 12) + "\n\n" + ("Второй абзац. " * 12)
    chunks = SmartChunker(chunk_size=180, overlap=30).chunk_pages([page(text)])
    assert len(chunks) >= 2
    assert all(len(chunk.text) <= 180 for chunk in chunks)


def test_chunk_ids_are_deterministic() -> None:
    chunker = SmartChunker(chunk_size=120, overlap=20)
    first = chunker.chunk_pages([page("Предложение. " * 30)])
    second = chunker.chunk_pages([page("Предложение. " * 30)])
    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]


def test_metadata_is_propagated_to_every_chunk() -> None:
    chunks = SmartChunker(chunk_size=120, overlap=20).chunk_pages([page("Текст. " * 50)])
    assert chunks
    assert all(chunk.filename == "wiki.pdf" for chunk in chunks)
    assert all(chunk.page_number == 7 for chunk in chunks)
    assert all(chunk.file_hash == "f" * 64 for chunk in chunks)
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))
