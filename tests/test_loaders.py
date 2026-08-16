from __future__ import annotations

from pathlib import Path

import pymupdf

from app.ingestion.pdf_loader import PDFLoader
from app.ingestion.txt_loader import TXTLoader


def test_pdf_loader_extracts_pages_and_metadata(tmp_path: Path) -> None:
    path = tmp_path / "русский.pdf"
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), "Baikal 1642 meters")
    document.save(path)
    document.close()

    pages = PDFLoader().load(path)

    assert len(pages) == 1
    assert pages[0].page_number == 1
    assert pages[0].filename == path.name
    assert pages[0].source == str(path.resolve())
    assert "1642" in pages[0].text
    assert len(pages[0].file_hash) == 64


def test_txt_loader_utf8_metadata(tmp_path: Path) -> None:
    path = tmp_path / "document.txt"
    path.write_text("Дата: 12.03.2025, № 42", encoding="utf-8")

    page = TXTLoader().load(path)[0]

    assert page.page_number is None
    assert page.filename == "document.txt"
    assert "№ 42" in page.text
    assert page.document_id == page.file_hash
