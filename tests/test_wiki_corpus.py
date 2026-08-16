from __future__ import annotations

import bz2
import json
from pathlib import Path
from types import MethodType
from typing import Any

from app.wiki.corpus import (
    WikiCorpusBuilder,
    WikiCorpusConfig,
    WikiCorpusStats,
    should_stop,
)


def test_target_size_and_limit_stop_logic(tmp_path: Path) -> None:
    by_size = WikiCorpusConfig(
        source=tmp_path / "dump.bz2", output=tmp_path / "out", target_bytes=100
    )
    by_limit = WikiCorpusConfig(
        source=tmp_path / "dump.bz2", output=tmp_path / "out", limit=2
    )
    assert should_stop(by_size, WikiCorpusStats(corpus_bytes=99)) is False
    assert should_stop(by_size, WikiCorpusStats(corpus_bytes=100)) is True
    assert should_stop(by_limit, WikiCorpusStats(generated_pdfs=1)) is False
    assert should_stop(by_limit, WikiCorpusStats(generated_pdfs=2)) is True


def test_streaming_generator_writes_manifest(tmp_path: Path, monkeypatch: Any) -> None:
    xml = """<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.11/">
    <page><title>Байкал</title><ns>0</ns><revision><text>
    Байкал — глубокое озеро. Максимальная глубина составляет 1642 метра.
    Дополнительный текст для прохождения минимальной длины статьи.
    </text></revision></page>
    <page><title>Википедия:Служебная</title><ns>4</ns><revision><text>skip</text></revision></page>
    </mediawiki>"""
    source = tmp_path / "wiki.xml.bz2"
    source.write_bytes(bz2.compress(xml.encode()))
    output = tmp_path / "corpus"
    config = WikiCorpusConfig(source=source, output=output, limit=1, min_article_chars=20)
    builder = object.__new__(WikiCorpusBuilder)
    builder.config = config
    builder.source = source.resolve()
    builder.output = output.resolve()
    builder.font_path = tmp_path / "font.ttf"

    def fake_save(self: WikiCorpusBuilder, title: str, text: str, index: int) -> Path:
        path = self.output / f"wiki_{index:06d}.pdf"
        path.write_bytes(f"{title}\n{text}".encode())
        return path

    builder._save_pdf = MethodType(fake_save, builder)
    monkeypatch.setattr(
        "app.wiki.corpus.collect_environment",
        lambda base_url, disk_path: type(
            "Snapshot", (), {"as_dict": lambda self: {"os": "test"}}
        )(),
    )
    stats = builder.build()
    manifest = json.loads((output / "corpus_manifest.json").read_text())
    assert stats.generated_pdfs == 1
    assert stats.corpus_bytes > 0
    assert manifest["generated_pdfs"] == 1
    assert manifest["aggregate_pdf_fingerprint"] == stats.aggregate_fingerprint
    assert manifest["source_dump_filename"] == source.name

