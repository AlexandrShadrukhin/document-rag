from __future__ import annotations

import bz2
import hashlib
import html
import json
import os
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import mwparserfromhell
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

from app.ingestion.common import file_sha256
from app.lab.environment import collect_environment
from app.lab.progress import ProgressCallback, ProgressEvent, emit
from app.wiki.fonts import FontResolutionError, resolve_cyrillic_font

BAD_PREFIXES = (
    "Википедия:",
    "Файл:",
    "Шаблон:",
    "Категория:",
    "Участник:",
    "Служебная:",
    "Портал:",
    "Обсуждение:",
)
SERVICE_SECTIONS = re.compile(
    r"==\s*(Литература|Примечания|Ссылки|Источники)\s*==", re.IGNORECASE
)


@dataclass(frozen=True)
class WikiCorpusConfig:
    source: Path
    output: Path
    limit: int | None = None
    target_bytes: int | None = None
    min_article_chars: int = 3000
    font_path: Path | None = None
    overwrite: bool = False
    resume: bool = True
    source_sha256: bool = False

    def __post_init__(self) -> None:
        if self.limit is not None and self.target_bytes is not None:
            raise ValueError("Use either limit or target size, not both")
        if self.limit is None and self.target_bytes is None:
            raise ValueError("Either limit or target size is required")
        if self.limit is not None and self.limit < 1:
            raise ValueError("limit must be positive")
        if self.target_bytes is not None and self.target_bytes < 1:
            raise ValueError("target size must be positive")
        if self.min_article_chars < 1:
            raise ValueError("min_article_chars must be positive")


@dataclass
class WikiCorpusStats:
    inspected_pages: int = 0
    accepted_articles: int = 0
    generated_pdfs: int = 0
    corpus_bytes: int = 0
    elapsed_seconds: float = 0.0
    aggregate_fingerprint: str = ""
    manifest_path: str = ""

    @property
    def corpus_gb(self) -> float:
        return self.corpus_bytes / 1024**3


def clean_wiki_text(text: str) -> str:
    if not text:
        return ""
    cleaned = mwparserfromhell.parse(text).strip_code()
    cleaned = SERVICE_SECTIONS.split(cleaned, maxsplit=1)[0]
    cleaned = re.sub(r"https?://\S+", "", cleaned)
    cleaned = re.sub(r"(мини|thumb)\|.*?\n", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"Файл:.*?\n", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.replace("|", " ")
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"([а-яё])([А-ЯЁ])", r"\1 \2", cleaned)
    cleaned = re.sub(r"\b([А-ЯЁ][а-яё]+)\s+\1\b", r"\1", cleaned)
    return cleaned.strip()


def should_stop(config: WikiCorpusConfig, stats: WikiCorpusStats) -> bool:
    if config.limit is not None:
        return stats.generated_pdfs >= config.limit
    return bool(config.target_bytes and stats.corpus_bytes >= config.target_bytes)


class WikiCorpusBuilder:
    version = "1"

    def __init__(self, config: WikiCorpusConfig) -> None:
        self.config = config
        self.source = config.source.expanduser().resolve()
        self.output = config.output.expanduser().resolve()
        if not self.source.is_file():
            raise FileNotFoundError(f"Wikipedia dump not found: {self.source}")
        if not self.source.name.lower().endswith((".xml.bz2", ".bz2")):
            raise ValueError(f"Expected an XML.BZ2 dump: {self.source}")
        self.font_path = resolve_cyrillic_font(config.font_path)
        self.font_name = f"WikiCorpus-{hashlib.sha1(str(self.font_path).encode()).hexdigest()[:8]}"
        if self.font_name not in pdfmetrics.getRegisteredFontNames():
            font = TTFont(self.font_name, str(self.font_path))
            glyphs = getattr(font.face, "charToGlyph", {})
            if any(not glyphs.get(ord(character)) for character in "АБЯабяЁё"):
                raise FontResolutionError(
                    f"Font does not contain required Cyrillic glyphs: {self.font_path}. "
                    "Set PDF_FONT_PATH to DejaVu Sans, Arial, or another Cyrillic TTF/OTF."
                )
            pdfmetrics.registerFont(font)

    def _styles(self) -> tuple[ParagraphStyle, ParagraphStyle]:
        styles = getSampleStyleSheet()
        return (
            ParagraphStyle(
                "WikiTitle",
                parent=styles["Title"],
                fontName=self.font_name,
                fontSize=18,
                leading=22,
            ),
            ParagraphStyle(
                "WikiBody",
                parent=styles["BodyText"],
                fontName=self.font_name,
                fontSize=11,
                leading=15,
            ),
        )

    def _save_pdf(self, title: str, text: str, index: int) -> Path:
        target = self.output / f"wiki_{index:06d}.pdf"
        temporary = target.with_suffix(".pdf.tmp")
        title_style, body_style = self._styles()
        story: list[object] = [
            Paragraph(f"Статья: {html.escape(title)}", title_style),
            Spacer(1, 15),
            Paragraph("Источник: Русская Википедия", body_style),
            Spacer(1, 20),
        ]
        paragraphs = re.split(r"\n+|(?<=[.!?])\s+(?=[А-ЯЁ])", text)
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if len(paragraph) < 50:
                continue
            story.extend([Paragraph(html.escape(paragraph), body_style), Spacer(1, 8)])
        if len(story) == 4:
            story.append(Paragraph(html.escape(text), body_style))
        SimpleDocTemplate(str(temporary)).build(story)
        os.replace(temporary, target)
        return target

    def _initial_state(self) -> tuple[WikiCorpusStats, hashlib._Hash]:
        stats = WikiCorpusStats()
        fingerprint = hashlib.sha256()
        existing = sorted(self.output.glob("wiki_*.pdf"))
        if not self.config.resume:
            existing = []
        for path in existing:
            digest = file_sha256(path)
            size = path.stat().st_size
            fingerprint.update(f"{path.name}:{size}:{digest}\n".encode())
            stats.generated_pdfs += 1
            stats.accepted_articles += 1
            stats.corpus_bytes += size
        return stats, fingerprint

    def build(self, callback: ProgressCallback | None = None) -> WikiCorpusStats:
        self.output.mkdir(parents=True, exist_ok=True)
        existing = list(self.output.glob("wiki_*.pdf"))
        if existing and self.config.overwrite:
            for path in existing:
                path.unlink()
        elif existing and not self.config.resume:
            raise FileExistsError(
                f"Output contains PDFs: {self.output}. Use resume or explicit overwrite."
            )

        stats, fingerprint = self._initial_state()
        if should_stop(self.config, stats):
            return self._write_manifest(stats, fingerprint.hexdigest())

        started = time.perf_counter()
        emit(callback, ProgressEvent("wiki_generation", "started", message="Reading dump"))
        accepted_seen = 0
        with bz2.open(self.source, "rb") as stream:
            context = ET.iterparse(stream, events=("start", "end"))
            _, root = next(context)
            for event, element in context:
                if event != "end" or not element.tag.endswith("page"):
                    continue
                stats.inspected_pages += 1
                title = ""
                raw_text = ""
                namespace = "0"
                for child in element.iter():
                    local_name = child.tag.rsplit("}", 1)[-1]
                    if local_name == "title":
                        title = child.text or ""
                    elif local_name == "ns":
                        namespace = child.text or "0"
                    elif local_name == "text":
                        raw_text = child.text or ""
                element.clear()
                root.clear()
                if namespace != "0" or any(title.startswith(prefix) for prefix in BAD_PREFIXES):
                    self._emit_progress(callback, stats, started, title)
                    continue
                cleaned = clean_wiki_text(raw_text)
                if len(cleaned) < self.config.min_article_chars:
                    self._emit_progress(callback, stats, started, title)
                    continue
                if accepted_seen < stats.generated_pdfs:
                    accepted_seen += 1
                    self._emit_progress(callback, stats, started, title)
                    continue
                accepted_seen += 1
                target = self._save_pdf(title, cleaned, stats.generated_pdfs)
                digest = file_sha256(target)
                size = target.stat().st_size
                fingerprint.update(f"{target.name}:{size}:{digest}\n".encode())
                stats.generated_pdfs += 1
                stats.accepted_articles += 1
                stats.corpus_bytes += size
                self._emit_progress(callback, stats, started, title)
                if should_stop(self.config, stats):
                    break
        stats.elapsed_seconds = time.perf_counter() - started
        emit(
            callback,
            ProgressEvent(
                "wiki_generation",
                "completed",
                current=(
                    stats.corpus_bytes
                    if self.config.target_bytes is not None
                    else stats.generated_pdfs
                ),
                total=(
                    self.config.target_bytes
                    if self.config.target_bytes is not None
                    else self.config.limit
                ),
                message="PDF corpus ready",
            ),
        )
        return self._write_manifest(stats, fingerprint.hexdigest())

    def _emit_progress(
        self,
        callback: ProgressCallback | None,
        stats: WikiCorpusStats,
        started: float,
        title: str,
    ) -> None:
        elapsed = max(time.perf_counter() - started, 1e-9)
        stats.elapsed_seconds = elapsed
        if self.config.target_bytes:
            current, total = stats.corpus_bytes, self.config.target_bytes
            rate = stats.corpus_bytes / elapsed
            eta = max(0.0, total - current) / rate if rate > 0 else None
        else:
            current, total = stats.generated_pdfs, self.config.limit
            rate = stats.generated_pdfs / elapsed
            eta = max(0.0, (total or 0) - current) / rate if rate > 0 else None
        emit(
            callback,
            ProgressEvent(
                "wiki_generation",
                "progress",
                current=current,
                total=total,
                message=title,
                details={
                    "generated_pdfs": stats.generated_pdfs,
                    "accepted_articles": stats.accepted_articles,
                    "inspected_pages": stats.inspected_pages,
                    "corpus_bytes": stats.corpus_bytes,
                    "elapsed_seconds": elapsed,
                    "throughput_bytes_per_second": stats.corpus_bytes / elapsed,
                    "eta_seconds": eta,
                },
            ),
        )

    def _write_manifest(self, stats: WikiCorpusStats, fingerprint: str) -> WikiCorpusStats:
        stats.aggregate_fingerprint = fingerprint
        manifest = self.output / "corpus_manifest.json"
        source_hash = file_sha256(self.source) if self.config.source_sha256 else None
        environment = collect_environment("http://localhost:11434", self.output)
        payload = {
            "generator_version": self.version,
            "generated_at": datetime.now(UTC).isoformat(),
            "source_dump_filename": self.source.name,
            "source_dump_path": str(self.source),
            "source_dump_size": self.source.stat().st_size,
            "source_dump_sha256": source_hash,
            "config": {
                **asdict(self.config),
                "source": str(self.source),
                "output": str(self.output),
                "font_path": str(self.font_path),
            },
            "inspected_pages": stats.inspected_pages,
            "accepted_articles": stats.accepted_articles,
            "generated_pdfs": stats.generated_pdfs,
            "total_corpus_bytes": stats.corpus_bytes,
            "total_corpus_gb": stats.corpus_gb,
            "aggregate_pdf_fingerprint": fingerprint,
            "platform": environment.as_dict(),
        }
        manifest.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        stats.manifest_path = str(manifest)
        return stats
