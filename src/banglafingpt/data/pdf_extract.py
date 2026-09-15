"""Convert NBR regulatory PDFs into structured text (paper Sec. 3.2, step 1).

Hierarchy (chapter / section / sub-section headings) and tables are preserved as
plain-text markers so that downstream segmentation never splits a table row from
its header and every segment can be traced back to a section number.
"""
from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

from ..config import DOMAINS
from .schema import SourceDocument, stable_id

# Matches "Section 12.", "ধারা ১২।", "Chapter III", "12.3 Heading".
HEADING_RE = re.compile(
    r"^\s*(?:(?:section|chapter|rule|article|ধারা|অধ্যায়|বিধি)\s*[-–]?\s*)?"
    r"(\d+(?:\.\d+)*|[IVXLC]+|[০-৯]+(?:\.[০-৯]+)*)\s*[.।:)-]\s+(\S.{0,120})$",
    re.IGNORECASE,
)
PAGE_NUMBER_RE = re.compile(r"^\s*(?:page\s*)?\d{1,4}\s*$", re.IGNORECASE)


def _load_pdf_pages(path: Path) -> list[str]:
    """Extract per-page text, preferring pdfplumber (keeps tables) then pypdf."""
    try:
        import pdfplumber

        pages: list[str] = []
        with pdfplumber.open(str(path)) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                for table in page.extract_tables() or []:
                    text += "\n" + _render_table(table)
                pages.append(text)
        return pages
    except ImportError:
        pass
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "PDF extraction needs `pdfplumber` or `pypdf`; `pip install pdfplumber`."
        ) from exc
    return [(p.extract_text() or "") for p in PdfReader(str(path)).pages]


def _render_table(table: Iterable[Iterable[str | None]]) -> str:
    rows = ["[TABLE]"]
    for row in table:
        cells = [(c or "").replace("\n", " ").strip() for c in row]
        if any(cells):
            rows.append(" | ".join(cells))
    rows.append("[/TABLE]")
    return "\n".join(rows)


def clean_page(text: str) -> str:
    """Drop page numbers and de-hyphenate words broken across line ends."""
    lines = [ln.rstrip() for ln in text.splitlines()]
    kept = [ln for ln in lines if ln.strip() and not PAGE_NUMBER_RE.match(ln)]
    joined = "\n".join(kept)
    return re.sub(r"(\w)-\n(\w)", r"\1\2", joined)


def structure_text(pages: list[str]) -> str:
    """Tag headings with ``[H] <number> <title>`` markers, preserving order."""
    out: list[str] = []
    for page in pages:
        for line in clean_page(page).splitlines():
            match = HEADING_RE.match(line)
            if match and len(line.split()) <= 14:
                out.append(f"[H] {match.group(1)} {match.group(2).strip()}")
            else:
                out.append(line)
    return "\n".join(out)


def extract_document(
    path: str | Path,
    domain: str,
    *,
    title: str | None = None,
    year: int | None = None,
    url: str | None = None,
) -> SourceDocument:
    """Read one regulatory PDF (or ``.txt``) into a :class:`SourceDocument`."""
    p = Path(path)
    if domain not in DOMAINS:
        raise ValueError(f"domain must be one of {DOMAINS}, got {domain!r}")
    if p.suffix.lower() == ".pdf":
        text = structure_text(_load_pdf_pages(p))
    else:
        text = structure_text(p.read_text(encoding="utf-8").split("\f"))
    return SourceDocument(
        doc_id=stable_id(p.name, domain),
        title=title or p.stem,
        domain=domain,
        year=year,
        url=url,
        text=text,
    )


def extract_corpus(manifest: list[dict[str, str]], raw_dir: str | Path) -> list[SourceDocument]:
    """Extract every document listed in a manifest (``path``, ``domain``, ...)."""
    root = Path(raw_dir)
    docs: list[SourceDocument] = []
    for entry in manifest:
        docs.append(
            extract_document(
                root / entry["path"],
                entry["domain"],
                title=entry.get("title"),
                year=int(entry["year"]) if entry.get("year") else None,
                url=entry.get("url"),
            )
        )
    return docs
