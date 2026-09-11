"""Split structured documents into coherent 150-300 word units (Sec. 3.2)."""
from __future__ import annotations

import re
from collections.abc import Iterable

from ..utils import word_count
from .schema import Segment, SourceDocument, stable_id

# Sentence boundary for Bangla (danda) and English (period/question/exclamation).
SENTENCE_RE = re.compile(r"(?<=[।?!.])\s+")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE_RE.split(text) if s.strip()]


def _blocks(text: str) -> list[tuple[str, str]]:
    """Group lines under their most recent ``[H]`` heading; keep tables intact."""
    section = ""
    buffer: list[str] = []
    out: list[tuple[str, str]] = []
    in_table = False
    for line in text.splitlines():
        if line.startswith("[H] ") and not in_table:
            if buffer:
                out.append((section, "\n".join(buffer)))
                buffer = []
            section = line[4:].strip()
            continue
        if line.strip() == "[TABLE]":
            in_table = True
        elif line.strip() == "[/TABLE]":
            in_table = False
        buffer.append(line)
    if buffer:
        out.append((section, "\n".join(buffer)))
    return out


def segment_document(doc: SourceDocument, min_words: int = 150,
                     max_words: int = 300) -> list[Segment]:
    """Pack sentences into segments of ``min_words``-``max_words``.

    Sentences are never split; a block shorter than ``min_words`` is still kept
    when it is a complete section, because short regulatory clauses (e.g. a rate
    table) are exactly the units we want to be able to cite.
    """
    if min_words > max_words:
        raise ValueError("min_words must not exceed max_words")
    segments: list[Segment] = []
    order = 0
    for section, block in _blocks(doc.text):
        current: list[str] = []
        current_words = 0
        for sentence in split_sentences(block):
            n = word_count(sentence)
            if current_words + n > max_words and current_words >= min_words:
                segments.append(_make(doc, section, current, order))
                order += 1
                current, current_words = [], 0
            current.append(sentence)
            current_words += n
        if current:
            if segments and current_words < min_words // 2 and segments[-1].section == section:
                segments[-1].text += " " + " ".join(current)  # absorb a dangling tail
            else:
                segments.append(_make(doc, section, current, order))
                order += 1
    return segments


def _make(doc: SourceDocument, section: str, sentences: Iterable[str], order: int) -> Segment:
    text = " ".join(sentences).strip()
    return Segment(
        segment_id=stable_id(doc.doc_id, section, str(order)),
        doc_id=doc.doc_id,
        domain=doc.domain,
        text=text,
        section=section or None,
        order=order,
    )


def segment_corpus(docs: Iterable[SourceDocument], min_words: int = 150,
                   max_words: int = 300) -> list[Segment]:
    out: list[Segment] = []
    for doc in docs:
        out.extend(segment_document(doc, min_words, max_words))
    return out
