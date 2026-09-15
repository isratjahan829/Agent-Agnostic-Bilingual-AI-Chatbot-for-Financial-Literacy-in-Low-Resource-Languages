"""Record types shared by the dataset pipeline.

Every QA pair keeps a pointer back to the document segment it came from
(``segment_id`` + ``source``), which is what makes the corpus auditable: an
annotator can always re-read the regulation the answer was taken from.
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any

from ..config import DOMAINS
from ..utils import detect_language, word_count


def stable_id(*parts: str) -> str:
    digest = hashlib.sha1("||".join(parts).encode("utf-8")).hexdigest()
    return digest[:16]


@dataclass
class SourceDocument:
    doc_id: str
    title: str
    domain: str
    year: int | None = None
    url: str | None = None
    text: str = ""

    def __post_init__(self) -> None:
        if self.domain not in DOMAINS:
            raise ValueError(f"domain must be one of {DOMAINS}, got {self.domain!r}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Segment:
    """A 150-300 word coherent unit of a regulatory document."""

    segment_id: str
    doc_id: str
    domain: str
    text: str
    section: str | None = None
    order: int = 0

    @property
    def n_words(self) -> int:
        return word_count(self.text)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class QAPair:
    qa_id: str
    question: str
    answer: str
    domain: str
    segment_id: str
    source: str
    language: str = ""
    generator: str = "template"  # template | llm | manual
    verified: bool = False
    split: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.domain not in DOMAINS:
            raise ValueError(f"domain must be one of {DOMAINS}, got {self.domain!r}")
        if not self.language:
            self.language = detect_language(self.question)

    @classmethod
    def create(cls, question: str, answer: str, domain: str, segment_id: str,
               source: str, **kw: Any) -> QAPair:
        return cls(
            qa_id=stable_id(question, answer, segment_id),
            question=question.strip(),
            answer=answer.strip(),
            domain=domain,
            segment_id=segment_id,
            source=source,
            **kw,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> QAPair:
        known = {k: v for k, v in row.items() if k in cls.__dataclass_fields__}
        return cls(**known)


def validate_pairs(pairs: list[QAPair]) -> list[str]:
    """Return human-readable problems; an empty list means the corpus is clean."""
    problems: list[str] = []
    seen: dict[str, str] = {}
    for pair in pairs:
        if not pair.question:
            problems.append(f"{pair.qa_id}: empty question")
        if not pair.answer:
            problems.append(f"{pair.qa_id}: empty answer")
        if not pair.segment_id:
            problems.append(f"{pair.qa_id}: missing segment provenance")
        if pair.qa_id in seen:
            problems.append(f"{pair.qa_id}: duplicate id (first seen for {seen[pair.qa_id]!r})")
        seen[pair.qa_id] = pair.question
    return problems
