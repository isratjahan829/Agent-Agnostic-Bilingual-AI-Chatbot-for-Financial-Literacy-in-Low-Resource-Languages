"""Load the released BanglaFinGPT spreadsheet into the pipeline's record types.

The distributed corpus is one row per QA pair with its source passage inline:

    QuestionID | Type (Bangla/English) | Topic | Context | Question | Answer

Passages repeat across rows, so retrieval chunks are built from the *unique*
contexts and every QA pair points at the chunk it came from.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from ..utils import word_count
from .schema import QAPair, Segment, stable_id

TOPIC_TO_DOMAIN = {
    "tax": "taxation", "taxation": "taxation",
    "vat": "vat",
    "customs": "customs",
    "finance": "finance", "general finance": "finance",
}
REQUIRED_COLUMNS = ("QuestionID", "Type", "Topic", "Context", "Question", "Answer")


def normalize_topic(topic: str) -> str:
    key = str(topic).strip().lower()
    if key not in TOPIC_TO_DOMAIN:
        raise ValueError(f"Unmapped topic {topic!r}; known: {sorted(set(TOPIC_TO_DOMAIN))}")
    return TOPIC_TO_DOMAIN[key]


def chunk_text(text: str, chunk_words: int = 220, overlap_words: int = 40) -> list[str]:
    """Sliding window over a long passage, on word boundaries.

    Overlap matters here: a rate and the condition that qualifies it often sit
    on either side of an arbitrary cut, and a chunk that holds only one of them
    produces an answer the grounding filter then rejects.
    """
    if chunk_words <= overlap_words:
        raise ValueError("chunk_words must exceed overlap_words")
    words = text.split()
    if len(words) <= chunk_words:
        return [text.strip()] if text.strip() else []
    step = chunk_words - overlap_words
    chunks = []
    for start in range(0, len(words), step):
        window = words[start : start + chunk_words]
        if len(window) < overlap_words and chunks:
            break
        chunks.append(" ".join(window))
    return chunks


def load_rows(path: str | Path, sheet: str | int = 0) -> list[dict[str, Any]]:
    """Read the spreadsheet with pandas and validate its columns."""
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("pandas and openpyxl are required to read the .xlsx") from exc

    frame = pd.read_excel(path, sheet_name=sheet)
    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"Spreadsheet is missing columns: {missing}")
    frame = frame.dropna(subset=["Question", "Answer", "Context", "Topic"])
    return frame.to_dict("records")


def build_corpus(
    rows: Iterable[dict[str, Any]],
    chunk_words: int = 220,
    overlap_words: int = 40,
) -> tuple[list[QAPair], list[Segment]]:
    """Split rows into retrieval segments and provenance-linked QA pairs."""
    segments: list[Segment] = []
    context_to_segment: dict[str, str] = {}
    pairs: list[QAPair] = []

    for row in rows:
        context = str(row["Context"]).strip()
        domain = normalize_topic(row["Topic"])
        language = "bn" if str(row["Type"]).strip().lower().startswith("bangla") else "en"
        doc_id = f"{domain}_{stable_id(context)[:8]}"

        if context not in context_to_segment:
            chunks = chunk_text(context, chunk_words, overlap_words)
            for order, chunk in enumerate(chunks):
                segments.append(
                    Segment(
                        segment_id=stable_id(doc_id, str(order)),
                        doc_id=doc_id,
                        domain=domain,
                        text=chunk,
                        section=None,
                        order=order,
                    )
                )
            # A pair is attributed to the first chunk of its passage; retrieval
            # is free to return any chunk, and grouping for the split uses the
            # passage rather than the chunk.
            context_to_segment[context] = stable_id(doc_id, "0")

        pairs.append(
            QAPair.create(
                question=str(row["Question"]).strip(),
                answer=str(row["Answer"]).strip(),
                domain=domain,
                segment_id=context_to_segment[context],
                source=doc_id,
                language=language,
                generator="released",
                verified=True,
                meta={"question_id": row.get("QuestionID"), "type": row.get("Type")},
            )
        )
    return pairs, segments


def load_corpus(
    path: str | Path,
    chunk_words: int = 220,
    overlap_words: int = 40,
) -> tuple[list[QAPair], list[Segment]]:
    return build_corpus(load_rows(path), chunk_words, overlap_words)


def corpus_overview(pairs: Sequence[QAPair]) -> list[dict[str, Any]]:
    """Table-2-shaped summary computed from the loaded pairs."""
    from collections import defaultdict

    by_domain: dict[str, list[QAPair]] = defaultdict(list)
    for pair in pairs:
        by_domain[pair.domain].append(pair)

    rows: list[dict[str, Any]] = []
    for domain in sorted(by_domain):
        group = by_domain[domain]
        rows.append({
            "domain": domain,
            "qa_pairs": len(group),
            "pct": round(100 * len(group) / max(1, len(pairs)), 2),
            "bangla": sum(1 for p in group if p.language == "bn"),
            "english": sum(1 for p in group if p.language == "en"),
            "avg_q_words": round(sum(word_count(p.question) for p in group) / len(group), 1),
            "avg_a_words": round(sum(word_count(p.answer) for p in group) / len(group), 1),
            "passages": len({p.segment_id for p in group}),
        })
    rows.append({
        "domain": "total",
        "qa_pairs": len(pairs),
        "pct": 100.0,
        "bangla": sum(1 for p in pairs if p.language == "bn"),
        "english": sum(1 for p in pairs if p.language == "en"),
        "avg_q_words": round(sum(word_count(p.question) for p in pairs) / max(1, len(pairs)), 1),
        "avg_a_words": round(sum(word_count(p.answer) for p in pairs) / max(1, len(pairs)), 1),
        "passages": len({p.segment_id for p in pairs}),
    })
    return rows
