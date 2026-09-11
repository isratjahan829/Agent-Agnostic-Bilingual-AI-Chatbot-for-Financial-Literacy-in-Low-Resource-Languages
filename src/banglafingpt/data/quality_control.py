"""Near-duplicate removal and answer-grounding checks (Sec. 3.2 quality control)."""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from ..utils import content_tokens, extract_numbers, ngrams, normalize_text
from .schema import QAPair, Segment


def char_ngrams(text: str, n: int = 4) -> set[str]:
    norm = normalize_text(text).replace(" ", "")
    return {norm[i : i + n] for i in range(max(0, len(norm) - n + 1))}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def drop_near_duplicates(pairs: Sequence[QAPair],
                         threshold: float = 0.92) -> tuple[list[QAPair], int]:
    """Remove questions that are near-identical to one already kept.

    Candidates are bucketed by their first content token, which keeps the
    comparison count near-linear instead of quadratic over the whole corpus.
    """
    kept: list[QAPair] = []
    buckets: dict[str, list[set[str]]] = defaultdict(list)
    dropped = 0
    for pair in pairs:
        signature = char_ngrams(pair.question)
        tokens = content_tokens(pair.question)
        keys = {tokens[0] if tokens else "", tokens[-1] if tokens else ""}
        if any(jaccard(signature, other) >= threshold
               for key in keys for other in buckets[key]):
            dropped += 1
            continue
        kept.append(pair)
        for key in keys:
            buckets[key].append(signature)
    return kept, dropped


def answer_is_grounded(pair: QAPair, segment_text: str) -> bool:
    """True when the answer text and all of its numbers occur in the segment."""
    haystack = normalize_text(segment_text)
    if normalize_text(pair.answer) not in haystack:
        return False
    source_numbers = set(extract_numbers(segment_text))
    return all(num in source_numbers for num in extract_numbers(pair.answer))


def filter_grounded(pairs: Sequence[QAPair],
                    segments: Sequence[Segment]) -> tuple[list[QAPair], int]:
    index = {s.segment_id: s.text for s in segments}
    kept = [p for p in pairs if answer_is_grounded(p, index.get(p.segment_id, ""))]
    return kept, len(pairs) - len(kept)


def corpus_statistics(pairs: Sequence[QAPair]) -> dict[str, object]:
    """Table-2-shaped statistics: counts and average question/answer lengths."""
    by_domain: dict[str, list[QAPair]] = defaultdict(list)
    for pair in pairs:
        by_domain[pair.domain].append(pair)
    stats: dict[str, object] = {}
    for domain, group in by_domain.items():
        stats[domain] = {
            "qa_pairs": len(group),
            "pct_of_dataset": round(100 * len(group) / max(1, len(pairs)), 2),
            "avg_q_words": round(sum(len(p.question.split()) for p in group) / len(group), 1),
            "avg_a_words": round(sum(len(p.answer.split()) for p in group) / len(group), 1),
            "sources": len({p.source for p in group}),
        }
    stats["total"] = {"qa_pairs": len(pairs), "sources": len({p.source for p in pairs})}
    return stats


def repetition_ratio(text: str, n: int = 3) -> float:
    """Fraction of repeated n-grams; a cheap degenerate-generation detector."""
    grams = ngrams(content_tokens(text), n)
    if not grams:
        return 0.0
    return 1.0 - len(set(grams)) / len(grams)
