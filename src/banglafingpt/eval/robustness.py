"""Query perturbations that probe real-world robustness (reviewer concern R1-C3).

Benchmark questions are clean and template-shaped. Real users type differently:
digits in the other script, no question mark, a typo, extra politeness, or only
the keywords. These transforms are deterministic and reversible in meaning, so a
large score drop under them is a robustness failure rather than a labelling one.
"""
from __future__ import annotations

import random
import re
from collections.abc import Callable, Sequence
from typing import Any

from ..utils import BENGALI_DIGITS, content_tokens, detect_language

POLITENESS = {"bn": ["দয়া করে বলুন, ", "একটু জানাবেন, ", "ভাই, "],
              "en": ["please tell me, ", "hi, quick question: ", "can you tell me "]}


def strip_question_mark(question: str) -> str:
    return question.rstrip(" ?？।").strip()


def to_bengali_digits(question: str) -> str:
    return re.sub(r"\d", lambda m: BENGALI_DIGITS[int(m.group())], question)


def to_ascii_digits(question: str) -> str:
    table = {ord(b): str(i) for i, b in enumerate(BENGALI_DIGITS)}
    return question.translate(table)


def keywords_only(question: str) -> str:
    """What a user types into a search box rather than a full sentence."""
    tokens = content_tokens(question)
    return " ".join(tokens[:6]) if tokens else question


def add_politeness(question: str, rng: random.Random) -> str:
    prefixes = POLITENESS.get(detect_language(question), POLITENESS["en"])
    return rng.choice(prefixes) + question


def introduce_typo(question: str, rng: random.Random) -> str:
    """Drop one character from a long word — the commonest real typing slip."""
    words = question.split()
    candidates = [i for i, w in enumerate(words) if len(w) > 4]
    if not candidates:
        return question
    i = rng.choice(candidates)
    word = words[i]
    cut = rng.randrange(1, len(word) - 1)
    words[i] = word[:cut] + word[cut + 1:]
    return " ".join(words)


PERTURBATIONS: dict[str, Callable[[str, random.Random], str]] = {
    "original": lambda q, rng: q,
    "no_question_mark": lambda q, rng: strip_question_mark(q),
    "digit_script_swapped": lambda q, rng: to_ascii_digits(q) if any(
        d in q for d in BENGALI_DIGITS) else to_bengali_digits(q),
    "keywords_only": lambda q, rng: keywords_only(q),
    "with_politeness": add_politeness,
    "single_typo": introduce_typo,
}


def perturb(question: str, kind: str, seed: int = 42) -> str:
    if kind not in PERTURBATIONS:
        raise ValueError(f"Unknown perturbation {kind!r}; known: {sorted(PERTURBATIONS)}")
    return PERTURBATIONS[kind](question, random.Random(seed))


def retrieval_robustness(
    retriever: Any,
    pairs: Sequence[Any],
    top_k: int = 5,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Passage-level recall@k under each perturbation, relative to the original."""
    rows: list[dict[str, Any]] = []
    baseline = None
    for kind in PERTURBATIONS:
        hits = 0
        for i, pair in enumerate(pairs):
            query = perturb(pair.question, kind, seed + i)
            docs = {c.doc_id for c in retriever.retrieve(query, top_k=top_k)}
            hits += pair.source in docs
        recall = hits / max(1, len(pairs))
        if baseline is None:
            baseline = recall
        rows.append({
            "perturbation": kind,
            f"recall@{top_k}": round(recall, 3),
            "delta": round(recall - baseline, 3),
            "relative_drop_pct": (round(100 * (baseline - recall) / baseline, 1)
                                  if baseline else 0.0),
        })
    return rows
