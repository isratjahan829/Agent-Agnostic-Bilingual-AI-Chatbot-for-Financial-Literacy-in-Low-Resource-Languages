"""Domain-balanced train/validation/test splits (paper Table 3)."""
from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Sequence

from ..config import DOMAINS
from ..utils import normalize_text
from .schema import QAPair


def split_pairs(
    pairs: Sequence[QAPair],
    train_size: int = 7412,
    val_size: int = 1000,
    test_size: int = 2000,
    seed: int = 42,
) -> dict[str, list[QAPair]]:
    """Split proportionally within each domain so every split keeps the domain mix.

    Splitting is done per *segment*, not per QA pair: two questions generated
    from the same passage must never straddle the train/test boundary, or the
    test set leaks the training context.
    """
    total = len(pairs)
    requested = train_size + val_size + test_size
    if requested > total:
        scale = total / requested
        train_size = int(train_size * scale)
        val_size = int(val_size * scale)
        test_size = total - train_size - val_size

    by_domain: dict[str, list[QAPair]] = defaultdict(list)
    for pair in pairs:
        by_domain[pair.domain].append(pair)

    rng = random.Random(seed)
    out: dict[str, list[QAPair]] = {"train": [], "validation": [], "test": []}
    for domain in DOMAINS:
        domain_pairs = by_domain.get(domain, [])
        if not domain_pairs:
            continue
        share = len(domain_pairs) / total
        quotas = {
            "train": round(train_size * share),
            "validation": round(val_size * share),
            "test": round(test_size * share),
        }
        for split, chunk in _split_by_segment(domain_pairs, quotas, rng).items():
            for pair in chunk:
                pair.split = split
            out[split].extend(chunk)
    return out


def _split_by_segment(
    pairs: list[QAPair], quotas: dict[str, int], rng: random.Random
) -> dict[str, list[QAPair]]:
    groups: dict[str, list[QAPair]] = defaultdict(list)
    for pair in pairs:
        groups[pair.segment_id].append(pair)
    segment_ids = sorted(groups)
    rng.shuffle(segment_ids)

    out: dict[str, list[QAPair]] = {"train": [], "validation": [], "test": []}
    order: list[tuple[str, int]] = [
        ("test", quotas["test"]), ("validation", quotas["validation"]),
        ("train", quotas["train"]),
    ]
    cursor = 0
    for split, quota in order:
        while cursor < len(segment_ids) and len(out[split]) < quota:
            out[split].extend(groups[segment_ids[cursor]])
            cursor += 1
    for remaining in segment_ids[cursor:]:  # leftovers go to train
        out["train"].extend(groups[remaining])
    return out


def describe(splits: dict[str, list[QAPair]]) -> list[dict[str, object]]:
    """Table-3-shaped summary: rows are domains, columns are splits."""
    rows: list[dict[str, object]] = []
    for domain in DOMAINS:
        row: dict[str, object] = {"domain": domain}
        for split, pairs in splits.items():
            row[split] = sum(1 for p in pairs if p.domain == domain)
        row["total"] = sum(int(row[s]) for s in splits)
        rows.append(row)
    totals: dict[str, object] = {"domain": "total"}
    for split in splits:
        totals[split] = sum(int(r[split]) for r in rows)
    totals["total"] = sum(int(r["total"]) for r in rows)
    rows.append(totals)
    return rows


def random_row_split(
    pairs: Sequence[QAPair],
    train_size: int = 7412,
    val_size: int = 1000,
    test_size: int = 2000,
    seed: int = 42,
) -> dict[str, list[QAPair]]:
    """Naive row-level shuffle split, kept for comparison only.

    This is what "randomly split the QA pairs" usually means, and it leaks: the
    same passage generates several questions, so a test question's own source
    passage — and often a near-paraphrase of the question itself — ends up in
    training. `split_pairs` groups by passage instead. Use this function to
    quantify the inflation, never to produce a reported result.
    """
    shuffled = list(pairs)
    random.Random(seed).shuffle(shuffled)
    total = len(shuffled)
    requested = train_size + val_size + test_size
    if requested > total:
        scale = total / requested
        train_size = int(train_size * scale)
        val_size = int(val_size * scale)
        test_size = total - train_size - val_size

    out = {
        "train": shuffled[:train_size],
        "validation": shuffled[train_size:train_size + val_size],
        "test": shuffled[train_size + val_size:train_size + val_size + test_size],
    }
    for name, group in out.items():
        for pair in group:
            pair.split = name
    return out


def leakage_report(splits: dict[str, list[QAPair]]) -> dict[str, object]:
    """How much of the test split shares a source passage with training.

    Also counts test questions whose exact text appears in training, which is
    the most direct form of the leak.
    """
    train_passages = {p.segment_id for p in splits.get("train", [])}
    train_questions = {normalize_text(p.question) for p in splits.get("train", [])}
    test = splits.get("test", [])
    if not test:
        return {"test_size": 0}

    shared_passage = sum(1 for p in test if p.segment_id in train_passages)
    duplicate_question = sum(1 for p in test if normalize_text(p.question) in train_questions)
    return {
        "test_size": len(test),
        "test_items_sharing_a_training_passage": shared_passage,
        "test_items_sharing_a_training_passage_pct": round(100 * shared_passage / len(test), 1),
        "test_questions_verbatim_in_training": duplicate_question,
        "test_questions_verbatim_in_training_pct": round(
            100 * duplicate_question / len(test), 1),
    }
