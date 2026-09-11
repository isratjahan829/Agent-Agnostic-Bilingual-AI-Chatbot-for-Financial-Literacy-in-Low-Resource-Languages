"""Hallucination annotation protocol and scoring (reviewer concern R1-C1b).

The paper reports a hallucination rate from three annotators but does not state
the sampling frame, the instructions, or the agreement. This module makes all
three explicit and reproducible: it draws a stratified sample from the *test
split only*, writes one blind sheet per annotator, and scores the returned
sheets into the three categories plus Fleiss' kappa.
"""
from __future__ import annotations

import csv
import random
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .human_eval import fleiss_kappa, interpret_kappa

CATEGORIES = ["fully_grounded", "partially_grounded", "hallucinated"]

INSTRUCTIONS = """\
Hallucination annotation — BanglaFinGPT

You will see a question, the system's answer, and the source passages the system
retrieved for it. Judge ONLY whether the answer is supported by those passages.
Do not judge style, completeness, or whether you personally know the rule.

Label each answer with exactly one category:

  fully_grounded      Every claim in the answer, including every number, rate,
                      date and section reference, appears in the passages.
  partially_grounded  The answer is mostly supported, but at least one claim or
                      number is not stated in the passages.
  hallucinated        The answer asserts something the passages do not support,
                      or invents a rate, date, section number or HS code.

A refusal ("the documents do not contain a specific answer ...") is
fully_grounded: it asserts nothing.

Work independently. Do not discuss items with the other annotators until all
sheets are returned.
"""


def build_annotation_sheets(
    records: Sequence[dict[str, Any]],
    out_dir: str | Path,
    annotators: Sequence[str] = ("A1", "A2", "A3"),
    sample_size: int = 200,
    stratify_by: str = "domain",
    seed: int = 42,
) -> dict[str, Any]:
    """Draw a stratified sample and write one blind CSV per annotator.

    The same items go to every annotator (so agreement is computable) but in a
    different order (so position effects do not correlate between sheets), and
    the system's own grounding verdict is never shown.
    """
    rng = random.Random(seed)
    strata: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        strata[record.get(stratify_by, "unknown")].append(record)

    sample: list[dict[str, Any]] = []
    total = sum(len(v) for v in strata.values())
    for key in sorted(strata):
        group = strata[key]
        quota = round(sample_size * len(group) / max(1, total))
        sample.extend(rng.sample(group, min(quota, len(group))))
    rng.shuffle(sample)
    sample = sample[:sample_size]

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "INSTRUCTIONS.txt").write_text(INSTRUCTIONS, encoding="utf-8")

    for annotator in annotators:
        order = list(sample)
        random.Random(f"{seed}:{annotator}").shuffle(order)
        path = out / f"annotations_{annotator}.csv"
        with path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["item_id", "domain", "question", "answer",
                             "retrieved_passages", "label", "note"])
            for record in order:
                passages = " ||| ".join(
                    str(c.get("text", c.get("chunk_id", "")))
                    for c in record.get("retrieved", [])
                )
                writer.writerow([record["qa_id"], record.get("domain", ""),
                                 record["question"], record["answer"],
                                 passages, "", ""])

    return {
        "sample_size": len(sample),
        "annotators": list(annotators),
        "stratified_by": stratify_by,
        "strata": {str(k): sum(1 for r in sample if r.get(stratify_by) == k)
                   for k in sorted(strata)},
        "sheets": [str(out / f"annotations_{a}.csv") for a in annotators],
        "instructions": str(out / "INSTRUCTIONS.txt"),
    }


def load_annotations(paths: Sequence[str | Path]) -> dict[str, dict[str, str]]:
    """Read returned sheets into {annotator: {item_id: label}}."""
    out: dict[str, dict[str, str]] = {}
    for path in paths:
        p = Path(path)
        annotator = p.stem.replace("annotations_", "")
        labels: dict[str, str] = {}
        with p.open(encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                label = (row.get("label") or "").strip()
                if label:
                    if label not in CATEGORIES:
                        raise ValueError(f"{p}: unknown label {label!r} for {row['item_id']}")
                    labels[row["item_id"]] = label
        out[annotator] = labels
    return out


def score_annotations(annotations: dict[str, dict[str, str]]) -> dict[str, Any]:
    """Majority-vote category shares plus Fleiss' kappa over the three raters."""
    annotators = sorted(annotations)
    common = set.intersection(*(set(annotations[a]) for a in annotators)) if annotators else set()
    if not common:
        raise ValueError("No item was labelled by every annotator")

    matrix, majority = [], Counter()
    disagreements = 0
    for item in sorted(common):
        votes = Counter(annotations[a][item] for a in annotators)
        matrix.append([votes.get(category, 0) for category in CATEGORIES])
        winner, count = votes.most_common(1)[0]
        majority[winner] += 1
        if count < len(annotators):
            disagreements += 1

    kappa = fleiss_kappa(matrix)
    n = len(common)
    return {
        "n_items": n,
        "n_annotators": len(annotators),
        "fleiss_kappa": round(kappa, 3),
        "agreement": interpret_kappa(kappa),
        "unanimous_pct": round(100 * (n - disagreements) / n, 1),
        "majority_vote": {category: {"count": majority.get(category, 0),
                                     "pct": round(100 * majority.get(category, 0) / n, 1)}
                          for category in CATEGORIES},
    }


def reduction_report(before_pct: float, after_pct: float) -> dict[str, float]:
    """Both framings of the filter's effect (reviewer concern R1-C1a).

    Reporting only the relative figure overstates a change measured from a small
    base; reporting both makes the size of the effect unambiguous.
    """
    return {
        "before_pct": round(before_pct, 2),
        "after_pct": round(after_pct, 2),
        "absolute_reduction_pp": round(before_pct - after_pct, 2),
        "relative_reduction_pct": round(100 * (before_pct - after_pct) / before_pct, 1)
        if before_pct else 0.0,
        "number_needed_to_filter": round(100 / (before_pct - after_pct), 1)
        if before_pct > after_pct else float("inf"),
    }
