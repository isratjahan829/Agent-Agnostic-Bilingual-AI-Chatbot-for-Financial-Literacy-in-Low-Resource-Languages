"""Automatic error typing and hallucination audit (paper Sec. 4.5-4.6).

Every failed prediction is assigned one of the five categories the paper
reports, using signals available at inference time.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from typing import Any

from ..utils import content_tokens, extract_numbers, normalize_text

ERROR_TYPES = ["outdated", "precision", "incomplete", "ambiguous", "formatting"]

# Year mentions that indicate the retrieved rule predates the latest amendment.
YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
AMBIGUITY_CUES = ("এই", "উক্ত", "উপরোক্ত", "সেক্ষেত্রে", "it", "this", "such", "above")


def classify_error(record: dict[str, Any], latest_year: int = 2025) -> str:
    """Heuristic error type for a wrong prediction.

    Order matters: a numeric mismatch is reported as ``precision`` even if the
    answer is also short, because the wrong rate is the actionable defect.
    """
    prediction = str(record.get("answer", ""))
    reference = str(record.get("reference", ""))
    retrieved = " ".join(
        str(c.get("text", "")) for c in record.get("retrieved", []) if isinstance(c, dict)
    )

    ref_numbers, pred_numbers = extract_numbers(reference), extract_numbers(prediction)
    if ref_numbers and pred_numbers != ref_numbers:
        pred_years = {int(y) for y in YEAR_RE.findall(prediction)}
        ref_years = {int(y) for y in YEAR_RE.findall(reference)}
        if pred_years and ref_years and max(pred_years) < max(ref_years):
            return "outdated"
        return "precision"

    ref_tokens, pred_tokens = set(content_tokens(reference)), set(content_tokens(prediction))
    if pred_tokens and ref_tokens:
        coverage = len(pred_tokens & ref_tokens) / len(ref_tokens)
        if coverage < 0.5:
            # Content is present in sources but missing from the answer -> the
            # model failed to synthesise across chunks.
            retrieved_tokens = set(content_tokens(retrieved))
            if retrieved and len(ref_tokens & retrieved_tokens) / len(ref_tokens) > 0.6:
                return "incomplete"
    question = str(record.get("question", ""))
    if any(cue in question for cue in AMBIGUITY_CUES) and len(question.split()) <= 8:
        return "ambiguous"
    if normalize_text(prediction) == normalize_text(reference):
        return "formatting"  # differs only in surface form
    return "incomplete"


def error_report(records: Sequence[dict[str, Any]], em_field: str = "em") -> dict[str, Any]:
    """Tables 9, 10 and 14: counts by type, and the domain x type heatmap."""
    failures = [r for r in records if float(r.get("scores", {}).get(em_field, 0.0)) < 1.0]
    typed = [(r.get("domain", "unknown"), classify_error(r)) for r in failures]

    overall = Counter(kind for _, kind in typed)
    total = sum(overall.values())
    by_domain: dict[str, Counter] = defaultdict(Counter)
    for domain, kind in typed:
        by_domain[domain][kind] += 1

    return {
        "total_errors": total,
        "by_type": [
            {
                "error_type": kind,
                "count": overall.get(kind, 0),
                "pct": round(100 * overall.get(kind, 0) / max(1, total), 1),
            }
            for kind in ERROR_TYPES
        ],
        "by_domain": [
            {
                "domain": domain,
                "total": sum(counts.values()),
                **{
                    kind: {
                        "count": counts.get(kind, 0),
                        "pct": round(100 * counts.get(kind, 0) / max(1, sum(counts.values())), 1),
                    }
                    for kind in ERROR_TYPES
                },
            }
            for domain, counts in sorted(by_domain.items())
        ],
    }


def hallucination_report(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Table 11: fully grounded / partially grounded / hallucinated shares.

    Grounding is read off the filter verdict, so the same thresholds that gate
    generation also define the audit categories. An empty answer is reported
    separately: it states nothing, so calling it a hallucination would overstate
    the fabrication rate of a system that simply produced no output.
    """
    fully = partial = hallucinated = empty = 0
    for record in records:
        grounding = record.get("grounding") or {}
        if not str(record.get("raw_answer", record.get("answer", ""))).strip():
            empty += 1
            continue
        overlap = float(grounding.get("keyword_overlap", 0.0))
        numeric = float(grounding.get("numeric_support", 1.0))
        if not record.get("answered", True):
            fully += 1  # a refusal states nothing, so it fabricates nothing
        elif grounding.get("grounded") and numeric >= 1.0:
            fully += 1
        elif overlap >= 0.4 and numeric >= 0.5:
            partial += 1
        else:
            hallucinated += 1
    n = max(1, len(records))
    return {
        "n": len(records),
        "fully_grounded_pct": round(100 * fully / n, 1),
        "partially_grounded_pct": round(100 * partial / n, 1),
        "hallucinated_pct": round(100 * hallucinated / n, 1),
        "empty_pct": round(100 * empty / n, 1),
    }


def compare_hallucination(variants: dict[str, Sequence[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows = [{"variant": name, **hallucination_report(records)}
            for name, records in variants.items()]
    baseline = next(
        (r for r in rows if "no_filter" in r["variant"] or "ft_rag" in r["variant"]), None
    )
    full = next((r for r in rows if "banglafingpt" in r["variant"]), None)
    if baseline and full and baseline["hallucinated_pct"]:
        reduction = 100 * (1 - full["hallucinated_pct"] / baseline["hallucinated_pct"])
        for row in rows:
            row["relative_reduction_pct"] = round(reduction, 1) if row is full else None
    return rows
