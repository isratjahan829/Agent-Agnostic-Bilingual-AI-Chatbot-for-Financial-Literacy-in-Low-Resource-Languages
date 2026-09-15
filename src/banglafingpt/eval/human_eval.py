"""Human evaluation scoring and inter-rater reliability (paper Sec. 4.2)."""
from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Sequence

DIMENSIONS = ["relevance", "coherence", "fluency", "accuracy", "creativity"]
SCALE = (1, 2, 3, 4, 5)


def mean_std(values: Sequence[float]) -> tuple[float, float]:
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    mean = sum(values) / n
    if n == 1:
        return mean, 0.0
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)  # sample std
    return mean, math.sqrt(variance)


def summarize_ratings(ratings: Iterable[dict[str, object]]) -> dict[str, dict[str, float]]:
    """``ratings``: rows of {query_id, rater_id, <dimension>: score}. -> Table 6."""
    buckets: dict[str, list[float]] = defaultdict(list)
    for row in ratings:
        for dimension in DIMENSIONS:
            if dimension in row and row[dimension] is not None:
                buckets[dimension].append(float(row[dimension]))  # type: ignore[arg-type]
    out: dict[str, dict[str, float]] = {}
    for dimension, values in buckets.items():
        mean, std = mean_std(values)
        out[dimension] = {"mean": round(mean, 2), "std": round(std, 2), "n": len(values)}
    return out


def fleiss_kappa(matrix: Sequence[Sequence[int]]) -> float:
    """Fleiss' kappa from an items x categories count matrix.

    Each row counts how many raters assigned each category to that item; rows
    must sum to the same number of raters.
    """
    rows = [list(r) for r in matrix if sum(r) > 0]
    if not rows:
        return 0.0
    n_raters = sum(rows[0])
    if any(sum(r) != n_raters for r in rows):
        raise ValueError("every item must be rated by the same number of raters")
    if n_raters < 2:
        raise ValueError("Fleiss' kappa needs at least two raters per item")
    n_items = len(rows)
    n_categories = len(rows[0])

    p_j = [sum(row[j] for row in rows) / (n_items * n_raters) for j in range(n_categories)]
    p_i = [
        (sum(count * count for count in row) - n_raters) / (n_raters * (n_raters - 1))
        for row in rows
    ]
    p_bar = sum(p_i) / n_items
    p_e = sum(p * p for p in p_j)
    if abs(1 - p_e) < 1e-12:
        return 1.0
    return (p_bar - p_e) / (1 - p_e)


def kappa_by_dimension(ratings: Sequence[dict[str, object]]) -> dict[str, float]:
    """Fleiss' kappa per dimension, computed over the 1-5 rating categories."""
    out: dict[str, float] = {}
    for dimension in DIMENSIONS:
        per_item: dict[object, list[int]] = defaultdict(lambda: [0] * len(SCALE))
        for row in ratings:
            score = row.get(dimension)
            if score is None:
                continue
            index = int(score) - SCALE[0]
            if 0 <= index < len(SCALE):
                per_item[row["query_id"]][index] += 1
        matrix = [counts for counts in per_item.values()]
        if not matrix:
            continue
        n_raters = max(sum(c) for c in matrix)
        matrix = [c for c in matrix if sum(c) == n_raters]
        try:
            out[dimension] = round(fleiss_kappa(matrix), 3)
        except ValueError:
            continue
    return out


def interpret_kappa(kappa: float) -> str:
    """Landis & Koch bands, as cited in the paper."""
    for threshold, label in ((0.0, "poor"), (0.20, "slight"), (0.40, "fair"),
                             (0.60, "moderate"), (0.80, "substantial")):
        if kappa <= threshold:
            return label
    return "almost perfect"
