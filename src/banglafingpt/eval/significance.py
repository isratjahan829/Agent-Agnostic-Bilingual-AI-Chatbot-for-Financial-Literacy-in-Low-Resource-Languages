"""Bootstrap CIs and paired significance tests (paper Table 12).

The paper reports 95% CIs and p-values against the base model; these are the
exact procedures that produce them from per-example scores.
"""
from __future__ import annotations

import math
import random
from collections.abc import Sequence


def bootstrap_ci(values: Sequence[float], n_resamples: int = 10000,
                 confidence: float = 0.95, seed: int = 42) -> tuple[float, float, float]:
    """Percentile bootstrap CI of the mean. Returns (mean, low, high)."""
    if not values:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    n = len(values)
    mean = sum(values) / n
    means: list[float] = []
    for _ in range(n_resamples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int((1 - confidence) / 2 * n_resamples)]
    hi = means[min(n_resamples - 1, int((1 + confidence) / 2 * n_resamples))]
    return mean, lo, hi


def mcnemar_test(baseline: Sequence[float], system: Sequence[float]) -> dict[str, float]:
    """Exact-match is binary and paired, so McNemar is the right test.

    Uses the binomial exact test on discordant pairs; falls back to the
    chi-square form with continuity correction when there are many.
    """
    if len(baseline) != len(system):
        raise ValueError("paired test needs equal-length score vectors")
    b = sum(1 for x, y in zip(baseline, system, strict=True) if x > 0.5 >= y)   # baseline only
    c = sum(1 for x, y in zip(baseline, system, strict=True) if y > 0.5 >= x)   # system only
    n = b + c
    if n == 0:
        return {"b": 0, "c": 0, "statistic": 0.0, "p_value": 1.0}
    if n < 25:
        tail = sum(math.comb(n, k) for k in range(min(b, c) + 1)) / (2**n)
        return {"b": b, "c": c, "statistic": float(min(b, c)), "p_value": min(1.0, 2 * tail)}
    stat = (abs(b - c) - 1) ** 2 / n
    return {"b": b, "c": c, "statistic": stat, "p_value": _chi2_sf_1df(stat)}


def paired_bootstrap_test(baseline: Sequence[float], system: Sequence[float],
                          n_resamples: int = 10000, seed: int = 42) -> dict[str, float]:
    """Two-sided paired bootstrap for continuous metrics (F1, BLEU, ROUGE)."""
    if len(baseline) != len(system):
        raise ValueError("paired test needs equal-length score vectors")
    diffs = [s - b for b, s in zip(baseline, system, strict=True)]
    observed = sum(diffs) / len(diffs)
    rng = random.Random(seed)
    centered = [d - observed for d in diffs]
    n = len(diffs)
    extreme = 0
    for _ in range(n_resamples):
        sample = sum(centered[rng.randrange(n)] for _ in range(n)) / n
        if abs(sample) >= abs(observed):
            extreme += 1
    return {
        "delta": observed,
        "p_value": (extreme + 1) / (n_resamples + 1),
        "n": n,
    }


def _chi2_sf_1df(x: float) -> float:
    """Survival function of chi-square with 1 df: erfc(sqrt(x/2))."""
    return math.erfc(math.sqrt(max(0.0, x) / 2))


def significance_table(
    scores_by_system: dict[str, Sequence[float]],
    baseline: str,
    metric: str = "em",
    seed: int = 42,
) -> list[dict[str, object]]:
    """Table-12-shaped rows: mean, 95% CI and p-value vs the baseline system."""
    if baseline not in scores_by_system:
        raise KeyError(f"baseline {baseline!r} not among {sorted(scores_by_system)}")
    base_scores = scores_by_system[baseline]
    rows: list[dict[str, object]] = []
    for name, values in scores_by_system.items():
        mean, lo, hi = bootstrap_ci(values, seed=seed)
        row: dict[str, object] = {
            "system": name,
            "metric": metric,
            "mean": round(100 * mean, 2) if metric == "em" else round(mean, 4),
            "ci95": [round(100 * lo, 2), round(100 * hi, 2)] if metric == "em"
                    else [round(lo, 4), round(hi, 4)],
        }
        if name == baseline:
            row["p_value"] = None
        else:
            test = (mcnemar_test(base_scores, values) if metric == "em"
                    else paired_bootstrap_test(base_scores, values, seed=seed))
            row["p_value"] = test["p_value"]
            row["significant_at_0.05"] = test["p_value"] < 0.05
        rows.append(row)
    return rows
