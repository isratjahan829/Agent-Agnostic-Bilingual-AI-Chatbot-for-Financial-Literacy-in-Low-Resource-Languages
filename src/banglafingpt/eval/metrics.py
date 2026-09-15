"""Automatic QA metrics with Bangla-aware normalisation (paper Sec. 4.1).

All metrics run on the normalisation in ``utils.normalize_text``: NFKC, Bengali
digits folded to ASCII, danda/punctuation stripped. Without it, exact match is
dominated by orthographic noise rather than answer correctness.
"""
from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

from ..utils import ngrams, normalize_text, tokenize


def exact_match(prediction: str, reference: str) -> float:
    return float(normalize_text(prediction) == normalize_text(reference))


def f1_score(prediction: str, reference: str) -> float:
    """Token-level F1 (SQuAD-style) over normalised tokens."""
    pred, ref = tokenize(prediction), tokenize(reference)
    if not pred or not ref:
        return float(pred == ref)
    common = Counter(pred) & Counter(ref)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(pred), overlap / len(ref)
    return 2 * precision * recall / (precision + recall)


def bleu4(prediction: str, reference: str, max_n: int = 4) -> float:
    """Sentence BLEU-4 with add-1 smoothing for higher-order n-grams."""
    pred, ref = tokenize(prediction), tokenize(reference)
    if not pred or not ref:
        return 0.0
    log_sum = 0.0
    for n in range(1, max_n + 1):
        pred_grams = Counter(ngrams(pred, n))
        ref_grams = Counter(ngrams(ref, n))
        matches = sum((pred_grams & ref_grams).values())
        total = max(1, len(pred) - n + 1)
        precision = (matches + (1 if n > 1 else 0)) / (total + (1 if n > 1 else 0))
        if precision <= 0:
            return 0.0
        log_sum += math.log(precision) / max_n
    brevity = 1.0 if len(pred) > len(ref) else math.exp(1 - len(ref) / max(1, len(pred)))
    return brevity * math.exp(log_sum)


def _lcs(a: Sequence[str], b: Sequence[str]) -> int:
    """Longest common subsequence length, O(len(a) * len(b)) time, O(len(b)) space."""
    previous = [0] * (len(b) + 1)
    for token_a in a:
        current = [0]
        for j, token_b in enumerate(b):
            match = previous[j] + 1 if token_a == token_b else max(previous[j + 1], current[j])
            current.append(match)
        previous = current
    return previous[-1]


def rouge_l(prediction: str, reference: str, beta: float = 1.2) -> float:
    pred, ref = tokenize(prediction), tokenize(reference)
    if not pred or not ref:
        return 0.0
    lcs = _lcs(pred, ref)
    if lcs == 0:
        return 0.0
    precision, recall = lcs / len(pred), lcs / len(ref)
    return ((1 + beta**2) * precision * recall) / (recall + beta**2 * precision)


def meteor(prediction: str, reference: str, alpha: float = 0.9,
           beta: float = 3.0, gamma: float = 0.5) -> float:
    """METEOR with exact + stem-prefix matching (no Bangla WordNet exists)."""
    pred, ref = tokenize(prediction), tokenize(reference)
    if not pred or not ref:
        return 0.0

    matches: list[tuple[int, int]] = []
    used_ref: set[int] = set()
    for i, token in enumerate(pred):                       # stage 1: exact
        for j, ref_token in enumerate(ref):
            if j not in used_ref and token == ref_token:
                matches.append((i, j))
                used_ref.add(j)
                break
    matched_pred = {i for i, _ in matches}
    for i, token in enumerate(pred):                       # stage 2: prefix/stem
        if i in matched_pred or len(token) < 4:
            continue
        for j, ref_token in enumerate(ref):
            if j in used_ref or len(ref_token) < 4:
                continue
            if token[:4] == ref_token[:4]:
                matches.append((i, j))
                used_ref.add(j)
                break
    m = len(matches)
    if m == 0:
        return 0.0

    precision, recall = m / len(pred), m / len(ref)
    fmean = precision * recall / (alpha * precision + (1 - alpha) * recall)

    matches.sort()
    chunks = 1
    for k in range(1, len(matches)):
        if not (matches[k][0] == matches[k - 1][0] + 1 and matches[k][1] == matches[k - 1][1] + 1):
            chunks += 1
    penalty = gamma * (chunks / m) ** beta
    return fmean * (1 - penalty)


METRIC_FUNCTIONS = {
    "em": exact_match, "f1": f1_score, "bleu4": bleu4,
    "rouge_l": rouge_l, "meteor": meteor,
}


def score_pair(prediction: str, reference: str) -> dict[str, float]:
    return {name: fn(prediction, reference) for name, fn in METRIC_FUNCTIONS.items()}


def aggregate_metrics(predictions: Sequence[str], references: Sequence[str]) -> dict[str, float]:
    """Corpus-level means; ``em`` is reported as a percentage like the paper."""
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have equal length")
    if not predictions:
        return {name: 0.0 for name in METRIC_FUNCTIONS} | {"n": 0}
    totals = {name: 0.0 for name in METRIC_FUNCTIONS}
    for prediction, reference in zip(predictions, references, strict=True):
        for name, value in score_pair(prediction, reference).items():
            totals[name] += value
    n = len(predictions)
    out = {name: round(total / n, 4) for name, total in totals.items()}
    out["em_pct"] = round(100 * out["em"], 2)
    out["n"] = n
    return out
