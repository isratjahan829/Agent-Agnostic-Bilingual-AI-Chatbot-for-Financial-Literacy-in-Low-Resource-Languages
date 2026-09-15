"""Post-generation grounding checks.

Stage 1 (system prompt) lives in ``models/prompts.py``. This module implements
stages 2 and 3: an answer is released only when

    cos(emb(a), emb(d*)) >= 0.70   AND   keyword_overlap(a, D_k) >= 0.70

and every numeric literal it contains occurs in the retrieved context. Anything
else is replaced by the fallback message (Eq. 19) — in a regulatory domain a
refusal is cheaper than a confident wrong rate.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

from ..config import HallucinationConfig
from ..retrieval.embedder import Embedder, HashingEmbedder
from ..utils import content_tokens, extract_numbers, normalize_text


@dataclass
class GroundingVerdict:
    grounded: bool
    cosine_similarity: float
    keyword_overlap: float
    numeric_support: float
    question_similarity: float = 1.0
    reason: str = ""
    best_chunk_id: str | None = None
    unsupported_numbers: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "grounded": self.grounded,
            "cosine_similarity": round(self.cosine_similarity, 4),
            "keyword_overlap": round(self.keyword_overlap, 4),
            "numeric_support": round(self.numeric_support, 4),
            "question_similarity": round(self.question_similarity, 4),
            "reason": self.reason,
            "best_chunk_id": self.best_chunk_id,
            "unsupported_numbers": self.unsupported_numbers,
        }


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def keyword_overlap(answer: str, contexts: Sequence[str]) -> float:
    """Share of the answer's content words that appear in the retrieved text.

    Directional on purpose: we ask whether the *answer* is covered by the
    sources, not whether the sources are covered by the answer.
    """
    answer_tokens = set(content_tokens(answer))
    if not answer_tokens:
        return 0.0
    context_tokens: set[str] = set()
    for context in contexts:
        context_tokens.update(content_tokens(context))
    return len(answer_tokens & context_tokens) / len(answer_tokens)


def numeric_support(answer: str, contexts: Sequence[str]) -> tuple[float, list[str]]:
    """Fraction of numbers in the answer that appear verbatim in the context.

    Fabricated rates and section numbers are the most damaging failure mode in
    this domain, and they are cheap to catch exactly.
    """
    numbers = extract_numbers(answer)
    if not numbers:
        return 1.0, []
    supported: set[str] = set()
    for context in contexts:
        supported.update(extract_numbers(context))
    missing = [n for n in numbers if n not in supported]
    return (len(numbers) - len(missing)) / len(numbers), missing


class HallucinationFilter:
    """Verifies a generated answer against the chunks that were retrieved for it."""

    def __init__(self, config: HallucinationConfig | None = None,
                 embedder: Embedder | None = None) -> None:
        self.config = config or HallucinationConfig()
        self.embedder = embedder or HashingEmbedder()

    def verify(self, answer: str, contexts: Sequence[str],
               chunk_ids: Sequence[str] | None = None,
               question: str | None = None) -> GroundingVerdict:
        if not normalize_text(answer):
            return GroundingVerdict(False, 0.0, 0.0, 0.0, reason="empty_answer")
        if not contexts:
            return GroundingVerdict(False, 0.0, 0.0, 0.0, reason="no_context_retrieved")

        payload = [answer, *contexts] + ([question] if question else [])
        vectors = self.embedder.encode(payload)
        answer_vec = vectors[0]
        context_vecs = vectors[1 : 1 + len(contexts)]
        sims = [cosine(answer_vec, v) for v in context_vecs]
        best_idx = max(range(len(sims)), key=sims.__getitem__)
        best_sim = sims[best_idx]

        # Grounding says the answer is *supported*; it does not say the sources are
        # *relevant*. A question no retrieved passage speaks to must be refused too.
        question_similarity = 1.0
        if question:
            question_similarity = max(cosine(vectors[-1], v) for v in context_vecs)

        overlap = keyword_overlap(answer, contexts)
        support, missing = numeric_support(answer, contexts)
        best_chunk_id = chunk_ids[best_idx] if chunk_ids and best_idx < len(chunk_ids) else None

        reasons: list[str] = []
        if best_sim < self.config.min_cosine_similarity:
            reasons.append(f"low_similarity({best_sim:.2f}<{self.config.min_cosine_similarity})")
        if overlap < self.config.min_keyword_overlap:
            reasons.append(f"low_keyword_overlap({overlap:.2f}<{self.config.min_keyword_overlap})")
        if support < self.config.min_numeric_support:
            reasons.append(f"unsupported_numbers({','.join(missing)})")
        if question and question_similarity < self.config.min_question_similarity:
            reasons.append(
                f"irrelevant_context({question_similarity:.2f}"
                f"<{self.config.min_question_similarity})"
            )

        return GroundingVerdict(
            grounded=not reasons,
            cosine_similarity=best_sim,
            keyword_overlap=overlap,
            numeric_support=support,
            question_similarity=question_similarity,
            reason=";".join(reasons) or "grounded",
            best_chunk_id=best_chunk_id,
            unsupported_numbers=missing,
        )
