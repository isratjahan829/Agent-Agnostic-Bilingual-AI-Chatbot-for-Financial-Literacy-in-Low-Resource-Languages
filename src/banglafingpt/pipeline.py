"""End-to-end BanglaFinGPT: retrieve -> generate -> verify -> answer or refuse.

This is the object under test in every table of the paper. The ablation variants
in Table 7 are the same object with components switched off:

    use_retrieval=False, use_filter=False -> base / fine-tuned only
    use_retrieval=True,  use_filter=False -> FT + RAG
    use_retrieval=True,  use_filter=True  -> BanglaFinGPT (full)
"""
from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .agents import Agent, build_agent
from .config import Config
from .hallucination.filters import GroundingVerdict, HallucinationFilter
from .models.prompts import build_prompt, fallback_message
from .retrieval.retriever import HybridRetriever, RetrievedChunk
from .utils import content_tokens, detect_language


@dataclass
class Answer:
    question: str
    text: str
    language: str
    answered: bool                     # False => refused, fallback returned
    citations: list[str] = field(default_factory=list)
    chunks: list[RetrievedChunk] = field(default_factory=list)
    verdict: GroundingVerdict | None = None
    raw_text: str = ""
    latency_ms: float = 0.0
    backend: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.text,
            "language": self.language,
            "answered": self.answered,
            "citations": self.citations,
            "retrieved": [
                {"chunk_id": c.chunk_id, "doc_id": c.doc_id, "score": round(c.score, 4)}
                for c in self.chunks
            ],
            "grounding": self.verdict.to_dict() if self.verdict else None,
            "raw_answer": self.raw_text,
            "latency_ms": round(self.latency_ms, 1),
            "backend": self.backend,
        }


class BanglaFinGPT:
    def __init__(
        self,
        config: Config,
        agent: Agent | None = None,
        retriever: HybridRetriever | None = None,
        hallucination_filter: HallucinationFilter | None = None,
        *,
        use_retrieval: bool = True,
        use_filter: bool = True,
    ) -> None:
        self.config = config
        self.agent = agent or build_agent(config.agent, config.qlora)
        self.retriever = retriever
        self.filter = hallucination_filter or HallucinationFilter(
            config.hallucination,
            embedder=getattr(retriever.index, "embedder", None) if retriever else None,
        )
        self.use_retrieval = use_retrieval and retriever is not None
        self.use_filter = use_filter and config.hallucination.enabled

    # ------------------------------------------------------------------ query
    def answer(self, question: str, *, domain: str | None = None,
               top_k: int | None = None) -> Answer:
        started = time.perf_counter()
        language = detect_language(question)

        chunks: list[RetrievedChunk] = []
        if self.use_retrieval and self.retriever is not None:
            chunks = self.retriever.retrieve(question, top_k=top_k or self.config.retrieval.top_k,
                                             domain=domain)

        contexts = [c.text for c in chunks]
        prompt = build_prompt(question, contexts, [c.citation() for c in chunks], language=language)

        gen = self.config.generation
        response = self.agent.generate(
            prompt,
            max_new_tokens=gen.max_new_tokens,
            temperature=gen.temperature,
            top_p=gen.top_p,
            repetition_penalty=gen.repetition_penalty,
            do_sample=gen.do_sample,
        )
        raw = response.text.strip()

        # The verdict is computed even when filtering is off, so an ablation variant
        # can be audited for grounding without changing what it answers.
        verdict = self.filter.verify(raw, contexts, [c.chunk_id for c in chunks],
                                     question=question)
        text, answered = raw, True
        if self.use_filter and not verdict.grounded:
            text, answered = fallback_message(language), False

        return Answer(
            question=question,
            text=text,
            language=language,
            answered=answered,
            citations=_supporting_citations(text, chunks) if answered else [],
            chunks=chunks,
            verdict=verdict,
            raw_text=raw,
            latency_ms=(time.perf_counter() - started) * 1000,
            backend=response.backend,
        )

    def batch_answer(self, questions: Sequence[str], **kwargs: Any) -> list[Answer]:
        return [self.answer(q, **kwargs) for q in questions]

    # ----------------------------------------------------------- ablation view
    def variant(self, *, use_retrieval: bool, use_filter: bool) -> BanglaFinGPT:
        """A shallow copy with components toggled, for Table 7 ablations."""
        clone = BanglaFinGPT.__new__(BanglaFinGPT)
        clone.__dict__.update(self.__dict__)
        clone.use_retrieval = use_retrieval and self.retriever is not None
        clone.use_filter = use_filter and self.config.hallucination.enabled
        return clone


def _supporting_citations(answer: str, chunks: Sequence[RetrievedChunk],
                          min_overlap: float = 0.4, max_citations: int = 3) -> list[str]:
    """Cite only the chunks the answer actually draws on.

    Listing every retrieved chunk would attach authoritative-looking references
    to text they do not support - worse than citing nothing in this domain. We
    keep the best-matching chunk and any other that is nearly as close, since a
    single answer can legitimately span two clauses.
    """
    answer_tokens = set(content_tokens(answer))
    if not answer_tokens or not chunks:
        return []
    scored = []
    for chunk in chunks:
        tokens = set(content_tokens(chunk.text))
        if not tokens:
            continue
        scored.append((len(answer_tokens & tokens) / len(answer_tokens), chunk.citation()))
    if not scored:
        return []
    scored.sort(reverse=True)
    best = scored[0][0]
    if best < min_overlap:
        return []
    keep = [citation for overlap, citation in scored if overlap >= max(min_overlap, 0.9 * best)]
    return list(dict.fromkeys(keep))[:max_citations]
