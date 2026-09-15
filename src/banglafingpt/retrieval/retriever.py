"""Hybrid dense+sparse retriever feeding the RAG prompt of Eq. (17)."""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from ..config import RetrievalConfig
from ..data.schema import Segment
from .embedder import Embedder, load_embedder
from .index import Chunk, DocumentIndex


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    doc_id: str
    domain: str
    score: float
    dense_score: float = 0.0
    sparse_score: float = 0.0
    section: str | None = None

    def citation(self) -> str:
        return f"{self.doc_id}#{self.section}" if self.section else self.doc_id


def chunks_from_segments(segments: Iterable[Segment]) -> list[Chunk]:
    return [
        Chunk(chunk_id=s.segment_id, text=s.text, doc_id=s.doc_id,
              domain=s.domain, section=s.section)
        for s in segments
    ]


def _minmax(scores: dict[int, float]) -> dict[int, float]:
    """Scale scores to [0, 1] so dense and BM25 values can be summed."""
    if not scores:
        return {}
    lo, hi = min(scores.values()), max(scores.values())
    if hi - lo < 1e-9:
        return {k: 1.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


class HybridRetriever:
    """score(d, q) = alpha * dense(d, q) + (1 - alpha) * bm25(d, q).

    Dense retrieval catches paraphrase, BM25 catches the exact strings that
    matter in regulation (section numbers, HS codes, rates) — the paper's
    failure analysis shows both are needed.
    """

    def __init__(self, index: DocumentIndex, config: RetrievalConfig | None = None) -> None:
        self.index = index
        self.config = config or RetrievalConfig()

    @classmethod
    def from_segments(cls, segments: Sequence[Segment], config: RetrievalConfig | None = None,
                      embedder: Embedder | None = None) -> HybridRetriever:
        config = config or RetrievalConfig()
        embedder = embedder or load_embedder(config.embedding_model,
                                             normalize=config.normalize_embeddings)
        chunks = chunks_from_segments(segments)
        # A corpus-trained encoder (TF-IDF+SVD) has to see the corpus before it
        # can encode anything; a pretrained one has no fit() and is used as is.
        fit = getattr(embedder, "fit", None)
        if callable(fit) and not getattr(embedder, "_fitted", True):
            fit([c.text for c in chunks])
        index = DocumentIndex(embedder).build(chunks)
        return cls(index, config)

    @classmethod
    def from_index_dir(cls, path: str, config: RetrievalConfig | None = None) -> HybridRetriever:
        config = config or RetrievalConfig()
        return cls(DocumentIndex.load(path, embedding_model=config.embedding_model), config)

    def retrieve(self, query: str, top_k: int | None = None,
                 domain: str | None = None) -> list[RetrievedChunk]:
        top_k = top_k or self.config.top_k
        pool = max(top_k * 4, 20)
        dense = _minmax(dict(self.index.dense_search(query, pool)))
        sparse = _minmax(dict(self.index.sparse_search(query, pool)))
        alpha = self.config.hybrid_alpha

        merged: list[RetrievedChunk] = []
        for idx in set(dense) | set(sparse):
            chunk = self.index.chunks[idx]
            if domain and chunk.domain != domain:
                continue
            d, s = dense.get(idx, 0.0), sparse.get(idx, 0.0)
            merged.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id, text=chunk.text, doc_id=chunk.doc_id,
                    domain=chunk.domain, section=chunk.section,
                    score=alpha * d + (1 - alpha) * s, dense_score=d, sparse_score=s,
                )
            )
        merged.sort(key=lambda c: c.score, reverse=True)
        return merged[:top_k]
