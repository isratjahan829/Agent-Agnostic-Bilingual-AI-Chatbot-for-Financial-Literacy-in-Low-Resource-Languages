"""Vector + BM25 index over regulatory chunks, with FAISS when available."""
from __future__ import annotations

import math
import pickle
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ..utils import content_tokens, write_json
from .embedder import Embedder, load_embedder


@dataclass
class Chunk:
    chunk_id: str
    text: str
    doc_id: str
    domain: str
    section: str | None = None


class BM25:
    """Okapi BM25 over content tokens (sparse half of the hybrid retriever)."""

    def __init__(self, corpus_tokens: Sequence[Sequence[str]], k1: float = 1.5,
                 b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.docs = [Counter(tokens) for tokens in corpus_tokens]
        self.lengths = [sum(c.values()) for c in self.docs]
        self.avg_len = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0
        self.df: Counter = Counter()
        for doc in self.docs:
            self.df.update(doc.keys())
        self.n = len(self.docs)

    def _idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log(1 + (self.n - df + 0.5) / (df + 0.5))

    def scores(self, query_tokens: Sequence[str]) -> list[float]:
        out = [0.0] * self.n
        for term in set(query_tokens):
            idf = self._idf(term)
            if idf <= 0:
                continue
            for i, doc in enumerate(self.docs):
                tf = doc.get(term, 0)
                if not tf:
                    continue
                norm_len = self.lengths[i] / max(1e-9, self.avg_len)
                denom = tf + self.k1 * (1 - self.b + self.b * norm_len)
                out[i] += idf * (tf * (self.k1 + 1)) / denom
        return out


class DocumentIndex:
    """Holds chunks, their dense vectors and a BM25 index; persists to disk."""

    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder
        self.chunks: list[Chunk] = []
        self.vectors: list[list[float]] = []
        self.bm25: BM25 | None = None
        self._faiss = None

    # ------------------------------------------------------------------ build
    def build(self, chunks: Sequence[Chunk], batch_size: int = 64) -> DocumentIndex:
        self.chunks = list(chunks)
        texts = [c.text for c in self.chunks]
        self.vectors = []
        for start in range(0, len(texts), batch_size):
            self.vectors.extend(self.embedder.encode(texts[start : start + batch_size]))
        self.bm25 = BM25([content_tokens(t) for t in texts])
        self._build_faiss()
        return self

    def _build_faiss(self) -> None:
        if not self.vectors:
            return
        try:
            import faiss
            import numpy as np
        except ImportError:
            self._faiss = None
            return
        matrix = np.asarray(self.vectors, dtype="float32")
        index = faiss.IndexFlatIP(matrix.shape[1])  # vectors are L2-normalised
        index.add(matrix)
        self._faiss = index

    # ----------------------------------------------------------------- search
    def dense_search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        if not self.vectors:
            return []
        encode = self.embedder.encode
        try:
            qvec = encode([query], is_query=True)[0]  # type: ignore[call-arg]
        except TypeError:
            qvec = encode([query])[0]
        if self._faiss is not None:
            import numpy as np

            query = np.asarray([qvec], dtype="float32")
            scores, idx = self._faiss.search(query, min(top_k, len(self.chunks)))
            return [(int(i), float(s)) for i, s in zip(idx[0], scores[0], strict=False) if i >= 0]
        sims = [(i, sum(a * b for a, b in zip(qvec, v, strict=False)))
                for i, v in enumerate(self.vectors)]
        sims.sort(key=lambda x: x[1], reverse=True)
        return sims[:top_k]

    def sparse_search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        if self.bm25 is None:
            return []
        scored = list(enumerate(self.bm25.scores(content_tokens(query))))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    # ------------------------------------------------------------ persistence
    def save(self, path: str | Path) -> None:
        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "index.pkl").open("wb") as fh:
            pickle.dump({"chunks": self.chunks, "vectors": self.vectors}, fh)
        write_json(directory / "meta.json", {
            "n_chunks": len(self.chunks),
            "dim": getattr(self.embedder, "dim", None),
            "embedder": type(self.embedder).__name__,
        })

    @classmethod
    def load(cls, path: str | Path, embedder: Embedder | None = None,
             embedding_model: str = "intfloat/multilingual-e5-base") -> DocumentIndex:
        directory = Path(path)
        with (directory / "index.pkl").open("rb") as fh:
            payload = pickle.load(fh)
        index = cls(embedder or load_embedder(embedding_model))
        index.chunks = payload["chunks"]
        index.vectors = payload["vectors"]
        index.bm25 = BM25([content_tokens(c.text) for c in index.chunks])
        index._build_faiss()
        return index

    def __len__(self) -> int:
        return len(self.chunks)
