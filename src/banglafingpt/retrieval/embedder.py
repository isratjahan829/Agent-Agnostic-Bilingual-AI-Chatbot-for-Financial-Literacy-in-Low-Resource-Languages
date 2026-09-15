"""Sentence embedding backends.

``SentenceTransformerEmbedder`` is the one used in the paper (multilingual E5).
``HashingEmbedder`` is a dependency-free fallback so the full pipeline, the unit
tests and CI can run on a machine without torch.
"""
from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from typing import Protocol

from ..utils import content_tokens, ngrams


class Embedder(Protocol):
    dim: int

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        ...


def _l2_normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec] if norm else vec


class HashingEmbedder:
    """Hashed character-trigram + word bag-of-features embedding.

    Not competitive with a trained encoder, but it is deterministic, needs no
    downloads, and is good enough to exercise every downstream code path.
    """

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def _bucket(self, feature: str) -> int:
        digest = hashlib.md5(feature.encode("utf-8")).digest()
        return int.from_bytes(digest[:4], "little") % self.dim

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self.dim
            tokens = content_tokens(text)
            features = list(tokens) + [" ".join(g) for g in ngrams(tokens, 2)]
            for feature in features:
                vec[self._bucket(feature)] += 1.0
            # sublinear scaling keeps long passages from dominating the score
            vec = [math.log1p(v) for v in vec]
            out.append(_l2_normalize(vec))
        return out


class SentenceTransformerEmbedder:
    """Wraps a `sentence-transformers` model (default: multilingual-e5-base)."""

    def __init__(self, model_name: str, normalize: bool = True, device: str | None = None) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise RuntimeError(
                "sentence-transformers is required for dense retrieval; "
                "`pip install sentence-transformers` or use HashingEmbedder."
            ) from exc
        self.model = SentenceTransformer(model_name, device=device)
        self.normalize = normalize
        self.dim = int(self.model.get_sentence_embedding_dimension())
        # E5 checkpoints expect "query: " / "passage: " prefixes.
        self.is_e5 = "e5" in model_name.lower()

    def encode(self, texts: Sequence[str], is_query: bool = False) -> list[list[float]]:
        payload = list(texts)
        if self.is_e5:
            prefix = "query: " if is_query else "passage: "
            payload = [prefix + t for t in payload]
        vectors = self.model.encode(
            payload, normalize_embeddings=self.normalize,
            convert_to_numpy=True, show_progress_bar=False,
        )
        return [v.tolist() for v in vectors]


def load_embedder(model_name: str, *, normalize: bool = True, fallback: bool = True) -> Embedder:
    """Load the configured encoder, falling back to hashing when torch is absent."""
    if model_name in {"hashing", "hash", ""}:
        return HashingEmbedder()
    try:
        return SentenceTransformerEmbedder(model_name, normalize=normalize)
    except RuntimeError:
        if not fallback:
            raise
        return HashingEmbedder()
