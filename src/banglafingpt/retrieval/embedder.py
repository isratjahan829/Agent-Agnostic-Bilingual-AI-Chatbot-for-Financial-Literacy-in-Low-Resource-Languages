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

from ..utils import content_tokens, ngrams, normalize_text


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


class TfidfSvdEmbedder:
    """Latent-semantic embedding fitted on the corpus itself (no model download).

    TF-IDF over character n-grams, reduced with truncated SVD. Character n-grams
    matter for Bangla: they absorb the inflectional suffixes that make two
    surface forms of the same term look unrelated to a word-level model. This is
    weaker than a trained multilingual encoder, but unlike the hashing fallback
    it learns the corpus's own term statistics, and it needs nothing but the
    documents already on disk.
    """

    def __init__(self, dim: int = 256, char_ngram_range: tuple = (2, 5),
                 min_df: int = 2, seed: int = 42) -> None:
        try:
            from sklearn.decomposition import TruncatedSVD
            from sklearn.feature_extraction.text import TfidfVectorizer
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise RuntimeError("scikit-learn is required for TfidfSvdEmbedder") from exc
        self.dim = dim
        self._vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=char_ngram_range, min_df=min_df,
            sublinear_tf=True, lowercase=True,
        )
        self._svd = TruncatedSVD(n_components=dim, random_state=seed)
        self._fitted = False

    def fit(self, documents: Sequence[str]) -> TfidfSvdEmbedder:
        """Fit on the corpus. Queries are then projected into the same space."""
        normalized = [normalize_text(d) for d in documents]
        matrix = self._vectorizer.fit_transform(normalized)
        n_components = min(self.dim, min(matrix.shape) - 1)
        if n_components < self.dim:
            self._svd.n_components = n_components
            self.dim = n_components
        self._svd.fit(matrix)
        self._fitted = True
        return self

    def encode(self, texts: Sequence[str], is_query: bool = False) -> list[list[float]]:
        if not self._fitted:
            raise RuntimeError("TfidfSvdEmbedder.encode called before fit()")
        matrix = self._vectorizer.transform([normalize_text(t) for t in texts])
        reduced = self._svd.transform(matrix)
        return [_l2_normalize(list(map(float, row))) for row in reduced]


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
    """Load the configured encoder.

    Falls back in order of quality when the requested model cannot be loaded:
    the pretrained sentence encoder, then a TF-IDF+SVD encoder fitted on the
    corpus itself, then hashed n-grams. The last two need no model download, so
    the pipeline runs unchanged on a machine with no GPU and no model access.
    """
    if model_name in {"hashing", "hash", ""}:
        return HashingEmbedder()
    if model_name in {"tfidf-svd", "lsa"}:
        return TfidfSvdEmbedder()
    try:
        return SentenceTransformerEmbedder(model_name, normalize=normalize)
    except RuntimeError:
        if not fallback:
            raise
    try:
        return TfidfSvdEmbedder()
    except RuntimeError:
        return HashingEmbedder()
