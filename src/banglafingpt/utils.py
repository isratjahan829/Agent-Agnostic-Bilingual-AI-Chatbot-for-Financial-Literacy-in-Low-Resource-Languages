"""Shared helpers: determinism, JSONL IO, and Bangla-aware text normalisation."""
from __future__ import annotations

import json
import os
import random
import re
import unicodedata
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

BENGALI_DIGITS = "০১২৩৪৫৬৭৮৯"
_DIGIT_MAP = {ord(b): str(i) for i, b in enumerate(BENGALI_DIGITS)}

# Punctuation that carries no meaning for exact-match comparison, including the
# Bangla danda (।) and double danda (॥).
_PUNCT = re.compile(r"[।॥,;:\"'`~!?()\[\]{}<>/\\|@#$%^&*_+=—–\-.]")
_WS = re.compile(r"\s+")

# Very common Bangla + English function words. Dropping them makes keyword
# overlap measure content words rather than grammar.
STOPWORDS = frozenset(
    """
    এই সেই ও এবং বা কিন্তু যে যা যার যাদের তার তাদের করা করে করার হবে হয় হয়েছে
    হচ্ছে ছিল থেকে জন্য মধ্যে সাথে উপর নিচে কোন কোনো একটি একটা কি কী কেন কীভাবে
    কিভাবে কত কবে আছে নেই না নয় আমি আমার আমরা আপনি আপনার তিনি এটি এটা এর
    the a an of for to in on at is are was were be been by with and or but this
    that these those what how why when where which who whom as from it its
    """.split()
)


# --------------------------------------------------------------------------- #
# determinism
# --------------------------------------------------------------------------- #
def set_seed(seed: int = 42) -> None:
    """Seed every RNG we might touch (stdlib, numpy, torch) for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# --------------------------------------------------------------------------- #
# io
# --------------------------------------------------------------------------- #
def read_jsonl(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def iter_jsonl(path: str | os.PathLike[str]) -> Iterator[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: malformed JSON line") from exc


def write_jsonl(path: str | os.PathLike[str], rows: Iterable[dict[str, Any]]) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with p.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_json(path: str | os.PathLike[str], obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: str | os.PathLike[str]) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# text
# --------------------------------------------------------------------------- #
def bn_to_en_digits(text: str) -> str:
    """Map Bengali digits to ASCII so ``৫০০`` and ``500`` compare equal."""
    return text.translate(_DIGIT_MAP)


def normalize_text(text: str, *, keep_digits_native: bool = False) -> str:
    """Unicode-normalise, unify digits, strip punctuation and collapse whitespace.

    NFKC is required for Bangla: the same grapheme can be encoded with different
    code point sequences, which would otherwise break exact match.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text).strip().lower()
    if not keep_digits_native:
        text = bn_to_en_digits(text)
    text = _PUNCT.sub(" ", text)
    return _WS.sub(" ", text).strip()


def tokenize(text: str) -> list[str]:
    """Whitespace tokenisation on normalised text (works for Bangla and English)."""
    normalized = normalize_text(text)
    return normalized.split() if normalized else []


def content_tokens(text: str) -> list[str]:
    return [t for t in tokenize(text) if t not in STOPWORDS and len(t) > 1]


def extract_numbers(text: str) -> list[str]:
    """Pull out numeric literals (percentages, amounts, HS codes, section numbers)."""
    return re.findall(r"\d+(?:[.,]\d+)*", bn_to_en_digits(text))


def word_count(text: str) -> int:
    return len(tokenize(text))


def ngrams(tokens: Sequence[str], n: int) -> list[tuple]:
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def detect_language(text: str) -> str:
    """Return ``bn``, ``en`` or ``mixed`` from Bengali-block character share."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return "unknown"
    bengali = sum(1 for c in letters if "ঀ" <= c <= "৿")
    ratio = bengali / len(letters)
    if ratio >= 0.6:
        return "bn"
    if ratio <= 0.1:
        return "en"
    return "mixed"
