"""Helpers that turn missing optional dependencies into actionable errors."""
from __future__ import annotations

from typing import Any


def _require(module: str, hint: str) -> Any:
    try:
        return __import__(module)
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(f"`{module}` is required for this feature. {hint}") from exc


def require_torch() -> Any:
    return _require("torch", "Install the training extras: pip install -r requirements-train.txt")


def require_transformers() -> Any:
    return _require("transformers", "pip install transformers")
