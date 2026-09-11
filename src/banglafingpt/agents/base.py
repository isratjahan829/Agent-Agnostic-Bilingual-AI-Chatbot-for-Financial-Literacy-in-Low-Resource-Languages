"""The single interface every generation backend implements."""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentResponse:
    text: str
    backend: str
    model: str = ""
    latency_ms: float = 0.0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    meta: dict[str, Any] = field(default_factory=dict)


class Agent(ABC):
    """Generates an answer given a question and retrieved context."""

    name: str = "agent"

    def __init__(self, model: str = "", **kwargs: Any) -> None:
        self.model = model
        self.options = kwargs

    @abstractmethod
    def generate(self, prompt: str, *, max_new_tokens: int = 256,
                 temperature: float = 0.2, **kwargs: Any) -> AgentResponse:
        """Return a completion for a fully rendered prompt."""

    def batch_generate(self, prompts: Sequence[str], **kwargs: Any) -> list[AgentResponse]:
        return [self.generate(p, **kwargs) for p in prompts]

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"{type(self).__name__}(model={self.model!r})"
