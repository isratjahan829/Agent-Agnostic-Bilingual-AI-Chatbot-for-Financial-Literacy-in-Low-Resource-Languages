"""Backend registry: name -> constructor, so configs stay declarative."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..config import AgentConfig, QLoRAConfig
from .base import Agent

_REGISTRY: dict[str, Callable[..., Agent]] = {}


def register_backend(name: str, factory: Callable[..., Agent]) -> None:
    _REGISTRY[name] = factory


def available_backends() -> list[str]:
    return sorted(_REGISTRY)


def _lazy(module: str, attr: str) -> Callable[..., Agent]:
    def factory(**kwargs: Any) -> Agent:
        import importlib

        return getattr(importlib.import_module(module, __package__), attr)(**kwargs)

    return factory


register_backend("local", _lazy(".local_agent", "LocalLlamaAgent"))
register_backend("openai", _lazy(".api_agents", "OpenAIAgent"))
register_backend("anthropic", _lazy(".api_agents", "AnthropicAgent"))
register_backend("gemini", _lazy(".api_agents", "GeminiAgent"))
register_backend("echo", _lazy(".api_agents", "EchoAgent"))


def build_agent(cfg: AgentConfig, qlora: QLoRAConfig | None = None) -> Agent:
    if cfg.backend not in _REGISTRY:
        raise ValueError(f"Unknown backend {cfg.backend!r}; available: {available_backends()}")
    kwargs: dict[str, Any] = {}
    if cfg.model:
        kwargs["model"] = cfg.model
    if cfg.api_key_env:
        kwargs["api_key_env"] = cfg.api_key_env
    if cfg.backend == "local":
        kwargs["adapter_path"] = cfg.adapter_path
        kwargs["qlora"] = qlora or QLoRAConfig()
    return _REGISTRY[cfg.backend](**kwargs)
