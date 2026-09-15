"""Agent-agnostic generation backends.

The RAG + hallucination machinery is identical no matter which model answers;
swapping `agent.backend` in the config is all it takes to evaluate GPT-4o,
Claude, Gemini or the local fine-tuned BanglaLLaMA on the same benchmark.
"""
from .base import Agent, AgentResponse
from .registry import available_backends, build_agent, register_backend

__all__ = ["Agent", "AgentResponse", "build_agent", "register_backend", "available_backends"]
