"""BanglaFinGPT: agent-agnostic bilingual RAG system for financial literacy in Bangla.

Reference implementation of the paper "Agent-Agnostic Bilingual AI Chatbot for
Financial Literacy in Low-Resource Languages" (BanglaFinGPT).
"""

__version__ = "0.1.0"

from .config import Config, load_config  # noqa: F401

__all__ = ["Config", "load_config", "__version__"]
