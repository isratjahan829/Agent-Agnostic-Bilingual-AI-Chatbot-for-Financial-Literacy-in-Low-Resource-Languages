"""Hosted baseline backends (GPT-4o, Claude, Gemini) used in paper Table 5.

They receive exactly the same RAG prompt as the local model, so the comparison
isolates the model rather than the prompt.
"""
from __future__ import annotations

import os
import re
import time
from typing import Any

from .base import Agent, AgentResponse


def _api_key(env_var: str) -> str:
    key = os.environ.get(env_var)
    if not key:
        raise RuntimeError(f"Environment variable {env_var} is not set.")
    return key


class OpenAIAgent(Agent):
    name = "openai"

    def __init__(self, model: str = "gpt-4o", api_key_env: str = "OPENAI_API_KEY",
                 **kw: Any) -> None:
        super().__init__(model=model, **kw)
        from openai import OpenAI

        self.client = OpenAI(api_key=_api_key(api_key_env))

    def generate(self, prompt: str, *, max_new_tokens: int = 256, temperature: float = 0.2,
                 **kwargs: Any) -> AgentResponse:
        started = time.perf_counter()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_new_tokens,
            temperature=temperature,
        )
        usage = response.usage
        return AgentResponse(
            text=(response.choices[0].message.content or "").strip(),
            backend=self.name, model=self.model,
            latency_ms=(time.perf_counter() - started) * 1000,
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
        )


class AnthropicAgent(Agent):
    name = "anthropic"

    def __init__(self, model: str = "claude-sonnet-4-5", api_key_env: str = "ANTHROPIC_API_KEY",
                 **kw: Any) -> None:
        super().__init__(model=model, **kw)
        import anthropic

        self.client = anthropic.Anthropic(api_key=_api_key(api_key_env))

    def generate(self, prompt: str, *, max_new_tokens: int = 256, temperature: float = 0.2,
                 **kwargs: Any) -> AgentResponse:
        started = time.perf_counter()
        message = self.client.messages.create(
            model=self.model,
            max_tokens=max_new_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(block.text for block in message.content if block.type == "text")
        return AgentResponse(
            text=text.strip(), backend=self.name, model=self.model,
            latency_ms=(time.perf_counter() - started) * 1000,
            prompt_tokens=message.usage.input_tokens,
            completion_tokens=message.usage.output_tokens,
        )


class GeminiAgent(Agent):
    name = "gemini"

    def __init__(self, model: str = "gemini-1.5-pro", api_key_env: str = "GOOGLE_API_KEY",
                 **kw: Any) -> None:
        super().__init__(model=model, **kw)
        import google.generativeai as genai

        genai.configure(api_key=_api_key(api_key_env))
        self.client = genai.GenerativeModel(model)

    def generate(self, prompt: str, *, max_new_tokens: int = 256, temperature: float = 0.2,
                 **kwargs: Any) -> AgentResponse:
        started = time.perf_counter()
        response = self.client.generate_content(
            prompt,
            generation_config={"max_output_tokens": max_new_tokens, "temperature": temperature},
        )
        return AgentResponse(
            text=(response.text or "").strip(), backend=self.name, model=self.model,
            latency_ms=(time.perf_counter() - started) * 1000,
        )


class EchoAgent(Agent):
    """Offline extractive stub: returns the retrieved sentence closest to the question.

    It is a genuine retrieval-only lower bound (no generation), and it lets the
    tests and the demo exercise every downstream code path without a model.
    """

    name = "echo"

    def __init__(self, model: str = "echo", **kw: Any) -> None:
        super().__init__(model=model, **kw)

    @staticmethod
    def _question_of(prompt: str) -> str:
        if "[INST]" in prompt and "[/INST]" in prompt:
            return prompt.split("[INST]", 1)[1].split("[/INST]", 1)[0].strip()
        return ""

    @staticmethod
    def _context_sentences(prompt: str) -> list[str]:
        from ..data.segment import split_sentences

        body = prompt.split("[CONTEXT]")[-1].split("[/CONTEXT]")[0]
        sentences: list[str] = []
        for block in body.split("\n\n"):
            # strip the "[1] (doc#section) " provenance prefix added by the prompt
            text = re.sub(r"^\s*\[\d+\]\s*(\([^)]*\))?\s*", "", block.strip())
            sentences.extend(split_sentences(text))
        return [s for s in sentences if s.strip()]

    def generate(self, prompt: str, *, max_new_tokens: int = 256, **kwargs: Any) -> AgentResponse:
        from ..utils import content_tokens

        started = time.perf_counter()
        question_tokens = set(content_tokens(self._question_of(prompt)))
        sentences = self._context_sentences(prompt)

        best, best_score = "", -1.0
        for sentence in sentences:
            tokens = set(content_tokens(sentence))
            if not tokens:
                continue
            overlap = len(question_tokens & tokens) / len(question_tokens | tokens)
            if overlap > best_score:
                best, best_score = sentence, overlap

        words = best.split()
        if len(words) > max_new_tokens:
            best = " ".join(words[:max_new_tokens])
        return AgentResponse(text=best.strip(), backend=self.name, model=self.model,
                             latency_ms=(time.perf_counter() - started) * 1000,
                             meta={"match_score": round(max(best_score, 0.0), 4)})
