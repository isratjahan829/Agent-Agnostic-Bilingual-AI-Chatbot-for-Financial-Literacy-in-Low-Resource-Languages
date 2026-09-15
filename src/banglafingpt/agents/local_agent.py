"""The paper's own system: fine-tuned BanglaLLaMA-3.2-3B served locally."""
from __future__ import annotations

import time
from typing import Any

from ..config import QLoRAConfig
from .base import Agent, AgentResponse


class LocalLlamaAgent(Agent):
    """Runs the 4-bit base model with (optionally) the trained LoRA adapters."""

    name = "local"

    def __init__(self, model: str = "", adapter_path: str | None = None,
                 qlora: QLoRAConfig | None = None, **kwargs: Any) -> None:
        super().__init__(model=model or (qlora or QLoRAConfig()).base_model, **kwargs)
        cfg = qlora or QLoRAConfig()
        if model:
            cfg.base_model = model
        from .._lazy import require_torch

        self.torch = require_torch()
        if adapter_path:
            from ..models.qlora import load_finetuned

            self.model, self.tokenizer = load_finetuned(adapter_path, cfg)
        else:
            from ..models.qlora import load_base_model

            self.model, self.tokenizer = load_base_model(cfg, for_training=False)
        self.adapter_path = adapter_path

    def generate(self, prompt: str, *, max_new_tokens: int = 256, temperature: float = 0.2,
                 top_p: float = 0.9, repetition_penalty: float = 1.05,
                 do_sample: bool = False, **kwargs: Any) -> AgentResponse:
        started = time.perf_counter()
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True,
                                max_length=kwargs.get("max_input_tokens", 3072))
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with self.torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                repetition_penalty=repetition_penalty,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        generated = output[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        return AgentResponse(
            text=text, backend=self.name, model=self.model_name,
            latency_ms=(time.perf_counter() - started) * 1000,
            prompt_tokens=int(inputs["input_ids"].shape[1]),
            completion_tokens=int(generated.shape[0]),
            meta={"adapter": self.adapter_path},
        )

    @property
    def model_name(self) -> str:
        return str(self.model) if isinstance(self.model, str) else type(self.model).__name__
