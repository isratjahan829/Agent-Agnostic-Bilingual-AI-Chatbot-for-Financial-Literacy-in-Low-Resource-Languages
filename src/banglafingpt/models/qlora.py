"""QLoRA setup for BanglaLLaMA-3.2-3B (paper Sec. 3.3.3).

4-bit NormalFloat quantisation (Eq. 9) with double quantisation keeps the 3B
base model inside ~6 GB of VRAM, which is what makes fine-tuning feasible on the
RTX 4050 laptop described in Sec. 3.4; LoRA adapters are the only trained
weights.
"""
from __future__ import annotations

from typing import Any

from ..config import QLoRAConfig


def supports_bf16() -> bool:
    """True when the current GPU can do bfloat16 (Ampere, SM 8.0, and newer).

    A Tesla T4 is Turing (SM 7.5) and cannot: asking for bfloat16 there fails at
    load time. An RTX 4050 is Ada (SM 8.9) and can.
    """
    try:
        import torch
    except ImportError:
        return False
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability()[0] >= 8


def _torch_dtype(name: str) -> Any:
    import torch

    if name == "auto":
        return torch.bfloat16 if supports_bf16() else torch.float16
    return {"bfloat16": torch.bfloat16, "float16": torch.float16,
            "float32": torch.float32}[name]


def build_bnb_config(cfg: QLoRAConfig) -> Any:
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=cfg.load_in_4bit,
        bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,       # "nf4"
        bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype=_torch_dtype(cfg.bnb_4bit_compute_dtype),
    )


def build_lora_config(cfg: QLoRAConfig) -> Any:
    from peft import LoraConfig

    return LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )


def load_tokenizer(model_name: str) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # left padding corrupts causal-LM training
    return tokenizer


def load_base_model(
    cfg: QLoRAConfig | None = None,
    *,
    for_training: bool = True,
    device_map: str = "auto",
) -> tuple[Any, Any]:
    """Load the 4-bit base model and tokenizer, LoRA-ready when training."""
    cfg = cfg or QLoRAConfig()
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "transformers is required to load the base model; "
            "`pip install -r requirements-train.txt`."
        ) from exc

    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        quantization_config=build_bnb_config(cfg) if cfg.load_in_4bit else None,
        device_map=device_map,
        torch_dtype=_torch_dtype(cfg.bnb_4bit_compute_dtype),
        trust_remote_code=False,
    )
    tokenizer = load_tokenizer(cfg.base_model)

    if for_training:
        from peft import get_peft_model, prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        model = get_peft_model(model, build_lora_config(cfg))
        model.config.use_cache = False  # incompatible with gradient checkpointing
    return model, tokenizer


def load_finetuned(adapter_path: str, cfg: QLoRAConfig | None = None,
                   device_map: str = "auto") -> tuple[Any, Any]:
    """Load the base model plus trained LoRA adapters for inference."""
    cfg = cfg or QLoRAConfig()
    from peft import PeftModel

    model, tokenizer = load_base_model(cfg, for_training=False, device_map=device_map)
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    model.config.use_cache = True
    return model, tokenizer


def trainable_parameter_summary(model: Any) -> dict:
    """Report the LoRA parameter fraction (the headline QLoRA efficiency claim)."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {
        "trainable": trainable,
        "total": total,
        "trainable_pct": round(100 * trainable / max(1, total), 4),
    }
