"""Typed configuration for every stage of the BanglaFinGPT pipeline.

All numbers default to the values reported in the paper so that a run with the
shipped `configs/default.yaml` reproduces the published setup.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

try:  # PyYAML is optional; JSON configs work without it.
    import yaml
except ImportError:  # pragma: no cover - exercised only without PyYAML
    yaml = None

DOMAINS: list[str] = ["taxation", "vat", "customs", "finance"]


@dataclass
class DataConfig:
    raw_dir: str = "data/raw"
    interim_dir: str = "data/interim"
    processed_dir: str = "data/processed"
    segment_min_words: int = 150
    segment_max_words: int = 300
    questions_per_segment: int = 3
    # Paper Table 3: domain-balanced 7,412 / 1,000 / 2,000 split.
    train_size: int = 7412
    val_size: int = 1000
    test_size: int = 2000
    near_duplicate_threshold: float = 0.92
    seed: int = 42


@dataclass
class RetrievalConfig:
    embedding_model: str = "intfloat/multilingual-e5-base"
    index_dir: str = "artifacts/index"
    chunk_words: int = 220
    chunk_overlap_words: int = 40
    top_k: int = 5
    # Hybrid dense+sparse scoring: score = alpha*dense + (1-alpha)*bm25.
    hybrid_alpha: float = 0.7
    normalize_embeddings: bool = True


@dataclass
class QLoRAConfig:
    base_model: str = "BanglaLLM/bangla-llama-3.2-3b-instruct"
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"  # NormalFloat-4, Eq. (9)
    bnb_4bit_use_double_quant: bool = True
    # "auto" resolves to bfloat16 on Ampere and newer, float16 otherwise.
    # A T4 (Turing) has no bfloat16 support, so a hard-coded bfloat16 fails there.
    bnb_4bit_compute_dtype: str = "auto"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )


@dataclass
class TrainConfig:
    output_dir: str = "artifacts/checkpoints/banglafingpt"
    max_seq_length: int = 1024
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    num_train_epochs: float = 3.0
    max_steps: int = 30000  # T = 3e4 steps, Sec. 3.3.4
    learning_rate: float = 5e-5  # alpha_max = 5e-5, Eq. (13)
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    optim: str = "paged_adamw_8bit"
    logging_steps: int = 25
    eval_steps: int = 250
    save_steps: int = 250
    early_stopping_patience: int = 3
    gradient_checkpointing: bool = True
    # None means "use bfloat16 if the GPU supports it, otherwise fp16".
    bf16: bool | None = None
    seed: int = 42


@dataclass
class GenerationConfig:
    max_new_tokens: int = 256
    temperature: float = 0.2
    top_p: float = 0.9
    repetition_penalty: float = 1.05
    do_sample: bool = False


@dataclass
class HallucinationConfig:
    # Sec. 4.6: cosine similarity >= 0.70 AND keyword overlap >= 0.70.
    min_cosine_similarity: float = 0.70
    min_keyword_overlap: float = 0.70
    min_numeric_support: float = 1.0  # every number must appear in a source
    # Relevance floor: how close the question must be to the best retrieved chunk.
    # 0.0 disables the check, which is the paper's configuration.
    min_question_similarity: float = 0.0
    enabled: bool = True


@dataclass
class AgentConfig:
    """Which backend answers a query. The pipeline is agent-agnostic."""
    backend: str = "local"  # local | openai | anthropic | gemini | echo
    model: str | None = None
    api_key_env: str | None = None
    adapter_path: str | None = None
    timeout_s: float = 60.0


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    qlora: QLoRAConfig = field(default_factory=QLoRAConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    hallucination: HallucinationConfig = field(default_factory=HallucinationConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    results_dir: str = "artifacts/results"
    figures_dir: str = "artifacts/figures"
    seed: int = 42

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_SECTIONS = {
    "data": DataConfig,
    "retrieval": RetrievalConfig,
    "qlora": QLoRAConfig,
    "train": TrainConfig,
    "generation": GenerationConfig,
    "hallucination": HallucinationConfig,
    "agent": AgentConfig,
}


def _read(path: str | os.PathLike[str]) -> dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    if str(path).endswith((".yaml", ".yml")):
        if yaml is None:
            raise RuntimeError("PyYAML is required to read YAML configs; `pip install pyyaml`.")
        return yaml.safe_load(text) or {}
    import json

    return json.loads(text)


def load_config(path: str | os.PathLike[str] | None = None, **overrides: Any) -> Config:
    """Load a config file and apply dotted overrides, e.g. ``train.max_steps=100``."""
    raw: dict[str, Any] = _read(path) if path else {}
    cfg = Config()
    for name, klass in _SECTIONS.items():
        section = raw.get(name) or {}
        known = {k: v for k, v in section.items() if k in klass.__dataclass_fields__}
        unknown = set(section) - set(known)
        if unknown:
            raise ValueError(f"Unknown keys in config section '{name}': {sorted(unknown)}")
        setattr(cfg, name, klass(**known))
    for key in ("results_dir", "figures_dir", "seed"):
        if key in raw:
            setattr(cfg, key, raw[key])
    for dotted, value in overrides.items():
        _set_dotted(cfg, dotted.replace("__", "."), value)
    return cfg


def _set_dotted(cfg: Config, dotted: str, value: Any) -> None:
    obj: Any = cfg
    parts = dotted.split(".")
    for part in parts[:-1]:
        obj = getattr(obj, part)
    if not hasattr(obj, parts[-1]):
        raise AttributeError(f"No config field named '{dotted}'")
    setattr(obj, parts[-1], value)
