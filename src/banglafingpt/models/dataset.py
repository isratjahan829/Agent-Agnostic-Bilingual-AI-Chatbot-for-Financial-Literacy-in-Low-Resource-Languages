"""Turn QA pairs into completion-masked training tensors."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..data.schema import QAPair
from ..retrieval.retriever import HybridRetriever
from ..utils import detect_language
from .prompts import build_training_example


def build_examples(
    pairs: Sequence[QAPair],
    retriever: HybridRetriever | None = None,
    top_k: int = 3,
    eos_token: str = "</s>",
) -> list[dict[str, str]]:
    """Render each QA pair as prompt/completion text.

    When a retriever is supplied the model is trained *with* retrieved context,
    so train-time and inference-time prompts match — training without context
    and serving with it is a distribution shift that costs several EM points.
    """
    examples: list[dict[str, str]] = []
    for pair in pairs:
        contexts: list[str] = []
        if retriever is not None:
            contexts = [c.text for c in retriever.retrieve(pair.question, top_k=top_k)]
        examples.append(
            build_training_example(
                question=pair.question,
                answer=pair.answer,
                contexts=contexts,
                language=detect_language(pair.question),
                eos_token=eos_token,
            )
        )
    return examples


class CompletionOnlyCollator:
    """Pads a batch and masks prompt tokens with -100 so loss covers the answer."""

    def __init__(self, tokenizer: Any, max_length: int = 1024) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, features: list[dict[str, str]]) -> dict[str, Any]:
        import torch

        input_ids: list[list[int]] = []
        labels: list[list[int]] = []
        for feature in features:
            prompt_ids = self.tokenizer(feature["prompt"], add_special_tokens=True)["input_ids"]
            completion_ids = self.tokenizer(
                feature["completion"], add_special_tokens=False
            )["input_ids"]
            ids = (prompt_ids + completion_ids)[: self.max_length]
            label = ([-100] * len(prompt_ids) + completion_ids)[: self.max_length]
            input_ids.append(ids)
            labels.append(label)

        width = max(len(ids) for ids in input_ids)
        pad_id = self.tokenizer.pad_token_id
        batch = {
            "input_ids": torch.tensor([ids + [pad_id] * (width - len(ids)) for ids in input_ids]),
            "attention_mask": torch.tensor(
                [[1] * len(ids) + [0] * (width - len(ids)) for ids in input_ids]
            ),
            "labels": torch.tensor([lb + [-100] * (width - len(lb)) for lb in labels]),
        }
        return batch


def to_hf_dataset(examples: Sequence[dict[str, str]]) -> Any:
    from datasets import Dataset

    return Dataset.from_list(list(examples))
