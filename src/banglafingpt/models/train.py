"""QLoRA fine-tuning loop: AdamW + cosine schedule (Eq. 13), early stopping."""
from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from ..config import Config, load_config
from ..data.schema import QAPair
from ..retrieval.retriever import HybridRetriever
from ..utils import read_jsonl, set_seed, write_json
from .dataset import CompletionOnlyCollator, build_examples, to_hf_dataset
from .qlora import load_base_model, trainable_parameter_summary


def train(
    cfg: Config,
    train_pairs: Sequence[QAPair],
    val_pairs: Sequence[QAPair],
    retriever: HybridRetriever | None = None,
) -> str:
    """Fine-tune the LoRA adapters and return the output directory."""
    from transformers import EarlyStoppingCallback, Trainer, TrainingArguments

    set_seed(cfg.train.seed)
    model, tokenizer = load_base_model(cfg.qlora, for_training=True)
    print("[qlora]", trainable_parameter_summary(model))

    eos = tokenizer.eos_token or "</s>"
    train_ds = to_hf_dataset(build_examples(train_pairs, retriever, cfg.retrieval.top_k, eos))
    val_ds = to_hf_dataset(build_examples(val_pairs, retriever, cfg.retrieval.top_k, eos))

    args = TrainingArguments(
        output_dir=cfg.train.output_dir,
        per_device_train_batch_size=cfg.train.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.train.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
        num_train_epochs=cfg.train.num_train_epochs,
        max_steps=cfg.train.max_steps,
        learning_rate=cfg.train.learning_rate,
        lr_scheduler_type=cfg.train.lr_scheduler_type,   # cosine decay, Eq. (13)
        warmup_ratio=cfg.train.warmup_ratio,
        weight_decay=cfg.train.weight_decay,
        optim=cfg.train.optim,                            # paged AdamW, Sec. 3.3.4
        logging_steps=cfg.train.logging_steps,
        eval_strategy="steps",
        eval_steps=cfg.train.eval_steps,
        save_steps=cfg.train.save_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        gradient_checkpointing=cfg.train.gradient_checkpointing,
        bf16=cfg.train.bf16,
        report_to=[],
        seed=cfg.train.seed,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=CompletionOnlyCollator(tokenizer, cfg.train.max_seq_length),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.train.early_stopping_patience)],
    )
    result = trainer.train()

    out = Path(cfg.train.output_dir)
    trainer.save_model(str(out))
    tokenizer.save_pretrained(str(out))
    write_json(out / "training_summary.json", {
        "metrics": result.metrics,
        "config": cfg.to_dict(),
        "parameters": trainable_parameter_summary(model),
    })
    return str(out)


def _load_pairs(path: str) -> list[QAPair]:
    return [QAPair.from_dict(row) for row in read_jsonl(path)]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="QLoRA fine-tune BanglaFinGPT")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--train-file", default="data/processed/train.jsonl")
    parser.add_argument("--val-file", default="data/processed/validation.jsonl")
    parser.add_argument("--index-dir", default=None,
                        help="retrieval index for context-aware training")
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if args.max_steps is not None:
        cfg.train.max_steps = args.max_steps
    retriever = (HybridRetriever.from_index_dir(args.index_dir, cfg.retrieval)
                 if args.index_dir else None)
    output = train(cfg, _load_pairs(args.train_file), _load_pairs(args.val_file), retriever)
    print(f"[done] adapters saved to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
