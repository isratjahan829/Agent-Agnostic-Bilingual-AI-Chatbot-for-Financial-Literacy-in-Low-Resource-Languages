"""Component ablation over the same test set (paper Table 7).

Variants: base, +fine-tuning, +RAG, +FT+RAG, +hallucination filtering. The first
four differ only in which adapter is loaded and which components are enabled, so
the comparison isolates each contribution.
"""
from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from ..config import Config, load_config
from ..data.schema import QAPair
from ..pipeline import BanglaFinGPT
from ..retrieval.retriever import HybridRetriever
from ..utils import set_seed, write_json, write_jsonl
from .evaluate import evaluate, load_pairs, per_example_records, run_system
from .significance import significance_table

VARIANTS: list[dict[str, object]] = [
    {"name": "base",          "adapter": None,  "retrieval": False, "filter": False},
    {"name": "ft",            "adapter": "use", "retrieval": False, "filter": False},
    {"name": "rag",           "adapter": None,  "retrieval": True,  "filter": False},
    {"name": "ft_rag",        "adapter": "use", "retrieval": True,  "filter": False},
    {"name": "banglafingpt",  "adapter": "use", "retrieval": True,  "filter": True},
]


def run_ablation(
    cfg: Config,
    pairs: Sequence[QAPair],
    index_dir: str | None,
    adapter_path: str | None,
    out_dir: str,
) -> dict[str, object]:
    retriever = HybridRetriever.from_index_dir(index_dir, cfg.retrieval) if index_dir else None
    reports: dict[str, object] = {}
    em_scores: dict[str, list[float]] = {}

    for variant in VARIANTS:
        name = str(variant["name"])
        cfg.agent.adapter_path = adapter_path if variant["adapter"] == "use" else None
        system = BanglaFinGPT(
            cfg, retriever=retriever,
            use_retrieval=bool(variant["retrieval"]),
            use_filter=bool(variant["filter"]),
        )
        print(f"[ablation] {name}", flush=True)
        answers = run_system(system, pairs)
        report = evaluate(pairs, answers)
        reports[name] = report
        records = per_example_records(pairs, answers)
        em_scores[name] = [float(r["scores"]["em"]) for r in records]
        write_jsonl(Path(out_dir) / f"ablation_{name}_predictions.jsonl", records)

    table = [
        {
            "configuration": name,
            "em_pct": reports[name]["overall"]["em_pct"],      # type: ignore[index]
            "f1": reports[name]["overall"]["f1"],              # type: ignore[index]
            "bleu4": reports[name]["overall"]["bleu4"],        # type: ignore[index]
            "rouge_l": reports[name]["overall"]["rouge_l"],    # type: ignore[index]
            "meteor": reports[name]["overall"]["meteor"],      # type: ignore[index]
            "refusal_rate_pct": reports[name]["refusal_rate_pct"],  # type: ignore[index]
        }
        for name in em_scores
    ]
    result = {
        "table": table,
        "significance": significance_table(em_scores, baseline="base", metric="em"),
        "reports": reports,
    }
    write_json(Path(out_dir) / "ablation.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the BanglaFinGPT ablation study")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--test-file", default="data/processed/test.jsonl")
    parser.add_argument("--index-dir", default="artifacts/index")
    parser.add_argument("--adapter", default="artifacts/checkpoints/banglafingpt")
    parser.add_argument("--backend", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if args.backend:
        cfg.agent.backend = args.backend
    set_seed(cfg.seed)
    out_dir = args.out_dir or cfg.results_dir
    result = run_ablation(cfg, load_pairs(args.test_file, args.limit), args.index_dir,
                          args.adapter, out_dir)
    for row in result["table"]:  # type: ignore[index]
        print(f"  {row['configuration']:<14} EM={row['em_pct']:>5}%  F1={row['f1']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
