"""Run a system over the held-out test set and produce Tables 5, 8, 11."""
from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..config import Config, load_config
from ..data.schema import QAPair
from ..pipeline import Answer, BanglaFinGPT
from ..utils import read_jsonl, set_seed, write_json, write_jsonl
from .metrics import aggregate_metrics, score_pair


def run_system(system: BanglaFinGPT, pairs: Sequence[QAPair],
               progress_every: int = 100) -> list[Answer]:
    answers: list[Answer] = []
    for i, pair in enumerate(pairs, 1):
        answers.append(system.answer(pair.question, domain=pair.domain))
        if progress_every and i % progress_every == 0:
            print(f"  [{i}/{len(pairs)}]", flush=True)
    return answers


def evaluate(
    pairs: Sequence[QAPair],
    answers: Sequence[Answer],
    *,
    score_refusals_as_wrong: bool = True,
) -> dict[str, Any]:
    """Overall + per-domain metrics, refusal rate and mean latency.

    A refused query scores zero rather than being dropped: the filter trades
    recall for precision, and hiding the refusals would hide that cost.
    """
    predictions = [
        (a.text if a.answered or not score_refusals_as_wrong else "") for a in answers
    ]
    references = [p.answer for p in pairs]
    overall = aggregate_metrics(predictions, references)

    by_domain: dict[str, list[int]] = defaultdict(list)
    for i, pair in enumerate(pairs):
        by_domain[pair.domain].append(i)

    domain_rows: dict[str, dict[str, float]] = {}
    for domain, idx in by_domain.items():
        domain_rows[domain] = aggregate_metrics(
            [predictions[i] for i in idx], [references[i] for i in idx]
        )

    refused = sum(1 for a in answers if not a.answered)
    return {
        "overall": overall,
        "by_domain": domain_rows,
        "refusal_rate_pct": round(100 * refused / max(1, len(answers)), 2),
        "mean_latency_ms": round(sum(a.latency_ms for a in answers) / max(1, len(answers)), 1),
        "n": len(answers),
    }


def per_example_records(pairs: Sequence[QAPair], answers: Sequence[Answer]) -> list[dict[str, Any]]:
    """Row-level dump used by the error analysis and significance tests."""
    rows: list[dict[str, Any]] = []
    for pair, answer in zip(pairs, answers, strict=True):
        scores = score_pair(answer.text if answer.answered else "", pair.answer)
        rows.append({
            "qa_id": pair.qa_id,
            "domain": pair.domain,
            "question": pair.question,
            "reference": pair.answer,
            **answer.to_dict(),
            "scores": {k: round(v, 4) for k, v in scores.items()},
        })
    return rows


def load_pairs(path: str, limit: int | None = None) -> list[QAPair]:
    rows = read_jsonl(path)
    if limit:
        rows = rows[:limit]
    return [QAPair.from_dict(r) for r in rows]


def build_system(cfg: Config, index_dir: str | None, *, use_retrieval: bool = True,
                 use_filter: bool = True) -> BanglaFinGPT:
    from ..retrieval.retriever import HybridRetriever

    retriever = None
    if index_dir and use_retrieval:
        retriever = HybridRetriever.from_index_dir(index_dir, cfg.retrieval)
    return BanglaFinGPT(cfg, retriever=retriever, use_retrieval=use_retrieval,
                        use_filter=use_filter)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a BanglaFinGPT configuration")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--test-file", default="data/processed/test.jsonl")
    parser.add_argument("--index-dir", default="artifacts/index")
    parser.add_argument("--backend", default=None, help="override agent.backend")
    parser.add_argument("--adapter", default=None, help="LoRA adapter path for the local backend")
    parser.add_argument("--no-retrieval", action="store_true")
    parser.add_argument("--no-filter", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--tag", default="banglafingpt")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if args.backend:
        cfg.agent.backend = args.backend
    if args.adapter:
        cfg.agent.adapter_path = args.adapter
    set_seed(cfg.seed)

    pairs = load_pairs(args.test_file, args.limit)
    system = build_system(cfg, args.index_dir, use_retrieval=not args.no_retrieval,
                          use_filter=not args.no_filter)
    print(f"[eval] {args.tag}: {len(pairs)} questions via {cfg.agent.backend}")
    answers = run_system(system, pairs)

    report = evaluate(pairs, answers)
    report["config"] = {
        "tag": args.tag, "backend": cfg.agent.backend,
        "retrieval": not args.no_retrieval, "filter": not args.no_filter,
    }
    out_dir = Path(args.out_dir or cfg.results_dir)
    write_json(out_dir / f"{args.tag}_metrics.json", report)
    write_jsonl(out_dir / f"{args.tag}_predictions.jsonl", per_example_records(pairs, answers))

    print(f"[eval] EM={report['overall']['em_pct']}%  F1={report['overall']['f1']}  "
          f"refused={report['refusal_rate_pct']}%")
    print(f"[eval] wrote {out_dir}/{args.tag}_metrics.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
