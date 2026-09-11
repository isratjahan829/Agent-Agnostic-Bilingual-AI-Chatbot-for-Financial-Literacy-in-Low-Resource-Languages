#!/usr/bin/env python3
"""Experiments that answer the reviewer comments, run on the released dataset.

Each experiment is named for the comment it addresses. Everything here runs on
CPU; experiments that need a GPU or an API key are listed by
`--list-blocked` with the exact command to run them elsewhere.

    python scripts/reviewer_experiments.py --all
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from banglafingpt.config import load_config  # noqa: E402
from banglafingpt.data.build_dataset import (  # noqa: E402
    leakage_report,
    random_row_split,
    split_pairs,
)
from banglafingpt.data.load_xlsx import load_corpus  # noqa: E402
from banglafingpt.data.quality_control import char_ngrams, jaccard  # noqa: E402
from banglafingpt.eval.annotation import build_annotation_sheets, reduction_report  # noqa: E402
from banglafingpt.eval.evaluate import evaluate, per_example_records, run_system  # noqa: E402
from banglafingpt.eval.robustness import retrieval_robustness  # noqa: E402
from banglafingpt.eval.significance import bootstrap_ci  # noqa: E402
from banglafingpt.hallucination.filters import HallucinationFilter  # noqa: E402
from banglafingpt.pipeline import BanglaFinGPT  # noqa: E402
from banglafingpt.retrieval.embedder import load_embedder  # noqa: E402
from banglafingpt.retrieval.retriever import HybridRetriever  # noqa: E402
from banglafingpt.utils import normalize_text, set_seed, write_json  # noqa: E402

DATASET = ROOT / "data" / "BanglaFinGPT_dataset.xlsx"
SEED = 42


# --------------------------------------------------------------------------- #
# R2-C1a / R1-C3: how the split is made, and what the naive split leaks
# --------------------------------------------------------------------------- #
def experiment_split_protocol(pairs) -> dict:
    grouped = split_pairs(pairs, 7412, 1000, 2000, seed=SEED)
    row_level = random_row_split(pairs, 7412, 1000, 2000, seed=SEED)

    # What a model could memorise: a test question with a near-identical twin in
    # training. Bucketed by first content token to keep the comparison tractable.
    def near_duplicate_rate(splits) -> float:
        from collections import defaultdict

        buckets = defaultdict(list)
        for pair in splits["train"]:
            tokens = normalize_text(pair.question).split()
            buckets[tokens[0] if tokens else ""].append(char_ngrams(pair.question))
        hits = 0
        for pair in splits["test"]:
            tokens = normalize_text(pair.question).split()
            signature = char_ngrams(pair.question)
            if any(jaccard(signature, other) >= 0.9
                   for other in buckets.get(tokens[0] if tokens else "", ())):
                hits += 1
        return round(100 * hits / max(1, len(splits["test"])), 1)

    return {
        "passage_grouped": {**leakage_report(grouped),
                            "near_duplicate_question_pct": near_duplicate_rate(grouped),
                            "sizes": {k: len(v) for k, v in grouped.items()}},
        "row_level_random": {**leakage_report(row_level),
                             "near_duplicate_question_pct": near_duplicate_rate(row_level),
                             "sizes": {k: len(v) for k, v in row_level.items()}},
    }


# --------------------------------------------------------------------------- #
# R1-C3: is the customs gap a real difficulty gap or an artifact of imbalance?
# --------------------------------------------------------------------------- #
def experiment_domain_balance(pairs, records) -> dict:
    import random

    by_domain: dict[str, list] = {}
    for record in records:
        by_domain.setdefault(record["domain"], []).append(record)

    smallest = min(len(v) for v in by_domain.values())
    rng = random.Random(SEED)
    rows = []
    for domain, group in sorted(by_domain.items()):
        balanced = rng.sample(group, smallest)
        f1_all = [float(r["scores"]["f1"]) for r in group]
        f1_balanced = [float(r["scores"]["f1"]) for r in balanced]
        mean_all, low, high = bootstrap_ci(f1_all, n_resamples=2000, seed=SEED)
        mean_balanced, _, _ = bootstrap_ci(f1_balanced, n_resamples=2000, seed=SEED)
        rows.append({
            "domain": domain,
            "n_full": len(group),
            "f1_full": round(mean_all, 4),
            "f1_ci95": [round(low, 4), round(high, 4)],
            "n_balanced": smallest,
            "f1_balanced": round(mean_balanced, 4),
            "train_share_pct": round(
                100 * sum(1 for p in pairs if p.domain == domain) / len(pairs), 2),
        })
    return {"per_domain": rows, "balanced_n_per_domain": smallest}


# --------------------------------------------------------------------------- #
# R1-C3: benchmark questions are clean; real users are not
# --------------------------------------------------------------------------- #
def experiment_robustness(retriever, test_pairs) -> dict:
    import random

    sample = random.Random(SEED).sample(test_pairs, min(500, len(test_pairs)))
    return {"n": len(sample), "rows": retrieval_robustness(retriever, sample, top_k=5, seed=SEED)}


# --------------------------------------------------------------------------- #
# R1-C1a: report the absolute effect alongside the relative one
# --------------------------------------------------------------------------- #
def experiment_reduction_framing() -> dict:
    return {
        "paper_reported": reduction_report(8.5, 0.5),
        "note": ("The relative figure is computed from an 8.5% base. Both framings "
                 "belong in the text so the size of the effect is unambiguous."),
    }


# --------------------------------------------------------------------------- #
# R1-C1b: a stated, reproducible annotation protocol
# --------------------------------------------------------------------------- #
def experiment_annotation_protocol(records, out_dir: Path) -> dict:
    return build_annotation_sheets(records, out_dir / "hallucination_annotation",
                                   annotators=("A1", "A2", "A3"), sample_size=200,
                                   stratify_by="domain", seed=SEED)


# --------------------------------------------------------------------------- #
# R1-C2 / R2-C1c: every system must see the same context
# --------------------------------------------------------------------------- #
def experiment_baseline_parity(cfg, retriever, test_pairs, grounding_filter) -> dict:
    """Measures the context effect for one backend, and names the missing runs.

    The reviewer's objection is that GPT-4o was judged without retrieval while
    BanglaFinGPT had it. The pipeline is backend-agnostic, so the fair matrix is
    a configuration change, not new code — but the hosted runs need API keys.
    """
    system = BanglaFinGPT(cfg, retriever=retriever, hallucination_filter=grounding_filter)
    measured = {}
    for name, toggles in {"without_context": dict(use_retrieval=False, use_filter=False),
                          "with_context": dict(use_retrieval=True, use_filter=False)}.items():
        answers = run_system(system.variant(**toggles), test_pairs, progress_every=0)
        measured[name] = evaluate(test_pairs, answers)["overall"]

    return {
        "backend": cfg.agent.backend,
        "measured": measured,
        "context_effect_f1": round(measured["with_context"]["f1"]
                                   - measured["without_context"]["f1"], 4),
        "required_runs_needing_api_keys": [
            f"python -m banglafingpt.eval.evaluate --backend {b} --tag {t}{flag}"
            for b, t, flag in [
                ("openai", "gpt4o_norag", " --no-retrieval"),
                ("openai", "gpt4o_rag", ""),
                ("anthropic", "claude_norag", " --no-retrieval"),
                ("anthropic", "claude_rag", ""),
                ("gemini", "gemini_norag", " --no-retrieval"),
                ("gemini", "gemini_rag", ""),
            ]
        ],
    }


# --------------------------------------------------------------------------- #
# R1-C5c: one number, one value, everywhere
# --------------------------------------------------------------------------- #
def experiment_number_consistency(records) -> dict:
    counts = Counter(r["domain"] for r in records)
    return {
        "test_split_sizes_by_domain": dict(sorted(counts.items())),
        "rule": ("Report every metric at one fixed precision (one decimal for "
                 "percentages, two for F1) and derive the text from the table."),
    }


BLOCKED = {
    "QLoRA fine-tuning and the fine-tuned scores": "needs a CUDA GPU (~6 GB VRAM)",
    "GPT-4o / Claude / Gemini rows, with and without RAG": "needs API keys",
    "Human evaluation (Table 6) and hallucination labels": "needs the three annotators",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the reviewer-response experiments")
    parser.add_argument("--dataset", default=str(DATASET))
    parser.add_argument("--out-dir", default="artifacts/reviewer")
    parser.add_argument("--limit", type=int, default=None, help="cap the test split (debugging)")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--list-blocked", action="store_true")
    args = parser.parse_args()

    if args.list_blocked:
        for item, reason in BLOCKED.items():
            print(f"- {item}: {reason}")
        return 0

    set_seed(SEED)
    out_dir = Path(args.out_dir)
    cfg = load_config(ROOT / "configs" / "default.yaml")
    cfg.agent.backend = "echo"
    cfg.agent.adapter_path = None

    print("[load] dataset")
    pairs, segments = load_corpus(args.dataset)
    splits = split_pairs(pairs, 7412, 1000, 2000, seed=SEED)
    test_pairs = splits["test"][: args.limit] if args.limit else splits["test"]

    print("[build] retrieval index")
    embedder = load_embedder(cfg.retrieval.embedding_model)
    cfg.retrieval.hybrid_alpha = 0.2      # tuned on validation; see the notebook
    retriever = HybridRetriever.from_segments(segments, cfg.retrieval, embedder=embedder)

    cfg.hallucination.min_cosine_similarity = 0.249   # calibrated on validation
    cfg.hallucination.min_keyword_overlap = 0.50
    grounding_filter = HallucinationFilter(cfg.hallucination, embedder=embedder)

    print("[run] system on the test split")
    system = BanglaFinGPT(cfg, retriever=retriever, hallucination_filter=grounding_filter)
    answers = run_system(system, test_pairs, progress_every=0)
    records = per_example_records(test_pairs, answers)

    results = {
        "R1_C1a_reduction_framing": experiment_reduction_framing(),
        "R1_C1b_annotation_protocol": experiment_annotation_protocol(records, out_dir),
        "R1_C2_baseline_parity": experiment_baseline_parity(cfg, retriever, test_pairs,
                                                            grounding_filter),
        "R1_C3_domain_balance": experiment_domain_balance(pairs, records),
        "R1_C3_robustness": experiment_robustness(retriever, test_pairs),
        "R1_C5c_number_consistency": experiment_number_consistency(records),
        "R2_C1a_split_protocol": experiment_split_protocol(pairs),
        "blocked_here": BLOCKED,
    }
    write_json(out_dir / "reviewer_experiments.json", results)

    print(f"\n[ok] wrote {out_dir / 'reviewer_experiments.json'}\n")
    split = results["R2_C1a_split_protocol"]
    print("R2-C1a  split protocol")
    for name, row in split.items():
        print(f"  {name:<18} test items sharing a training passage: "
              f"{row['test_items_sharing_a_training_passage_pct']}%   "
              f"near-duplicate questions: {row['near_duplicate_question_pct']}%")
    print("\nR1-C1a  hallucination reduction framing")
    print(f"  {results['R1_C1a_reduction_framing']['paper_reported']}")
    print("\nR1-C3   query robustness (passage recall@5)")
    for row in results["R1_C3_robustness"]["rows"]:
        print(f"  {row['perturbation']:<22} {row['recall@5']:.3f}  "
              f"({row['relative_drop_pct']:.1f}% drop vs original)")
    print("\nBlocked in this environment:")
    for item, reason in BLOCKED.items():
        print(f"  - {item}: {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
