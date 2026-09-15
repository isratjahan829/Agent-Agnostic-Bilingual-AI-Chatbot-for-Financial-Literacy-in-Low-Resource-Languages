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
from dataclasses import replace
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
from banglafingpt.utils import (  # noqa: E402
    extract_numbers,
    normalize_text,
    set_seed,
    word_count,
    write_json,
)  # noqa: E402

DATASET = ROOT / "data" / "BanglaFinGPT_dataset.xlsx"
SEED = 42


TARGET_REFUSAL = 0.123   # paper Sec. 3.6: 12.3% of queries are declined


def calibrate_threshold(cfg, retriever, embedder, val_pairs, sample: int = 300) -> float:
    """Pick the cosine threshold that reproduces the paper's refusal rate.

    Runs the pipeline unfiltered over validation questions, then takes the
    quantile of the observed grounding similarities that would decline
    TARGET_REFUSAL of them.
    """
    import random

    probe_cfg = replace(cfg.hallucination, min_cosine_similarity=0.0,
                        min_keyword_overlap=0.0, min_question_similarity=0.0)
    probe = BanglaFinGPT(cfg, retriever=retriever,
                         hallucination_filter=HallucinationFilter(probe_cfg, embedder=embedder))
    subset = random.Random(SEED).sample(val_pairs, min(sample, len(val_pairs)))
    cosines = sorted(probe.answer(p.question).verdict.cosine_similarity for p in subset)
    index = min(len(cosines) - 1, int(TARGET_REFUSAL * len(cosines)))
    return round(cosines[index], 3)


def tune_alpha(retriever, val_pairs, ks=(5,), seed: int = SEED,
               grid=(0.0, 0.2, 0.3, 0.5, 0.7, 1.0), sample: int = 400) -> float:
    """Pick the dense/sparse weight by passage-level recall@5 on validation."""
    import random

    subset = random.Random(seed).sample(val_pairs, min(sample, len(val_pairs)))
    best_alpha, best_recall = grid[0], -1.0
    for alpha in grid:
        retriever.config.hybrid_alpha = alpha
        hits = sum(p.source in {c.doc_id for c in retriever.retrieve(p.question, top_k=5)}
                   for p in subset)
        recall = hits / len(subset)
        if recall > best_recall:
            best_alpha, best_recall = alpha, recall
    retriever.config.hybrid_alpha = best_alpha
    return best_alpha


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
def experiment_domain_difficulty(pairs, records) -> dict:
    """Evidence on whether corpus imbalance explains the per-domain accuracy gap.

    Note on what this can and cannot show. Per-domain accuracy does not depend on
    how many items *other* domains contribute to the test split, so re-balancing
    the test set answers nothing; the imbalance the reviewer means is in the
    training data, and testing it properly requires retraining on a balanced
    training set, which needs a GPU. What can be established without one is
    whether the observed ordering is consistent with the imbalance hypothesis at
    all, and whether a content-difficulty explanation fits better.
    """
    from statistics import mean

    by_domain: dict[str, list] = {}
    for record in records:
        by_domain.setdefault(record["domain"], []).append(record)

    corpus_share = {
        domain: 100 * sum(1 for p in pairs if p.domain == domain) / len(pairs)
        for domain in by_domain
    }

    rows = []
    for domain, group in sorted(by_domain.items()):
        f1 = [float(r["scores"]["f1"]) for r in group]
        mean_f1, low, high = bootstrap_ci(f1, n_resamples=2000, seed=SEED)
        references = [r["reference"] for r in group]
        rows.append({
            "domain": domain,
            "corpus_share_pct": round(corpus_share[domain], 1),
            "n_test": len(group),
            "f1": round(mean_f1, 4),
            "f1_ci95": [round(low, 4), round(high, 4)],
            # Content-difficulty proxies: longer answers with denser numeric
            # content are harder to match exactly and harder to generate.
            "avg_answer_words": round(mean(len(a.split()) for a in references), 1),
            "numbers_per_answer": round(
                mean(len(extract_numbers(a)) for a in references), 2),
        })

    ranked_by_f1 = sorted(rows, key=lambda r: r["f1"])
    weakest = ranked_by_f1[0]
    largest = max(rows, key=lambda r: r["corpus_share_pct"])
    smallest = min(rows, key=lambda r: r["corpus_share_pct"])

    return {
        "per_domain": rows,
        "weakest_domain": weakest["domain"],
        "largest_domain": largest["domain"],
        "smallest_domain": smallest["domain"],
        "imbalance_hypothesis_consistent": weakest["domain"] == smallest["domain"],
        "interpretation": (
            f"The weakest domain is {weakest['domain']}, which is the "
            f"{'least' if weakest is smallest else 'most'} represented domain in the "
            f"corpus ({weakest['corpus_share_pct']}%). "
            + ("This is consistent with the imbalance hypothesis."
               if weakest["domain"] == smallest["domain"]
               else "This is the opposite of what the imbalance hypothesis predicts.")
        ),
        "caveat": ("Per-domain accuracy is independent of the other domains' test "
                   "sizes, so this is evidence about the direction of the effect, not "
                   "a controlled test. A controlled test requires retraining on a "
                   "domain-balanced training set, which needs a GPU."),
    }


# --------------------------------------------------------------------------- #
# R2-C1b: how much of retrieval quality comes from the dense half?
# --------------------------------------------------------------------------- #
def experiment_encoder_comparison(segments, val_pairs, cfg) -> dict:
    """Compare the available dense encoders against BM25 and the hybrid.

    The paper specifies a pretrained multilingual encoder. This environment has
    no access to model weights, so the comparison covers what can be built from
    the corpus alone: hashed n-grams, and TF-IDF reduced by SVD. Both are weaker
    than a trained encoder; reporting them bounds how much of the retrieval
    result rests on the dense half rather than on lexical matching.
    """
    import random

    from banglafingpt.retrieval.embedder import HashingEmbedder, TfidfSvdEmbedder

    subset = random.Random(SEED).sample(val_pairs, min(400, len(val_pairs)))
    grid = (1.0, 0.7, 0.5, 0.3, 0.0)
    rows = []
    for name, embedder in (("hashed n-grams", HashingEmbedder()),
                           ("TF-IDF + SVD (corpus-fitted)", TfidfSvdEmbedder())):
        retriever = HybridRetriever.from_segments(segments, cfg.retrieval, embedder=embedder)
        recalls = {}
        for alpha in grid:
            retriever.config.hybrid_alpha = alpha
            hits = sum(p.source in {c.doc_id for c in retriever.retrieve(p.question, top_k=5)}
                       for p in subset)
            recalls[alpha] = round(hits / len(subset), 3)
        rows.append({
            "encoder": name,
            "dense_only_recall@5": recalls[1.0],
            "bm25_only_recall@5": recalls[0.0],
            "best_hybrid_recall@5": max(recalls.values()),
            "best_alpha": max(recalls, key=recalls.get),
            "by_alpha": recalls,
        })
    return {
        "n": len(subset),
        "rows": rows,
        "note": ("alpha = 1.0 is dense only, 0.0 is BM25 only. The pretrained encoder "
                 "named in the manuscript could not be evaluated here: model weights "
                 "are unreachable from this environment."),
    }


# --------------------------------------------------------------------------- #
# R1-C5c / R1-C3: do the manuscript's dataset tables match the released dataset?
# --------------------------------------------------------------------------- #
# Table 2 of the submitted manuscript, transcribed.
PAPER_TABLE_2 = {
    "taxation": {"pairs": 2823, "avg_q_words": 9.4, "avg_a_words": 28.3},
    "vat": {"pairs": 2641, "avg_q_words": 8.1, "avg_a_words": 22.1},
    "customs": {"pairs": 3475, "avg_q_words": 10.7, "avg_a_words": 31.7},
    "finance": {"pairs": 1473, "avg_q_words": 7.9, "avg_a_words": 25.8},
}


def experiment_table2_audit(pairs) -> dict:
    """Compare the manuscript's dataset table against the released dataset.

    A reviewer can download the Kaggle release and recompute these in a minute,
    so any discrepancy is worth finding before they do.
    """
    from statistics import mean

    rows = []
    for domain, claimed in sorted(PAPER_TABLE_2.items()):
        group = [p for p in pairs if p.domain == domain]
        actual = {
            "pairs": len(group),
            "avg_q_words": round(mean(word_count(p.question) for p in group), 1),
            "avg_a_words": round(mean(word_count(p.answer) for p in group), 1),
        }
        rows.append({
            "domain": domain,
            "claimed": claimed,
            "actual": actual,
            "pairs_delta": actual["pairs"] - claimed["pairs"],
            "matches": actual["pairs"] == claimed["pairs"],
        })

    # The manuscript explains the customs gap by answer length. Does the released
    # dataset support that ordering?
    longest = max(rows, key=lambda r: r["actual"]["avg_a_words"])["domain"]
    claimed_longest = max(rows, key=lambda r: r["claimed"]["avg_a_words"])["domain"]
    return {
        "per_domain": rows,
        "total_claimed": sum(v["pairs"] for v in PAPER_TABLE_2.values()),
        "total_actual": len(pairs),
        "all_domains_match": all(r["matches"] for r in rows),
        "longest_answers_claimed": claimed_longest,
        "longest_answers_actual": longest,
        "length_explanation_supported": longest == claimed_longest,
    }


# --------------------------------------------------------------------------- #
# R1-C3: benchmark questions are clean; real users are not
# --------------------------------------------------------------------------- #
def experiment_robustness(retriever, test_pairs) -> dict:
    # The whole test split, not a sample: a sampled recall differs between runs by
    # more than the effect being measured, and this number goes into the paper.
    return {"n": len(test_pairs),
            "rows": retrieval_robustness(retriever, test_pairs, top_k=5, seed=SEED)}


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
    retriever = HybridRetriever.from_segments(segments, cfg.retrieval, embedder=embedder)

    # Tune the dense/sparse mix on validation, exactly as the notebook does, so both
    # artifacts quote the same retrieval configuration and therefore the same numbers.
    print("[tune] dense/sparse mix on validation")
    cfg.retrieval.hybrid_alpha = tune_alpha(retriever, splits["validation"], seed=SEED)
    print(f"       selected hybrid_alpha = {cfg.retrieval.hybrid_alpha}")

    # Calibrate the filter on validation. The cosine threshold is scale-dependent:
    # every encoder puts similarity on a different scale, so a value tuned for one
    # silently changes the refusal rate under another.
    print("[tune] hallucination threshold on validation")
    cfg.hallucination.min_keyword_overlap = 0.50
    cfg.hallucination.min_cosine_similarity = calibrate_threshold(
        cfg, retriever, embedder, splits["validation"])
    print(f"       min_cosine_similarity = {cfg.hallucination.min_cosine_similarity} "
          f"(targeting a {100 * TARGET_REFUSAL:.1f}% refusal rate)")
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
        "R1_C3_domain_difficulty": experiment_domain_difficulty(pairs, records),
        "R1_C5c_table2_audit": experiment_table2_audit(pairs),
        "R2_C1b_encoder_comparison": experiment_encoder_comparison(
            segments, splits["validation"], cfg),
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
    audit = results["R1_C5c_table2_audit"]
    print("\nR2-C1b  retrieval encoders (validation, recall@5)")
    for row in results["R2_C1b_encoder_comparison"]["rows"]:
        print(f"  {row['encoder']:<30} dense-only {row['dense_only_recall@5']:.3f}  "
              f"BM25-only {row['bm25_only_recall@5']:.3f}  "
              f"best hybrid {row['best_hybrid_recall@5']:.3f} at alpha={row['best_alpha']}")

    print("\nR1-C5c  manuscript Table 2 vs the released dataset")
    for row in audit["per_domain"]:
        c, a = row["claimed"], row["actual"]
        flag = "ok" if row["matches"] else f"differs by {row['pairs_delta']:+d}"
        print(f"  {row['domain']:<10} pairs {c['pairs']:>5} claimed / {a['pairs']:>5} actual "
              f"({flag});  avg answer {c['avg_a_words']:>5} / {a['avg_a_words']:>5} words")
    print(f"  total {audit['total_claimed']} claimed / {audit['total_actual']} actual")
    print(f"  longest answers: {audit['longest_answers_claimed']} claimed, "
          f"{audit['longest_answers_actual']} actual -> length explanation "
          f"{'supported' if audit['length_explanation_supported'] else 'NOT supported'}")

    print("\nR1-C3   domain difficulty")
    for row in results["R1_C3_domain_difficulty"]["per_domain"]:
        print(f"  {row['domain']:<10} corpus {row['corpus_share_pct']:>5}%  "
              f"F1 {row['f1']:.3f} {row['f1_ci95']}  "
              f"answer {row['avg_answer_words']:>5} words, "
              f"{row['numbers_per_answer']} numbers")
    print(f"  -> {results['R1_C3_domain_difficulty']['interpretation']}")

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
