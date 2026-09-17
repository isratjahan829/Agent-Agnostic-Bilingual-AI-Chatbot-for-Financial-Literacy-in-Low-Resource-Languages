#!/usr/bin/env python3
"""Collect every measured number into one readable report plus CSV tables.

Reads whatever result JSON exists and writes artifacts/RESULTS_SUMMARY.md with a
CSV beside each table. Numbers that were never measured are listed as pending
rather than omitted, so the report cannot be mistaken for a complete result set.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def table(rows: list[dict], columns: list[str] | None = None) -> str:
    if not rows:
        return "_(not measured)_\n"
    columns = columns or list(rows[0])
    out = ["| " + " | ".join(columns) + " |",
           "|" + "|".join("---" for _ in columns) + "|"]
    for row in rows:
        out.append("| " + " | ".join(str(row.get(c, "")) for c in columns) + " |")
    return "\n".join(out) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarise all measured results")
    parser.add_argument("--out", default=str(ROOT / "artifacts" / "RESULTS_SUMMARY.md"))
    parser.add_argument("--csv-dir", default=str(ROOT / "artifacts" / "tables"))
    args = parser.parse_args()

    nb = load(ROOT / "artifacts" / "results" / "notebook_results.json")
    rev = load(ROOT / "artifacts" / "reviewer" / "reviewer_experiments.json")
    gpu = load(ROOT / "artifacts" / "gpu_results.json")
    csv_dir = Path(args.csv_dir)

    md: list[str] = [
        "# BanglaFinGPT — measured results",
        "",
        "Every number here was produced by running the released code on "
        "`data/BanglaFinGPT_dataset.xlsx`. Anything not measured is listed at the end as "
        "pending, not estimated.",
        "",
    ]

    if nb:
        env = nb.get("environment", {})
        data = nb.get("dataset", {})
        md += [
            "## Run environment",
            "",
            table([{
                "GPU": "yes" if env.get("gpu") else "no",
                "dense encoder": env.get("encoder", "?"),
                "generator backend": env.get("backend", "?"),
                "QA pairs": data.get("pairs", "?"),
                "passages": data.get("passages", "?"),
                "retrieval chunks": data.get("chunks", "?"),
            }]),
            "The generator is the offline extractive backend: no GPU, no API keys and no "
            "model weights were reachable, so the generation scores below are a retrieval "
            "floor, not the fine-tuned system.",
            "",
            "## Splits",
            "",
            table([{"split": k, "pairs": v} for k, v in nb.get("splits", {}).items()]),
        ]

        ablation = nb.get("ablation", [])
        if ablation:
            write_csv(csv_dir / "ablation.csv", ablation)
            md += ["## Ablation (2,036-question test split)", "", table(ablation), ""]

        sig = nb.get("significance", [])
        if sig:
            def _p(value) -> str:
                # A baseline row carries no p-value; pandas may have turned it into NaN.
                if value is None or value != value:
                    return "—"
                return "< 1e-12" if value < 1e-12 else f"{value:.2e}"

            rows = [{"system": r["system"], "EM %": r["mean"],
                     "95% CI": f"[{r['ci95'][0]}, {r['ci95'][1]}]",
                     "p vs baseline": _p(r.get("p_value"))} for r in sig]
            write_csv(csv_dir / "significance.csv", rows)
            md += ["## Significance (McNemar on exact match, bootstrap CIs)", "",
                   table(rows), ""]

        by_domain = nb.get("by_domain", {})
        if by_domain:
            rows = [{"domain": d, "n": int(v["n"]), "EM %": v["em_pct"], "F1": v["f1"],
                     "BLEU-4": v["bleu4"], "ROUGE-L": v["rouge_l"], "METEOR": v["meteor"]}
                    for d, v in by_domain.items()]
            write_csv(csv_dir / "by_domain.csv", rows)
            md += ["## Per domain", "", table(rows), ""]

        retrieval = nb.get("retrieval", {})
        if retrieval:
            md += ["## Retrieval", "",
                   table([{"tuned alpha": retrieval.get("alpha"),
                           **{k: v for k, v in retrieval.get("test_recall", {}).items()}}]),
                   ""]

    if rev:
        split_audit = rev.get("R2_C1a_split_protocol", {})
        if split_audit:
            rows = [{"protocol": k.replace("_", " "),
                     "test items sharing a training passage":
                         f"{v['test_items_sharing_a_training_passage_pct']}%",
                     "near-duplicate questions": f"{v['near_duplicate_question_pct']}%"}
                    for k, v in split_audit.items()]
            write_csv(csv_dir / "split_leakage.csv", rows)
            md += ["## Split-protocol audit", "", table(rows), ""]

        enc = rev.get("R2_C1b_encoder_comparison", {}).get("rows", [])
        if enc:
            rows = [{"encoder": r["encoder"], "dense only": r["dense_only_recall@5"],
                     "BM25 only": r["bm25_only_recall@5"],
                     "best hybrid": r["best_hybrid_recall@5"], "best alpha": r["best_alpha"]}
                    for r in enc]
            write_csv(csv_dir / "encoders.csv", rows)
            md += ["## Retrieval encoders (validation recall@5)", "", table(rows), ""]

        rob = rev.get("R1_C3_robustness", {})
        if rob.get("rows"):
            rows = [{"query form": r["perturbation"].replace("_", " "),
                     "recall@5": r["recall@5"], "drop": f"{r['relative_drop_pct']}%"}
                    for r in rob["rows"]]
            write_csv(csv_dir / "robustness.csv", rows)
            md += [f"## Query robustness (all {rob.get('n', '?')} test questions)", "",
                   table(rows), ""]

        dom = rev.get("R1_C3_domain_difficulty", {})
        if dom.get("per_domain"):
            rows = [{"domain": r["domain"], "corpus share %": r["corpus_share_pct"],
                     "F1": r["f1"], "95% CI": f"[{r['f1_ci95'][0]}, {r['f1_ci95'][1]}]",
                     "avg answer words": r["avg_answer_words"],
                     "numbers per answer": r["numbers_per_answer"]}
                    for r in dom["per_domain"]]
            write_csv(csv_dir / "domain_difficulty.csv", rows)
            md += ["## Domain difficulty", "", table(rows), "",
                   dom.get("interpretation", ""), "", dom.get("caveat", ""), ""]

        audit = rev.get("R1_C5c_table2_audit", {})
        if audit.get("per_domain"):
            rows = [{"domain": r["domain"], "pairs (manuscript)": r["claimed"]["pairs"],
                     "pairs (dataset)": r["actual"]["pairs"], "delta": r["pairs_delta"],
                     "avg answer words (manuscript)": r["claimed"]["avg_a_words"],
                     "avg answer words (dataset)": r["actual"]["avg_a_words"]}
                    for r in audit["per_domain"]]
            write_csv(csv_dir / "table2_audit.csv", rows)
            md += ["## Manuscript Table 2 vs the released dataset", "", table(rows),
                   f"\nTotals agree at {audit['total_actual']}. Longest answers: "
                   f"{audit['longest_answers_claimed']} per the manuscript, "
                   f"{audit['longest_answers_actual']} in the dataset.", ""]

        framing = rev.get("R1_C1a_reduction_framing", {}).get("paper_reported", {})
        if framing:
            md += ["## Hallucination reduction, both framings", "", table([framing]), ""]

    md += ["## Figures", "",
           table([{"file": p.name, "KB": round(p.stat().st_size / 1024)}
                  for p in sorted((ROOT / "artifacts" / "figures").glob("*.png"))]), ""]

    pending = [
        "Fine-tuned model scores — needs a GPU (notebooks/BanglaFinGPT_gpu_run.ipynb)",
        "GPT-4o / Claude / Gemini, with and without retrieval — needs API keys",
        "Baseline hallucination rates — needs the same API keys",
        "Hallucination annotation labels and Fleiss' kappa — needs the three annotators",
        "Human evaluation (Table 6) — needs the three domain experts",
    ]
    if gpu:
        pending = pending[3:]
        md += ["## GPU run", "", table(gpu.get("ablation", [])), ""]
    md += ["## Not measured", ""] + [f"- {item}" for item in pending] + [""]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md), encoding="utf-8")
    print(f"[ok] {out}")
    for path in sorted(csv_dir.glob("*.csv")):
        print(f"     {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
