#!/usr/bin/env python3
"""End-to-end smoke run on the bundled sample corpus — no GPU, no API key.

Exercises retrieval, prompting, the hallucination filter, metrics, the ablation
table and the significance test, so a reviewer can verify the pipeline works
before committing to a full training run.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from banglafingpt.config import load_config  # noqa: E402
from banglafingpt.data.schema import QAPair, Segment  # noqa: E402
from banglafingpt.eval.error_analysis import error_report, hallucination_report  # noqa: E402
from banglafingpt.eval.evaluate import evaluate, per_example_records, run_system  # noqa: E402
from banglafingpt.eval.significance import significance_table  # noqa: E402
from banglafingpt.pipeline import BanglaFinGPT  # noqa: E402
from banglafingpt.retrieval.retriever import HybridRetriever  # noqa: E402
from banglafingpt.utils import read_jsonl, set_seed, write_json  # noqa: E402


def main() -> int:
    cfg = load_config(ROOT / "configs/demo.yaml")
    set_seed(cfg.seed)

    segments = [Segment(**{k: v for k, v in row.items() if k in Segment.__dataclass_fields__})
                for row in read_jsonl(ROOT / "data/sample/segments.jsonl")]
    pairs = [QAPair.from_dict(row) for row in read_jsonl(ROOT / "data/sample/qa_pairs.jsonl")]
    print(f"[demo] {len(segments)} segments, {len(pairs)} QA pairs")

    retriever = HybridRetriever.from_segments(segments, cfg.retrieval)
    system = BanglaFinGPT(cfg, retriever=retriever)

    variants = {
        "no_retrieval": system.variant(use_retrieval=False, use_filter=False),
        "rag_no_filter": system.variant(use_retrieval=True, use_filter=False),
        "banglafingpt": system.variant(use_retrieval=True, use_filter=True),
    }

    em_scores, table, records_by_variant = {}, [], {}
    for name, variant in variants.items():
        answers = run_system(variant, pairs, progress_every=0)
        report = evaluate(pairs, answers)
        records = per_example_records(pairs, answers)
        records_by_variant[name] = records
        em_scores[name] = [float(r["scores"]["em"]) for r in records]
        table.append({"variant": name, **report["overall"],
                      "refusal_rate_pct": report["refusal_rate_pct"]})
        print(f"  {name:<15} EM={report['overall']['em_pct']:>6}%  "
              f"F1={report['overall']['f1']:<6} refused={report['refusal_rate_pct']}%")

    sample = system.answer("ভ্যাটের আদর্শ হার কত?")
    print(f"\n[demo] Q: {sample.question}\n       A: {sample.text}")
    print(f"       sources: {', '.join(dict.fromkeys(sample.citations)) or 'n/a'}")

    # Show the filter rejecting a fabricated rate: the number 37 appears in no source.
    fabricated = "আমদানি শুল্কের হার ৩৭ শতাংশ এবং ইহা সকল পণ্যের ক্ষেত্রে প্রযোজ্য।"
    contexts = [c.text for c in retriever.retrieve("আমদানি শুল্কের হার কত?")]
    verdict = system.filter.verify(fabricated, contexts)
    print(f"\n[demo] fabricated answer: {fabricated}\n"
          f"       grounded={verdict.grounded}  reason={verdict.reason}")

    out = Path(cfg.results_dir)
    write_json(out / "demo_summary.json", {
        "table": table,
        "significance": significance_table(em_scores, baseline="no_retrieval", metric="em"),
        "errors": error_report(records_by_variant["banglafingpt"]),
        "hallucination": {name: hallucination_report(records)
                          for name, records in records_by_variant.items()},
    })
    print(f"\n[demo] wrote {out / 'demo_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
