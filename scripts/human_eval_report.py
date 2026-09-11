#!/usr/bin/env python3
"""Table 6 + Fleiss' kappa from an expert ratings file.

Input is CSV or JSONL with columns: query_id, rater_id, relevance, coherence,
fluency, accuracy, creativity (1-5).
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from banglafingpt.eval.human_eval import (  # noqa: E402
    DIMENSIONS,
    interpret_kappa,
    kappa_by_dimension,
    summarize_ratings,
)
from banglafingpt.utils import read_jsonl, write_json  # noqa: E402


def load_ratings(path: Path) -> list[dict]:
    if path.suffix.lower() == ".jsonl":
        return read_jsonl(path)
    with path.open(encoding="utf-8") as fh:
        rows = []
        for row in csv.DictReader(fh):
            parsed = {"query_id": row["query_id"], "rater_id": row.get("rater_id", "")}
            for dimension in DIMENSIONS:
                if row.get(dimension) not in (None, ""):
                    parsed[dimension] = float(row[dimension])
            rows.append(parsed)
        return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarise human evaluation ratings")
    parser.add_argument("ratings", help="CSV or JSONL of expert ratings")
    parser.add_argument("--out", default="artifacts/results/human_eval.json")
    args = parser.parse_args()

    ratings = load_ratings(Path(args.ratings))
    if not ratings:
        print("[error] no ratings found", file=sys.stderr)
        return 1

    summary = summarize_ratings(ratings)
    kappas = kappa_by_dimension(ratings)
    overall = sum(kappas.values()) / len(kappas) if kappas else 0.0

    print(f"{'Dimension':<12} {'Mean ± SD':<14} {'kappa'}")
    for dimension in DIMENSIONS:
        if dimension not in summary:
            continue
        stats = summary[dimension]
        kappa = kappas.get(dimension)
        print(f"{dimension:<12} {stats['mean']:.2f} ± {stats['std']:.2f}   "
              f"{kappa if kappa is not None else '-'}")
    print(f"\nMean Fleiss' kappa = {overall:.3f} ({interpret_kappa(overall)} agreement)")

    write_json(args.out, {"summary": summary, "kappa": kappas,
                          "mean_kappa": round(overall, 3),
                          "interpretation": interpret_kappa(overall),
                          "n_ratings": len(ratings)})
    print(f"[ok] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
