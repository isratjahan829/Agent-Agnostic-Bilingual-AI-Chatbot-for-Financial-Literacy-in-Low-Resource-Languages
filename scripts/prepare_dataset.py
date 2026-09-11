#!/usr/bin/env python3
"""Raw NBR PDFs -> segments -> QA candidates -> QC -> train/val/test splits."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from banglafingpt.config import load_config  # noqa: E402
from banglafingpt.data.build_dataset import describe, split_pairs  # noqa: E402
from banglafingpt.data.pdf_extract import extract_corpus  # noqa: E402
from banglafingpt.data.qa_generate import generate_candidates  # noqa: E402
from banglafingpt.data.quality_control import (  # noqa: E402
    corpus_statistics,
    drop_near_duplicates,
    filter_grounded,
)
from banglafingpt.data.schema import validate_pairs  # noqa: E402
from banglafingpt.data.segment import segment_corpus  # noqa: E402
from banglafingpt.utils import read_json, set_seed, write_json, write_jsonl  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the BanglaFinGPT QA dataset")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--manifest", default="data/raw/manifest.json",
                        help="list of {path, domain, title, year, url}")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.data.seed)
    out_dir = Path(args.out_dir or cfg.data.processed_dir)

    manifest = read_json(args.manifest)
    docs = extract_corpus(manifest, cfg.data.raw_dir)
    segments = segment_corpus(docs, cfg.data.segment_min_words, cfg.data.segment_max_words)
    print(f"[1/4] {len(docs)} documents -> {len(segments)} segments")

    pairs = generate_candidates(segments, questions_per_segment=cfg.data.questions_per_segment)
    print(f"[2/4] {len(pairs)} candidate QA pairs")

    pairs, n_dup = drop_near_duplicates(pairs, cfg.data.near_duplicate_threshold)
    pairs, n_ungrounded = filter_grounded(pairs, segments)
    print(f"[3/4] dropped {n_dup} near-duplicates, {n_ungrounded} ungrounded -> {len(pairs)} kept")

    problems = validate_pairs(pairs)
    if problems:
        print(f"[warn] {len(problems)} validation problems; first 5: {problems[:5]}",
              file=sys.stderr)

    splits = split_pairs(pairs, cfg.data.train_size, cfg.data.val_size,
                         cfg.data.test_size, cfg.data.seed)
    write_jsonl(out_dir / "segments.jsonl", [s.to_dict() for s in segments])
    for name, group in splits.items():
        write_jsonl(out_dir / f"{name}.jsonl", [p.to_dict() for p in group])
    write_json(out_dir / "dataset_card.json", {
        "statistics": corpus_statistics(pairs),
        "splits": describe(splits),
    })
    print(f"[4/4] wrote splits to {out_dir}: "
          + ", ".join(f"{k}={len(v)}" for k, v in splits.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
