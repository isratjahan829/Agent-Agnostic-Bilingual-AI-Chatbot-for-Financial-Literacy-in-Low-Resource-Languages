#!/usr/bin/env python3
"""Build the retrieval index from processed segments."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from banglafingpt.config import load_config  # noqa: E402
from banglafingpt.data.schema import Segment  # noqa: E402
from banglafingpt.retrieval.embedder import load_embedder  # noqa: E402
from banglafingpt.retrieval.index import DocumentIndex  # noqa: E402
from banglafingpt.retrieval.retriever import chunks_from_segments  # noqa: E402
from banglafingpt.utils import read_jsonl, set_seed  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the BanglaFinGPT retrieval index")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--segments", default="data/processed/segments.jsonl")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--hashing", action="store_true",
                        help="use the dependency-free hashing embedder")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    segments = [Segment(**{k: v for k, v in row.items() if k in Segment.__dataclass_fields__})
                for row in read_jsonl(args.segments)]
    if not segments:
        print(f"[error] no segments found in {args.segments}", file=sys.stderr)
        return 1

    if args.hashing:
        from banglafingpt.retrieval.embedder import HashingEmbedder

        embedder = HashingEmbedder()
    else:
        embedder = load_embedder(cfg.retrieval.embedding_model,
                                 normalize=cfg.retrieval.normalize_embeddings)

    out_dir = args.out_dir or cfg.retrieval.index_dir
    index = DocumentIndex(embedder).build(chunks_from_segments(segments))
    index.save(out_dir)
    print(f"[ok] indexed {len(index)} chunks with {type(embedder).__name__} -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
