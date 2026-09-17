#!/usr/bin/env python3
"""Assemble the artifact a reviewer or editor may ask for, as one zip.

Reviewers who ask for "the data" usually want four things: the corpus and the
exact splits used, the per-example predictions behind every reported score, the
code and configuration that produced them, and enough environment detail to tell
whether a rerun should match. This collects all of that from whatever runs have
actually been made, and writes a MANIFEST that states plainly what is present and
what is still missing, so nothing is implied that was not measured.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from banglafingpt.config import load_config  # noqa: E402
from banglafingpt.data.build_dataset import leakage_report, split_pairs  # noqa: E402
from banglafingpt.data.load_xlsx import load_corpus  # noqa: E402
from banglafingpt.utils import set_seed, write_json, write_jsonl  # noqa: E402

SEED = 42


def git_revision() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def build(out_dir: Path, dataset: Path) -> dict:
    set_seed(SEED)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(ROOT / "configs" / "default.yaml")

    pairs, segments = load_corpus(dataset)
    splits = split_pairs(pairs, 7412, 1000, 2000, seed=SEED)

    # 1. The exact splits, so a reviewer can verify any per-domain or per-split claim.
    for name, group in splits.items():
        write_jsonl(out_dir / "splits" / f"{name}.jsonl", [p.to_dict() for p in group])
    write_jsonl(out_dir / "splits" / "passages.jsonl", [s.to_dict() for s in segments])

    # 2. The leakage audit, which is the claim most worth checking.
    write_json(out_dir / "splits" / "leakage_audit.json", {
        "passage_grouped": leakage_report(splits),
        "note": ("10,412 QA pairs are derived from "
                 f"{len({s.doc_id for s in segments})} distinct passages, so splitting "
                 "by row would place a test question's own source passage in training."),
    })

    # 3. Whatever results exist. Nothing is invented for results that were never run.
    present, missing = [], []
    for label, source in [
        ("cpu_notebook_results", ROOT / "artifacts" / "results" / "notebook_results.json"),
        ("revision_experiments", ROOT / "artifacts" / "reviewer" / "reviewer_experiments.json"),
        ("gpu_run_results", ROOT / "artifacts" / "gpu_results.json"),
    ]:
        if source.exists():
            (out_dir / "results").mkdir(parents=True, exist_ok=True)
            (out_dir / "results" / f"{label}.json").write_text(
                source.read_text(encoding="utf-8"), encoding="utf-8")
            present.append(label)
        else:
            missing.append(label)

    for pattern, target in [("artifacts/results/*predictions*.jsonl", "predictions"),
                            ("artifacts/reviewer/hallucination_annotation/*", "annotation")]:
        for path in sorted(ROOT.glob(pattern)):
            destination = out_dir / target / path.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(path.read_bytes())

    # 4. Configuration and environment.
    write_json(out_dir / "configuration.json", {
        "config": cfg.to_dict(),
        "seed": SEED,
        "git_revision": git_revision(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })

    manifest = {
        "dataset": {"path": dataset.name, "qa_pairs": len(pairs),
                    "passages": len({s.doc_id for s in segments}),
                    "chunks": len(segments)},
        "splits": {name: len(group) for name, group in splits.items()},
        "results_present": present,
        "results_missing": missing,
        "still_to_measure": _pending(missing),
    }
    write_json(out_dir / "MANIFEST.json", manifest)
    (out_dir / "README.txt").write_text(_readme(manifest), encoding="utf-8")
    return manifest


GPU_PENDING = [
    "Fine-tuned model scores (Tables 5, 7, 8, 12)"
    " — run notebooks/BanglaFinGPT_gpu_run.ipynb",
    "Hosted baselines with and without retrieval (Table 5)"
    " — same notebook, API keys required",
    "Baseline hallucination rates (Table 13) — same notebook",
]
HUMAN_PENDING = [
    "Hallucination annotation labels and Fleiss' kappa (Table 11, Section 3.6.1)"
    " — three annotators",
    "Human evaluation (Table 6) — three domain experts",
]


def _pending(missing: list[str]) -> list[str]:
    """What has not been measured yet, so the package never implies more than it holds."""
    pending = list(HUMAN_PENDING)
    if "gpu_run_results" in missing:
        pending = GPU_PENDING + pending
    return pending


def _readme(manifest: dict) -> str:
    lines = [
        "BanglaFinGPT — reproducibility package",
        "=" * 38,
        "",
        "Contents",
        "  splits/       the exact train/validation/test splits used, plus the source",
        "                passages and the leakage audit",
        "  results/      the measured results, one JSON per run",
        "  predictions/  per-example predictions behind every reported score",
        "  annotation/   the blind sheets and instructions for the hallucination study",
        "  configuration.json  full configuration, seed, git revision, environment",
        "",
        f"Dataset: {manifest['dataset']['qa_pairs']} QA pairs over "
        f"{manifest['dataset']['passages']} distinct passages.",
        "Splits: " + ", ".join(f"{k}={v}" for k, v in manifest["splits"].items()) + ".",
        "",
        "Splitting is by source passage, not by row: several questions derive from each",
        "passage, so a row-level split would put a test question's own source passage in",
        "training. leakage_audit.json quantifies both.",
        "",
        "Not yet measured (listed so nothing here is mistaken for a completed result):",
    ]
    lines += [f"  - {item}" for item in manifest["still_to_measure"]]
    lines += ["", "Code: see the repository at the git revision in configuration.json."]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the reviewer package")
    parser.add_argument("--dataset", default=str(ROOT / "data" / "BanglaFinGPT_dataset.xlsx"))
    parser.add_argument("--out-dir", default=str(ROOT / "artifacts" / "reviewer_package"))
    parser.add_argument("--zip", default=str(
        ROOT / "artifacts" / "BanglaFinGPT_reviewer_package.zip"))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    manifest = build(out_dir, Path(args.dataset))

    archive = Path(args.zip)
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(out_dir.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(out_dir))

    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"\n[ok] {archive}  ({archive.stat().st_size / 1e6:.1f} MB)")
    if manifest["results_missing"]:
        print(f"[note] not yet run: {', '.join(manifest['results_missing'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
