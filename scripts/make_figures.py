#!/usr/bin/env python3
"""Regenerate the paper's result figures from the JSON written by the eval runs."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from banglafingpt.config import load_config  # noqa: E402
from banglafingpt.utils import read_json  # noqa: E402

METRICS = [("em_pct", "Exact Match (%)"), ("f1", "F1"), ("bleu4", "BLEU-4"),
           ("rouge_l", "ROUGE-L")]


def _plt():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise SystemExit("matplotlib is required: pip install matplotlib") from exc


def ablation_figure(ablation: dict, out_dir: Path) -> Path:
    """Figure 5: EM and F1 progression across ablation configurations."""
    plt = _plt()
    rows = ablation["table"]
    names = [r["configuration"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key, label in ((axes[0], "em_pct", "Exact Match (%)"), (axes[1], "f1", "F1")):
        ax.bar(names, [r[key] for r in rows], color="#3b6ea5")
        ax.set_ylabel(label)
        ax.set_title(f"{label} by configuration")
        ax.tick_params(axis="x", rotation=30)
        for i, row in enumerate(rows):
            ax.text(i, row[key], f"{row[key]}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    path = out_dir / "ablation.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def significance_figure(ablation: dict, out_dir: Path) -> Path:
    """Figure 8: EM with 95% bootstrap confidence intervals."""
    plt = _plt()
    rows = ablation["significance"]
    names = [r["system"] for r in rows]
    means = [r["mean"] for r in rows]
    lows = [m - r["ci95"][0] for m, r in zip(means, rows, strict=True)]
    highs = [r["ci95"][1] - m for m, r in zip(means, rows, strict=True)]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(names, means, yerr=[lows, highs], capsize=5, color="#4c9a6b")
    ax.set_ylabel("Exact Match (%)")
    ax.set_title("Performance with 95% bootstrap confidence intervals")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    path = out_dir / "significance.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def domain_figure(metrics: dict, out_dir: Path) -> Path:
    """Figure 6: per-domain EM and F1."""
    plt = _plt()
    by_domain = metrics["by_domain"]
    domains = sorted(by_domain)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(domains, [by_domain[d]["em_pct"] for d in domains], color="#3b6ea5")
    axes[0].set_ylabel("Exact Match (%)")
    axes[1].bar(domains, [by_domain[d]["f1"] for d in domains], color="#a5643b")
    axes[1].set_ylabel("F1")
    for ax, title in zip(axes, ("EM by domain", "F1 by domain"), strict=True):
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    path = out_dir / "by_domain.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def error_figure(errors: dict, out_dir: Path) -> Path:
    """Figure 7: error type distribution."""
    plt = _plt()
    rows = [r for r in errors["by_type"] if r["count"]]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh([r["error_type"] for r in rows], [r["pct"] for r in rows], color="#a53b52")
    ax.set_xlabel("Share of errors (%)")
    ax.set_title(f"Error types (n={errors['total_errors']})")
    fig.tight_layout()
    path = out_dir / "error_types.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Regenerate result figures")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--metrics", default="banglafingpt_metrics.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    results = Path(args.results_dir or cfg.results_dir)
    out_dir = Path(cfg.figures_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written = []
    ablation_path = results / "ablation.json"
    if ablation_path.exists():
        ablation = read_json(ablation_path)
        written += [ablation_figure(ablation, out_dir), significance_figure(ablation, out_dir)]
        last = ablation["reports"].get("banglafingpt")
        if last:
            written.append(domain_figure(last, out_dir))
    metrics_path = results / args.metrics
    if metrics_path.exists():
        written.append(domain_figure(read_json(metrics_path), out_dir))

    if not written:
        print(f"[error] no result JSON found under {results}; run `make eval` or "
              "`make ablation` first", file=sys.stderr)
        return 1
    for path in written:
        print(f"[ok] {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
