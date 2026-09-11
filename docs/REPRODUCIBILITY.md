# Reproducibility map

Every quantitative claim in the paper, the code that computes it, and the command
that regenerates it. Run everything from the repository root.

## Environment

| Item | Value |
|---|---|
| Python | ≥ 3.10 |
| Fine-tuning hardware | 1× NVIDIA RTX 4050 (6 GB) for the 3B model in 4-bit NF4; A100/T4 for full runs |
| Seeds | `seed: 42` in `configs/default.yaml`, applied via `utils.set_seed` (stdlib, numpy, torch) |
| Determinism caveat | Metrics and splits are deterministic. GPU generation is not bit-exact across driver/kernel versions; EM varies by ≲0.3 points between runs. |

## Paper → code

| Paper element | Code | Command |
|---|---|---|
| Whole study, executed end-to-end on the released dataset | `notebooks/BanglaFinGPT_reproduction.ipynb` | open the notebook (outputs committed) |
| Released `.xlsx` → pipeline records | `data/load_xlsx.py::load_corpus` | used by the notebook |
| Sec. 3.2 — PDF → structured text | `data/pdf_extract.py` | `make dataset` |
| Sec. 3.2 — 150–300 word segments | `data/segment.py::segment_document` | `make dataset` |
| Sec. 3.2 — template + LLM QA generation | `data/qa_generate.py` | `make dataset` |
| Sec. 3.2 — dedup + grounding QC | `data/quality_control.py` | `make dataset` |
| Table 2 — dataset composition | `data/quality_control.py::corpus_statistics` | `data/processed/dataset_card.json` |
| Table 3 — domain-balanced splits | `data/build_dataset.py::split_pairs` | `make dataset` |
| Eq. 9 — NF4 quantisation | `models/qlora.py::build_bnb_config` | `make train` |
| Sec. 3.3.3 — LoRA adapters | `models/qlora.py::build_lora_config` | `make train` |
| Eq. 13 — cosine LR schedule, AdamW | `models/train.py` (`lr_scheduler_type="cosine"`, `optim="paged_adamw_8bit"`) | `make train` |
| Eq. 16 — RAG marginalisation | `retrieval/retriever.py::HybridRetriever.retrieve` | `make index` |
| Eq. 17 — prompt assembly | `models/prompts.py::build_prompt` | — |
| Eq. 19 — hallucination predicate | `hallucination/filters.py::HallucinationFilter.verify` | — |
| Table 5 — baseline comparison | `eval/evaluate.py` | `python -m banglafingpt.eval.evaluate --backend {local,openai,anthropic,gemini} --tag <name>` |
| Table 6 — human evaluation | `eval/human_eval.py::summarize_ratings` | `python -c "..."` over your ratings file |
| Sec. 4.2 — Fleiss' κ | `eval/human_eval.py::fleiss_kappa` | as above |
| Table 7 — ablation | `eval/ablation.py` | `make ablation` |
| Table 8 — per-domain results | `eval/evaluate.py::evaluate` (`by_domain`) | `make eval` |
| Tables 9, 10, 14 — error analysis | `eval/error_analysis.py::error_report` | `make ablation` |
| Table 11 — hallucination audit | `eval/error_analysis.py::hallucination_report` | `make ablation` |
| Table 12 — 95% CIs and p-values | `eval/significance.py::significance_table` | `make ablation` |
| Table 13 — latency | `pipeline.Answer.latency_ms`, aggregated in `evaluate` | `make eval` |
| Split-protocol audit (revision, Sec. 3.2.1) | `data/build_dataset.py::leakage_report` | `python scripts/reviewer_experiments.py --all` |
| Query robustness (revision, Sec. 4.5) | `eval/robustness.py` | `python scripts/reviewer_experiments.py --all` |
| Hallucination annotation protocol (revision, Sec. 4.6.1) | `eval/annotation.py` | `python scripts/reviewer_experiments.py --all` |
| Domain-balanced re-evaluation (revision, Sec. 4.4) | `scripts/reviewer_experiments.py` | `python scripts/reviewer_experiments.py --all` |

## Metric definitions

All metrics run over `utils.normalize_text`: NFKC normalisation, Bengali digits
folded to ASCII, danda and punctuation stripped, whitespace collapsed. Without
this, exact match measures orthography rather than correctness.

- **EM** — normalised string equality.
- **F1** — SQuAD-style token overlap.
- **BLEU-4** — sentence BLEU, add-1 smoothing on n > 1, brevity penalty.
- **ROUGE-L** — LCS-based F-measure, β = 1.2.
- **METEOR** — exact + 4-character stem matching with a fragmentation penalty
  (no Bangla WordNet exists, so the synonym stage is omitted).

Refused queries are scored as wrong, not dropped. The filter buys precision with
recall, and dropping refusals would hide that cost.

## Statistical procedures

- **CIs** — percentile bootstrap over per-example scores, 10,000 resamples, seed 42.
- **EM comparisons** — McNemar on discordant pairs (binomial exact below 25
  discordants, χ² with continuity correction above).
- **Continuous metrics** — two-sided paired bootstrap, 10,000 resamples.
- **Inter-rater agreement** — Fleiss' κ over the 1–5 rating categories, interpreted
  with the Landis & Koch bands.

## What the committed notebook run does and does not show

`notebooks/BanglaFinGPT_reproduction.ipynb` was executed on a CPU-only machine. Its
dataset audit, splits, retrieval scores, filter behaviour, metrics, significance tests
and error analysis are real measurements on all 10,412 pairs. Its **generation** scores
come from the offline extractive backend, because QLoRA fine-tuning needs a GPU — they
are a floor, not a reproduction of the paper's 90% EM. Two consequences worth naming:

- Dense retrieval uses a hashed n-gram fallback rather than multilingual-E5, which is why
  the tuned `hybrid_alpha` leans on BM25 in that run.
- The hallucination audit starts from a near-zero fabrication rate, because an extractive
  backend copies from its sources by construction. The paper's 8.5% → 0.5% is measured on
  a generative model.

## Known deviations from the paper text

- The paper cites `max_steps = 3 × 10⁴` (Sec. 3.3.4) alongside 3 epochs. With
  7,412 training pairs and an effective batch of 32, 3 epochs is ≈700 steps;
  `max_steps` therefore acts as an upper bound and early stopping on validation
  loss (patience 3) terminates first.
- Table 12 reports `+ RAG only` at 52.1% EM while the surrounding text at one
  point describes fine-tuning as being "boosted" by RAG to that figure. The code
  treats 52.1% as RAG-only over the base model, consistent with Tables 5 and 7.
- Retrieval uses hybrid dense + BM25 scoring. The paper describes dense retrieval;
  BM25 is included because exact strings (section numbers, HS codes) are what the
  customs error analysis shows dense retrieval missing. Set
  `retrieval.hybrid_alpha: 1.0` for dense-only behaviour.
