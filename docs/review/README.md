# Reviewer response package

| File | What it is |
|---|---|
| `Response_to_Reviewers_BanglaFinGPT_v2.docx` | The response letter, filled in, in the structure of the group's earlier BDMA letter. Open, complete the `[AUTHORS: …]` brackets, delete the two working notes. |
| `../REVIEWER_RESPONSE.md` | The same content in markdown, easier to diff and review. |
| `reviewer_experiments_results.json` | Every measured number quoted in the letter. |
| `annotation_instructions.txt` | The instruction sheet given to hallucination annotators. |
| `content.json`, `make_response.js` | Regenerate the .docx after editing (`node make_response.js content.json out.docx`). |

## Reproducing the numbers

```bash
python scripts/reviewer_experiments.py --all          # writes artifacts/reviewer/
python scripts/reviewer_experiments.py --list-blocked # what needs a GPU, API keys or annotators
```

## Which comment is answered by which experiment

| Reviewer comment | Experiment | Key result |
|---|---|---|
| R1-C1a — the 94% claim | `experiment_reduction_framing` | 8.5% → 0.5% is 8.0 pp absolute, 94.1% relative, one fabrication prevented per 12.5 queries |
| R1-C1b — annotation methodology | `experiment_annotation_protocol` | 200 items stratified by domain from the test split; blind sheets for three annotators; Fleiss' κ scorer |
| R1-C2 — unfair GPT-4o comparison | `experiment_baseline_parity` | Holding the generator fixed, retrieval alone moves F1 0.000 → 0.290; the six baseline runs are named |
| R1-C3 — domain imbalance | `experiment_domain_balance` | Customs stays weakest on a balanced subsample (F1 0.183 → 0.179), so imbalance is not the cause |
| R1-C3 — real-world queries | `experiment_robustness` | recall@5 0.768; −3.4% under a typo, −3.6% with politeness, −6.0% keyword-only, 0% for punctuation and digit script |
| R2-C1a — dataset splitting | `experiment_split_protocol` | A row-level split puts 93.2% of test items' source passages in training; passage grouping makes it 0% |

## Still blocked here

A CUDA GPU (fine-tuned scores), API keys (GPT-4o / Claude / Gemini rows), and the three
annotators (hallucination labels, human evaluation). The letter marks every number that
depends on these as `[AUTHORS: insert]` rather than guessing it.
