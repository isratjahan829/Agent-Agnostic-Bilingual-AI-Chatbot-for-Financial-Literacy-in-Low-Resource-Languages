# BanglaFinGPT

Reference implementation of **"Agent-Agnostic Bilingual AI Chatbot for Financial
Literacy in Low-Resource Languages"** — a retrieval-augmented, QLoRA-fine-tuned
system that answers Bangla and English questions about Bangladesh taxation, VAT,
customs and financial regulation, grounded in National Board of Revenue (NBR)
documents.

The system has three parts, matching the paper:

1. **Dataset construction** — NBR PDFs → structured text → 150–300 word segments
   → template + LLM question generation → deduplication and grounding checks →
   domain-balanced train/validation/test splits.
2. **Model** — BanglaLLaMA-3.2-3B loaded in 4-bit NF4 and fine-tuned with QLoRA
   adapters, served behind a hybrid dense + BM25 retriever.
3. **Hallucination mitigation** — a grounding system prompt, then a post-hoc
   filter that releases an answer only if it is semantically close to a retrieved
   chunk, lexically covered by it, and contains no number absent from the sources.
   Otherwise the user gets a fallback pointing to official NBR documentation.

## Quick start

```bash
pip install -r requirements.txt -r requirements-dev.txt
make test     # unit tests, no GPU or API key required
```

## Reproduction notebook

[`notebooks/BanglaFinGPT_reproduction.ipynb`](notebooks/BanglaFinGPT_reproduction.ipynb)
runs the whole study on the released dataset (`data/BanglaFinGPT_dataset.xlsx`,
10,412 QA pairs) and is **committed with its outputs**: dataset audit, leakage-free
splits, retrieval tuning, filter calibration, ablation, significance tests, error
analysis and figures.

It was executed on a **CPU-only machine**, so the QLoRA cell is gated behind
`RUN_TRAINING` and the generation scores in it come from the offline extractive
backend — a floor, not the paper's 90% EM. Open it on a GPU runtime, install
`requirements-train.txt`, set `RUN_TRAINING = True` and `RUN_FINETUNED_EVAL = True`,
and re-run from section 5 for the fine-tuned numbers.

```bash
pip install -r requirements-dev.txt jupyter openpyxl
jupyter lab notebooks/BanglaFinGPT_reproduction.ipynb
```

## Full pipeline

The released spreadsheet loads directly:

```python
from banglafingpt.data.load_xlsx import load_corpus
pairs, segments = load_corpus("data/BanglaFinGPT_dataset.xlsx")
```

To rebuild the corpus from the original regulatory PDFs instead:

```bash
# 1. Dataset: put the NBR PDFs in data/raw/ and describe them in a manifest
cat > data/raw/manifest.json <<'JSON'
[{"path": "vat_act_2012.pdf", "domain": "vat", "title": "VAT Act 2012", "year": 2024}]
JSON
make dataset                       # -> data/processed/{train,validation,test}.jsonl

# 2. Retrieval index over the segments
make index                         # -> artifacts/index/

# 3. QLoRA fine-tuning (CUDA GPU; ~6 GB VRAM for the 3B model in 4-bit)
pip install -r requirements-train.txt
make train                         # -> artifacts/checkpoints/banglafingpt/

# 4. Evaluation, ablation and significance tables
make eval
make ablation

# 5. Use it
make chat                          # interactive bilingual CLI
make serve                         # FastAPI on :8000, POST /ask
```

## Agent-agnostic backends

The RAG and grounding machinery is independent of who generates the text, so the
same benchmark runs against any backend by changing one config field:

```bash
python -m banglafingpt.eval.evaluate --backend local  --adapter artifacts/checkpoints/banglafingpt
python -m banglafingpt.eval.evaluate --backend openai    # gpt-4o      (OPENAI_API_KEY)
python -m banglafingpt.eval.evaluate --backend anthropic # claude      (ANTHROPIC_API_KEY)
python -m banglafingpt.eval.evaluate --backend gemini    # gemini      (GOOGLE_API_KEY)
python -m banglafingpt.eval.evaluate --backend echo      # extractive, offline
```

Add your own by implementing `Agent.generate` and calling `register_backend`.

## Layout

```
src/banglafingpt/
  config.py            typed config for every stage; configs/default.yaml holds paper values
  utils.py             Bangla-aware normalisation (NFKC, digit folding, danda), seeding, IO
  data/                PDF extraction, segmentation, QA generation, QC, splits, xlsx loader
  retrieval/           embedders, BM25 + dense index (FAISS when available), hybrid retriever
  models/              QLoRA setup, prompt construction (Eq. 17), completion-masked training
  hallucination/       grounding verdicts: cosine + keyword overlap + numeric support
  agents/              local / OpenAI / Anthropic / Gemini / echo backends behind one interface
  eval/                EM, F1, BLEU-4, ROUGE-L, METEOR, ablation, bootstrap CIs, McNemar, Fleiss' κ
  app/                 CLI and FastAPI service
  pipeline.py          retrieve → generate → verify → answer or refuse

notebooks/             executed reproduction notebook
scripts/               dataset build, index build, evaluation figures, revision experiments
tests/                 64 unit tests; fixtures are self-contained
```

## Revision experiments

`scripts/reviewer_experiments.py --all` reproduces the analyses added in revision:
the split-protocol audit (a row-level split puts 93.2% of test items' source
passages into training; grouping by passage makes it 0%), query-robustness under
user-style perturbations, the domain-balanced re-evaluation, and the hallucination
annotation sheets. `--list-blocked` names what needs a GPU, API keys or annotators.

See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for the mapping from each
paper claim to the code and command that produces it, and
[`docs/DATASET_CARD.md`](docs/DATASET_CARD.md) for dataset provenance.

## Limitations

- Answers reflect the regulations that were indexed. Rules change with each
  budget; a stale index produces confidently outdated answers, which the paper's
  error analysis identifies as the largest error class (38%).
- The filter trades recall for precision: roughly 12% of queries are refused
  rather than answered.
- Output is informational and is not professional tax advice.

## Citation

```bibtex
@article{zafor2026banglafingpt,
  title  = {Agent-Agnostic Bilingual AI Chatbot for Financial Literacy in Low-Resource Languages},
  author = {Zafor, Md. Abu and Jahan, Israt and Rafi, Md. Saleh and Hossain, Farzia
            and Kaisar, Shahriar and Chowdhury, Abdullahi},
  year   = {2026}
}
```

## License

MIT for the code. The NBR source documents remain under their own terms.
