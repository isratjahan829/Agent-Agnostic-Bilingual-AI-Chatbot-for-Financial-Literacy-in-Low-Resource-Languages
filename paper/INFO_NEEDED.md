# What is still missing — reviewer-scope only

The manuscript now answers the reviewers' comments and nothing beyond them. Optional
experiments that no reviewer requested have been removed: the "+ RAG context" rows in
Table 5, the query-robustness subsection, and the baselines' hallucination-rate row in
Table 13.

**No GPU run is required for most of it.** 9 markers remain (2 more are in a preamble
comment). Seven are facts from your existing records; two are decisions that depend on
how the reported results were produced.

| # | Marker | Section | What to supply | Reviewer comment |
|---|---|---|---|---|
| 1 | annotator background | 3.6.1 | Who the three annotators were — background, not names. e.g. "two graduate students in computer science and one practising tax practitioner". | R1 #1b, asked directly |
| 2 | Fleiss' κ | 3.6.1 | The inter-annotator agreement from the study you already ran. | R1 #1b, asked directly |
| 3 | unanimous % | 3.6.1 | Share of the 200 items all three agreed on. | R1 #1b |
| 4 | human-evaluation κ | 3.2 | One value. The text says 0.72, Figure 2(b) says 0.68–0.71 — pick one, or say the figure is per-dimension and the text the mean. | R1 #5c, number consistency |
| 5–6 | template vs GPT-4 split | 2.2 | The percentage of retained questions from each generator. **"Not recorded" is an acceptable answer** — write that rather than estimate. | R1 #3b |
| 7 | wall-clock training time | 2.4 | How long the reported fine-tuning run took. | R2 (b), implementation details |

## The one marker that is a decision

Section 2.2.1 carries a marker asking for **the partition unit**. Tables 2 and 3 and the
reported N = 2,000 are now mutually consistent and consistent with the released dataset,
so this is the only thing left that depends on how the experiment was run:

- **Questions grouped by source passage** → write the sentence offered as option (a) in
  the marker. Nothing needs re-running.
- **The 10,412 pairs partitioned individually** → the scores are inflated and have to be
  regenerated before the sentence can be written. The corpus has 1,451 distinct passages,
  so a question-level partition puts 93.2% of test items' source passages into training.

## If your original annotation records are lost

Only then do you need to redo the study. `scripts/reviewer_experiments.py --all`
regenerates blind sheets for three annotators, and:

```python
from banglafingpt.eval.annotation import load_annotations, score_annotations
score_annotations(load_annotations(sheets["sheets"]))   # rates + Fleiss' kappa
```

## One question that decides whether anything must be re-run

**How was the original 90% EM experiment split?**

- **By passage / context** → nothing to re-run. Section 2.2.1 describes the method, which
  is what Reviewer 2 (a) asked for.
- **By row** (a plain shuffle of the 10,412 pairs) → the scores are inflated and must be
  regenerated. 10,412 QA pairs come from 1,451 distinct passages, so a row-level split
  puts 93.2% of test items' source passages into training. In that case run
  `notebooks/BanglaFinGPT_gpu_run.ipynb` and replace Tables 5, 7, 8 and 12.

## Outside the `.tex`

| What | Reviewer comment |
|---|---|
| `ref.bib`: a DOI on every entry that has one, one consistent preprint format, and `Bisk2020` completed with venue, pages and DOI | R1 #5b |
| Rename the statistical-significance figure to `08_statistical_significance.png` — the submitted path had spaces and parentheses, which breaks `\includegraphics` | R1 #5d |

## Before submitting

```bash
grep -o 'needsnum{' main.tex | wc -l    # must be 0
```

Then set `\revisionfalse` in the preamble for a clean copy, and compile:
`pdflatex main` → `biber main` → `pdflatex main` ×2. Fill the same facts into the
`[AUTHORS: …]` brackets in the response letter so the two documents agree.
