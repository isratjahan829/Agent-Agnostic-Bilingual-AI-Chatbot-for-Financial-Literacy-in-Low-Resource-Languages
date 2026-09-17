# What changed in `main.tex`, by reviewer comment

Compile: `pdflatex main` → `biber main` → `pdflatex main` ×2.
Revision text prints **blue** (`\revised{}`); values you must still regenerate print
**red** (`\needsnum{}`). There are **34** red placeholders — search `needsnum` and clear
every one before submitting. Set `\revisionfalse` in the preamble for a clean copy.

## Reviewer 1

| Comment | Section touched | Change |
|---|---|---|
| 1a — 94% misleading | Abstract, 3.6, 4 | Absolute reduction (8.0 pp) stated first, relative second; "one fabricated answer per 12.5 queries" added; "94%" no longer a headline |
| 1b — annotation method | new **3.6.1** | Sampling frame, annotator blinding, category definitions, Fleiss' κ, released code |
| 2 — GPT-4o unfair | 3.1 (Table 5), 3.8, 3.11 | Three "+ RAG context" rows added to Table 5; scale claim withdrawn in 3.11 and the Conclusion; parameter-count and training-volume rows deleted from Table 13 |
| 2 — perplexity conflation | 2.3 | Sentence tying GPT-4's Bengali perplexity to accuracy removed; perplexity now explicitly "not a measure of answer correctness" |
| 3a — single-source corpus | new **3.5.3**, new **3.11.1** | Query-robustness table under five user-style perturbations; Limitations subsection |
| 3b — GPT-4 question bias | 2.2 | Generator split stated, verbatim-answer + numeric checks described as the mitigation |
| 3c — domain imbalance | 3.4, 3.11 | Length-based explanation replaced (see below); the ordering is shown to contradict the imbalance hypothesis, and a controlled test is named as future work |
| 4a — math decoration | 2.3.x | Only NF4, cosine schedule, RAG marginalisation, prompt construction and the hallucination predicate kept |
| 4b — KL 12.4 nats | 2.3.1 | **Equation and figure removed**; replaced by a plain statement of the distribution gap |
| 4c — Chinchilla | old 2.3.2 | **Removed**; subsection retitled "Model Selection under Deployment Constraints" with the real 6 GB reason |
| 4d — Eqs. 11–15 | 2.3.3, 2.3.4 | **Removed**; cited to Dettmers et al. instead of restated |
| 5a — grammar | 1 | Three named sentences rewritten; "billions of people" quantifier softened (cite Findex or delete) |
| 5b — references | `ref.bib` | **Not done in the .tex** — see below |
| 5c — 45.20/45.21/45.2 | Abstract, 3.1, 3.8 | 45.2% everywhere; percentages to one decimal, F1 to two |
| 5d — figures | all | `\Description{}` removed (an `acmart` command — it does not compile under `article`); figure paths fixed |

## Reviewer 2

| Comment | Section | Change |
|---|---|---|
| (a) dataset splitting | **2.2.1** | Passage-grouped split described; 93.2% vs 0% leakage figure stated |
| (b) implementation details | **2.4**, Table 6 | Full hyperparameter table; seed; context-in-training-prompts rationale |
| (c) fair baselines | 3.1 | Same as R1-C2 |
| (d) statistics | 3.7 | Bootstrap CIs, McNemar, paired bootstrap named; refusals scored as wrong |
| (d) hallucination | 3.6.1 | Protocol as above |

## Compile errors fixed in your original source

1. **`\Description{...}`** on two figures — an `acmart` command, undefined in `article`.
   This alone stops the document compiling. Removed.
2. **`\graphicspath{ {./figures/} }` vs `\includegraphics{Fig/...}`** — the graphics path
   and the include paths disagreed. Now `\graphicspath{ {./Fig/} {./figures/} }` and the
   includes are bare filenames.
3. **`{BanglaFinGPT/Fig/Figure 8 (2).png}`** — spaces and parentheses in a path break
   `\includegraphics`. Referenced as `08_statistical_significance.png`;
   **rename the file on Overleaf to match.**
4. **`[[42.3–47.9]`** in Table 12 — stray bracket and an en-dash inside a numeric range.
   All CIs now `[42.3, 47.9]`.
5. Added `\usepackage{xcolor}` (needed for the revision markup) and `booktabs`.

## Two findings that change the paper, not just its wording

**1. Table 2 does not match the released dataset.** Recomputing from
`data/BanglaFinGPT_dataset.xlsx`:

| Domain | Pairs (Table 2) | Pairs (actual) | Avg answer words (Table 2) | Avg answer words (actual) |
|---|---|---|---|---|
| Taxation | 2,823 | 2,800 | 28.3 | 25.9 |
| VAT | 2,641 | 2,620 | 22.1 | 26.5 |
| Customs | 3,475 | 3,448 | 31.7 | 25.1 |
| Finance | 1,473 | 1,544 | 25.8 | 22.2 |
| **Total** | **10,412** | **10,412** | 26.8 | 25.2 |

Totals agree; every per-domain figure does not. Table 3's splits derive from Table 2 and
inherit the same counts. Either the tables were computed from an earlier version of the
corpus, or the Kaggle release is not the corpus you ran on — the second would be worse,
since the release is what reviewers download. `\needsnum` note added to Table 2.

**2. The customs explanation in Section 3.4 does not survive that check.** The submitted
text attributed the customs gap to longer answers ("31.7 words vs. 22.1 for VAT"). In the
released dataset VAT answers are *longer* than customs answers (26.5 vs 25.1), so the
explanation is backwards. Replaced with a multi-condition-reasoning explanation, which
the error analysis in Table 9 already supports.

**3. The domain-imbalance analysis was rebuilt twice, and the surviving claim is
narrow.** The first version re-balanced the *test* set, which cannot answer a question
about *training* imbalance — per-domain accuracy does not depend on the other domains'
test sizes. The second version claimed the ordering contradicted the imbalance
hypothesis (customs largest and weakest). That claim did not survive either: once the
hallucination filter's cosine threshold was calibrated per encoder rather than
hard-coded, customs (F1 0.193) and general finance (0.176) fell inside each other's
confidence intervals, and their ordering flips with the retrieval encoder.

What is reported now is what holds in every configuration tested: corpus share does not
track accuracy. The ordering by accuracy is taxation, VAT, customs, general finance;
the ordering by corpus share is customs, taxation, VAT, general finance. The largest
domain is not the most accurate and the most accurate is not the largest. A controlled
test needs retraining on a domain-balanced training set, and is named as future work
rather than claimed.

**4. Retrieval numbers now come from a corpus-fitted encoder.** The dense half was
hashed n-grams; it is now TF-IDF over character n-grams reduced by SVD and fitted on the
corpus, which lifts dense-only recall@5 from 0.530 to 0.598 on validation. BM25 alone
still reaches 0.802, so the tuned mix stays lexical — the benchmark questions reuse the
exact regulatory terminology of their source passages. That is a property of the corpus,
not of a weak encoder. The pretrained multilingual encoder the manuscript specifies could
not be evaluated: model weights are unreachable from the environment these runs were
made in.

## Inconsistencies to settle yourself

- **Fleiss' κ**: the text says 0.72, Figure 2(b)'s caption says 0.68–0.71. One value,
  or state clearly that the caption reports per-dimension κ and the text the mean.
- **Table 9 vs Table 10**: error counts total 100 in both, so "Count" and "(%)" are the
  same column. Fine, but say "N = 100 errors" (now added) or convert one to raw counts.
- **Table 13 latency**: 450 ms for your system against 2000 ms for GPT-4o compares a
  local model with an API round-trip. The caption now says so; consider dropping the
  row if you cannot measure both the same way.
- **`Training Data Size (hrs)`** was measured in *hours* for text corpora. Row removed.
- **Section 2.3** originally said "best characterized these five major points" and then
  gave one. Rewritten to a single sentence.

## Reference list (R1-5b) — still to do

Not fixable from the `.tex`; it needs `ref.bib`. Add a DOI to every entry that has one,
use a consistent preprint format (`arXiv:XXXX.XXXXX`), and complete entry `Bisk2020`
("Experience grounds language") with venue, pages and DOI. Also check `Pakray2025` and
`karpukhin2020` exist in your `.bib` — the original text cited them inconsistently
(one citation was empty in the submitted draft).
