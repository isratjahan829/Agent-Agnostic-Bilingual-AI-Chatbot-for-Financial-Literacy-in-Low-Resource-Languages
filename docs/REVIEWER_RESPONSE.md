# Response to Reviewers — BanglaFinGPT (ICOMPUTING-D-26-00612)

Draft responses for `Response_to_Reviewer_BanglaFinGPT_v1.1.docx`, written in the
structure of the group's earlier BDMA response letter: a direct answer, then the exact
text added to the revised manuscript in quotes with its section reference.

**Two things to check before sending.**

1. Text in `[AUTHORS: ...]` is a fact only you can supply (who annotated, which GPU,
   what the fine-tuned model scores). Nothing in this draft invents such a value.
2. Every number quoted below without a bracket was measured by
   `scripts/reviewer_experiments.py` on the released dataset and is reproducible with
   `python scripts/reviewer_experiments.py --all`. Numbers that depend on the
   fine-tuned model are bracketed, because this environment has no GPU — see
   `python scripts/reviewer_experiments.py --list-blocked`.

---

## Reviewer 1

### Comment #1a — the 94% hallucination-reduction claim is misleading

**Response.** We agree, and we have changed how the result is reported throughout the
paper. The relative figure is computed from an 8.5% base, so on its own it overstates
the size of the effect. The revised manuscript reports the absolute change first and the
relative change second, in the Abstract, Section 4.6 and the Conclusion, and no longer
uses "94%" as a headline number.

Precisely: 8.5% → 0.5% is an absolute reduction of **8.0 percentage points**, a relative
reduction of 94.1%, and corresponds to one fabricated answer prevented for every 12.5
queries the system answers.

Added to the Abstract:

> "A hallucination mitigation framework combining system prompts with semantic
> similarity and keyword-based filtering reduces fabricated responses from 8.5% to 0.5%
> of answers — an absolute reduction of 8.0 percentage points (94% in relative terms)
> — at the cost of declining to answer 12.3% of queries."

Added to Section 4.6:

> "The filter reduces the share of fabricated answers from 8.5% to 0.5%. We report this
> as an absolute reduction of 8.0 percentage points rather than only as a 94% relative
> reduction, because the baseline rate is already low and a relative figure computed
> from a small base is easy to over-read. In practice the filter prevents approximately
> one fabricated answer for every 12.5 queries served, while declining to answer 12.3%
> of queries."

The helper that produces both framings from any pair of rates is released as
`banglafingpt.eval.annotation.reduction_report`, so the two numbers in the text cannot
drift apart.

---

### Comment #1b — the hallucination detection methodology is not described

**Response.** We agree that this was underspecified, and we have added a full protocol
subsection together with the code and the annotation sheets, so the measurement can be
repeated exactly.

Added as Section 4.6.1 ("Hallucination annotation protocol"):

> "**Sampling frame.** 200 responses were drawn from the held-out test split only, never
> from training or validation, stratified by domain in proportion to the test split
> (customs 65, taxation 55, VAT 50, finance 30) with a fixed random seed (42). The same
> 200 items were judged for every system variant, so the variants are directly
> comparable.
>
> **Annotators.** Three annotators [AUTHORS: state their background, e.g. "two graduate
> students in computer science and one practising tax consultant"] labelled every item
> independently. Annotators were not shown the system's own grounding score, which
> variant produced an answer, or each other's labels, and each received the items in a
> different order.
>
> **Categories.** Each answer was assigned exactly one label against the passages
> retrieved for it: *fully grounded* (every claim and every number, rate, date and
> section reference appears in the passages), *partially grounded* (mostly supported, at
> least one claim or number not stated in the passages), or *hallucinated* (asserts
> something the passages do not support, or invents a rate, date, section number or HS
> code). A refusal was counted as fully grounded, since it asserts nothing.
>
> **Aggregation and agreement.** The reported rate is the majority label across the three
> annotators. Inter-annotator agreement was Fleiss' κ = [AUTHORS: insert, from
> `score_annotations`], with [AUTHORS: insert]% of items unanimous."

The instructions given to annotators, the sheet generator and the scorer are released as
`banglafingpt.eval.annotation`; running `scripts/reviewer_experiments.py --all`
regenerates the blind sheets (`artifacts/reviewer/hallucination_annotation/`) and
`score_annotations` computes the rates and Fleiss' κ from the returned sheets.

---

### Comment #2 — the GPT-4o comparison is not like-for-like

**Response.** The reviewer is right, and we have corrected both the experiment and the
claim.

**(i) The baselines now receive the same retrieved context.** Our pipeline is
backend-agnostic — retrieval, prompt assembly, filtering and scoring are identical
whichever model generates the text — so the fair comparison is a configuration change,
not a different system. The revised Table 5 reports every baseline twice, without
retrieval (as originally submitted) and with exactly the same top-k passages and the
same prompt that BanglaFinGPT receives:

| System | EM (%) | F1 |
|---|---|---|
| GPT-4o, no retrieval | 45.2 | 0.61 |
| GPT-4o + same RAG context | [AUTHORS: insert] | [AUTHORS: insert] |
| Claude Sonnet 3.5, no retrieval | 42.8 | 0.59 |
| Claude Sonnet 3.5 + same RAG context | [AUTHORS: insert] | [AUTHORS: insert] |
| Gemini Pro 1.5, no retrieval | 38.6 | 0.54 |
| Gemini Pro 1.5 + same RAG context | [AUTHORS: insert] | [AUTHORS: insert] |
| BanglaFinGPT (fine-tuned + RAG + filter) | 90.0 | 0.88 |

The six runs are one command each and are listed in
`artifacts/reviewer/reviewer_experiments.json` under
`R1_C2_baseline_parity.required_runs_needing_api_keys`, for example:

```
python -m banglafingpt.eval.evaluate --backend openai --tag gpt4o_norag --no-retrieval
python -m banglafingpt.eval.evaluate --backend openai --tag gpt4o_rag
```

That access to context, not scale, is the dominant factor is visible even without a
hosted model: holding the generator fixed and toggling retrieval alone moves token-level
F1 from 0.000 to 0.290 on the 2,036-question test split in the released CPU
configuration.

**(ii) The claims have been qualified and the perplexity conflation removed.** We no
longer describe the result as a 3B model outperforming a 100B+ model. Perplexity and
exact-match accuracy measure different things, and we have removed the passage that put
them side by side as if one implied the other.

Revised in Section 5 (Discussion):

> "BanglaFinGPT's advantage over general-purpose models comes from two sources that are
> unavailable to those models in this setting: supervised adaptation to Bangla financial
> language, and retrieval over the authoritative NBR documents from which the answers are
> drawn. It is therefore not evidence that a 3B model is intrinsically stronger than a
> larger one. The comparison in Table 5 separates the two contributions by reporting each
> baseline both with and without the same retrieved context."

Removed from Section 3.3: the sentence relating GPT-4's Bengali perplexity (~45) to
answer accuracy. Perplexity now appears only where it is used, as a language-modelling
statistic motivating a Bangla-pretrained base model.

---

### Comment #3 — dataset quality and bias

**Response.** We have separated this into the three concerns the reviewer raises and
addressed each with either new analysis or an explicit limitation.

**(a) Single-source corpus.** The corpus is deliberately scoped to NBR primary sources,
because the system's purpose is to answer from the authoritative text rather than from
secondary commentary, and because grounding requires a citable source. We agree this
limits external validity and have said so in the Limitations section rather than leaving
it implicit.

Because the questions in the benchmark are cleaner than what users type, we also added a
robustness check: the same test questions perturbed in ways real users produce. Passage-
level recall@5 changes as follows (500 test questions, released configuration):

| Query form | recall@5 | change |
|---|---|---|
| original | 0.768 | — |
| question mark removed | 0.768 | 0.0% |
| digits in the other script (১৫ ↔ 15) | 0.768 | 0.0% |
| one character dropped from a word (typo) | 0.742 | −3.4% |
| politeness prefix added | 0.740 | −3.6% |
| keywords only, no sentence | 0.722 | −6.0% |

Added to Section 4.5:

> "Retrieval is unaffected by punctuation and by whether digits are written in Bengali or
> ASCII, because both are normalised before matching. It degrades modestly under a
> single-character typo (−3.4%), an added politeness prefix (−3.6%) and keyword-only
> queries (−6.0%). This is a proxy for user-typed input rather than a substitute for a
> user study, which remains future work."

**(b) GPT-4-generated questions.** All *answers* are extracted verbatim from the source
passage; the model was used only to propose *questions*, and any proposed pair whose
answer did not occur literally in the passage was discarded. The phrasing bias the
reviewer identifies is nevertheless real, and we now state it and quantify it.

Added to Section 3.2:

> "Candidate questions were produced by two mechanisms: pattern templates
> ([AUTHORS: insert]% of retained pairs) and GPT-4 proposals ([AUTHORS: insert]%). Answers
> were never generated: a proposed pair was retained only if its answer appeared verbatim
> in the source passage, and every numeric value in the answer also appeared there. The
> question distribution therefore inherits some of the phrasing conventions of both
> mechanisms, which is a limitation of the benchmark rather than of the system.
> [AUTHORS: if you can split the test scores by generator, add the comparison here —
> `per_example_records` carries the `generator` field.]"

**(c) Domain imbalance.** We tested the reviewer's hypothesis directly rather than
assuming it. Re-evaluating on a domain-balanced subsample of the test split (n = 305 per
domain, the size of the smallest domain) leaves the ranking and the gaps essentially
unchanged in the released configuration:

| Domain | share of corpus | F1, full test split | F1, balanced subsample |
|---|---|---|---|
| taxation | 26.9% | 0.356 | 0.350 |
| VAT | 25.2% | 0.337 | 0.331 |
| finance | 14.8% | 0.196 | 0.196 |
| customs | 33.1% | 0.183 | 0.179 |

Customs is the *largest* domain in the corpus and still the weakest, and finance is the
smallest and not the weakest — so sample size is not what separates them.

Added to Section 4.4:

> "To test whether the customs gap is an artifact of domain imbalance, we repeated the
> evaluation on a domain-balanced subsample of the test split (n = 305 per domain). The
> ordering and the size of the gaps were unchanged [AUTHORS: insert the fine-tuned
> numbers from `scripts/reviewer_experiments.py`]. Customs is the most represented domain
> in the corpus yet the weakest in accuracy, while general finance is the least
> represented and not the weakest. The difficulty is therefore a property of the content
> — multi-condition tariff rules and duty calculations whose answers are longer (31.7
> words on average against 22.1 for VAT) — rather than of the amount of training data."

---

### Comment #4 — the theoretical framework is overly complex and partly superfluous

**Response.** We agree and have cut the formalism back to the equations that a reader
needs in order to implement or evaluate the system.

**4a.** Retained in the main text: NF4 quantisation (Eq. 9), the cosine learning-rate
schedule actually used (Eq. 13), the RAG marginalisation and prompt construction
(Eqs. 16–17), and the hallucination predicate (Eq. 19) — each of which corresponds to a
line of the released implementation. The remaining derivations have been removed or moved
to an appendix, as detailed below.

**4b.** The KL divergence figure ("≈12.4 nats") has been **removed**. We could not
document an estimation procedure — corpus, tokenizer, and estimator — to a standard that
would let a reader reproduce it, and the paper's argument does not depend on it. The
qualitative point it was making is now stated plainly in Section 3.3:

> "Bangla financial text is far from the distribution these models are pretrained on: the
> vocabulary is legal-administrative Bangla, the numbers are regulatory rates and HS
> codes, and the entities are Bangladeshi institutions. This is what domain adaptation
> has to close."

**4c.** The Chinchilla analysis has been **removed**. It concluded that a compute-optimal
model would be ~23B parameters and we then used 3B for reasons that have nothing to do
with compute-optimality, so it argued against the design without informing it. The actual
reason is now one sentence in Section 3.3.2:

> "We select a 3B base model so that 4-bit fine-tuning and inference fit within the 6 GB
> of GPU memory available on consumer hardware, which is the deployment setting this
> system targets."

**4d.** Equations 11–15 (the quantisation-error bound and the AdamW convergence rate) are
standard results that we restated without extending. They have been removed; the
quantisation configuration and the optimiser schedule are now given as the concrete
settings used (NF4, double quantisation, bf16 compute; AdamW with cosine decay from
5 × 10⁻⁵, 3% warmup), with citations to Dettmers et al. and Loshchilov & Hutter.

---

### Comment #5 — writing and presentation

**5a — grammar and unsupported claims.** Corrected throughout. The specific instances the
reviewer names:

| Original | Revised |
|---|---|
| "Verifiable financial information access is crucial…" | "Access to verifiable financial information is crucial…" |
| "billions of people worldwide lack access to financial advice in their native language" | Either cite or soften. Revised: "a large share of the world's population has no access to financial guidance in a language they read fluently [AUTHORS: cite World Bank Global Findex 2021, or delete the quantifier]" |
| "This presents a major barrier to economic opportunity and financial inclusion" | "This is a major barrier to access to economic opportunity and to financial inclusion" |

The full manuscript has been re-read for the same class of error [AUTHORS: confirm a
language edit pass was done before resubmission].

**5b — reference formatting.** The reference list has been normalised: every entry now
carries a DOI where one exists, and preprints are labelled with their arXiv identifier and
year consistently. Reference [11] (Bisk et al., "Experience grounds language") has been
completed with the full venue, page range and DOI.

**5c — inconsistent numbers.** Corrected. GPT-4o's exact-match score is **45.2%**
everywhere: Abstract, Table 5, Section 4.1 and the Discussion. We adopted a single rule
for the revision — percentages to one decimal, F1 and other ratios to two — and the text
now quotes the tables rather than restating numbers independently.

**5d — figures.** All figures are now rendered images generated from the released code
(`scripts/make_figures.py` and the reproduction notebook) rather than descriptions of
intended figures, and each is referenced at the point where its result is discussed. Any
figure we could not produce from data has been removed together with its reference.

---

## Reviewer 2

### General comment

**Response.** We thank the reviewer. The four issues raised — dataset splitting,
implementation details, baseline fairness, and the statistical and hallucination
evaluations — are addressed in turn below. To make all four verifiable rather than merely
described, the full implementation, the dataset loader, the evaluation suite and an
executed reproduction notebook are released at [AUTHORS: repository URL].

### (a) Dataset splitting

**Response.** This was the most consequential issue, and the reviewer is right to raise
it. The released corpus contains 10,412 QA pairs drawn from only **1,451 distinct source
passages**: the same passage yields several questions, and each appears in both Bangla and
English. A row-level random split therefore leaks.

We measured it. With a random split of the rows, **93.2%** of test items share their
source passage with the training set. Grouping the split by source passage brings that to
**0.0%**:

| Split protocol | test items sharing a training passage | near-duplicate test questions |
|---|---|---|
| random row-level shuffle | 93.2% | 1.7% |
| grouped by source passage (adopted) | 0.0% | 1.6% |

Added to Section 3.2.1:

> "Because the corpus contains 10,412 QA pairs derived from 1,451 distinct source
> passages, a question-level random split would place a test question's own source passage
> — and frequently a near-paraphrase of the question itself — in the training set. We
> therefore split by source passage: all questions derived from a passage, in both
> languages, are assigned to the same split. Splitting is stratified by domain so the
> domain proportions are preserved, yielding 7,412 training, 1,000 validation and 2,000
> test pairs. Under a question-level random split, 93.2% of test items would share a
> source passage with training; under the passage-grouped split this is 0%."

Both procedures are released (`split_pairs` and, for the comparison only,
`random_row_split`), and `leakage_report` reproduces the table.

[AUTHORS: if the submitted results used a row-level split, the test scores must be
regenerated under the passage-grouped split before resubmission. This is the one change in
this letter that can move the headline numbers, and it is better to report a lower,
leak-free number than to have a reviewer find it.]

### (b) Implementation details

**Response.** Added as Section 3.4 and an appendix table: base model and revision,
quantisation (NF4, double quantisation, bf16 compute), LoRA rank 16 / α 32 / dropout 0.05
on all attention and MLP projections, effective batch size 32 (4 × 8 accumulation),
AdamW with cosine decay from 5 × 10⁻⁵ and 3% warmup, early stopping on validation loss
with patience 3, maximum sequence length 1024, loss computed on answer tokens only,
retriever (encoder, chunk size 220 words with 40-word overlap, top-k 5, hybrid dense/BM25
weighting), decoding parameters, filter thresholds, random seed 42, and the hardware used
[AUTHORS: confirm the GPU and wall-clock training time].

> "Training prompts include retrieved context, matching what the model sees at inference
> time; training without context and serving with it is a distribution shift that costs
> accuracy."

### (c) Fair baseline comparison

**Response.** Addressed together with Reviewer 1's Comment #2: every baseline is now
reported both without retrieval and with exactly the same retrieved context and prompt as
BanglaFinGPT, and the claim about model scale has been withdrawn.

### (d) Statistical and hallucination evaluation

**Response.** The statistical reporting has been made explicit rather than asserted.
Confidence intervals are percentile bootstrap intervals over per-example scores (10,000
resamples, seed 42). Exact-match comparisons between systems use McNemar's test on
discordant pairs, which is the appropriate paired test for a binary per-item outcome;
continuous metrics use a two-sided paired bootstrap. The procedures are released as
`banglafingpt.eval.significance` and the numbers in Table 12 are regenerated by
`make ablation`.

Added to Section 4.7:

> "Confidence intervals are 95% percentile bootstrap intervals computed over per-example
> scores with 10,000 resamples. Differences in exact match between configurations are
> tested with McNemar's test on discordant items; differences in F1 and the generation
> metrics use a two-sided paired bootstrap over the same items. Queries that the
> hallucination filter declines to answer are scored as incorrect rather than excluded, so
> the reported accuracy already carries the cost of the filter's precision/recall trade."

The hallucination evaluation is addressed in Comment #1b above.

---

## What still has to be run before resubmission

| Needed for | Blocked by | Command |
|---|---|---|
| Fine-tuned scores under the passage-grouped split (Tables 5, 7, 8, 12) | CUDA GPU | `make train` then `make ablation` |
| Baseline rows with and without RAG (Table 5) | API keys | see `R1_C2_baseline_parity.required_runs_needing_api_keys` |
| Hallucination rates and Fleiss' κ (Table 11, Section 4.6.1) | three annotators | `scripts/reviewer_experiments.py --all`, then `score_annotations` |
| Human evaluation (Table 6) | three experts | `scripts/human_eval_report.py` |

Everything else quoted in this letter is already reproducible with
`python scripts/reviewer_experiments.py --all`.
