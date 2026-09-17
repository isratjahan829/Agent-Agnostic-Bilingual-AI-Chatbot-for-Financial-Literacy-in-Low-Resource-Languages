# Manuscript and revision

| File | What it is |
|---|---|
| `main.tex` | Revised manuscript. Revision text prints **blue** (`\revised{}`); values still to be supplied print **red** (`\needsnum{}`). `\revisionfalse` in the preamble gives a clean copy. |
| `Response_to_Reviewers_BanglaFinGPT_v2.docx` | Point-by-point response letter, in the structure of the group's earlier BDMA letter. |
| `INFO_NEEDED.md` | The 10 markers still open, what each needs, and which reviewer comment it answers. |
| `CHANGES.md` | Every edit mapped to its comment, plus the findings that changed results rather than wording. |

## Scope

The manuscript answers the reviewers' comments and nothing beyond them. Three additions
that no reviewer requested were removed before submission: the "+ RAG context" rows in
Table 5, the query-robustness subsection, and the baselines' hallucination-rate row in
Table 13.

## Before submitting

```bash
grep -o 'needsnum{' main.tex | wc -l    # must be 0
pdflatex main && biber main && pdflatex main && pdflatex main
```

One marker is a blocking note at the head of Section 3: every reported score must come
from the passage-grouped split described in Section 2.2.1. If the submitted results used
a question-level partition they are inflated — 93.2% of test items would share a source
passage with the training set — and Tables 5, 7, 8 and 12 have to be regenerated with
`notebooks/BanglaFinGPT_gpu_run.ipynb`. Delete the note once the numbers and Section
2.2.1 agree.
