# Dataset card — BanglaFinGPT QA corpus

## Summary

Bilingual (Bangla/English) question–answer pairs about Bangladesh taxation, VAT,
customs and general financial procedure, extracted from official National Board
of Revenue publications. Target size: 10,412 pairs across 18 source documents.

| Domain | QA pairs | Share | Avg Q | Avg A | Sources |
|---|---|---|---|---|---|
| Taxation | 2,823 | 27.12% | 9.4 w | 28.3 w | 6 |
| VAT | 2,641 | 25.36% | 8.1 w | 22.1 w | 3 |
| Customs | 3,475 | 33.37% | 10.7 w | 31.7 w | 5 |
| Finance (general) | 1,473 | 14.15% | 7.9 w | 25.8 w | 4 |
| **Total** | **10,412** | 100% | 8.95 w | 26.8 w | 18 |

Splits: 7,412 train / 1,000 validation / 2,000 test, balanced by domain and split
**by source segment**, so no test question shares a passage with a training one.

## Sources

- NBR circulars and guidelines, 2020–2025 (18 documents)
- Income Tax Ordinance 1984 (as amended 2024)
- VAT Act 2012 (as amended 2024)
- Customs Act 1969 (as amended 2024)
- Bangladesh Financial Regulations Guide (2024)

## Construction

1. PDFs converted to structured text, preserving headings, tables and cross-references.
2. Segmentation into coherent 150–300 word units, each retaining its section number.
3. 2–3 candidate questions per segment: pattern templates plus an LLM proposer.
4. **Answers are always extracted verbatim** — an LLM-proposed pair is discarded
   unless its answer occurs literally in the segment, and every number in the
   answer must occur in the segment.
5. Near-duplicate questions removed (character 4-gram Jaccard ≥ 0.92).
6. Manual review for naturalness and ambiguity; a subset re-checked by a domain
   expert against the original regulation.

Every pair stores `segment_id` and `source`, so any answer can be traced to the
clause it came from.

## Intended use and limits

- **Intended**: research on domain adaptation, RAG and hallucination mitigation in
  low-resource languages; benchmarking financial QA in Bangla.
- **Not intended**: as a source of tax advice, or as ground truth for any current
  filing. Rates and thresholds change with every budget; pairs are valid only for
  the vintage of the source document.
- **Coverage bias**: customs is over-represented (33%) and general finance
  under-represented (14%), which the paper's per-domain results reflect.
- **Language mix**: Bangla dominates; English pairs come from the English-language
  sections of the source acts.

## License

Derived from Bangladesh government publications, which remain under their own
terms. The extraction and annotation layer is released for research use; see the
Kaggle release referenced in the paper.
