import pytest

from banglafingpt.data.build_dataset import describe, split_pairs
from banglafingpt.data.pdf_extract import structure_text
from banglafingpt.data.qa_generate import template_questions
from banglafingpt.data.quality_control import corpus_statistics, drop_near_duplicates
from banglafingpt.data.schema import QAPair, validate_pairs
from banglafingpt.data.segment import segment_document, split_sentences


def test_sentence_split_handles_danda():
    assert len(split_sentences("প্রথম বাক্য। দ্বিতীয় বাক্য। Third sentence.")) == 3


def test_structure_text_tags_headings():
    assert "[H] 15" in structure_text(["15. VAT rate\nSome body text here."])


def test_segmentation_respects_the_word_budget():
    from banglafingpt.data.schema import SourceDocument

    sentence = " ".join(["শব্দ"] * 12) + "। "
    doc = SourceDocument(doc_id="d", title="t", domain="vat", text=sentence * 30)
    segments = segment_document(doc, min_words=30, max_words=60)
    assert len(segments) > 1
    # A segment may overshoot by at most the length of the sentence that closed it.
    assert all(s.n_words <= 60 + 12 for s in segments)


def test_a_single_oversized_sentence_is_never_split():
    from banglafingpt.data.schema import SourceDocument

    doc = SourceDocument(doc_id="d", title="t", domain="vat",
                         text=" ".join(["শব্দ"] * 500) + "।")
    segments = segment_document(doc, min_words=30, max_words=60)
    assert len(segments) == 1


def test_template_questions_are_extractive(sample_segments):
    pairs = template_questions(sample_segments[0], max_questions=3)
    assert pairs
    assert all(p.answer in sample_segments[0].text for p in pairs)


def test_near_duplicate_removal():
    pairs = [
        QAPair.create("ভ্যাটের হার কত?", "১৫ শতাংশ", "vat", "s1", "d1"),
        QAPair.create("ভ্যাটের হার কত ?", "১৫ শতাংশ", "vat", "s1", "d1"),
        QAPair.create("আয়করের সীমা কত?", "৩ লক্ষ", "taxation", "s2", "d1"),
    ]
    kept, dropped = drop_near_duplicates(pairs, threshold=0.9)
    assert dropped == 1 and len(kept) == 2


def test_split_keeps_segments_together(sample_pairs):
    splits = split_pairs(sample_pairs, train_size=10, val_size=2, test_size=4, seed=1)
    assignments = {}
    for name, pairs in splits.items():
        for pair in pairs:
            assignments.setdefault(pair.segment_id, name)
            assert assignments[pair.segment_id] == name  # no leakage across splits


def test_split_totals_match_input(sample_pairs):
    splits = split_pairs(sample_pairs, train_size=10, val_size=2, test_size=4, seed=1)
    assert sum(len(v) for v in splits.values()) == len(sample_pairs)
    assert describe(splits)[-1]["domain"] == "total"


def test_invalid_domain_is_rejected():
    with pytest.raises(ValueError):
        QAPair.create("q", "a", "sports", "s1", "d1")


def test_validate_pairs_flags_empty_answers():
    bad = QAPair.create("প্রশ্ন", "উত্তর", "vat", "s1", "d1")
    bad.answer = ""
    assert validate_pairs([bad])


def test_corpus_statistics_shape(sample_pairs):
    stats = corpus_statistics(sample_pairs)
    assert stats["total"]["qa_pairs"] == len(sample_pairs)


def test_chunk_text_overlaps_and_covers():
    from banglafingpt.data.load_xlsx import chunk_text

    words = [f"w{i}" for i in range(500)]
    chunks = chunk_text(" ".join(words), chunk_words=200, overlap_words=40)
    assert len(chunks) > 1
    assert chunks[0].split()[-40:] == chunks[1].split()[:40]  # windows overlap
    assert chunks[-1].split()[-1] == "w499"                   # nothing is dropped


def test_short_text_is_a_single_chunk():
    from banglafingpt.data.load_xlsx import chunk_text

    assert chunk_text("ছোট একটি অনুচ্ছেদ", chunk_words=200, overlap_words=40) == [
        "ছোট একটি অনুচ্ছেদ"
    ]


def test_topics_map_to_paper_domains():
    from banglafingpt.data.load_xlsx import normalize_topic

    assert normalize_topic("Tax") == "taxation"
    assert normalize_topic("Vat") == "vat"
    with pytest.raises(ValueError):
        normalize_topic("Sports")
