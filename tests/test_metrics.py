import pytest

from banglafingpt.eval.metrics import (
    aggregate_metrics,
    bleu4,
    exact_match,
    f1_score,
    meteor,
    rouge_l,
)


def test_exact_match_ignores_punctuation_and_digit_script():
    assert exact_match("হার ১৫ শতাংশ।", "হার 15 শতাংশ") == 1.0
    assert exact_match("হার ১০ শতাংশ", "হার ১৫ শতাংশ") == 0.0


def test_f1_is_partial_credit():
    score = f1_score("মূল্য সংযোজন করের হার ১৫ শতাংশ", "করের হার ১৫ শতাংশ")
    assert 0.5 < score < 1.0


@pytest.mark.parametrize("metric", [f1_score, bleu4, rouge_l, meteor])
def test_identical_strings_score_near_one(metric):
    text = "আয়কর রিটার্ন দাখিলের সময়সীমা ৩০ নভেম্বর"
    assert metric(text, text) > 0.95


@pytest.mark.parametrize("metric", [f1_score, bleu4, rouge_l, meteor])
def test_disjoint_strings_score_zero(metric):
    assert metric("কাস্টমস শুল্ক", "শিক্ষা সেবা") == 0.0


def test_empty_prediction_scores_zero():
    assert f1_score("", "কিছু উত্তর") == 0.0


def test_aggregate_reports_percentage_and_count():
    out = aggregate_metrics(["ক", "খ"], ["ক", "গ"])
    assert out["em_pct"] == 50.0 and out["n"] == 2


def test_aggregate_rejects_length_mismatch():
    with pytest.raises(ValueError):
        aggregate_metrics(["ক"], ["ক", "খ"])
