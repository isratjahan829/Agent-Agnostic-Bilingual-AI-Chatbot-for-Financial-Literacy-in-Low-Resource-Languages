import pytest

from banglafingpt.eval.human_eval import fleiss_kappa, interpret_kappa, summarize_ratings
from banglafingpt.eval.significance import bootstrap_ci, mcnemar_test, significance_table


def test_bootstrap_ci_brackets_the_mean():
    values = [1.0] * 90 + [0.0] * 10
    mean, low, high = bootstrap_ci(values, n_resamples=500)
    assert low <= mean <= high and 0.85 < mean < 0.95


def test_mcnemar_detects_a_clear_improvement():
    baseline = [0.0] * 100
    system = [1.0] * 80 + [0.0] * 20
    assert mcnemar_test(baseline, system)["p_value"] < 0.001


def test_mcnemar_is_insignificant_when_systems_tie():
    scores = [1.0, 0.0] * 20
    assert mcnemar_test(scores, scores)["p_value"] == 1.0


def test_significance_table_marks_the_baseline_row():
    table = significance_table(
        {"base": [0.0] * 50, "full": [1.0] * 50}, baseline="base"
    )
    base_row = next(r for r in table if r["system"] == "base")
    full_row = next(r for r in table if r["system"] == "full")
    assert base_row["p_value"] is None
    assert full_row["significant_at_0.05"] is True


def test_perfect_agreement_gives_kappa_one():
    assert fleiss_kappa([[0, 0, 3, 0, 0], [0, 0, 0, 3, 0]]) == pytest.approx(1.0)


def test_kappa_requires_equal_rater_counts():
    with pytest.raises(ValueError):
        fleiss_kappa([[3, 0], [2, 0]])


def test_kappa_interpretation_bands():
    assert interpret_kappa(0.72) == "substantial"
    assert interpret_kappa(0.1) == "slight"


def test_summarize_ratings_reports_mean_and_std():
    rows = [
        {"query_id": 1, "rater_id": "a", "accuracy": 4, "relevance": 4},
        {"query_id": 1, "rater_id": "b", "accuracy": 3, "relevance": 4},
    ]
    summary = summarize_ratings(rows)
    assert summary["accuracy"]["mean"] == 3.5
    assert summary["relevance"]["std"] == 0.0
