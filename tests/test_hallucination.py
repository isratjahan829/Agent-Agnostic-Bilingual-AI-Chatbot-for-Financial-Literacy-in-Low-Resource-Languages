import pytest

from banglafingpt.config import HallucinationConfig
from banglafingpt.hallucination.filters import (
    HallucinationFilter,
    keyword_overlap,
    numeric_support,
)

CONTEXT = ["বাংলাদেশে মূল্য সংযোজন করের আদর্শ হার ১৫ শতাংশ এবং নিবন্ধনের সীমা ৩ কোটি টাকা।"]


@pytest.fixture()
def strict_filter():
    return HallucinationFilter(HallucinationConfig(min_cosine_similarity=0.3,
                                                   min_keyword_overlap=0.5))


def test_grounded_answer_passes(strict_filter):
    verdict = strict_filter.verify("মূল্য সংযোজন করের আদর্শ হার ১৫ শতাংশ।", CONTEXT)
    assert verdict.grounded


def test_fabricated_number_is_rejected(strict_filter):
    verdict = strict_filter.verify("মূল্য সংযোজন করের আদর্শ হার ৩৭ শতাংশ।", CONTEXT)
    assert not verdict.grounded and "37" in verdict.unsupported_numbers


def test_off_topic_answer_is_rejected(strict_filter):
    verdict = strict_filter.verify("ঢাকার আবহাওয়া আজ রৌদ্রোজ্জ্বল।", CONTEXT)
    assert not verdict.grounded


def test_empty_answer_and_missing_context_are_rejected(strict_filter):
    assert strict_filter.verify("", CONTEXT).reason == "empty_answer"
    assert strict_filter.verify("যেকোনো উত্তর", []).reason == "no_context_retrieved"


def test_keyword_overlap_is_directional():
    assert keyword_overlap("ভ্যাট হার", ["ভ্যাট হার নিবন্ধন সীমা টার্নওভার"]) == 1.0


def test_numeric_support_reports_missing_numbers():
    support, missing = numeric_support("হার ১৫ ও ৯৯ শতাংশ", CONTEXT)
    assert support == 0.5 and missing == ["99"]


def test_answer_without_numbers_is_numerically_supported():
    support, missing = numeric_support("কোনো সংখ্যা নেই", CONTEXT)
    assert support == 1.0 and missing == []


def test_relevance_check_refuses_an_unrelated_question(strict_filter):
    strict_filter.config.min_question_similarity = 0.5
    verdict = strict_filter.verify(
        "মূল্য সংযোজন করের আদর্শ হার ১৫ শতাংশ।", CONTEXT,
        question="সিঙ্গাপুরে ক্রিপ্টোকারেন্সির মূলধনী লাভ কর কত?",
    )
    assert not verdict.grounded and "irrelevant_context" in verdict.reason


def test_relevance_check_is_off_by_default(strict_filter):
    assert strict_filter.config.min_question_similarity == 0.0
    verdict = strict_filter.verify("মূল্য সংযোজন করের আদর্শ হার ১৫ শতাংশ।", CONTEXT,
                                   question="ভ্যাটের হার কত?")
    assert verdict.grounded
