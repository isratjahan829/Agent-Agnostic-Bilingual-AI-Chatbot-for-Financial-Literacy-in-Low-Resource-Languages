from banglafingpt.utils import (
    bn_to_en_digits,
    content_tokens,
    detect_language,
    extract_numbers,
    normalize_text,
)


def test_bengali_digits_fold_to_ascii():
    assert bn_to_en_digits("১৫৭") == "157"


def test_normalisation_makes_digit_variants_equal():
    assert normalize_text("হার ১৫ শতাংশ।") == normalize_text("হার 15 শতাংশ")


def test_normalisation_is_unicode_stable():
    # Same grapheme, different code point sequences.
    assert normalize_text("ক্ষ") == normalize_text("ক্ষ")


def test_detect_language():
    assert detect_language("ভ্যাটের হার কত?") == "bn"
    assert detect_language("What is the VAT rate?") == "en"
    assert detect_language("VAT হার কত") == "mixed"


def test_extract_numbers_handles_both_scripts():
    assert extract_numbers("১৫% এবং 7.5%") == ["15", "7.5"]


def test_content_tokens_drop_stopwords():
    assert "এই" not in content_tokens("এই ভ্যাট হার")
