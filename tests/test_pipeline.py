from banglafingpt.models.prompts import build_prompt, fallback_message
from banglafingpt.pipeline import BanglaFinGPT


def test_prompt_contains_instruction_and_context_blocks():
    prompt = build_prompt("ভ্যাট হার কত?", ["ভ্যাট হার ১৫ শতাংশ"], ["vat_act#15"])
    assert "[INST]" in prompt and "[/INST]" in prompt
    assert "[CONTEXT]" in prompt and "vat_act#15" in prompt


def test_prompt_language_switches_system_message():
    assert "You are BanglaFinGPT" in build_prompt("What is VAT?", language="en")
    assert "আপনি BanglaFinGPT" in build_prompt("ভ্যাট কী?", language="bn")


def test_pipeline_answers_a_known_question(system):
    answer = system.answer("ভ্যাটের আদর্শ হার কত?")
    assert answer.answered
    assert "১৫ শতাংশ" in answer.text
    assert answer.citations and answer.citations[0].startswith("vat_act_2012")


def test_pipeline_detects_english_questions(system):
    answer = system.answer("What is the advance income tax rate at the import stage?")
    assert answer.language == "en"


def test_refusal_returns_the_fallback_message(demo_config, retriever):
    strict = BanglaFinGPT(demo_config, retriever=retriever)
    strict.filter.config.min_keyword_overlap = 1.01  # impossible to satisfy
    answer = strict.answer("ভ্যাটের আদর্শ হার কত?")
    assert not answer.answered
    assert answer.text == fallback_message("bn")
    assert answer.citations == []


def test_variant_toggles_components(system):
    no_rag = system.variant(use_retrieval=False, use_filter=False)
    assert not no_rag.use_retrieval and not no_rag.use_filter
    assert system.use_retrieval  # original is unchanged
