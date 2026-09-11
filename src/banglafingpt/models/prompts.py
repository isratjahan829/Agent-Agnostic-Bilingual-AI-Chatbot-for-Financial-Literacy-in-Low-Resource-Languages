"""Prompt construction (Eq. 17) and the grounding system prompt (Sec. 3.3.5).

The system prompt is the first of the three hallucination defences: it forbids
answering outside the retrieved context and mandates the fallback message when
the context is insufficient.
"""
from __future__ import annotations

from collections.abc import Sequence

INSTRUCTION_OPEN, INSTRUCTION_CLOSE = "[INST]", "[/INST]"
CONTEXT_OPEN, CONTEXT_CLOSE = "[CONTEXT]", "[/CONTEXT]"

SYSTEM_PROMPT_BN = """আপনি BanglaFinGPT — বাংলাদেশ জাতীয় রাজস্ব বোর্ড (NBR) এর সরকারি
নথির ভিত্তিতে কর, ভ্যাট, কাস্টমস ও আর্থিক নিয়ম সম্পর্কে তথ্য দেন।

কঠোরভাবে মেনে চলুন:
১. শুধুমাত্র নিচে দেওয়া [CONTEXT] অংশের তথ্য ব্যবহার করে উত্তর দিন।
২. কোনো হার, তারিখ, ধারা নম্বর বা HS কোড নিজে থেকে তৈরি করবেন না; হুবহু নথি থেকে নিন।
৩. প্রসঙ্গে উত্তর না থাকলে লিখুন: "প্রদত্ত নথিতে এই প্রশ্নের সুনির্দিষ্ট উত্তর নেই।
   অনুগ্রহ করে NBR এর সরকারি ওয়েবসাইট (https://nbr.gov.bd) দেখুন বা একজন
   নিবন্ধিত কর পরামর্শকের সাথে যোগাযোগ করুন।"
৪. সংক্ষিপ্ত ও নির্ভুল উত্তর দিন এবং প্রযোজ্য ধারা/অনুচ্ছেদ উল্লেখ করুন।
৫. প্রশ্ন যে ভাষায় করা হয়েছে, সেই ভাষাতেই উত্তর দিন।"""

SYSTEM_PROMPT_EN = """You are BanglaFinGPT, answering questions about Bangladesh
taxation, VAT, customs and financial regulation strictly from official National
Board of Revenue (NBR) documents.

Rules:
1. Use only the information inside [CONTEXT] below.
2. Never invent a rate, date, section number or HS code — copy them from the context.
3. If the context does not answer the question, reply: "The provided documents do
   not contain a specific answer. Please consult the official NBR website
   (https://nbr.gov.bd) or a registered tax practitioner."
4. Be concise and factual, and cite the applicable section.
5. Answer in the same language as the question."""

FALLBACK_BN = ("প্রদত্ত নথিতে এই প্রশ্নের সুনির্দিষ্ট উত্তর নেই। অনুগ্রহ করে NBR এর "
               "সরকারি ওয়েবসাইট (https://nbr.gov.bd) দেখুন বা একজন নিবন্ধিত কর "
               "পরামর্শকের সাথে যোগাযোগ করুন।")
FALLBACK_EN = ("The provided documents do not contain a specific answer. Please "
               "consult the official NBR website (https://nbr.gov.bd) or a "
               "registered tax practitioner.")


def fallback_message(language: str) -> str:
    return FALLBACK_EN if language == "en" else FALLBACK_BN


def system_prompt(language: str = "bn") -> str:
    return SYSTEM_PROMPT_EN if language == "en" else SYSTEM_PROMPT_BN


def format_context(contexts: Sequence[str], citations: Sequence[str] | None = None) -> str:
    """Concatenate the top-k retrieved chunks, numbered so the model can cite them."""
    lines: list[str] = []
    for i, text in enumerate(contexts, 1):
        tag = f" ({citations[i - 1]})" if citations and i <= len(citations) else ""
        lines.append(f"[{i}]{tag} {text.strip()}")
    return "\n\n".join(lines)


def build_prompt(
    question: str,
    contexts: Sequence[str] = (),
    citations: Sequence[str] | None = None,
    language: str = "bn",
) -> str:
    """prompt = [INST] q [/INST] [CONTEXT] ⊕ d_i [/CONTEXT]  (Eq. 17)."""
    instruction = f"{INSTRUCTION_OPEN} {question.strip()} {INSTRUCTION_CLOSE}"
    parts = [system_prompt(language), "", instruction]
    if contexts:
        parts += [CONTEXT_OPEN, format_context(contexts, citations), CONTEXT_CLOSE]
    return "\n".join(parts)


def build_training_example(
    question: str,
    answer: str,
    contexts: Sequence[str] = (),
    language: str = "bn",
    eos_token: str = "</s>",
) -> dict:
    """A supervised example whose loss is masked to the answer tokens only."""
    prompt = build_prompt(question, contexts, language=language)
    return {
        "prompt": prompt + "\n",
        "completion": answer.strip() + eos_token,
        "text": prompt + "\n" + answer.strip() + eos_token,
    }
