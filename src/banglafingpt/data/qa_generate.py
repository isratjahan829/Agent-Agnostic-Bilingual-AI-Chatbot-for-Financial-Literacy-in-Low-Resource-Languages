"""Semi-automated QA generation: templates + an LLM proposer (Sec. 3.2).

Both paths only ever *extract* the answer from the segment text; nothing is
free-written by a model. That is what keeps the answers in the original
regulatory wording, numbers and provisions included.
"""
from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence

from ..utils import extract_numbers, normalize_text
from .schema import QAPair, Segment
from .segment import split_sentences

# Bangla question templates keyed by the cue that must appear in the sentence.
TEMPLATES: dict[str, list[tuple[str, str]]] = {
    "rate": [
        (r"(হার|শতাংশ|%|rate)", "{subject} এর হার কত?"),
        (r"(হার|শতাংশ|%|rate)", "What is the applicable rate for {subject}?"),
    ],
    "exemption": [
        (r"(অব্যাহতি|রেয়াত|exempt)", "{subject} কি ভ্যাট/কর অব্যাহতিপ্রাপ্ত?"),
        (r"(অব্যাহতি|রেয়াত|exempt)", "Is {subject} exempt from tax or VAT?"),
    ],
    "deadline": [
        (r"(তারিখ|সময়সীমা|মধ্যে|deadline|within)", "{subject} এর সময়সীমা কত?"),
    ],
    "definition": [
        (r"(অর্থ|সংজ্ঞা|বলিতে|means|defined)", "{subject} বলতে কী বোঝায়?"),
        (r"(অর্থ|সংজ্ঞা|বলিতে|means|defined)", "What does {subject} mean?"),
    ],
    "procedure": [
        (r"(দাখিল|আবেদন|নিবন্ধন|procedure|apply|register)", "{subject} এর প্রক্রিয়া কী?"),
    ],
    "penalty": [
        (r"(জরিমানা|দণ্ড|penalty|fine)", "{subject} লঙ্ঘনের জরিমানা কী?"),
    ],
}

SUBJECT_STOP = re.compile(r"^\s*(?:[০-৯\d]+[.)]?\s*)")


def _subject_of(sentence: str, section: str | None) -> str:
    """Best-effort topic phrase: the section title, else the sentence head."""
    if section:
        cleaned = SUBJECT_STOP.sub("", section).strip()
        if len(cleaned.split()) >= 2:
            return cleaned
    words = sentence.split()
    return " ".join(words[: min(6, len(words))]).rstrip(":,।")


def template_questions(segment: Segment, max_questions: int = 3) -> list[QAPair]:
    """Generate template-based QA pairs whose answers are verbatim sentences."""
    pairs: list[QAPair] = []
    seen: set[str] = set()
    for sentence in split_sentences(segment.text):
        if len(sentence.split()) < 5 or sentence.startswith("[TABLE]"):
            continue
        for cue, templates in TEMPLATES.items():
            for pattern, template in templates:
                if not re.search(pattern, sentence, re.IGNORECASE):
                    continue
                question = template.format(subject=_subject_of(sentence, segment.section))
                key = normalize_text(question)
                if key in seen:
                    continue
                seen.add(key)
                pairs.append(
                    QAPair.create(
                        question=question,
                        answer=sentence,
                        domain=segment.domain,
                        segment_id=segment.segment_id,
                        source=segment.doc_id,
                        generator="template",
                        meta={"cue": cue, "section": segment.section,
                              "numbers": extract_numbers(sentence)},
                    )
                )
                if len(pairs) >= max_questions:
                    return pairs
    return pairs


LLM_PROMPT = """You are building a Bangla financial QA dataset from official
Bangladesh National Board of Revenue (NBR) regulation text.

Rules:
1. Write {n} questions a taxpayer would realistically ask about the passage.
2. The answer MUST be copied verbatim from the passage (one or two sentences).
   Never paraphrase numbers, rates, dates, section references or HS codes.
3. If the passage does not support a self-contained answer, return fewer pairs.
4. Keep the question in the same language as the passage.

Return JSON only: {{"pairs": [{{"question": "...", "answer": "..."}}]}}

Passage (domain: {domain}, section: {section}):
\"\"\"{passage}\"\"\"
"""

LLMFn = Callable[[str], str]


def llm_questions(segment: Segment, llm: LLMFn, max_questions: int = 3) -> list[QAPair]:
    """Ask an LLM for candidate pairs, then keep only verbatim-grounded answers."""
    prompt = LLM_PROMPT.format(
        n=max_questions, domain=segment.domain,
        section=segment.section or "n/a", passage=segment.text,
    )
    try:
        payload = json.loads(_extract_json(llm(prompt)))
    except (json.JSONDecodeError, ValueError):
        return []
    haystack = normalize_text(segment.text)
    pairs: list[QAPair] = []
    for item in payload.get("pairs", [])[:max_questions]:
        question, answer = (item.get("question") or "").strip(), (item.get("answer") or "").strip()
        if not question or not answer:
            continue
        if normalize_text(answer) not in haystack:
            continue  # reject anything the model invented
        pairs.append(
            QAPair.create(
                question=question, answer=answer, domain=segment.domain,
                segment_id=segment.segment_id, source=segment.doc_id,
                generator="llm",
                meta={"section": segment.section, "numbers": extract_numbers(answer)},
            )
        )
    return pairs


def _extract_json(text: str) -> str:
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object in LLM response")
    return text[start : end + 1]


def generate_candidates(
    segments: Sequence[Segment],
    llm: LLMFn | None = None,
    questions_per_segment: int = 3,
) -> list[QAPair]:
    """Hybrid generation over a corpus; the LLM path is optional and additive."""
    out: list[QAPair] = []
    for segment in segments:
        pairs = template_questions(segment, questions_per_segment)
        if llm is not None and len(pairs) < questions_per_segment:
            pairs += llm_questions(segment, llm, questions_per_segment - len(pairs))
        out.extend(pairs[:questions_per_segment])
    return out
