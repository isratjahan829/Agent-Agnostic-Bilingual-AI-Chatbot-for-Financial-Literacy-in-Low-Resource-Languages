"""FastAPI service exposing the pipeline (``uvicorn banglafingpt.app.api:app``)."""
from __future__ import annotations

import os

from pydantic import BaseModel, Field

from ..config import load_config
from ..eval.evaluate import build_system


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    domain: str | None = None
    top_k: int | None = Field(default=None, ge=1, le=20)


class Citation(BaseModel):
    chunk_id: str
    doc_id: str
    score: float


class AskResponse(BaseModel):
    answer: str
    language: str
    answered: bool
    citations: list[Citation]
    grounding: dict | None = None
    latency_ms: float
    disclaimer: str


DISCLAIMER = ("এই উত্তর শুধুমাত্র তথ্যগত; চূড়ান্ত সিদ্ধান্তের জন্য NBR এর সরকারি নথি বা "
              "নিবন্ধিত কর পরামর্শকের পরামর্শ নিন। / Informational only; verify with "
              "official NBR documents or a registered tax practitioner.")


def create_app():
    from fastapi import FastAPI, HTTPException

    cfg = load_config(os.environ.get("BANGLAFINGPT_CONFIG", "configs/default.yaml"))
    system = build_system(cfg, os.environ.get("BANGLAFINGPT_INDEX", "artifacts/index"))
    app = FastAPI(title="BanglaFinGPT", version="0.1.0")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "backend": cfg.agent.backend,
                "chunks": len(system.retriever.index) if system.retriever else 0}

    @app.post("/ask", response_model=AskResponse)
    def ask(request: AskRequest) -> AskResponse:
        try:
            answer = system.answer(request.question, domain=request.domain, top_k=request.top_k)
        except Exception as exc:  # surface a clean error instead of a stack trace
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return AskResponse(
            answer=answer.text,
            language=answer.language,
            answered=answer.answered,
            citations=[
                Citation(chunk_id=c.chunk_id, doc_id=c.doc_id, score=round(c.score, 4))
                for c in (answer.chunks if answer.answered else [])
            ],
            grounding=answer.verdict.to_dict() if answer.verdict else None,
            latency_ms=round(answer.latency_ms, 1),
            disclaimer=DISCLAIMER,
        )

    return app


app = None
if os.environ.get("BANGLAFINGPT_EAGER_APP", "1") == "1":  # pragma: no cover
    try:
        app = create_app()
    except Exception:  # import-time failures shouldn't break `import banglafingpt`
        app = None
