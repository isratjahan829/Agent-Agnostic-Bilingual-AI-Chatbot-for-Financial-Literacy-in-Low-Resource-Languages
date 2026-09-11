"""Interactive bilingual chat over the BanglaFinGPT pipeline."""
from __future__ import annotations

import argparse
from collections.abc import Sequence

from ..config import load_config
from ..eval.evaluate import build_system
from ..utils import set_seed

BANNER = """BanglaFinGPT — NBR-grounded financial QA (বাংলা / English)
Type a question, or /quit to exit, /domain <taxation|vat|customs|finance|any>.
Answers are informational only and are not a substitute for professional advice.
"""


def format_answer(answer) -> str:
    lines = [answer.text]
    if answer.answered and answer.citations:
        lines.append("\nসূত্র / Sources: " + ", ".join(dict.fromkeys(answer.citations)))
    if not answer.answered and answer.verdict:
        lines.append(f"\n[filtered: {answer.verdict.reason}]")
    lines.append(f"[{answer.latency_ms:.0f} ms]")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Chat with BanglaFinGPT")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--index-dir", default="artifacts/index")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--backend", default=None)
    parser.add_argument("--question", default=None, help="answer one question and exit")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if args.backend:
        cfg.agent.backend = args.backend
    if args.adapter:
        cfg.agent.adapter_path = args.adapter
    set_seed(cfg.seed)
    system = build_system(cfg, args.index_dir)

    if args.question:
        print(format_answer(system.answer(args.question)))
        return 0

    print(BANNER)
    domain: str | None = None
    while True:
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line:
            continue
        if line in {"/quit", "/exit"}:
            return 0
        if line.startswith("/domain"):
            value = line.split(maxsplit=1)[-1].strip()
            domain = None if value in {"any", "/domain"} else value
            print(f"[domain filter: {domain or 'any'}]")
            continue
        print(format_answer(system.answer(line, domain=domain)))


if __name__ == "__main__":
    raise SystemExit(main())
