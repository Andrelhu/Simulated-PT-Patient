from __future__ import annotations

import argparse
import asyncio
import os

from .console_app import main


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="ana_agent")
    p.add_argument("--model", required=True, help="Model name, e.g. llama3.2:8b or claude-3-5-haiku-20241022")
    p.add_argument(
        "--backend",
        default=os.getenv("ANA_BACKEND", "ollama"),
        choices=["ollama", "claude"],
        help="Inference backend (default: ollama)",
    )
    p.add_argument("--session-id", default="console", help="Session key for in-memory history")
    p.add_argument("--no-guardrail", action="store_true", help="Skip the guardrail stage (single-pass mode)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(
        main(
            model=args.model,
            backend=args.backend,
            session_id=args.session_id,
            guardrail=not args.no_guardrail,
        )
    )
