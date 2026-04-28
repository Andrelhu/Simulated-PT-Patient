from __future__ import annotations

import argparse
import asyncio

from .console_app import main


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="ana_agent")
    p.add_argument("--model", required=True, help="e.g. gpt-4o-mini")
    p.add_argument("--session-id", default="console", help="session key for in-memory history")
    p.add_argument("--retry", type=int, default=3, help="retry attempts per stage")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(model=args.model, session_id=args.session_id, retry_attempts=args.retry))
