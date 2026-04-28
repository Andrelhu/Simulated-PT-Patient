from __future__ import annotations

import asyncio
import datetime as dt
from pathlib import Path

from .config import load_settings
from .llm import build_llm
from .chains import build_chains


async def _ainput(prompt: str) -> str:
    return await asyncio.to_thread(input, prompt)


def _new_log_path(logs_dir: Path) -> Path:
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return logs_dir / f"ana_console_{ts}.txt"


def _append_log(path: Path, user_text: str, ana_text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(f"USER: {user_text}\n")
        f.write(f"ANA: {ana_text}\n\n")


async def main(*, model: str, session_id: str = "console", retry_attempts: int = 3):
    s = load_settings()

    llm = build_llm(api_key=s.openai_api_key, model=model, temperature=s.temperature)
    chain = build_chains(
        llm=llm,
        behavior_file=s.behavior_file,
        character_file=s.character_file,
        retry_attempts=retry_attempts,
    )

    cfg = {"configurable": {"session_id": session_id}}
    log_path = _new_log_path(s.logs_dir)

    print("Ana console. Type /exit.\n")
    print(f"model: {model}")
    print(f"session_id: {session_id}")
    print(f"log: {log_path}\n")

    while True:
        user = (await _ainput("you> ")).strip()
        if not user:
            continue
        if user.lower() in {"/exit", "exit", "quit"}:
            break

        msg = await chain.ainvoke({"input": user}, config=cfg)
        text = msg.content

        print(f"ana> {text}\n")
        _append_log(log_path, user, text)
