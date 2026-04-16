from __future__ import annotations

import asyncio
import datetime as dt
import re
from pathlib import Path

from .config import load_settings
from .providers import build_provider
from .pipeline import run_pipeline
from .session import ConversationHistory


def _new_log_path(logs_dir: Path, backend: str, model: str) -> Path:
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_safe = re.sub(r"[^\w\-.]", "_", model)
    return logs_dir / f"{backend}_{model_safe}_{ts}.txt"


def _append_log(path: Path, user_text: str, ana_text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(f"USER: {user_text}\n")
        f.write(f"ANA: {ana_text}\n\n")


async def main(
    *,
    model: str,
    backend: str,
    session_id: str = "console",
    guardrail: bool = True,
) -> None:
    s = load_settings(backend=backend)

    provider = build_provider(
        backend=s.backend,
        model=model,
        anthropic_api_key=s.anthropic_api_key,
        ollama_base_url=s.ollama_base_url,
    )

    history = ConversationHistory()
    log_path = _new_log_path(s.logs_dir, backend, model)

    print("Ana console. Type /exit to quit.\n")
    print(f"backend:    {backend}")
    print(f"model:      {model}")
    print(f"guardrail:  {guardrail}")
    print(f"session_id: {session_id}")
    print(f"log:        {log_path}\n")

    while True:
        user = (await asyncio.to_thread(input, "you> ")).strip()
        if not user:
            continue
        if user.lower() in {"/exit", "exit", "quit"}:
            break

        text = await asyncio.to_thread(
            run_pipeline,
            user,
            history,
            provider,
            s.spec_file,
            s.temperature,
            guardrail,
            s.guardrail_refresh_turns,
        )

        history.append_user(user)
        history.append_assistant(text)

        print(f"ana> {text}\n")
        _append_log(log_path, user, text)
