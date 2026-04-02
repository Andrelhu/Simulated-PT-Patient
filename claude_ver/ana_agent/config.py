from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _read_key_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return ""
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8-sig").strip()


@dataclass(frozen=True)
class Settings:
    repo_root: Path
    data_dir: Path
    specs_dir: Path
    logs_dir: Path

    behavior_file: Path
    character_file: Path

    backend: str          # "ollama" | "claude"
    temperature: float

    # Claude backend
    anthropic_api_key: str

    # Ollama backend
    ollama_base_url: str

    # Guardrail tuning
    guardrail_refresh_turns: int


def load_settings(*, backend: str | None = None) -> Settings:
    repo_root = Path(__file__).resolve().parent.parent

    data_dir = Path(os.getenv("ANA_DATA_DIR", str(repo_root.parent / "data"))).resolve()
    specs_dir = Path(os.getenv("ANA_SPECS_DIR", str(data_dir / "specs"))).resolve()
    logs_dir = Path(os.getenv("ANA_LOGS_DIR", str(data_dir / "session_logs"))).resolve()

    behavior_file = Path(os.getenv("ANA_BEHAVIOR_FILE", str(specs_dir / "behavior.txt"))).resolve()
    character_file = Path(os.getenv("ANA_CHARACTER_FILE", str(specs_dir / "character.txt"))).resolve()

    resolved_backend = backend or os.getenv("ANA_BACKEND", "ollama")

    # Anthropic API key: env var → key file → empty
    key_file = Path(os.getenv("ANA_ANTHROPIC_KEY_FILE", str(repo_root.parent / "anthropic_token.txt")))
    anthropic_api_key = (os.getenv("ANTHROPIC_API_KEY") or "").strip()
    if not anthropic_api_key:
        anthropic_api_key = _read_key_file(key_file)

    ollama_base_url = os.getenv("ANA_OLLAMA_BASE_URL", "http://localhost:11434")
    temperature = float(os.getenv("ANA_TEMPERATURE", "0.7"))
    guardrail_refresh_turns = int(os.getenv("ANA_GUARDRAIL_REFRESH_TURNS", "3"))

    logs_dir.mkdir(parents=True, exist_ok=True)
    specs_dir.mkdir(parents=True, exist_ok=True)

    return Settings(
        repo_root=repo_root,
        data_dir=data_dir,
        specs_dir=specs_dir,
        logs_dir=logs_dir,
        behavior_file=behavior_file,
        character_file=character_file,
        backend=resolved_backend,
        temperature=temperature,
        anthropic_api_key=anthropic_api_key,
        ollama_base_url=ollama_base_url,
        guardrail_refresh_turns=guardrail_refresh_turns,
    )
