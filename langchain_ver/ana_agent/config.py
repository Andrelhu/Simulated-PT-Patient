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

    openai_api_key: str
    openai_key_file: Path

    temperature: float


def load_settings() -> Settings:
    repo_root = Path(__file__).resolve().parent.parent

    data_dir = Path(os.getenv("ANA_DATA_DIR", str(repo_root / "data"))).resolve()
    specs_dir = Path(os.getenv("ANA_SPECS_DIR", str(data_dir / "specs"))).resolve()
    logs_dir = Path(os.getenv("ANA_LOGS_DIR", str(data_dir / "session_logs"))).resolve()

    behavior_file = Path(os.getenv("ANA_BEHAVIOR_FILE", str(specs_dir / "behavior.txt"))).resolve()
    character_file = Path(os.getenv("ANA_CHARACTER_FILE", str(specs_dir / "character.txt"))).resolve()

    # Key: env var takes priority; else read from file (default D:\openAI_token.txt)
    key_file = Path(os.getenv("ANA_OPENAI_KEY_FILE", r"D:\openAI_token.txt"))
    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        api_key = _read_key_file(key_file) or ""

    temperature = float(os.getenv("ANA_TEMPERATURE", "0.7"))

    logs_dir.mkdir(parents=True, exist_ok=True)
    specs_dir.mkdir(parents=True, exist_ok=True)

    return Settings(
        repo_root=repo_root,
        data_dir=data_dir,
        specs_dir=specs_dir,
        logs_dir=logs_dir,
        behavior_file=behavior_file,
        character_file=character_file,
        openai_api_key=api_key,
        openai_key_file=key_file,
        temperature=temperature,
    )
