from __future__ import annotations

import re
from pathlib import Path

from .providers import LLMProvider
from .session import ConversationHistory

# Patterns that indicate the model broke character.
_BROKEN_CHARACTER_PATTERNS = re.compile(
    r"\bAs an AI\b|\bI'm an AI\b|\bI am an AI\b|\bI'm a language model\b",
    re.IGNORECASE,
)

_CONSTRAINT_REMINDER = (
    "\n\n[REMINDER — obey these at all times:\n"
    "- Answer ONLY what was directly asked\n"
    "- No medical diagnoses or clinical jargon\n"
    "- Stay in character as the patient\n"
    "- Ask at most ONE follow-up question per reply]"
)

_STAGE1_SYSTEM = """\
You are Ana, a physical therapy patient.

BEHAVIOR RULES:
{behavior}

CHARACTER DESCRIPTION:
{character}

Reply in character as Ana. Follow every behavior rule strictly. Answer only what \
the student directly asks. Do not volunteer extra information. Use plain, everyday \
language — no medical terms or diagnoses. Ask at most one follow-up question per reply. \
No meta-commentary or out-of-character text.\
{reminder}"""

_STAGE2_SYSTEM = """\
You are a strict compliance editor for a clinical simulation.

BEHAVIOR RULES:
{behavior}

A PT student asked: "{user_input}"

Below is Ana's draft response. Check it against these four rules:
1. Answers only what was directly asked (no volunteered extra information)
2. Contains no medical jargon or self-diagnosis
3. Is fully in character as a patient (no "As an AI" or meta-text)
4. Asks at most one follow-up question

If the draft violates any rule, rewrite ONLY the offending part. \
If the draft is fully compliant, return it unchanged. \
Return ONLY the final patient message — no explanations, no labels."""


def _read_text(p: Path) -> str:
    if not p.exists():
        raise FileNotFoundError(f"Missing spec file: {p}")
    return p.read_text(encoding="utf-8").strip()


def _trim_extra_questions(text: str) -> str:
    """Keep only the first question mark and everything before the second."""
    parts = text.split("?")
    if len(parts) <= 2:
        return text
    # Rejoin first question + its question mark, discard the rest.
    return parts[0] + "?"


def run_pipeline(
    user_input: str,
    history: ConversationHistory,
    provider: LLMProvider,
    behavior_file: Path,
    character_file: Path,
    temperature: float,
    guardrail: bool = True,
    refresh_turns: int = 3,
) -> str:
    behavior = _read_text(behavior_file)
    character = _read_text(character_file)

    # Inject a constraint reminder every `refresh_turns` turns to counter attention decay.
    reminder = ""
    turn = history.turn_count()
    if turn > 0 and turn % refresh_turns == 0:
        reminder = _CONSTRAINT_REMINDER

    system_s1 = _STAGE1_SYSTEM.format(behavior=behavior, character=character, reminder=reminder)
    messages_s1 = history.to_messages() + [{"role": "user", "content": user_input}]

    draft = provider.complete(system=system_s1, messages=messages_s1, temperature=temperature)

    # Programmatic checks — no extra LLM call needed.
    if _BROKEN_CHARACTER_PATTERNS.search(draft):
        # Regenerate once without the offending context.
        draft = provider.complete(system=system_s1, messages=messages_s1, temperature=temperature)

    draft = _trim_extra_questions(draft)

    if not guardrail:
        return draft

    # Stage 2: repairable compliance check (low temperature for deterministic correction).
    system_s2 = _STAGE2_SYSTEM.format(behavior=behavior, user_input=user_input)
    messages_s2 = [{"role": "user", "content": draft}]

    final = provider.complete(system=system_s2, messages=messages_s2, temperature=0.2)
    return final
