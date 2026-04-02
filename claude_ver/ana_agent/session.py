from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ConversationHistory:
    """Plain in-memory conversation history. No global state, no framework magic."""

    _messages: list[dict] = field(default_factory=list)
    _turn_count: int = 0

    def append_user(self, text: str) -> None:
        self._messages.append({"role": "user", "content": text})

    def append_assistant(self, text: str) -> None:
        self._messages.append({"role": "assistant", "content": text})
        self._turn_count += 1

    def to_messages(self) -> list[dict]:
        return list(self._messages)

    def turn_count(self) -> int:
        return self._turn_count
