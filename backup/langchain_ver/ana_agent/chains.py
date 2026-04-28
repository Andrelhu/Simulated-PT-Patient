from __future__ import annotations

from pathlib import Path
from typing import Dict

from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory


_STORE: Dict[str, BaseChatMessageHistory] = {}


def _get_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in _STORE:
        _STORE[session_id] = InMemoryChatMessageHistory()
    return _STORE[session_id]


def _read_text(p: Path) -> str:
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")
    return p.read_text(encoding="utf-8").strip()


def _retry(runnable, attempts: int):
    # LCEL-native retry (no RetryPolicy import).
    return runnable.with_retry(
        stop_after_attempt=attempts,
        wait_exponential_jitter=True,
    )


def build_chains(
    *,
    llm,
    behavior_file: Path,
    character_file: Path,
    retry_attempts: int = 3,
) -> RunnableWithMessageHistory:
    """Builds the 4-stage chain and wraps it with in-session history."""
    behavior = _read_text(behavior_file)
    character = _read_text(character_file)

    # Stage 1: Draft
    s1_prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
             "You are Ana (the patient).\n\nBEHAVIOR RULES:\n{behavior}\n\nCHARACTER:\n{character}\n\n"
             "Write a DRAFT reply. Stay in character. No meta."),
            MessagesPlaceholder("history"),
            ("human", "{input}"),
        ]
    ).partial(behavior=behavior, character=character)
    s1 = _retry(s1_prompt | llm | StrOutputParser(), retry_attempts)

    # Stage 2: Guardrail rewrite
    s2_prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
             "You are a strict editor for Ana.\n\nBEHAVIOR RULES:\n{behavior}\n\nCHARACTER:\n{character}\n\n"
             "Rewrite the draft to obey rules (answer only what’s asked, no medical jargon/diagnosis, "
             "one question at a time). Return ONLY the corrected answer."),
            MessagesPlaceholder("history"),
            ("human", "USER:\n{input}\n\nDRAFT:\n{draft}"),
        ]
    ).partial(behavior=behavior, character=character)
    s2 = _retry(s2_prompt | llm | StrOutputParser(), retry_attempts)

    # Stage 3: Style fidelity
    s3_prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
             "Style pass: increase Ana fidelity without adding new facts.\n\nCHARACTER:\n{character}\n\n"
             "Return ONLY the revised text."),
            MessagesPlaceholder("history"),
            ("human", "{text}"),
        ]
    ).partial(character=character)
    s3 = _retry(s3_prompt | llm | StrOutputParser(), retry_attempts)

    # Stage 4: Final answer (AIMessage)
    s4_prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
             "Final pass.\n\nBEHAVIOR RULES:\n{behavior}\n\nCHARACTER:\n{character}\n\n"
             "Return ONLY the final patient message. No meta."),
            MessagesPlaceholder("history"),
            ("human", "{text}"),
        ]
    ).partial(behavior=behavior, character=character)
    s4 = _retry(s4_prompt | llm, retry_attempts)

    core_chain = (
        RunnablePassthrough()
        .assign(draft=s1)    # adds "draft"
        .assign(text=s2)     # sets "text" to corrected
        .assign(text=s3)     # sets "text" to style-adjusted
        | s4                 # returns AIMessage
    )

    return RunnableWithMessageHistory(
        core_chain,
        _get_history,
        input_messages_key="input",
        history_messages_key="history",
    )
