from __future__ import annotations

from langchain_openai import ChatOpenAI
from langchain_core.rate_limiters import InMemoryRateLimiter


def build_llm(*, api_key: str, model: str, temperature: float = 0.7) -> ChatOpenAI:
    api_key = (api_key or "").strip()
    if not api_key:
        raise RuntimeError("Missing OpenAI API key (env OPENAI_API_KEY or ANA_OPENAI_KEY_FILE).")

    # Optional pacing (prevents bursts). You can tweak or remove.
    rate_limiter = InMemoryRateLimiter(
        requests_per_second=0.5,   # ~1 request / 2 seconds
        check_every_n_seconds=0.1,
        max_bucket_size=1,         # no bursts
    )

    return ChatOpenAI(
        model=model,
        temperature=temperature,
        openai_api_key=api_key,
        rate_limiter=rate_limiter,
    )
