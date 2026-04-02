from __future__ import annotations

from typing import Protocol

from tenacity import retry, stop_after_attempt, wait_exponential


class LLMProvider(Protocol):
    """Common interface for all inference backends."""

    def complete(self, *, system: str, messages: list[dict], temperature: float) -> str:
        ...


def _retry_decorator():
    return retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )


class ClaudeProvider:
    """Anthropic Claude API backend."""

    def __init__(self, api_key: str, model: str) -> None:
        if not api_key:
            raise RuntimeError(
                "Missing Anthropic API key. Set ANTHROPIC_API_KEY or ANA_ANTHROPIC_KEY_FILE."
            )
        try:
            import anthropic as _anthropic
        except ImportError as e:
            raise RuntimeError(
                "anthropic package not installed. Run: pip install anthropic"
            ) from e

        self._client = _anthropic.Anthropic(api_key=api_key)
        self._model = model

    @_retry_decorator()
    def complete(self, *, system: str, messages: list[dict], temperature: float) -> str:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=system,
            messages=messages,
            temperature=temperature,
        )
        return response.content[0].text


class OllamaProvider:
    """Local Ollama backend (100% private, no data egress)."""

    def __init__(self, base_url: str, model: str) -> None:
        try:
            import ollama as _ollama
        except ImportError as e:
            raise RuntimeError(
                "ollama package not installed. Run: pip install ollama"
            ) from e

        self._client = _ollama.Client(host=base_url)
        self._model = model

    @_retry_decorator()
    def complete(self, *, system: str, messages: list[dict], temperature: float) -> str:
        full_messages = [{"role": "system", "content": system}] + messages
        response = self._client.chat(
            model=self._model,
            messages=full_messages,
            options={"temperature": temperature},
        )
        return response["message"]["content"]


def build_provider(*, backend: str, model: str, anthropic_api_key: str, ollama_base_url: str) -> LLMProvider:
    if backend == "claude":
        return ClaudeProvider(api_key=anthropic_api_key, model=model)
    elif backend == "ollama":
        return OllamaProvider(base_url=ollama_base_url, model=model)
    else:
        raise ValueError(f"Unknown backend: {backend!r}. Choose 'ollama' or 'claude'.")
