"""
LLM Client — Grok (xAI) with OpenAI-compatible API
─────────────────────────────────────────────────────
Designed as a pluggable interface: swap provider by changing settings.llm.provider
or by subclassing BaseLLMClient.

Supports:
  • chat completions (messages list)
  • streaming (optional)
  • exponential-backoff retry on transient errors
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import List, Dict, Optional

import httpx

from config.settings import settings

logger = logging.getLogger(__name__)


# ── Abstract base ─────────────────────────────────────────────────────────────

class BaseLLMClient(ABC):
    """All LLM clients must implement this interface."""

    @abstractmethod
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        system: str | None = None,
    ) -> str:
        """Send a chat request, return the assistant's reply as a plain string."""
        ...

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Estimate token count for *text*."""
        ...


# ── Token counting utility ────────────────────────────────────────────────────

def count_tokens_simple(text: str) -> int:
    """
    Fast heuristic: 1 token ≈ 4 characters for English text.
    Replace with tiktoken or the model provider's tokeniser for accuracy.
    """
    return max(1, len(text) // settings.tokens.chars_per_token)


def count_messages_tokens(messages: List[Dict[str, str]]) -> int:
    """Sum token estimates across all messages."""
    total = 0
    for msg in messages:
        total += count_tokens_simple(msg.get("content", ""))
        total += 4   # role + formatting overhead per message
    total += 2       # reply primer
    return total


# ── Grok / xAI client ────────────────────────────────────────────────────────

class GrokClient(BaseLLMClient):
    """
    Calls the xAI Grok API using its OpenAI-compatible /v1/chat/completions endpoint.

    Retry logic: exponential backoff on HTTP 429 / 5xx.
    """

    def __init__(self) -> None:
        cfg = settings.llm
        if not cfg.api_key:
            logger.warning(
                "GROK_API_KEY is not set — LLM calls will fail. "
                "Set the environment variable before starting."
            )
        self._api_key = cfg.api_key
        self._base_url = cfg.base_url.rstrip("/")
        self._model = cfg.model
        self._default_temp = cfg.temperature
        self._max_retries = cfg.max_retries
        self._client = httpx.Client(timeout=60.0)
        logger.info("GrokClient initialised (model=%s)", self._model)

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        system: str | None = None,
    ) -> str:
        """
        Send messages to Grok and return the assistant text.

        Args:
            messages: List of {"role": ..., "content": ...} dicts.
            temperature: Override default temperature.
            max_tokens: Override default max output tokens.
            system: If provided, prepended as a system message.
        """
        full_messages: List[Dict[str, str]] = []
        if system:
            full_messages.append({"role": "system", "content": system})
        full_messages.extend(messages)

        payload = {
            "model": self._model,
            "messages": full_messages,
            "temperature": temperature if temperature is not None else self._default_temp,
            "max_tokens": max_tokens or settings.tokens.max_response_tokens,
        }

        return self._call_with_retry(payload)

    def _call_with_retry(self, payload: dict) -> str:
        url = f"{self._base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        for attempt in range(1, self._max_retries + 1):
            try:
                response = self._client.post(url, json=payload, headers=headers)
                if response.status_code == 200:
                    data = response.json()
                    text = data["choices"][0]["message"]["content"]
                    logger.debug("GrokClient: received %d chars", len(text))
                    return text.strip()

                if response.status_code in (429, 500, 502, 503):
                    wait = 2 ** attempt
                    logger.warning(
                        "GrokClient HTTP %d — retrying in %ds (attempt %d/%d)",
                        response.status_code, wait, attempt, self._max_retries,
                    )
                    time.sleep(wait)
                    continue

                response.raise_for_status()

            except httpx.RequestError as exc:
                logger.error("GrokClient network error: %s", exc)
                if attempt == self._max_retries:
                    raise
                time.sleep(2 ** attempt)

        raise RuntimeError(f"GrokClient: all {self._max_retries} retries failed.")

    def count_tokens(self, text: str) -> int:
        return count_tokens_simple(text)

    def __repr__(self) -> str:
        return f"GrokClient(model={self._model})"


# ── Factory ───────────────────────────────────────────────────────────────────

def get_llm_client() -> BaseLLMClient:
    """
    Return the configured LLM client.
    Extend this factory to add OpenAI, Anthropic, Ollama, etc.
    """
    provider = settings.llm.provider.lower()
    if provider == "grok":
        return GrokClient()
    raise ValueError(f"Unknown LLM provider: '{provider}'. Add it to llm/grok_client.py.")
