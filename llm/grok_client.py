"""
LLM Client — Groq API (via official groq-python SDK)
──────────────────────────────────────────────────────
Root fix: chat() now accepts temperature and max_tokens kwargs
so that summarizer.py and fact_extractor.py can call them without crashing.
"""

import os
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Optional

from groq import Groq
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


# ── Abstract base ─────────────────────────────────────────────────────────────

class BaseLLMClient(ABC):
    @abstractmethod
    def chat(
        self,
        prompt: str = None,
        history: list = None,
        messages: list = None,
        system: str = None,
        temperature: float = None,
        max_tokens: int = None,
    ) -> str:
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        pass


# ── Groq client ───────────────────────────────────────────────────────────────

class GroqClient(BaseLLMClient):
    """
    Concrete Groq implementation using the official groq-python SDK.

    Accepts temperature and max_tokens so that all internal services
    (Summarizer, FactExtractor, ContextAssembler) can call .chat()
    without triggering unexpected keyword argument errors.
    """

    DEFAULT_TEMPERATURE = 0.7
    DEFAULT_MAX_TOKENS = 2048

    def __init__(self, model_name: str = "llama-3.3-70b-versatile"):
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GROQ_API_KEY not found in environment variables. "
                "Add it to your .env file as: GROQ_API_KEY=gsk_..."
            )
        self.client = Groq(api_key=self.api_key)
        self.model_name = model_name
        logger.info("GroqClient initialised (model=%s)", self.model_name)

    def chat(
        self,
        prompt: str = None,
        history: list = None,
        messages: list = None,
        system: str = None,
        temperature: float = None,
        max_tokens: int = None,
    ) -> str:
        """
        Flexible chat method that handles all calling conventions used
        across MemoryAgent, Summarizer, FactExtractor, and ContextAssembler.

        Priority order:
          1. system  -> prepended as {"role": "system", ...}
          2. messages -> used directly if provided  (MemoryAgent / ContextAssembler)
          3. history + prompt -> fallback for simple callers
        """
        final_messages: List[Dict[str, str]] = []

        # 1. System prompt
        if system:
            final_messages.append({"role": "system", "content": system})

        # 2. Structured messages list (primary path from MemoryAgent)
        if messages:
            final_messages.extend(messages)

        # 3. Fallback: history + single prompt
        else:
            if history:
                final_messages.extend(history)
            if prompt:
                final_messages.append({"role": "user", "content": prompt})

        if not final_messages:
            logger.warning("GroqClient.chat() called with no messages.")
            return ""

        resolved_temp = temperature if temperature is not None else self.DEFAULT_TEMPERATURE
        resolved_tokens = max_tokens if max_tokens is not None else self.DEFAULT_MAX_TOKENS

        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=final_messages,
                temperature=resolved_temp,
                max_tokens=resolved_tokens,
            )
            reply = completion.choices[0].message.content
            logger.debug("GroqClient reply: %d chars", len(reply))
            return reply.strip()

        except Exception as e:
            logger.error("Groq API error: %s", e)
            return f"Error communicating with Groq: {e}"

    def count_tokens(self, text: str) -> int:
        return count_tokens_simple(text)

    def __repr__(self) -> str:
        return f"GroqClient(model={self.model_name})"


# ── Utilities ─────────────────────────────────────────────────────────────────

def count_tokens_simple(text: str) -> int:
    """1 token approximately 4 characters for English text."""
    if not text:
        return 1
    return max(1, len(text) // 4)


def count_messages_tokens(messages: List[Dict[str, str]]) -> int:
    """Sum token estimates for a list of message dicts."""
    total = sum(len(m.get("content", "")) // 4 for m in messages)
    total += len(messages) * 4
    total += 2
    return total


# ── Factory ───────────────────────────────────────────────────────────────────

def get_llm_client() -> BaseLLMClient:
    """
    Returns the configured LLM client.
    To swap providers, replace GroqClient() with your own BaseLLMClient subclass.
    """
    return GroqClient()