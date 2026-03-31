"""
Short-Term (Working) Memory
───────────────────────────
Stores the last N conversation turns verbatim using a FIFO deque.
This is the "scratchpad" that always appears in the LLM prompt.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Deque, List, Literal

from config.settings import settings

logger = logging.getLogger(__name__)

Role = Literal["user", "assistant", "system"]


@dataclass
class Turn:
    """A single conversation turn."""
    role: Role
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    turn_id: int = 0
    importance: float = 1.0          # elevated later by ImportanceScorer


class ShortTermMemory:
    """
    Fixed-size FIFO buffer of recent conversation turns.

    When the buffer is full, the oldest turns are evicted and returned
    so the caller can move them to episodic / semantic memory.
    """

    def __init__(self, max_turns: int | None = None) -> None:
        self._max = max_turns or settings.memory.short_term_max_turns
        self._buffer: Deque[Turn] = deque(maxlen=self._max)
        self._turn_counter: int = 0
        logger.info("ShortTermMemory initialised (max_turns=%d)", self._max)

    # ── Public API ─────────────────────────────────────────────────────────

    def add(self, role: Role, content: str) -> Turn:
        """Append a new turn; returns the Turn object."""
        self._turn_counter += 1
        turn = Turn(role=role, content=content, turn_id=self._turn_counter)

        if len(self._buffer) == self._max:
            evicted = self._buffer[0]
            logger.debug("STM evicting turn_id=%d (role=%s)", evicted.turn_id, evicted.role)

        self._buffer.append(turn)
        logger.debug("STM add  turn_id=%d role=%s len=%d", turn.turn_id, role, len(self._buffer))
        return turn

    def get_all(self) -> List[Turn]:
        """Return all buffered turns in chronological order."""
        return list(self._buffer)

    def get_recent(self, n: int) -> List[Turn]:
        """Return the most recent *n* turns."""
        return list(self._buffer)[-n:]

    def pop_oldest_chunk(self, chunk_size: int) -> List[Turn]:
        """
        Remove and return the oldest *chunk_size* turns from the buffer.
        Used when the context window threshold is exceeded.
        """
        chunk: List[Turn] = []
        for _ in range(min(chunk_size, len(self._buffer))):
            chunk.append(self._buffer.popleft())
        logger.info("STM popped %d turns for episodic processing", len(chunk))
        return chunk

    def as_messages(self) -> List[dict]:
        """Convert buffer to OpenAI-style messages list."""
        return [{"role": t.role, "content": t.content} for t in self._buffer]

    def token_estimate(self) -> int:
        """Rough token count for the current buffer contents."""
        chars = sum(len(t.content) for t in self._buffer)
        return chars // settings.tokens.chars_per_token

    def clear(self) -> None:
        """Reset buffer (e.g. on session start)."""
        self._buffer.clear()
        logger.info("STM cleared")

    def __len__(self) -> int:
        return len(self._buffer)

    def __repr__(self) -> str:
        return f"ShortTermMemory(turns={len(self._buffer)}, max={self._max})"
