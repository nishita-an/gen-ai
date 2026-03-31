"""
Memory Agent — Core Orchestrator
──────────────────────────────────
Wires all memory components together and implements the full
7-step context management loop for each incoming message.

Step 1  Store message in short-term memory
Step 2  Estimate token usage
Step 3  If threshold exceeded → compress oldest chunk to episodic + semantic
Step 4  Retrieve hybrid context (STM + summaries + vector + profile)
Step 5  Assemble structured prompt
Step 6  Generate LLM response
Step 7  Store assistant response in memory + extract profile updates
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

from memory.short_term import ShortTermMemory
from memory.episodic_memory import EpisodicMemory
from memory.semantic_memory import SemanticMemory
from memory.user_profile_memory import UserProfileMemory
from retrieval.embedder import Embedder
from retrieval.vector_store import VectorStore
from llm.grok_client import BaseLLMClient, get_llm_client, count_messages_tokens
from services.summarizer import Summarizer
from services.fact_extractor import FactExtractor
from services.context_assembler import ContextAssembler
from services.importance_scorer import ImportanceScorer
from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class AgentResponse:
    """Returned by MemoryAgent.chat()."""
    reply: str
    session_id: str
    turn_count: int
    token_estimate: int
    compression_triggered: bool
    latency_ms: float


class MemoryAgent:
    """
    The central orchestrator for the persistent memory system.

    One instance per session. Create a new instance (or pass a fresh session_id)
    to start a new session while reusing the same persistent stores.
    """

    def __init__(
        self,
        session_id: str = "default",
        llm: Optional[BaseLLMClient] = None,
    ) -> None:
        self.session_id = session_id

        # ── Memory layers ────────────────────────────────────────────────
        self.stm = ShortTermMemory()
        self.episodic = EpisodicMemory()
        self.embedder = Embedder()
        self.semantic = SemanticMemory()
        self.vector_store = VectorStore(embedder=self.embedder, semantic_memory=self.semantic)
        self.user_profile = UserProfileMemory()

        # ── Services ─────────────────────────────────────────────────────
        self._llm = llm or get_llm_client()
        self._summarizer = Summarizer(llm=self._llm)
        self._fact_extractor = FactExtractor(llm=self._llm, session_id=session_id)
        self._assembler = ContextAssembler(
            episodic=self.episodic,
            vector_store=self.vector_store,
            user_profile=self.user_profile,
            session_id=session_id,
        )
        self._scorer = ImportanceScorer()

        self._turn_count = 0
        logger.info("MemoryAgent started (session=%s)", session_id)

    # ── Main entry point ──────────────────────────────────────────────────────

    def chat(self, user_message: str) -> AgentResponse:
        """
        Process one user message through the full memory pipeline.
        Returns an AgentResponse with the LLM reply and metadata.
        """
        t0 = time.monotonic()
        compression_triggered = False

        # Step 1 — Store user message in short-term memory
        self.stm.add("user", user_message)
        self._turn_count += 1
        logger.info("Turn %d — user message stored (len=%d)", self._turn_count, len(user_message))

        # Step 2 — Estimate token usage
        current_tokens = self._estimate_total_tokens(user_message)
        threshold = int(settings.tokens.context_window * settings.tokens.summarisation_threshold)
        logger.debug("Token estimate: %d / %d (threshold %d)", current_tokens, settings.tokens.context_window, threshold)

        # Step 3 — Compress if over threshold OR topic shift OR chunk size limit
        chunk_size = settings.memory.episodic_chunk_size
        topic_shift = self.vector_store.detect_topic_shift(user_message)
        should_compress = (
            current_tokens > threshold
            or topic_shift
            or len(self.stm) >= chunk_size * 2
        )

        if should_compress:
            compression_triggered = True
            self._compress_oldest_chunk()

        # Step 4 + 5 — Assemble context
        system_prompt, messages = self._assembler.assemble(
            query=user_message,
            recent_turns=self.stm.get_all(),
        )

        # Step 6 — Generate LLM response
        reply = self._llm.chat(messages=messages, system=system_prompt)
        logger.info("Turn %d — LLM responded (%d chars)", self._turn_count, len(reply))

        # Step 7 — Store assistant reply + update profile
        self.stm.add("assistant", reply)
        self._maybe_update_user_profile(user_message, reply)

        latency = (time.monotonic() - t0) * 1000
        return AgentResponse(
            reply=reply,
            session_id=self.session_id,
            turn_count=self._turn_count,
            token_estimate=current_tokens,
            compression_triggered=compression_triggered,
            latency_ms=round(latency, 1),
        )

    # ── Compression pipeline ──────────────────────────────────────────────────

    def _compress_oldest_chunk(self) -> None:
        """
        Pop the oldest chunk from STM, summarise it, extract facts,
        and store everything in episodic + semantic memory.
        """
        chunk_size = settings.memory.episodic_chunk_size
        chunk = self.stm.pop_oldest_chunk(chunk_size)
        if not chunk:
            return

        logger.info("Compressing chunk of %d turns into episodic memory", len(chunk))

        # 3a. Episodic summary
        try:
            entry = self._summarizer.summarise_chunk(chunk, session_id=self.session_id)
            self.episodic.store(entry)
        except Exception as exc:
            logger.error("Summarization failed: %s", exc)

        # 3b. Semantic fact extraction + vector storage
        try:
            facts = self._fact_extractor.extract_from_turns(chunk)
            if facts:
                self.vector_store.add_facts_batch(facts)
                logger.info("Stored %d semantic facts", len(facts))
        except Exception as exc:
            logger.error("Fact extraction failed: %s", exc)

        # 3c. Apply decay to older memories
        self.vector_store.decay()
        self.episodic.decay_old_entries()

    # ── Profile updates ───────────────────────────────────────────────────────

    def _maybe_update_user_profile(self, user_msg: str, assistant_msg: str) -> None:
        """
        Periodically extract user profile facts from the conversation.
        Runs every 10 turns to avoid LLM call overhead.
        """
        if self._turn_count % 10 != 0:
            return

        _PROFILE_EXTRACTION_PROMPT = """
Extract long-term facts about the user from this conversation exchange.
Return ONLY valid JSON with these keys (omit keys with empty values):
{
  "goals": ["..."],
  "projects": ["..."],
  "technical_interests": ["..."],
  "constraints": ["..."],
  "preferences": {"key": "value"},
  "custom_facts": {"key": "value"}
}
If nothing notable, return {}.

User: {user}
Assistant: {assistant}
"""
        try:
            raw = self._llm.chat(
                messages=[{
                    "role": "user",
                    "content": _PROFILE_EXTRACTION_PROMPT.format(
                        user=user_msg, assistant=assistant_msg
                    )
                }],
                temperature=0.2,
            )
            import json
            cleaned = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            extracted = json.loads(cleaned) if cleaned else {}
            if extracted:
                self.user_profile.merge_from_llm_extraction(extracted)
        except Exception as exc:
            logger.warning("Profile update failed: %s", exc)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _estimate_total_tokens(self, latest_message: str) -> int:
        from llm.grok_client import count_tokens_simple
        stm_tokens = self.stm.token_estimate()
        msg_tokens = count_tokens_simple(latest_message)
        profile_tokens = count_tokens_simple(self.user_profile.format_for_prompt())
        return stm_tokens + msg_tokens + profile_tokens

    def status(self) -> dict:
        """Return a snapshot of memory system state — useful for debugging."""
        return {
            "session_id": self.session_id,
            "turn_count": self._turn_count,
            "stm_turns": len(self.stm),
            "episodic_entries": self.episodic.count(self.session_id),
            "semantic_facts": self.vector_store.count(),
            "stm_token_estimate": self.stm.token_estimate(),
        }
