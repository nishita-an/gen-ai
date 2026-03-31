"""
Context Assembler
──────────────────
Combines all four memory types into a single, structured LLM prompt.

Prompt layout (top → bottom):
  [SYSTEM]
    § User Profile
    § Relevant Summaries (episodic)
    § Relevant Facts (semantic)
    § Recent Conversation (short-term)
  [MESSAGES]   ← short-term turns passed as the actual messages list

Also contains:
  • token budget management (prune if over limit)
  • hybrid retrieval ranking (recency + semantic + importance)
"""

from __future__ import annotations

import logging
from typing import List, Optional

from memory.short_term import Turn
from memory.episodic_memory import EpisodicMemory, EpisodicEntry
from memory.user_profile_memory import UserProfileMemory
from retrieval.vector_store import VectorStore
from llm.grok_client import count_tokens_simple
from config.settings import settings

logger = logging.getLogger(__name__)

# ── Prompt section templates ──────────────────────────────────────────────────

_SYSTEM_HEADER = """You are an expert AI study assistant with persistent memory.
You have access to the user's profile, past conversation summaries, semantic knowledge, and the current conversation.
Use all this context to provide accurate, personalised, and coherent answers.
Never pretend you don't remember something that appears in the context below."""

_SECTION_DIVIDER = "\n" + "─" * 60 + "\n"

_PROFILE_SECTION = """
## USER PROFILE
{profile}
"""

_SUMMARIES_SECTION = """
## RELEVANT CONVERSATION SUMMARIES
{summaries}
"""

_SEMANTIC_SECTION = """
## RELEVANT KNOWLEDGE (from memory)
{facts}
"""

_RECENT_SECTION = """
## RECENT CONVERSATION
(The last {n} turns are provided below as structured messages.)
"""


class ContextAssembler:
    """
    Builds a complete LLM prompt from all memory sources.

    Usage:
        assembler = ContextAssembler(...)
        system, messages = assembler.assemble(
            query="explain attention mechanisms",
            recent_turns=stm.get_all(),
        )
        reply = llm.chat(messages, system=system)
    """

    def __init__(
        self,
        episodic: Optional[EpisodicMemory] = None,
        vector_store: Optional[VectorStore] = None,
        user_profile: Optional[UserProfileMemory] = None,
        session_id: str = "default",
    ) -> None:
        self._episodic = episodic or EpisodicMemory()
        self._vector_store = vector_store or VectorStore()
        self._profile = user_profile or UserProfileMemory()
        self._session_id = session_id
        logger.info("ContextAssembler initialised (session=%s)", session_id)

    # ── Public API ────────────────────────────────────────────────────────────

    def assemble(
        self,
        query: str,
        recent_turns: List[Turn],
    ) -> tuple[str, List[dict]]:
        """
        Build the (system_prompt, messages_list) pair for the LLM.

        Returns:
            system_prompt: Injected as the system role.
            messages: List of {"role":..., "content":...} dicts.
        """
        budget = settings.tokens.context_window - settings.tokens.max_response_tokens

        # 1. User profile (always included — immutable)
        profile_text = self._profile.format_for_prompt()

        # 2. Relevant episodic summaries
        summaries_text = self._fetch_relevant_summaries(query)

        # 3. Relevant semantic facts
        facts_text = self._fetch_relevant_facts(query)

        # 4. Build system prompt
        system = self._build_system(profile_text, summaries_text, facts_text, len(recent_turns))

        # 5. Convert recent turns to messages
        messages = [{"role": t.role, "content": t.content} for t in recent_turns]

        # 6. Token budget check — prune oldest messages if needed
        system_tokens = count_tokens_simple(system)
        messages = self._fit_messages_to_budget(messages, budget - system_tokens)

        logger.debug(
            "ContextAssembler: system=%d tok, messages=%d turns",
            system_tokens, len(messages),
        )
        return system, messages

    # ── Private helpers ───────────────────────────────────────────────────────

    def _fetch_relevant_summaries(self, query: str) -> str:
        """Retrieve top-k episodic summaries relevant to the current query."""
        entries: List[EpisodicEntry] = self._episodic.get_recent_summaries(
            session_id=self._session_id,
            n=settings.memory.summary_top_k,
        )
        if not entries:
            return "No conversation summaries available yet."

        parts = []
        for i, e in enumerate(entries, 1):
            parts.append(f"[Summary {i}] {e.to_text()}")
            self._episodic.increment_access(e.entry_id)

        return "\n\n".join(parts)

    def _fetch_relevant_facts(self, query: str) -> str:
        """Retrieve top-k semantic facts relevant to the query."""
        results = self._vector_store.query(query, top_k=settings.memory.semantic_top_k)
        if not results:
            return "No semantic memories available yet."

        lines = []
        for meta, score in results:
            cat = meta.get("category", "fact")
            text = meta.get("text", "")
            lines.append(f"• [{cat}] {text}  (relevance: {score:.2f})")

        return "\n".join(lines)

    @staticmethod
    def _build_system(
        profile: str,
        summaries: str,
        facts: str,
        n_recent: int,
    ) -> str:
        sections = [
            _SYSTEM_HEADER,
            _SECTION_DIVIDER,
            _PROFILE_SECTION.format(profile=profile),
            _SECTION_DIVIDER,
            _SUMMARIES_SECTION.format(summaries=summaries),
            _SECTION_DIVIDER,
            _SEMANTIC_SECTION.format(facts=facts),
            _SECTION_DIVIDER,
            _RECENT_SECTION.format(n=n_recent),
        ]
        return "".join(sections)

    @staticmethod
    def _fit_messages_to_budget(
        messages: List[dict],
        token_budget: int,
    ) -> List[dict]:
        """
        Drop oldest messages (from the front) until the list fits the budget.
        Always preserves at least the last 2 turns (most recent exchange).
        """
        while len(messages) > 2:
            total = sum(count_tokens_simple(m["content"]) for m in messages)
            if total <= token_budget:
                break
            messages = messages[1:]   # drop oldest
            logger.debug("Budget trim: dropped oldest message, %d remain", len(messages))
        return messages

    def get_full_context_token_estimate(self, query: str, recent_turns: List[Turn]) -> int:
        """Estimate total tokens that will be sent to the LLM."""
        system, messages = self.assemble(query, recent_turns)
        total = count_tokens_simple(system)
        total += sum(count_tokens_simple(m["content"]) for m in messages)
        return total
