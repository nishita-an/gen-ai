"""
Summarizer
───────────
Uses the LLM to compress conversation chunks into structured summaries.

Output is a structured EpisodicEntry preserving:
  • key decisions
  • technical facts
  • user goals
  • constraints
  • unresolved questions
  • topic tags (for future retrieval scoring)

Parsing: asks the LLM for JSON, handles partial/malformed JSON gracefully.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import List, Optional

from llm.grok_client import BaseLLMClient, get_llm_client
from memory.episodic_memory import EpisodicEntry
from memory.short_term import Turn
from services.importance_scorer import ImportanceScorer
from config.settings import settings

logger = logging.getLogger(__name__)

_CHUNK_SUMMARY_PROMPT = """\
You are a conversation memory manager. Below is a segment of a study-assistant conversation.
Extract a structured summary in valid JSON with exactly these keys:

{{
  "summary": "<2-4 sentence overview of what was discussed>",
  "decisions": ["<decision 1>", ...],
  "technical_facts": ["<concrete fact 1>", ...],
  "constraints": ["<constraint 1>", ...],
  "user_goals": ["<goal 1>", ...],
  "unresolved_questions": ["<open question 1>", ...],
  "topic_tags": ["<1-word tag>", ...]
}}

Rules:
- Preserve ALL technical details, code mentions, algorithms, tools, frameworks.
- Capture any explicit decisions the user or assistant made.
- List constraints (things to avoid, performance limits, etc.).
- Note any unresolved questions the user raised but did not get answered.
- topic_tags should be 1-3 lowercase words per tag, e.g. ["backpropagation", "pytorch", "memory"].
- Output ONLY valid JSON — no markdown fences, no extra text.

Conversation segment:
{conversation}
"""

_SESSION_SUMMARY_PROMPT = """\
You are a conversation memory manager. Below are summaries from multiple conversation chunks
in a single study session. Create a unified session-level summary in valid JSON:

{{
  "summary": "<3-5 sentence high-level overview of the entire session>",
  "decisions": ["<key decision>", ...],
  "technical_facts": ["<important fact>", ...],
  "constraints": ["<constraint>", ...],
  "user_goals": ["<goal>", ...],
  "unresolved_questions": ["<open question>", ...],
  "topic_tags": ["<tag>", ...]
}}

Chunk summaries:
{summaries}
"""


def _parse_json_response(raw: str) -> dict:
    """Parse LLM JSON output, stripping markdown fences if present."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        cleaned = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        logger.warning("JSON parse failed: %s\nRaw: %.200s", exc, raw)
        # Fallback: return minimal valid dict
        return {
            "summary": raw[:500],
            "decisions": [],
            "technical_facts": [],
            "constraints": [],
            "user_goals": [],
            "unresolved_questions": [],
            "topic_tags": [],
        }


class Summarizer:
    """
    Converts raw conversation turns or existing summaries
    into structured EpisodicEntry objects via LLM.
    """

    def __init__(self, llm: Optional[BaseLLMClient] = None) -> None:
        self._llm = llm or get_llm_client()
        self._scorer = ImportanceScorer()
        logger.info("Summarizer initialised")

    def summarise_chunk(
        self,
        turns: List[Turn],
        session_id: str,
    ) -> EpisodicEntry:
        """
        Produce a chunk-level EpisodicEntry from a list of conversation turns.
        """
        if not turns:
            raise ValueError("Cannot summarise an empty chunk.")

        conversation_text = "\n".join(
            f"[{t.role.upper()}]: {t.content}" for t in turns
        )
        prompt = _CHUNK_SUMMARY_PROMPT.format(conversation=conversation_text)

        raw = self._llm.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,   # lower temp for structured extraction
        )
        parsed = _parse_json_response(raw)

        # Compute importance from the raw text
        full_text = conversation_text + " " + parsed.get("summary", "")
        importance = self._scorer.score(full_text, total_turns=len(turns))

        entry = EpisodicEntry(
            entry_id=str(uuid.uuid4()),
            session_id=session_id,
            level="chunk",
            summary=parsed.get("summary", ""),
            decisions=parsed.get("decisions", []),
            technical_facts=parsed.get("technical_facts", []),
            constraints=parsed.get("constraints", []),
            user_goals=parsed.get("user_goals", []),
            unresolved_questions=parsed.get("unresolved_questions", []),
            topic_tags=parsed.get("topic_tags", []),
            importance_score=importance,
            start_turn=turns[0].turn_id,
            end_turn=turns[-1].turn_id,
        )
        logger.info(
            "Summarizer: chunk summary created (turns %d→%d, importance=%.2f)",
            entry.start_turn, entry.end_turn, importance,
        )
        return entry

    def summarise_session(
        self,
        chunk_summaries: List[str],
        session_id: str,
    ) -> EpisodicEntry:
        """
        Roll up multiple chunk summaries into one session-level summary.
        Called when the number of chunks exceeds a threshold.
        """
        if not chunk_summaries:
            raise ValueError("No summaries to roll up.")

        summaries_text = "\n\n---\n\n".join(
            f"Chunk {i+1}:\n{s}" for i, s in enumerate(chunk_summaries)
        )
        prompt = _SESSION_SUMMARY_PROMPT.format(summaries=summaries_text)

        raw = self._llm.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        parsed = _parse_json_response(raw)

        entry = EpisodicEntry(
            entry_id=str(uuid.uuid4()),
            session_id=session_id,
            level="session",
            summary=parsed.get("summary", ""),
            decisions=parsed.get("decisions", []),
            technical_facts=parsed.get("technical_facts", []),
            constraints=parsed.get("constraints", []),
            user_goals=parsed.get("user_goals", []),
            unresolved_questions=parsed.get("unresolved_questions", []),
            topic_tags=parsed.get("topic_tags", []),
            importance_score=0.9,    # session summaries always highly important
            start_turn=0,
            end_turn=0,
        )
        logger.info("Summarizer: session summary created for session_id=%s", session_id)
        return entry
