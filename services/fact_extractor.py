"""
Fact Extractor
───────────────
Uses the LLM to pull structured, reusable facts out of conversation text.
These facts are stored in FAISS for semantic retrieval.

Fact types extracted:
  • technical   — algorithms, code, APIs, tools, frameworks
  • goal        — user learning objectives
  • constraint  — limits, requirements, things to avoid
  • preference  — user style / approach preferences
  • definition  — "X is defined as Y" statements
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import List, Optional

from llm.grok_client import BaseLLMClient, get_llm_client
from memory.semantic_memory import SemanticFact
from services.importance_scorer import ImportanceScorer
from config.settings import settings

logger = logging.getLogger(__name__)

_FACT_EXTRACTION_PROMPT = """\
You are a knowledge extraction engine for a study-assistant memory system.
Given the conversation below, extract a list of stable, reusable facts worth memorising.

Return ONLY a valid JSON array (no markdown, no preamble). Each element:
{{
  "text": "<the fact in one clear sentence>",
  "category": "<one of: technical | goal | constraint | preference | definition>"
}}

Rules:
- Extract only facts that would be useful in a future conversation.
- Do NOT extract filler, pleasantries, or transient state.
- Be specific: prefer "User is learning PyTorch for LSTM models" over "User is learning ML".
- Maximum 10 facts per call.
- If nothing worth extracting exists, return [].

Conversation:
{conversation}
"""


def _parse_facts_response(raw: str) -> List[dict]:
    """Parse LLM JSON array, handle markdown fences."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        cleaned = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    # Strip leading/trailing brackets if wrapped in object
    if not cleaned.startswith("["):
        # Try to find a JSON array anywhere in the response
        start = cleaned.find("[")
        end = cleaned.rfind("]")
        if start != -1 and end != -1:
            cleaned = cleaned[start : end + 1]
    try:
        data = json.loads(cleaned)
        if isinstance(data, list):
            return data
        return []
    except json.JSONDecodeError as exc:
        logger.warning("Fact extraction JSON parse failed: %s", exc)
        return []


class FactExtractor:
    """
    Extracts structured semantic facts from conversation turns
    and wraps them as SemanticFact objects ready for FAISS ingestion.
    """

    VALID_CATEGORIES = {"technical", "goal", "constraint", "preference", "definition"}

    def __init__(
        self,
        llm: Optional[BaseLLMClient] = None,
        session_id: str = "default",
    ) -> None:
        self._llm = llm or get_llm_client()
        self._scorer = ImportanceScorer()
        self._session_id = session_id
        logger.info("FactExtractor initialised")

    def extract(
        self,
        conversation_text: str,
        source_turn: int = 0,
    ) -> List[SemanticFact]:
        """
        Run LLM extraction on *conversation_text*.
        Returns a list of SemanticFact objects (without embeddings — caller must embed).
        """
        prompt = _FACT_EXTRACTION_PROMPT.format(conversation=conversation_text)
        raw = self._llm.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,   # deterministic extraction
        )
        raw_facts = _parse_facts_response(raw)

        facts: List[SemanticFact] = []
        for item in raw_facts:
            text = item.get("text", "").strip()
            category = item.get("category", "technical").lower()
            if not text:
                continue
            if category not in self.VALID_CATEGORIES:
                category = "technical"

            importance = self._scorer.score(text)
            fact = SemanticFact(
                fact_id=str(uuid.uuid4()),
                session_id=self._session_id,
                text=text,
                category=category,
                importance=importance,
                source_turn=source_turn,
            )
            facts.append(fact)

        logger.info("FactExtractor: extracted %d facts from conversation", len(facts))
        return facts

    def extract_from_turns(self, turns) -> List[SemanticFact]:
        """Convenience: extract from a list of Turn objects."""
        conversation_text = "\n".join(
            f"[{t.role.upper()}]: {t.content}" for t in turns
        )
        source_turn = turns[-1].turn_id if turns else 0
        return self.extract(conversation_text, source_turn=source_turn)
