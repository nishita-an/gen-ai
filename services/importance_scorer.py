"""
Importance Scorer
──────────────────
Assigns a composite importance score [0..1] to a piece of text / memory.

Score = recency_weight  × recency_score
      + semantic_weight × semantic_density_score
      + keyword_weight  × keyword_score

Used by:
  • Summarizer — to decide which details to preserve
  • SemanticMemory — stored alongside every fact
  • EpisodicMemory — for memory decay eligibility
"""

from __future__ import annotations

import logging
import math
from typing import List

from config.settings import settings

logger = logging.getLogger(__name__)

# High-value keywords that boost importance
IMPORTANT_KEYWORDS = frozenset({
    "goal", "objective", "requirement", "must", "critical", "important",
    "deadline", "constraint", "decision", "conclusion", "problem", "error",
    "bug", "fix", "always", "never", "prefer", "avoid", "key", "remember",
    "fact", "define", "specification", "architecture", "design",
})


class ImportanceScorer:
    """
    Stateless scorer — call score() on any text snippet.
    """

    def score(
        self,
        text: str,
        turn_index: int = 0,
        total_turns: int = 1,
    ) -> float:
        """
        Compute composite importance score.

        Args:
            text: The text to score.
            turn_index: Position of this turn in the conversation (0 = first).
            total_turns: Total turns seen so far.

        Returns:
            Float in [0, 1].
        """
        recency = self._recency_score(turn_index, total_turns)
        density = self._semantic_density(text)
        keywords = self._keyword_score(text)

        cfg = settings.importance
        composite = (
            cfg.recency_weight  * recency
            + cfg.semantic_weight * density
            + cfg.keyword_weight  * keywords
        )
        return round(min(1.0, max(0.0, composite)), 4)

    # ── Sub-scores ────────────────────────────────────────────────────────

    @staticmethod
    def _recency_score(turn_index: int, total_turns: int) -> float:
        """
        Exponential recency: recent turns score higher.
        Score = exp(-λ * (total - index - 1))
        λ = 0.1 gives ≈ 0.9 for the last turn, ~0.1 for 20 turns ago.
        """
        if total_turns <= 1:
            return 1.0
        distance = total_turns - turn_index - 1
        return math.exp(-0.1 * distance)

    @staticmethod
    def _semantic_density(text: str) -> float:
        """
        Proxy for information density: unique content words / total words.
        Dense technical text scores higher than filler phrases.
        """
        words = text.lower().split()
        if not words:
            return 0.0
        stop_words = {
            "the", "a", "an", "is", "it", "in", "of", "to", "and", "or",
            "that", "this", "with", "for", "on", "at", "by", "i", "you",
            "we", "they", "he", "she", "was", "are", "be", "been", "have",
            "has", "do", "does", "not", "no", "so", "but", "as", "from",
        }
        content_words = [w for w in words if w not in stop_words]
        unique = len(set(content_words))
        return min(1.0, unique / max(len(words), 1))

    @staticmethod
    def _keyword_score(text: str) -> float:
        """Fraction of important keywords found in the text."""
        words = set(text.lower().split())
        hits = words.intersection(IMPORTANT_KEYWORDS)
        if not hits:
            return 0.0
        # Soft cap: diminishing returns after 3 keywords
        return min(1.0, len(hits) / 3)

    def score_batch(self, texts: List[str]) -> List[float]:
        """Score a list of texts with uniform positional weighting."""
        n = len(texts)
        return [self.score(t, i, n) for i, t in enumerate(texts)]
