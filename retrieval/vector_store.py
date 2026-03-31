"""
VectorStore
────────────
High-level retrieval facade combining the Embedder and SemanticMemory.

Adds:
  • importance-weighted re-ranking of raw FAISS scores
  • topic-shift detection using embedding cosine similarity
  • hybrid retrieval that fuses vector results with recency
"""

from __future__ import annotations

import logging
from typing import List, Tuple

from retrieval.embedder import Embedder
from memory.semantic_memory import SemanticMemory, SemanticFact
from config.settings import settings

logger = logging.getLogger(__name__)

# Topic-shift threshold: if similarity drops below this, a new topic has begun
TOPIC_SHIFT_THRESHOLD = 0.45


class VectorStore:
    """
    Retrieval layer sitting on top of SemanticMemory.

    Typical flow:
        vs = VectorStore()
        vs.add_fact(fact)
        results = vs.query("explain transformer attention", top_k=5)
    """

    def __init__(
        self,
        embedder: Embedder | None = None,
        semantic_memory: SemanticMemory | None = None,
    ) -> None:
        self.embedder = embedder or Embedder()
        self.semantic = semantic_memory or SemanticMemory()
        self._last_topic_vec: List[float] | None = None
        logger.info("VectorStore initialised")

    # ── Ingestion ───────────────────────────────────────────────────────────

    def add_fact(self, fact: SemanticFact) -> None:
        """Embed a fact (if needed) and store it."""
        if fact.embedding is None:
            fact.embedding = self.embedder.encode(fact.text)
        self.semantic.add(fact)

    def add_facts_batch(self, facts: List[SemanticFact]) -> None:
        """Batch-embed and store."""
        no_embed = [f for f in facts if f.embedding is None]
        if no_embed:
            vecs = self.embedder.encode_batch([f.text for f in no_embed])
            for f, v in zip(no_embed, vecs):
                f.embedding = v
        self.semantic.add_batch(facts)

    # ── Retrieval ────────────────────────────────────────────────────────────

    def query(
        self,
        text: str,
        top_k: int | None = None,
        min_score: float = 0.2,
    ) -> List[Tuple[dict, float]]:
        """
        Retrieve top-k semantically relevant facts for *text*.
        Returns list of (metadata_dict, importance_weighted_score).
        """
        k = top_k or settings.memory.semantic_top_k
        q_vec = self.embedder.encode(text)
        raw = self.semantic.search(q_vec, top_k=k * 2, min_score=min_score)

        # Re-rank: blend cosine score with stored importance
        reranked = []
        for meta, cos_score in raw:
            importance = meta.get("importance", 0.5)
            combined = (
                settings.importance.semantic_weight * cos_score
                + settings.importance.keyword_weight * importance
            )
            reranked.append((meta, combined))

        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked[:k]

    def query_texts(self, text: str, top_k: int | None = None) -> List[str]:
        """Convenience: return fact texts only."""
        return [r[0]["text"] for r in self.query(text, top_k=top_k)]

    # ── Topic detection ─────────────────────────────────────────────────────

    def detect_topic_shift(self, new_message: str) -> bool:
        """
        Returns True if the new message is significantly different
        from the previous conversation topic.

        Used to trigger early summarisation.
        """
        new_vec = self.embedder.encode(new_message)
        if self._last_topic_vec is None:
            self._last_topic_vec = new_vec
            return False

        import numpy as np
        a = np.array(self._last_topic_vec)
        b = np.array(new_vec)
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        similarity = float(np.dot(a, b) / denom) if denom > 0 else 1.0

        # Update rolling topic vector (EMA)
        alpha = 0.4
        self._last_topic_vec = (alpha * b + (1 - alpha) * a).tolist()

        shifted = similarity < TOPIC_SHIFT_THRESHOLD
        if shifted:
            logger.info("Topic shift detected (sim=%.3f < %.3f)", similarity, TOPIC_SHIFT_THRESHOLD)
        return shifted

    def decay(self) -> None:
        self.semantic.decay(settings.importance.decay_factor)

    def count(self) -> int:
        return self.semantic.count()

    def __repr__(self) -> str:
        return f"VectorStore(facts={self.count()})"
