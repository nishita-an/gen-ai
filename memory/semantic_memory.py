"""
Semantic Memory (Vector Memory)
────────────────────────────────
Stores important facts extracted from conversations as dense embeddings.
Uses FAISS for fast approximate nearest-neighbour search.

On disk:
  • faiss.index   – the raw FAISS flat/IVF index
  • faiss_meta.json – parallel list of metadata dicts

Thread safety: FAISS writes are not thread-safe; use an asyncio lock or
               a process-level lock for concurrent FastAPI workers.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import faiss
import numpy as np

from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class SemanticFact:
    """One extractable fact stored in vector memory."""
    fact_id: str
    session_id: str
    text: str                       # the fact in natural language
    category: str                   # e.g. "technical", "goal", "constraint"
    importance: float               # 0..1 composite score
    source_turn: int
    created_at: float = 0.0
    access_count: int = 0
    embedding: Optional[List[float]] = None  # not serialised to JSON

    def __post_init__(self) -> None:
        if self.created_at == 0.0:
            self.created_at = time.time()

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("embedding", None)
        return d


class SemanticMemory:
    """
    FAISS-backed vector store for semantic facts.

    Architecture:
      IndexFlatIP (inner-product) on L2-normalised vectors → cosine similarity.
    For large corpora, swap to IndexIVFFlat with nlist clusters.
    """

    def __init__(
        self,
        index_path: str | None = None,
        meta_path: str | None = None,
        dim: int | None = None,
    ) -> None:
        self._dim = dim or settings.embedding.embedding_dim
        self._index_path = Path(index_path or settings.memory.faiss_index_path)
        self._meta_path = Path(meta_path or settings.memory.faiss_meta_path)

        self._index: faiss.Index = self._load_or_create_index()
        self._meta: List[dict] = self._load_meta()

        logger.info(
            "SemanticMemory init: dim=%d, stored=%d facts",
            self._dim, len(self._meta),
        )

    # ── Index lifecycle ─────────────────────────────────────────────────────

    def _load_or_create_index(self) -> faiss.Index:
        if self._index_path.exists():
            idx = faiss.read_index(str(self._index_path))
            logger.debug("FAISS index loaded from %s", self._index_path)
            return idx
        idx = faiss.IndexFlatIP(self._dim)  # cosine via normalised vectors
        logger.debug("FAISS index created (dim=%d)", self._dim)
        return idx

    def _load_meta(self) -> List[dict]:
        if self._meta_path.exists():
            with open(self._meta_path, "r") as f:
                return json.load(f)
        return []

    def _save(self) -> None:
        faiss.write_index(self._index, str(self._index_path))
        with open(self._meta_path, "w") as f:
            json.dump(self._meta, f, indent=2)

    # ── Core operations ─────────────────────────────────────────────────────

    @staticmethod
    def _normalise(vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        return (vec / norm) if norm > 0 else vec

    def add(self, fact: SemanticFact) -> None:
        """Store a fact and its embedding in the index."""
        if fact.embedding is None:
            raise ValueError("SemanticFact must have a non-None embedding before add().")

        vec = np.array(fact.embedding, dtype="float32").reshape(1, -1)
        vec = self._normalise(vec)
        self._index.add(vec)
        self._meta.append(fact.to_dict())
        self._save()
        logger.debug("SemanticMemory added fact_id=%s", fact.fact_id)

    def add_batch(self, facts: List[SemanticFact]) -> None:
        """Batch-add for efficiency."""
        if not facts:
            return
        vecs = np.array(
            [self._normalise(np.array(f.embedding, dtype="float32")) for f in facts],
            dtype="float32",
        )
        self._index.add(vecs)
        self._meta.extend(f.to_dict() for f in facts)
        self._save()
        logger.info("SemanticMemory batch-added %d facts", len(facts))

    def search(
        self,
        query_embedding: List[float],
        top_k: int | None = None,
        min_score: float = 0.0,
    ) -> List[Tuple[dict, float]]:
        """
        Return top-k (metadata, score) pairs for a query embedding.
        Score is cosine similarity (0..1).
        """
        k = top_k or settings.memory.semantic_top_k
        if self._index.ntotal == 0:
            return []

        k = min(k, self._index.ntotal)
        q = np.array(query_embedding, dtype="float32").reshape(1, -1)
        q = self._normalise(q)

        scores, indices = self._index.search(q, k)
        results: List[Tuple[dict, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or score < min_score:
                continue
            meta = self._meta[idx]
            # bump access count in-memory (persisted on next write)
            self._meta[idx]["access_count"] = meta.get("access_count", 0) + 1
            results.append((meta, float(score)))

        logger.debug("SemanticMemory search returned %d results", len(results))
        return results

    def decay(self, decay_factor: float = 0.05) -> None:
        """Reduce importance of zero-access facts (memory decay)."""
        changed = False
        for m in self._meta:
            if m.get("access_count", 0) == 0:
                m["importance"] = max(0.0, m.get("importance", 1.0) - decay_factor)
                changed = True
        if changed:
            self._save()
        logger.debug("SemanticMemory decay applied")

    def count(self) -> int:
        return self._index.ntotal

    def __repr__(self) -> str:
        return f"SemanticMemory(facts={self.count()}, dim={self._dim})"
