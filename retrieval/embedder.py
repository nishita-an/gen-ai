"""
Embedder
─────────
Wraps sentence-transformers to produce dense vector embeddings.

Design:
  • Model is loaded once (lazy singleton) to avoid repeated I/O.
  • encode() accepts a single string or a list of strings.
  • Always returns List[float] (single) or List[List[float]] (batch).
"""

from __future__ import annotations

import logging
from typing import List, Union

from config.settings import settings

logger = logging.getLogger(__name__)

_model_cache: dict = {}


def _get_model(model_name: str):
    """Load and cache the sentence-transformer model."""
    if model_name not in _model_cache:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model: %s", model_name)
        _model_cache[model_name] = SentenceTransformer(model_name)
        logger.info("Embedding model loaded.")
    return _model_cache[model_name]


class Embedder:
    """
    Thin wrapper around sentence-transformers.

    Usage:
        embedder = Embedder()
        vec = embedder.encode("What is backpropagation?")
        batch = embedder.encode_batch(["fact one", "fact two"])
    """

    def __init__(self, model_name: str | None = None) -> None:
        self._model_name = model_name or settings.embedding.model_name
        self._model = None   # lazy load on first use

    def _load(self):
        if self._model is None:
            self._model = _get_model(self._model_name)
        return self._model

    def encode(self, text: str) -> List[float]:
        """Embed a single string → List[float] (length = embedding_dim)."""
        model = self._load()
        vec = model.encode(text, normalize_embeddings=True)
        return vec.tolist()

    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple strings in one forward pass."""
        if not texts:
            return []
        model = self._load()
        vecs = model.encode(texts, normalize_embeddings=True, batch_size=32)
        return [v.tolist() for v in vecs]

    def similarity(self, text_a: str, text_b: str) -> float:
        """
        Cosine similarity between two texts.
        Returns a float in [-1, 1]; typically [0, 1] for sentence-transformers.
        Used for topic-shift detection.
        """
        import numpy as np
        va = self.encode(text_a)
        vb = self.encode(text_b)
        a, b = np.array(va), np.array(vb)
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        return float(np.dot(a, b) / denom) if denom > 0 else 0.0

    @property
    def dim(self) -> int:
        return settings.embedding.embedding_dim

    def __repr__(self) -> str:
        return f"Embedder(model={self._model_name}, dim={self.dim})"
