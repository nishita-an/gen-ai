"""
utils/embedder.py
─────────────────
Wraps a SentenceTransformer model and provides batched, normalised embeddings.

Also exposes a ChromaDB-compatible EmbeddingFunction subclass so the same
model can be plugged directly into the ChromaDB collection.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
except ImportError as exc:
    raise RuntimeError("sentence-transformers not installed. Run: pip install sentence-transformers") from exc

try:
    from chromadb import EmbeddingFunction, Embeddings
    _CHROMA_AVAILABLE = True
except ImportError:
    _CHROMA_AVAILABLE = False

from config import EMBEDDING_MODEL, EMBEDDING_BATCH_SIZE


# ── Singleton model cache ─────────────────────────────────────────────────────
_model_cache: dict[str, SentenceTransformer] = {}


def get_model(model_name: str = EMBEDDING_MODEL) -> SentenceTransformer:
    """Load (and cache) the SentenceTransformer model."""
    if model_name not in _model_cache:
        logger.info("Loading embedding model: %s", model_name)
        _model_cache[model_name] = SentenceTransformer(model_name)
        logger.info("Embedding model loaded.")
    return _model_cache[model_name]


# ── Main embedder class ───────────────────────────────────────────────────────

class Embedder:
    """
    Batched, L2-normalised text embedder.

    Parameters
    ──────────
    model_name : HuggingFace model ID (default from config).
    batch_size : texts per encoding call (default from config).
    """

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL,
        batch_size: int = EMBEDDING_BATCH_SIZE,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = get_model(self.model_name)
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of texts.

        Returns
        ───────
        List of float lists (one per input text), L2-normalised.
        """
        if not texts:
            return []

        all_embeddings: list[np.ndarray] = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            try:
                vecs = self.model.encode(
                    batch,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )
                all_embeddings.append(vecs)
            except Exception as exc:
                logger.error(
                    "Embedding failed for batch %d-%d: %s",
                    i, i + len(batch), exc,
                )
                # Return zero vectors for failed batch to keep index aligned
                dim = self.model.get_sentence_embedding_dimension() or 768
                zero = np.zeros((len(batch), dim), dtype=np.float32)
                all_embeddings.append(zero)

        combined = np.vstack(all_embeddings)
        return combined.tolist()

    def embed_one(self, text: str) -> list[float]:
        """Convenience wrapper for a single text."""
        return self.embed([text])[0]


# ── ChromaDB-compatible EmbeddingFunction ─────────────────────────────────────

if _CHROMA_AVAILABLE:
    class ChromaEmbeddingFunction(EmbeddingFunction):
        """
        Drop-in ChromaDB EmbeddingFunction backed by a SentenceTransformer.

        Usage:
            ef = ChromaEmbeddingFunction()
            collection = client.get_or_create_collection("name", embedding_function=ef)
        """

        def __init__(self, model_name: str = EMBEDDING_MODEL) -> None:
            self._embedder = Embedder(model_name=model_name)

        def __call__(self, input: list[str]) -> Embeddings:  # noqa: A002
            return self._embedder.embed(input)

else:
    # Fallback stub so imports don't break when chromadb isn't installed yet
    class ChromaEmbeddingFunction:  # type: ignore[no-redef]
        def __init__(self, model_name: str = EMBEDDING_MODEL) -> None:
            self._embedder = Embedder(model_name=model_name)

        def __call__(self, input: list[str]) -> list[list[float]]:  # noqa: A002
            return self._embedder.embed(input)
