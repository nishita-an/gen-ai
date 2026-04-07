"""
retriever.py
────────────
Query pipeline for the Multilingual RAG system.

Handles:
• Cross-language retrieval — the multilingual embedding model maps queries
  and documents into a shared semantic space regardless of language.
• Multi-document answer spanning — up to TOP_K chunks from different files
  are merged into one context window.
• Source deduplication — consecutive chunks from the same page are merged
  before being sent to the LLM to reduce redundancy.
• Structured source citations returned alongside the LLM answer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import chromadb

from config import (
    CHROMA_DB_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    TOP_K,
    GROQ_API_KEY,
    GROQ_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    SYSTEM_PROMPT,
    RAG_PROMPT_TEMPLATE,
)
from utils.embedder import ChromaEmbeddingFunction, Embedder

logger = logging.getLogger(__name__)


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    text       : str
    source     : str
    filepath   : str
    page_num   : int
    language   : str
    distance   : float
    has_tables : bool = False
    has_images : bool = False


@dataclass
class RAGResponse:
    answer   : str
    sources  : list[RetrievedChunk]
    query    : str
    model    : str
    error    : str | None = None


# ── Retriever class ───────────────────────────────────────────────────────────

class MultilingualRetriever:
    """
    Retrieves relevant chunks from ChromaDB and generates an answer via Groq.

    Parameters
    ──────────
    top_k     : number of chunks to retrieve per query
    """

    def __init__(self, top_k: int = TOP_K) -> None:
        self.top_k    = top_k
        self._collection: chromadb.Collection | None = None
        self._embedder: Embedder | None = None
        self._groq_client = None

    # ── Lazy initialisation ───────────────────────────────────────────────────

    @property
    def collection(self) -> chromadb.Collection:
        if self._collection is None:
            ef = ChromaEmbeddingFunction(model_name=EMBEDDING_MODEL)
            client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
            try:
                self._collection = client.get_collection(
                    name=COLLECTION_NAME,
                    embedding_function=ef,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Vector store '{COLLECTION_NAME}' not found at {CHROMA_DB_PATH}. "
                    "Did you run `python ingest.py` first?"
                ) from exc
        return self._collection

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = Embedder()
        return self._embedder

    @property
    def groq_client(self):
        if self._groq_client is None:
            try:
                from groq import Groq
                if not GROQ_API_KEY:
                    raise ValueError("GROQ_API_KEY is not set in .env")
                self._groq_client = Groq(api_key=GROQ_API_KEY)
            except ImportError as exc:
                raise RuntimeError("groq package not installed. Run: pip install groq") from exc
        return self._groq_client

    # ── Core retrieval ────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """
        Embed *query* and retrieve the top-k most similar chunks.

        The multilingual embedding model ensures that a Hindi query can
        retrieve English (or Kannada) documents if the semantic content matches.

        Returns a list of RetrievedChunk, sorted by ascending cosine distance
        (i.e., best match first).
        """
        k = top_k or self.top_k

        query_vec = self.embedder.embed_one(query)

        try:
            results = self.collection.query(
                query_embeddings=[query_vec],
                n_results=min(k, self.collection.count()),
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            logger.error("ChromaDB query failed: %s", exc)
            return []

        chunks: list[RetrievedChunk] = []
        docs       = results.get("documents", [[]])[0]
        metas      = results.get("metadatas", [[]])[0]
        distances  = results.get("distances", [[]])[0]

        for doc, meta, dist in zip(docs, metas, distances):
            chunks.append(
                RetrievedChunk(
                    text       = doc,
                    source     = meta.get("source", "unknown"),
                    filepath   = meta.get("filepath", ""),
                    page_num   = int(meta.get("page_num", 0)),
                    language   = meta.get("language", "en"),
                    distance   = float(dist),
                    has_tables = bool(meta.get("has_tables", False)),
                    has_images = bool(meta.get("has_images", False)),
                )
            )

        return chunks

    # ── Context formatting ────────────────────────────────────────────────────

    @staticmethod
    def format_context(chunks: list[RetrievedChunk]) -> str:
        """
        Format retrieved chunks into an LLM-ready context string.

        Consecutive chunks from the same source + page are merged to reduce
        redundant preamble lines and save tokens.
        """
        if not chunks:
            return "(No relevant documents found.)"

        merged = _merge_adjacent_chunks(chunks)

        context_parts: list[str] = []
        for idx, chunk in enumerate(merged, start=1):
            lang_note = f" [{chunk.language.upper()}]" if chunk.language != "en" else ""
            header = (
                f"[Source {idx}] {chunk.source}{lang_note} — page {chunk.page_num}"
            )
            if chunk.has_tables:
                header += " [contains table]"
            context_parts.append(f"{header}\n{chunk.text}")

        return "\n\n---\n\n".join(context_parts)

    # ── LLM generation ────────────────────────────────────────────────────────

    def generate(self, query: str, chunks: list[RetrievedChunk]) -> str:
        """Call Groq to generate an answer given the query and retrieved chunks."""
        context = self.format_context(chunks)
        user_message = RAG_PROMPT_TEMPLATE.format(
            context  = context,
            question = query,
        )

        try:
            response = self.groq_client.chat.completions.create(
                model    = GROQ_MODEL,
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_message},
                ],
                temperature = LLM_TEMPERATURE,
                max_tokens  = LLM_MAX_TOKENS,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            logger.error("Groq generation failed: %s", exc)
            return f"⚠️ LLM generation error: {exc}"

    # ── Full pipeline ─────────────────────────────────────────────────────────

    def query(self, user_query: str) -> RAGResponse:
        """
        End-to-end: retrieve → generate → return RAGResponse.

        Parameters
        ──────────
        user_query : the user's question (any supported language)

        Returns
        ───────
        RAGResponse with answer text and source citations.
        """
        if not user_query.strip():
            return RAGResponse(
                answer  = "Please enter a question.",
                sources = [],
                query   = user_query,
                model   = GROQ_MODEL,
                error   = "empty query",
            )

        logger.info("Query: %s", user_query[:120])

        try:
            chunks = self.retrieve(user_query)
        except RuntimeError as exc:
            return RAGResponse(
                answer  = str(exc),
                sources = [],
                query   = user_query,
                model   = GROQ_MODEL,
                error   = str(exc),
            )

        if not chunks:
            return RAGResponse(
                answer  = "No relevant documents found in the knowledge base.",
                sources = [],
                query   = user_query,
                model   = GROQ_MODEL,
            )

        answer = self.generate(user_query, chunks)

        logger.info("Generated answer (%d chars) from %d chunks.", len(answer), len(chunks))

        return RAGResponse(
            answer  = answer,
            sources = chunks,
            query   = user_query,
            model   = GROQ_MODEL,
        )

    def collection_stats(self) -> dict[str, Any]:
        """Return basic stats about the vector store."""
        try:
            count = self.collection.count()
            return {"total_chunks": count, "collection": COLLECTION_NAME, "db_path": CHROMA_DB_PATH}
        except Exception as exc:
            return {"error": str(exc)}


# ── Utility ───────────────────────────────────────────────────────────────────

def _merge_adjacent_chunks(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """
    Merge consecutive chunks that share the same source AND page_num.
    This reduces repeated header noise and token usage.
    """
    if not chunks:
        return chunks

    merged: list[RetrievedChunk] = []
    current = chunks[0]

    for nxt in chunks[1:]:
        if nxt.source == current.source and nxt.page_num == current.page_num:
            # Merge text; keep the better (lower) distance score
            current = RetrievedChunk(
                text       = current.text + "\n" + nxt.text,
                source     = current.source,
                filepath   = current.filepath,
                page_num   = current.page_num,
                language   = current.language,
                distance   = min(current.distance, nxt.distance),
                has_tables = current.has_tables or nxt.has_tables,
                has_images = current.has_images or nxt.has_images,
            )
        else:
            merged.append(current)
            current = nxt

    merged.append(current)
    return merged
