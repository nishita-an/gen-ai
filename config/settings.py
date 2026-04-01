"""
Central configuration for the Memory Agent system.
All tuneable parameters live here — no magic numbers elsewhere.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)


@dataclass
class MemorySettings:
    # Short-term (working) memory
    short_term_max_turns: int = 10          # keep last N turns exactly

    # Episodic memory
    episodic_chunk_size: int = 6            # turns before a chunk is summarised
    episodic_db_path: str = str(DATA_DIR / "episodic.db")

    # Semantic memory (FAISS)
    faiss_index_path: str = str(DATA_DIR / "faiss.index")
    faiss_meta_path: str = str(DATA_DIR / "faiss_meta.json")
    semantic_top_k: int = 5                 # chunks retrieved per query

    # User profile
    profile_path: str = str(DATA_DIR / "user_profile.json")

    # Retrieval
    summary_top_k: int = 3                  # summaries retrieved per query


@dataclass
class TokenSettings:
    # Grok / OpenAI-compatible window sizes (adjust per model)
    context_window: int = 8_192
    summarisation_threshold: float = 0.75  # trigger at 75 % usage
    max_response_tokens: int = 1_024

    # Simple token estimator — 1 token ≈ 4 chars (English)
    chars_per_token: int = 4


@dataclass
class LLMSettings:
    provider: str = "groq"
    model: str = "llama-3.3-70b-versatile"      # or "llama3-8b-8192" for faster/cheaper
    api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    base_url: str = "https://api.groq.com/openai/v1"

@dataclass
class EmbeddingSettings:
    model_name: str = "all-MiniLM-L6-v2"           # fast, local, 384-dim
    embedding_dim: int = 384


@dataclass
class ImportanceSettings:
    # Weights for composite importance score (all sum to 1.0)
    recency_weight: float = 0.30
    semantic_weight: float = 0.40
    keyword_weight: float = 0.30
    # Score threshold below which facts are eligible for decay
    decay_threshold: float = 0.25
    # How much importance decays per retrieval cycle
    decay_factor: float = 0.05


@dataclass
class AppSettings:
    memory: MemorySettings = field(default_factory=MemorySettings)
    tokens: TokenSettings = field(default_factory=TokenSettings)
    llm: LLMSettings = field(default_factory=LLMSettings)
    embedding: EmbeddingSettings = field(default_factory=EmbeddingSettings)
    importance: ImportanceSettings = field(default_factory=ImportanceSettings)
    log_level: str = "INFO"
    session_id: str = "default"


# Singleton — import and use `settings` everywhere
settings = AppSettings()
