"""
config.py — Central configuration for the Multilingual RAG Pipeline.
All tuneable parameters live here; .env overrides are applied at the bottom.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Directory layout ────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent.resolve()
PDF_FOLDER = Path(os.getenv("PDF_FOLDER", BASE_DIR / "pdfs"))
CHROMA_DB_PATH = str(Path(os.getenv("CHROMA_DB_PATH", BASE_DIR / "chroma_db")))
HASHES_FILE    = BASE_DIR / "ingested_hashes.json"
LOG_FILE       = Path(os.getenv("INGEST_LOG", BASE_DIR / "ingest.log"))

# ── Embedding model ─────────────────────────────────────────────────────────
# paraphrase-multilingual-mpnet-base-v2 → 768-dim, 50+ languages
EMBEDDING_MODEL     = os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-mpnet-base-v2")
EMBEDDING_DIMENSION = 768
EMBEDDING_BATCH_SIZE = 32   # texts per encoding call

# ── ChromaDB ────────────────────────────────────────────────────────────────
COLLECTION_NAME = "multilingual_rag"

# ── LLM (Groq) ──────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL   = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
LLM_TEMPERATURE  = 0.1
LLM_MAX_TOKENS   = 1536

# ── Chunking ─────────────────────────────────────────────────────────────────
# Characters (not tokens) per chunk; generous for CJK / Indic scripts
LANG_CHUNK_SIZES: dict[str, int] = {
    "en": 900,   # English  — longer sentences
    "fr": 900,   # French
    "de": 900,
    "es": 900,
    "hi": 700,   # Hindi    — Devanagari, compact glyphs
    "kn": 700,   # Kannada  — Brahmic script
    "ta": 700,   # Tamil
    "te": 700,   # Telugu
    "ar": 700,
    "zh": 500,   # Chinese  — each char ≈ word
    "ja": 500,
    "ko": 500,
}
DEFAULT_CHUNK_SIZE    = 900
CHUNK_OVERLAP_CHARS   = 100   # sliding-window overlap
MIN_CHUNK_LENGTH      = 60    # discard chunks shorter than this

# ── Retrieval ────────────────────────────────────────────────────────────────
TOP_K = int(os.getenv("TOP_K", 5))

# ── PDF parsing ──────────────────────────────────────────────────────────────
SCANNED_TEXT_THRESHOLD = 20   # chars per page; below → treat as scanned image
MAX_TABLE_ROWS_INLINE  = 40   # render tables > this as truncated markdown

# ── Prompt template ──────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are an expert multilingual research assistant. You will be given retrieved \
document excerpts (context) and a user question. Your job is to:
1. Answer the question accurately using ONLY the provided context.
2. If the context doesn't contain the answer, say so honestly.
3. Cite your sources by referencing [Source N] inline.
4. Respond in the SAME LANGUAGE as the user's question.
5. Be concise but complete. Do not hallucinate.
"""

RAG_PROMPT_TEMPLATE = """\
## Retrieved Context

{context}

---

## Question
{question}

## Instructions
Answer the question using the context above. Cite sources as [Source N].
If the answer cannot be found in the context, say: \
"I could not find sufficient information in the provided documents."
"""
