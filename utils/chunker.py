"""
utils/chunker.py
────────────────
Multilingual text chunker.

Strategy
────────
1. Detect the language of each page's text (langdetect).
2. Select a per-language chunk-size (characters, not tokens).
3. Sentence-aware splitting:
   • European / Latin-script languages  → NLTK sent_tokenize with the
     appropriate language model.
   • Indic / CJK scripts (Kannada, Hindi, Tamil, Chinese, …) → split on
     sentence-ending punctuation (।, ?, !, ., …) then on newlines, then
     fall back to hard character windows.
4. Apply a sliding-window overlap between consecutive chunks.
5. Filter out chunks that are too short to be meaningful.
"""

from __future__ import annotations

import logging
import re
from typing import Iterator

logger = logging.getLogger(__name__)

# ── langdetect ───────────────────────────────────────────────────────────────
try:
    from langdetect import detect as _langdetect_detect, LangDetectException
    _LANGDETECT_AVAILABLE = True
except ImportError:
    _LANGDETECT_AVAILABLE = False
    logger.warning("langdetect not installed — language detection disabled; defaulting to 'en'.")

# ── NLTK ─────────────────────────────────────────────────────────────────────
import nltk

def _ensure_nltk_data() -> None:
    for resource in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except Exception:
            try:
                nltk.download(resource, quiet=True, force=True)
            except Exception:
                pass

_ensure_nltk_data()

from nltk.tokenize import sent_tokenize

from config import (
    LANG_CHUNK_SIZES,
    DEFAULT_CHUNK_SIZE,
    CHUNK_OVERLAP_CHARS,
    MIN_CHUNK_LENGTH,
)

# ── Language → NLTK tokenizer name mapping ────────────────────────────────────
_NLTK_LANG_MAP: dict[str, str] = {
    "en": "english",
    "fr": "french",
    "de": "german",
    "es": "spanish",
    "pt": "portuguese",
    "it": "italian",
    "nl": "dutch",
    "da": "danish",
    "fi": "finnish",
    "hu": "hungarian",
    "nb": "norwegian",
    "pl": "polish",
    "ro": "romanian",
    "ru": "russian",
    "sl": "slovene",
    "sv": "swedish",
    "tr": "turkish",
}

# Languages whose scripts don't rely on spaces → use punctuation splitting
_PUNCTUATION_SPLIT_LANGS = {"hi", "kn", "ta", "te", "ml", "mr", "bn", "gu", "pa",
                             "zh", "zh-cn", "zh-tw", "ja", "ko", "ar", "fa", "ur"}

# Regex: Indic danda + common end-of-sentence punctuation
_SENTENCE_END_RE = re.compile(
    r"(?<=[।॥\?\!\.\n])\s*",
    re.UNICODE,
)


# ── Public API ────────────────────────────────────────────────────────────────

def detect_language(text: str) -> str:
    """Return ISO-639-1 language code, or 'en' on failure."""
    if not _LANGDETECT_AVAILABLE or len(text.strip()) < 20:
        return "en"
    try:
        return _langdetect_detect(text[:2000])
    except LangDetectException:
        return "en"
    except Exception as exc:
        logger.debug("Language detection error: %s", exc)
        return "en"


def chunk_document(pages: list[dict], source: str) -> list[dict]:
    """
    Chunk a list of page dicts (from pdf_parser) into overlapping text chunks.

    Parameters
    ──────────
    pages  : list of page dicts produced by pdf_parser.parse_pdf
    source : display name (PDF stem) used in metadata

    Returns
    ───────
    list of chunk dicts:
    {
        "text"       : str   — chunk text
        "chunk_id"   : str   — "<source>_p<page>_c<idx>"
        "source"     : str   — file stem
        "filepath"   : str
        "page_num"   : int
        "language"   : str   — ISO-639-1 code
        "has_tables" : bool
        "has_images" : bool
        "chunk_index": int   — index within the document
    }
    """
    chunks: list[dict] = []
    global_idx = 0

    for page in pages:
        text       = page["text"].strip()
        page_num   = page["page_num"]
        filepath   = page.get("filepath", "")
        has_tables = page.get("has_tables", False)
        has_images = page.get("has_images", False)

        if not text:
            continue

        lang       = detect_language(text)
        chunk_size = LANG_CHUNK_SIZES.get(lang, DEFAULT_CHUNK_SIZE)

        raw_chunks = list(_split_text(text, lang, chunk_size))

        for local_idx, chunk_text in enumerate(raw_chunks):
            chunk_text = chunk_text.strip()
            if len(chunk_text) < MIN_CHUNK_LENGTH:
                continue

            chunk_id = f"{source}_p{page_num}_c{local_idx}"
            chunks.append({
                "text"        : chunk_text,
                "chunk_id"    : chunk_id,
                "source"      : source,
                "filepath"    : filepath,
                "page_num"    : page_num,
                "language"    : lang,
                "has_tables"  : has_tables,
                "has_images"  : has_images,
                "chunk_index" : global_idx,
            })
            global_idx += 1

    return chunks


# ── Internal splitting logic ──────────────────────────────────────────────────

def _split_text(text: str, lang: str, chunk_size: int) -> Iterator[str]:
    """
    Split *text* into overlapping chunks of ≤ chunk_size characters.

    Strategy (in order):
    1. Split into sentences (language-aware).
    2. Greedily merge sentences up to chunk_size.
    3. Apply CHUNK_OVERLAP_CHARS sliding window between consecutive chunks.
    """
    sentences = _to_sentences(text, lang)

    if not sentences:
        return

    # ── Greedy sentence packing ──────────────────────────────────────────────
    buffer: list[str]   = []
    buffer_len: int     = 0
    previous_tail: str  = ""   # last CHUNK_OVERLAP_CHARS chars of prev chunk

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue

        # If a single sentence exceeds chunk_size, hard-split it
        if len(sent) > chunk_size:
            # First flush any existing buffer
            if buffer:
                chunk_text = previous_tail + " ".join(buffer)
                yield chunk_text
                previous_tail = chunk_text[-CHUNK_OVERLAP_CHARS:] if len(chunk_text) > CHUNK_OVERLAP_CHARS else chunk_text
                buffer = []
                buffer_len = 0
            # Hard split the long sentence
            for sub in _hard_split(sent, chunk_size, previous_tail):
                yield sub
                previous_tail = sub[-CHUNK_OVERLAP_CHARS:] if len(sub) > CHUNK_OVERLAP_CHARS else sub
            continue

        if buffer_len + len(sent) + 1 > chunk_size and buffer:
            # Emit current buffer as a chunk
            chunk_text = previous_tail + " ".join(buffer)
            yield chunk_text
            previous_tail = chunk_text[-CHUNK_OVERLAP_CHARS:] if len(chunk_text) > CHUNK_OVERLAP_CHARS else chunk_text
            buffer = []
            buffer_len = 0

        buffer.append(sent)
        buffer_len += len(sent) + 1

    # Emit remaining buffer
    if buffer:
        yield previous_tail + " ".join(buffer)


def _to_sentences(text: str, lang: str) -> list[str]:
    """
    Split text into sentences using the best available method for the language.
    """
    # Punctuation-based splitting for Indic / CJK languages
    if lang in _PUNCTUATION_SPLIT_LANGS:
        return _punctuation_split(text)

    # NLTK sent_tokenize for European languages
    nltk_lang = _NLTK_LANG_MAP.get(lang, "english")
    try:
        return sent_tokenize(text, language=nltk_lang)
    except Exception:
        try:
            return sent_tokenize(text, language="english")
        except Exception:
            return _punctuation_split(text)


def _punctuation_split(text: str) -> list[str]:
    """
    Split on Indic dandas, common end-of-sentence punctuation, or newlines.
    Falls back to returning the whole text as one unit.
    """
    # Split on sentence-ending characters
    parts = _SENTENCE_END_RE.split(text)
    # Also split on double-newlines (paragraph breaks)
    expanded: list[str] = []
    for part in parts:
        sub = [p.strip() for p in part.split("\n\n") if p.strip()]
        expanded.extend(sub)
    return expanded if expanded else [text]


def _hard_split(text: str, chunk_size: int, prefix: str) -> Iterator[str]:
    """
    Split a single long string into windows of chunk_size chars,
    prepending the overlap prefix to each window.
    """
    start = 0
    while start < len(text):
        end = start + chunk_size
        window = text[start:end]
        if start == 0:
            yield prefix + window
        else:
            overlap_start = max(0, start - CHUNK_OVERLAP_CHARS)
            yield text[overlap_start:end]
        start = end
