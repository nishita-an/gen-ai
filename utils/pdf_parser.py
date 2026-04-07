"""
utils/pdf_parser.py
───────────────────
Parses PDF files using PyMuPDF (fitz).

Features
────────
• Page-by-page processing (handles 100+ page PDFs without OOM)
• Scanned-image detection → logs warning and skips
• Empty-page skipping
• Table extraction (PyMuPDF ≥ 1.23 find_tables; fallback to block text)
• Figure / image block detection (noted in metadata)
• MD5 content-hash deduplication (caller decides whether to skip)
• Returns structured page-level dicts for the chunker
"""

from __future__ import annotations

import hashlib
import logging
import re
import sys
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)

# ── Try importing fitz (PyMuPDF) ────────────────────────────────────────────
try:
    import fitz  # type: ignore
except ImportError:
    logger.critical("PyMuPDF (fitz) not installed. Run: pip install pymupdf")
    sys.exit(1)

from config import SCANNED_TEXT_THRESHOLD, MAX_TABLE_ROWS_INLINE


# ── Public API ───────────────────────────────────────────────────────────────

def compute_file_hash(path: Path) -> str:
    """Return MD5 hex-digest of the raw file bytes (fast dedup check)."""
    md5 = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            md5.update(chunk)
    return md5.hexdigest()


def parse_pdf(path: Path) -> tuple[list[dict], str | None]:
    """
    Parse a single PDF and return (pages, content_hash).

    Parameters
    ──────────
    path : Path to the PDF file.

    Returns
    ───────
    pages : list of page dicts  (empty if the file is unreadable / scanned-only)
    content_hash : MD5 of the file bytes (used for deduplication upstream)

    Page dict schema
    ────────────────
    {
        "text"       : str   — plain text of the page (tables embedded as markdown)
        "page_num"   : int   — 1-based
        "source"     : str   — file stem (no extension)
        "filepath"   : str   — absolute path string
        "has_tables" : bool
        "has_images" : bool
        "char_count" : int
    }
    """
    content_hash = compute_file_hash(path)
    source_name  = path.stem

    pages: list[dict] = []

    try:
        doc = fitz.open(str(path))
    except Exception as exc:
        logger.error("Cannot open %s: %s", path.name, exc)
        return [], content_hash

    total_pages  = len(doc)
    scanned_pages = 0

    for page_idx in range(total_pages):
        try:
            page_data = _process_page(doc[page_idx], page_idx + 1, source_name, str(path))
        except Exception as exc:
            logger.warning("Error processing page %d of %s: %s", page_idx + 1, path.name, exc)
            continue

        if page_data is None:
            # Empty page — silently skip
            continue

        if page_data.get("_scanned"):
            scanned_pages += 1
            continue

        pages.append(page_data)

    doc.close()

    if scanned_pages > 0 and not pages:
        logger.warning(
            "SKIPPED %s — all %d pages appear to be scanned images with no extractable text.",
            path.name, scanned_pages,
        )
    elif scanned_pages > 0:
        logger.info(
            "%s — %d/%d pages skipped (scanned images).",
            path.name, scanned_pages, total_pages,
        )

    return pages, content_hash


# ── Internal helpers ──────────────────────────────────────────────────────────

def _process_page(page: fitz.Page, page_num: int, source: str, filepath: str) -> dict | None:
    """
    Extract text + tables from a single page.

    Returns None  → empty page (skip silently)
    Returns dict  with _scanned=True → scanned image page (caller logs warning)
    Returns dict  → normal page with text
    """
    # ── 1. Raw text extraction ───────────────────────────────────────────────
    raw_text: str = page.get_text("text")  # type: ignore[attr-defined]
    raw_text = _clean_text(raw_text)

    # ── 2. Check for empty page ──────────────────────────────────────────────
    if not raw_text.strip():
        return None

    # ── 3. Check if scanned (very few chars for page area) ──────────────────
    if len(raw_text.strip()) < SCANNED_TEXT_THRESHOLD:
        image_list = page.get_images(full=False)
        if image_list:                        # has images but no text → scanned
            return {"_scanned": True}
        # Has almost no text and no images → genuinely empty; skip
        return None

    # ── 4. Table extraction ──────────────────────────────────────────────────
    has_tables  = False
    table_texts: list[str] = []

    try:
        tables = page.find_tables()  # PyMuPDF ≥ 1.23
        for tbl in tables:
            has_tables = True
            md = _table_to_markdown(tbl)
            if md:
                table_texts.append(md)
    except AttributeError:
        # Older PyMuPDF — fall back: tables appear as text blocks already
        pass
    except Exception as exc:
        logger.debug("Table extraction error on page %d: %s", page_num, exc)

    # ── 5. Image-block detection ─────────────────────────────────────────────
    has_images = bool(page.get_images(full=False))

    # ── 6. Compose final page text ───────────────────────────────────────────
    combined = raw_text
    if table_texts:
        combined = combined + "\n\n" + "\n\n".join(table_texts)

    combined = _clean_text(combined)

    if len(combined.strip()) < SCANNED_TEXT_THRESHOLD:
        return None

    return {
        "text"       : combined,
        "page_num"   : page_num,
        "source"     : source,
        "filepath"   : filepath,
        "has_tables" : has_tables,
        "has_images" : has_images,
        "char_count" : len(combined),
    }


def _table_to_markdown(table) -> str:
    """Convert a PyMuPDF Table object to a Markdown-formatted string."""
    try:
        rows = table.extract()
    except Exception:
        return ""

    if not rows:
        return ""

    # Truncate very large tables
    truncated = False
    if len(rows) > MAX_TABLE_ROWS_INLINE + 1:
        rows = rows[: MAX_TABLE_ROWS_INLINE + 1]
        truncated = True

    def cell(v) -> str:
        return str(v).replace("|", "\\|").replace("\n", " ").strip() if v is not None else ""

    lines: list[str] = []
    header = rows[0]
    lines.append("| " + " | ".join(cell(c) for c in header) + " |")
    lines.append("| " + " | ".join("---" for _ in header) + " |")

    for row in rows[1:]:
        lines.append("| " + " | ".join(cell(c) for c in row) + " |")

    if truncated:
        lines.append(f"| *…table truncated at {MAX_TABLE_ROWS_INLINE} rows…* |")

    return "\n".join(lines)


def _clean_text(text: str) -> str:
    """Normalise whitespace; collapse excessive blank lines."""
    # Remove null bytes and control chars (except newlines/tabs)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    # Collapse 3+ newlines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse multiple spaces (but not newlines)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()
