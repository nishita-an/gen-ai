"""
ingest.py
─────────
Ingestion pipeline for the Multilingual RAG system.

Run once (or re-run to add new PDFs):
    python ingest.py [--pdf-folder ./pdfs] [--reset]

Features
────────
• Scans PDF_FOLDER recursively for *.pdf files
• Deduplicates by MD5 file hash (skips already-ingested hashes)
• Skips scanned-image-only PDFs with a log warning
• Processes page-by-page (memory-safe for 100+ page PDFs)
• Chunks with multilingual sentence-aware splitter
• Embeds in batches via SentenceTransformer
• Stores in persistent ChromaDB with full metadata
• Resumable: re-running skips already-indexed chunk IDs
• Progress bars via tqdm
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import chromadb
from tqdm import tqdm

# ── Local imports ─────────────────────────────────────────────────────────────
import config
from config import (
    PDF_FOLDER,
    CHROMA_DB_PATH,
    COLLECTION_NAME,
    HASHES_FILE,
    LOG_FILE,
    EMBEDDING_MODEL,
)
from utils.pdf_parser import parse_pdf, compute_file_hash
from utils.chunker import chunk_document
from utils.embedder import ChromaEmbeddingFunction, Embedder


# ── Logging setup ─────────────────────────────────────────────────────────────

def _setup_logging(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(log_file), encoding="utf-8"),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
    )

logger = logging.getLogger("ingest")


# ── Hash persistence ──────────────────────────────────────────────────────────

def _load_hashes(path: Path) -> set[str]:
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return set(json.load(fh))
        except Exception:
            return set()
    return set()


def _save_hashes(path: Path, hashes: set[str]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(sorted(hashes), fh, indent=2)


# ── ChromaDB helpers ──────────────────────────────────────────────────────────

def _get_collection(reset: bool = False) -> chromadb.Collection:
    """Return (or create) the persistent ChromaDB collection."""
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    ef = ChromaEmbeddingFunction(model_name=EMBEDDING_MODEL)

    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
            logger.info("Existing collection '%s' deleted (--reset flag).", COLLECTION_NAME)
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def _existing_chunk_ids(collection: chromadb.Collection) -> set[str]:
    """Return all chunk IDs already stored in the collection (for resumability)."""
    try:
        result = collection.get(include=[])  # IDs only, no embeddings
        return set(result["ids"])
    except Exception:
        return set()


# ── Batch upsert ──────────────────────────────────────────────────────────────

def _upsert_chunks(
    collection: chromadb.Collection,
    embedder: Embedder,
    chunks: list[dict],
    existing_ids: set[str],
) -> int:
    """
    Embed and upsert a list of chunk dicts into ChromaDB.

    Skips chunks whose IDs are already present.
    Returns the number of newly added chunks.
    """
    new_chunks = [c for c in chunks if c["chunk_id"] not in existing_ids]
    if not new_chunks:
        return 0

    texts     = [c["text"]     for c in new_chunks]
    ids       = [c["chunk_id"] for c in new_chunks]

    embeddings = embedder.embed(texts)

    # ChromaDB metadata values must be str | int | float | bool
    metadatas = [
        {
            "source"     : str(c["source"]),
            "filepath"   : str(c["filepath"]),
            "page_num"   : int(c["page_num"]),
            "language"   : str(c["language"]),
            "has_tables" : bool(c["has_tables"]),
            "has_images" : bool(c["has_images"]),
            "chunk_index": int(c["chunk_index"]),
        }
        for c in new_chunks
    ]

    # ChromaDB upsert in sub-batches of 500 (API limit safety)
    BATCH = 500
    added = 0
    for i in range(0, len(ids), BATCH):
        sl = slice(i, i + BATCH)
        try:
            collection.upsert(
                ids        = ids[sl],
                embeddings = embeddings[sl],
                documents  = texts[sl],
                metadatas  = metadatas[sl],
            )
            added += len(ids[sl])
        except Exception as exc:
            logger.error("ChromaDB upsert error at batch %d: %s", i, exc)

    return added


# ── Main ingestion logic ───────────────────────────────────────────────────────

def ingest(pdf_folder: Path, reset: bool = False) -> None:
    """
    Full ingestion pipeline.

    1. Discover PDFs
    2. Deduplicate by MD5
    3. Parse → chunk → embed → store
    """
    _setup_logging(LOG_FILE)
    logger.info("═" * 60)
    logger.info("Starting ingestion from: %s", pdf_folder)

    if not pdf_folder.exists():
        logger.error("PDF folder does not exist: %s", pdf_folder)
        sys.exit(1)

    # Discover PDFs (case-insensitive glob)
    pdf_files: list[Path] = []
    for pattern in ("*.pdf", "*.PDF"):
        pdf_files.extend(pdf_folder.rglob(pattern))
    pdf_files = sorted(set(pdf_files))   # deduplicate paths (rglob can double-count on some FS)

    if not pdf_files:
        logger.error("No PDF files found in %s", pdf_folder)
        sys.exit(0)

    logger.info("Found %d PDF file(s).", len(pdf_files))

    # Load persisted deduplication hashes
    ingested_hashes: set[str] = set() if reset else _load_hashes(HASHES_FILE)
    logger.info("Already-ingested file hashes in cache: %d", len(ingested_hashes))

    # Set up ChromaDB
    collection     = _get_collection(reset=reset)
    existing_ids   = set() if reset else _existing_chunk_ids(collection)
    logger.info("Chunks already in vector store: %d", len(existing_ids))

    # Embedder (loaded once, shared across all PDFs)
    embedder = Embedder()

    # Statistics
    total_pdfs_skipped   = 0
    total_pdfs_processed = 0
    total_chunks_added   = 0

    for pdf_path in tqdm(pdf_files, desc="Ingesting PDFs", unit="pdf"):
        # ── Deduplication check ──────────────────────────────────────────────
        file_hash = compute_file_hash(pdf_path)
        if file_hash in ingested_hashes:
            logger.info("SKIP (duplicate hash): %s", pdf_path.name)
            total_pdfs_skipped += 1
            continue

        logger.info("Processing: %s", pdf_path.name)

        # ── Parse ────────────────────────────────────────────────────────────
        pages, _ = parse_pdf(pdf_path)   # hash already computed above

        if not pages:
            logger.warning("No extractable text pages in: %s — skipping.", pdf_path.name)
            ingested_hashes.add(file_hash)   # avoid re-processing scanned file each run
            _save_hashes(HASHES_FILE, ingested_hashes)
            total_pdfs_skipped += 1
            continue

        logger.info("  Parsed %d page(s) from %s", len(pages), pdf_path.name)

        # ── Chunk ─────────────────────────────────────────────────────────────
        chunks = chunk_document(pages, source=pdf_path.stem)
        logger.info("  Produced %d chunk(s)", len(chunks))

        if not chunks:
            logger.warning("  No valid chunks from %s", pdf_path.name)
            continue

        # ── Embed & store ─────────────────────────────────────────────────────
        added = _upsert_chunks(collection, embedder, chunks, existing_ids)
        # Update existing_ids in-place so duplicates within a batch are handled
        for c in chunks:
            existing_ids.add(c["chunk_id"])

        logger.info("  Added %d new chunk(s) to vector store.", added)
        total_chunks_added   += added
        total_pdfs_processed += 1

        # Persist hash immediately so interrupted runs don't re-process
        ingested_hashes.add(file_hash)
        _save_hashes(HASHES_FILE, ingested_hashes)

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info("═" * 60)
    logger.info("Ingestion complete.")
    logger.info("  PDFs processed  : %d", total_pdfs_processed)
    logger.info("  PDFs skipped    : %d (duplicates / scanned)", total_pdfs_skipped)
    logger.info("  Chunks added    : %d", total_chunks_added)
    logger.info("  Vector store    : %s", CHROMA_DB_PATH)
    logger.info("  Total chunks in store: %d", collection.count())


# ── CLI entry-point ───────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest PDF documents into the multilingual RAG vector store."
    )
    parser.add_argument(
        "--pdf-folder",
        type=Path,
        default=PDF_FOLDER,
        help=f"Folder containing PDF files (default: {PDF_FOLDER})",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete the existing vector store and re-ingest everything from scratch.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    ingest(args.pdf_folder, reset=args.reset)
