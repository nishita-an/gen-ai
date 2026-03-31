"""
Episodic Memory (Summarised Memory)
────────────────────────────────────
Stores compressed summaries of older conversation chunks.
Maintains a two-level hierarchy:
  • chunk-level  – summary of a single episodic chunk (6 turns)
  • session-level – rolled-up summary of many chunks

Backed by SQLite so data survives process restarts.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Generator, List, Optional

from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class EpisodicEntry:
    """One summarised chunk of conversation history."""
    entry_id: str                   # UUID-style unique key
    session_id: str
    level: str                      # "chunk" | "session"
    summary: str
    decisions: List[str]
    technical_facts: List[str]
    constraints: List[str]
    user_goals: List[str]
    unresolved_questions: List[str]
    topic_tags: List[str]
    importance_score: float
    start_turn: int
    end_turn: int
    created_at: str = ""
    access_count: int = 0

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()

    def to_text(self) -> str:
        """Human-readable representation used for retrieval scoring."""
        parts = [f"Summary: {self.summary}"]
        if self.decisions:
            parts.append("Decisions: " + "; ".join(self.decisions))
        if self.technical_facts:
            parts.append("Facts: " + "; ".join(self.technical_facts))
        if self.user_goals:
            parts.append("Goals: " + "; ".join(self.user_goals))
        if self.unresolved_questions:
            parts.append("Open Qs: " + "; ".join(self.unresolved_questions))
        return "\n".join(parts)


_LIST_FIELDS = {
    "decisions", "technical_facts", "constraints",
    "user_goals", "unresolved_questions", "topic_tags",
}

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS episodic_memory (
    entry_id            TEXT PRIMARY KEY,
    session_id          TEXT NOT NULL,
    level               TEXT NOT NULL,
    summary             TEXT NOT NULL,
    decisions           TEXT NOT NULL DEFAULT '[]',
    technical_facts     TEXT NOT NULL DEFAULT '[]',
    constraints         TEXT NOT NULL DEFAULT '[]',
    user_goals          TEXT NOT NULL DEFAULT '[]',
    unresolved_questions TEXT NOT NULL DEFAULT '[]',
    topic_tags          TEXT NOT NULL DEFAULT '[]',
    importance_score    REAL NOT NULL DEFAULT 0.5,
    start_turn          INTEGER NOT NULL DEFAULT 0,
    end_turn            INTEGER NOT NULL DEFAULT 0,
    created_at          TEXT NOT NULL,
    access_count        INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_session ON episodic_memory(session_id);
CREATE INDEX IF NOT EXISTS idx_level   ON episodic_memory(level);
"""


class EpisodicMemory:
    """
    Persistent episodic memory backed by SQLite.

    Thread-safety: each call opens a short-lived connection.
    For high-concurrency use, wrap in a connection pool (e.g. aiosqlite).
    """

    def __init__(self, db_path: str | None = None) -> None:
        self._db = db_path or settings.memory.episodic_db_path
        self._init_db()
        logger.info("EpisodicMemory initialised (db=%s)", self._db)

    # ── DB helpers ──────────────────────────────────────────────────────────

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self._db)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(_CREATE_SQL)

    # ── Serialisation ───────────────────────────────────────────────────────

    @staticmethod
    def _row_to_entry(row: sqlite3.Row) -> EpisodicEntry:
        d = dict(row)
        for f in _LIST_FIELDS:
            d[f] = json.loads(d[f])
        return EpisodicEntry(**d)

    @staticmethod
    def _entry_to_row(e: EpisodicEntry) -> dict:
        d = asdict(e)
        for f in _LIST_FIELDS:
            d[f] = json.dumps(d[f])
        return d

    # ── Public API ──────────────────────────────────────────────────────────

    def store(self, entry: EpisodicEntry) -> None:
        """Insert or replace an episodic entry."""
        row = self._entry_to_row(entry)
        placeholders = ", ".join(f":{k}" for k in row)
        columns = ", ".join(row.keys())
        sql = f"INSERT OR REPLACE INTO episodic_memory ({columns}) VALUES ({placeholders})"
        with self._conn() as conn:
            conn.execute(sql, row)
        logger.info("Episodic stored entry_id=%s level=%s", entry.entry_id, entry.level)

    def get_by_session(
        self,
        session_id: str,
        level: Optional[str] = None,
        limit: int = 20,
    ) -> List[EpisodicEntry]:
        """Retrieve all entries for a session, optionally filtered by level."""
        sql = "SELECT * FROM episodic_memory WHERE session_id = ?"
        params: list = [session_id]
        if level:
            sql += " AND level = ?"
            params.append(level)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        with self._conn() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._row_to_entry(r) for r in rows]

    def get_recent_summaries(
        self,
        session_id: str,
        n: int = 5,
    ) -> List[EpisodicEntry]:
        """Return the n most recent chunk-level summaries for a session."""
        return self.get_by_session(session_id, level="chunk", limit=n)

    def get_all_summaries_text(self, session_id: str) -> List[str]:
        """All summaries as plain text — used for session-level re-summarisation."""
        entries = self.get_by_session(session_id)
        return [e.to_text() for e in entries]

    def increment_access(self, entry_id: str) -> None:
        sql = "UPDATE episodic_memory SET access_count = access_count + 1 WHERE entry_id = ?"
        with self._conn() as conn:
            conn.execute(sql, (entry_id,))

    def update_importance(self, entry_id: str, score: float) -> None:
        sql = "UPDATE episodic_memory SET importance_score = ? WHERE entry_id = ?"
        with self._conn() as conn:
            conn.execute(sql, (score, entry_id))

    def decay_old_entries(self, decay_factor: float = 0.05) -> None:
        """
        Apply memory decay: reduce importance of low-access entries.
        Entries with importance < threshold may be pruned in future.
        """
        sql = """
            UPDATE episodic_memory
               SET importance_score = MAX(0.0, importance_score - ?)
             WHERE access_count = 0
        """
        with self._conn() as conn:
            conn.execute(sql, (decay_factor,))
        logger.debug("EpisodicMemory decay applied (factor=%.2f)", decay_factor)

    def count(self, session_id: str) -> int:
        sql = "SELECT COUNT(*) FROM episodic_memory WHERE session_id = ?"
        with self._conn() as conn:
            return conn.execute(sql, (session_id,)).fetchone()[0]

    def __repr__(self) -> str:
        return f"EpisodicMemory(db={self._db})"
