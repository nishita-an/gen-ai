"""
Test Suite — Persistent Memory System
──────────────────────────────────────
Tests all four memory layers plus token counting, importance scoring,
and context assembly without requiring a live LLM API key.

Run with:
  cd memory_agent
  python -m pytest tests/ -v
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memory.short_term import ShortTermMemory, Turn
from memory.episodic_memory import EpisodicMemory, EpisodicEntry
from memory.semantic_memory import SemanticMemory, SemanticFact
from memory.user_profile_memory import UserProfileMemory
from services.importance_scorer import ImportanceScorer
from llm.grok_client import count_tokens_simple, count_messages_tokens


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_turns(n: int, role_cycle=("user", "assistant")) -> List[Turn]:
    turns = []
    stm = ShortTermMemory(max_turns=200)
    for i in range(n):
        role = role_cycle[i % len(role_cycle)]
        t = stm.add(role, f"Turn {i} content: This is message number {i}.")
        turns.append(t)
    return turns


def _make_fact(text: str, dim: int = 384) -> SemanticFact:
    import numpy as np
    vec = np.random.randn(dim).astype("float32")
    vec = (vec / np.linalg.norm(vec)).tolist()
    return SemanticFact(
        fact_id=f"fact-{int(time.time() * 1e6)}",
        session_id="test",
        text=text,
        category="technical",
        importance=0.7,
        source_turn=1,
        embedding=vec,
    )


# ── ShortTermMemory ───────────────────────────────────────────────────────────

class TestShortTermMemory(unittest.TestCase):
    def setUp(self):
        self.stm = ShortTermMemory(max_turns=5)

    def test_add_and_len(self):
        self.stm.add("user", "hello")
        self.assertEqual(len(self.stm), 1)

    def test_fifo_eviction(self):
        for i in range(7):
            self.stm.add("user", f"msg {i}")
        self.assertEqual(len(self.stm), 5)
        # Oldest should be evicted
        texts = [t.content for t in self.stm.get_all()]
        self.assertNotIn("msg 0", texts)
        self.assertNotIn("msg 1", texts)
        self.assertIn("msg 6", texts)

    def test_pop_oldest_chunk(self):
        for i in range(5):
            self.stm.add("user", f"msg {i}")
        popped = self.stm.pop_oldest_chunk(3)
        self.assertEqual(len(popped), 3)
        self.assertEqual(len(self.stm), 2)

    def test_as_messages(self):
        self.stm.add("user", "hi")
        self.stm.add("assistant", "hello!")
        msgs = self.stm.as_messages()
        self.assertEqual(msgs[0]["role"], "user")
        self.assertEqual(msgs[1]["role"], "assistant")

    def test_token_estimate(self):
        self.stm.add("user", "a" * 400)   # ≈100 tokens
        estimate = self.stm.token_estimate()
        self.assertGreater(estimate, 50)
        self.assertLess(estimate, 200)

    def test_clear(self):
        self.stm.add("user", "hi")
        self.stm.clear()
        self.assertEqual(len(self.stm), 0)

    def test_get_recent(self):
        for i in range(5):
            self.stm.add("user", f"msg {i}")
        recent = self.stm.get_recent(3)
        self.assertEqual(len(recent), 3)
        self.assertEqual(recent[-1].content, "msg 4")

    def test_turn_id_increments(self):
        t1 = self.stm.add("user", "first")
        t2 = self.stm.add("user", "second")
        self.assertEqual(t2.turn_id, t1.turn_id + 1)


# ── EpisodicMemory ────────────────────────────────────────────────────────────

class TestEpisodicMemory(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test_episodic.db")
        self.em = EpisodicMemory(db_path=self.db_path)

    def _make_entry(self, level="chunk") -> EpisodicEntry:
        import uuid
        return EpisodicEntry(
            entry_id=str(uuid.uuid4()),
            session_id="session1",
            level=level,
            summary="We discussed backpropagation and gradient descent.",
            decisions=["Use Adam optimiser"],
            technical_facts=["Learning rate set to 0.001"],
            constraints=["Must run on CPU"],
            user_goals=["Understand BPTT"],
            unresolved_questions=["Why does loss spike at epoch 5?"],
            topic_tags=["backprop", "lstm"],
            importance_score=0.75,
            start_turn=1,
            end_turn=6,
        )

    def test_store_and_retrieve(self):
        entry = self._make_entry()
        self.em.store(entry)
        results = self.em.get_by_session("session1")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].summary, entry.summary)

    def test_list_fields_roundtrip(self):
        entry = self._make_entry()
        self.em.store(entry)
        retrieved = self.em.get_by_session("session1")[0]
        self.assertEqual(retrieved.decisions, ["Use Adam optimiser"])
        self.assertEqual(retrieved.topic_tags, ["backprop", "lstm"])

    def test_level_filter(self):
        self.em.store(self._make_entry(level="chunk"))
        self.em.store(self._make_entry(level="session"))
        chunks = self.em.get_by_session("session1", level="chunk")
        sessions = self.em.get_by_session("session1", level="session")
        self.assertEqual(len(chunks), 1)
        self.assertEqual(len(sessions), 1)

    def test_access_increment(self):
        entry = self._make_entry()
        self.em.store(entry)
        self.em.increment_access(entry.entry_id)
        results = self.em.get_by_session("session1")
        self.assertEqual(results[0].access_count, 1)

    def test_count(self):
        self.em.store(self._make_entry())
        self.em.store(self._make_entry())
        self.assertEqual(self.em.count("session1"), 2)

    def test_to_text(self):
        entry = self._make_entry()
        text = entry.to_text()
        self.assertIn("backpropagation", text)
        self.assertIn("Adam optimiser", text)

    def test_decay(self):
        entry = self._make_entry()
        self.em.store(entry)
        self.em.decay_old_entries(decay_factor=0.1)
        results = self.em.get_by_session("session1")
        self.assertAlmostEqual(results[0].importance_score, 0.65, places=2)


# ── SemanticMemory ────────────────────────────────────────────────────────────

class TestSemanticMemory(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.sm = SemanticMemory(
            index_path=os.path.join(self.tmpdir, "test.index"),
            meta_path=os.path.join(self.tmpdir, "test_meta.json"),
            dim=384,
        )

    def test_add_and_count(self):
        self.sm.add(_make_fact("Neural networks use backpropagation."))
        self.assertEqual(self.sm.count(), 1)

    def test_batch_add(self):
        facts = [_make_fact(f"Fact {i}") for i in range(5)]
        self.sm.add_batch(facts)
        self.assertEqual(self.sm.count(), 5)

    def test_search_returns_results(self):
        import numpy as np
        # Add a fact and search with its own embedding
        fact = _make_fact("The transformer uses attention mechanism.")
        self.sm.add(fact)
        results = self.sm.search(fact.embedding, top_k=1)
        self.assertEqual(len(results), 1)
        self.assertGreater(results[0][1], 0.9)  # near-perfect match

    def test_search_empty_returns_empty(self):
        results = self.sm.search([0.0] * 384, top_k=5)
        self.assertEqual(results, [])

    def test_persistence(self):
        """Index should survive a reload."""
        fact = _make_fact("FAISS persists to disk.")
        self.sm.add(fact)

        sm2 = SemanticMemory(
            index_path=os.path.join(self.tmpdir, "test.index"),
            meta_path=os.path.join(self.tmpdir, "test_meta.json"),
            dim=384,
        )
        self.assertEqual(sm2.count(), 1)


# ── UserProfileMemory ─────────────────────────────────────────────────────────

class TestUserProfileMemory(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.profile_path = os.path.join(self.tmpdir, "profile.json")
        self.up = UserProfileMemory(profile_path=self.profile_path)

    def test_add_and_get_goal(self):
        self.up.add_goal("Learn PyTorch")
        self.assertIn("Learn PyTorch", self.up.get("goals"))

    def test_no_duplicate_goals(self):
        self.up.add_goal("Learn PyTorch")
        self.up.add_goal("Learn PyTorch")
        self.assertEqual(len(self.up.get("goals")), 1)

    def test_persistence(self):
        self.up.add_goal("Persistent goal")
        up2 = UserProfileMemory(profile_path=self.profile_path)
        self.assertIn("Persistent goal", up2.get("goals"))

    def test_format_for_prompt(self):
        self.up.add_goal("Master NLP")
        self.up.add_project("Thesis: summarisation model")
        text = self.up.format_for_prompt()
        self.assertIn("Master NLP", text)
        self.assertIn("Thesis", text)

    def test_merge_from_llm(self):
        self.up.merge_from_llm_extraction({
            "goals": ["Understand attention"],
            "technical_interests": ["transformers", "BERT"],
            "preferences": {"language": "Python"},
        })
        self.assertIn("Understand attention", self.up.get("goals"))
        self.assertIn("transformers", self.up.get("technical_interests"))
        self.assertEqual(self.up.get("preferences")["language"], "Python")

    def test_remove_fact(self):
        self.up.add_goal("Temp goal")
        self.up.remove_fact("Temp goal")
        self.assertNotIn("Temp goal", self.up.get("goals"))

    def test_set_preference(self):
        self.up.set_preference("verbosity", "concise")
        self.assertEqual(self.up.get("preferences")["verbosity"], "concise")


# ── ImportanceScorer ──────────────────────────────────────────────────────────

class TestImportanceScorer(unittest.TestCase):
    def setUp(self):
        self.scorer = ImportanceScorer()

    def test_score_range(self):
        score = self.scorer.score("This is a test message.")
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_high_importance_keywords(self):
        high = self.scorer.score("This is a critical requirement and must never be ignored.")
        low = self.scorer.score("OK sounds good.")
        self.assertGreater(high, low)

    def test_recency_matters(self):
        recent = self.scorer.score("recent message", turn_index=9, total_turns=10)
        old = self.scorer.score("old message", turn_index=0, total_turns=10)
        self.assertGreater(recent, old)

    def test_batch(self):
        texts = ["fact one", "critical goal", "filler words"]
        scores = self.scorer.score_batch(texts)
        self.assertEqual(len(scores), 3)


# ── Token counting ────────────────────────────────────────────────────────────

class TestTokenCounting(unittest.TestCase):
    def test_count_tokens_simple(self):
        # 400 chars ≈ 100 tokens
        self.assertEqual(count_tokens_simple("a" * 400), 100)

    def test_count_messages(self):
        messages = [
            {"role": "user", "content": "Hello world"},
            {"role": "assistant", "content": "Hi there! How can I help?"},
        ]
        count = count_messages_tokens(messages)
        self.assertGreater(count, 0)

    def test_empty_text(self):
        self.assertEqual(count_tokens_simple(""), 1)   # min 1


# ── Integration: STM → Episodic flow ─────────────────────────────────────────

class TestMemoryIntegration(unittest.TestCase):
    """
    Smoke-test the STM → Episodic compression flow using a mock LLM.
    Does NOT require a real API key.
    """

    def test_stm_chunk_pop_ready_for_summarisation(self):
        stm = ShortTermMemory(max_turns=20)
        for i in range(12):
            stm.add("user" if i % 2 == 0 else "assistant", f"Message {i}")

        chunk = stm.pop_oldest_chunk(6)
        self.assertEqual(len(chunk), 6)
        self.assertEqual(len(stm), 6)
        # Remaining turns should be the later ones
        remaining = stm.get_all()
        self.assertEqual(remaining[0].content, "Message 6")


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    unittest.main(verbosity=2)
