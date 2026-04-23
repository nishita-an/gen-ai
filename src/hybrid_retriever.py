import re
from src.config import TOP_K

# Matches entity IDs like INV001, V002, PAY003, EXP004
ID_PATTERN = re.compile(r'\b(INV\d+|V\d+|PAY\d+|EXP\d+)\b')


class HybridRetriever:

    def __init__(self, vector_db, bm25):
        self.vector_db = vector_db
        self.bm25 = bm25

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _extract_ids(self, texts: list[str]) -> set[str]:
        """Pull entity IDs out of a list of strings."""
        ids = set()
        for text in texts:
            ids.update(ID_PATTERN.findall(text.upper()))
        return ids

    def _merge(self, *doc_lists):
        """Deduplicate docs by page_content, preserving first-seen order."""
        seen = {}
        for docs in doc_lists:
            for doc in docs:
                if doc.page_content not in seen:
                    seen[doc.page_content] = doc
        return list(seen.values())

    # ------------------------------------------------------------------ #
    #  Main retrieval                                                      #
    # ------------------------------------------------------------------ #

    def retrieve(self, query: str) -> list:

        # ── Step 1: Standard hybrid retrieval ──────────────────────────
        dense_docs  = self.vector_db.similarity_search(query, k=TOP_K)
        sparse_docs = self.bm25.retrieve(query, k=TOP_K)
        initial     = self._merge(dense_docs, sparse_docs)

        # ── Step 2: Entity expansion ────────────────────────────────────
        # Collect IDs from the query AND from the initially retrieved chunks
        all_texts = [query] + [d.page_content for d in initial]
        entity_ids = self._extract_ids(all_texts)

        expansion_docs = []
        for eid in entity_ids:
            # BM25 is fast and exact-match-friendly — ideal for ID lookups
            hits = self.bm25.retrieve(eid, k=3)
            expansion_docs.extend(hits)

        # ── Step 3: Merge everything ────────────────────────────────────
        final = self._merge(initial, expansion_docs)

        print(f"  [Retriever] query IDs found: {entity_ids or 'none'}")
        print(f"  [Retriever] chunks returned: {len(final)}")

        return final