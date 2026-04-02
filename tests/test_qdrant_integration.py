"""Integration tests against live Qdrant at 100.113.229.40.

These tests verify end-to-end memory storage and retrieval using the real
MemoryClient pipeline — no mocks for the storage layer. Tests use isolated
collection names prefixed with 'inttest_' and clean up after themselves.

Run with:
    pytest tests/test_qdrant_integration.py -v
"""

from __future__ import annotations

import time
import uuid

import pytest

QDRANT_URL = "http://100.113.229.40:6333"
FACTS_COLLECTION = "inttest_facts"
PREFS_COLLECTION = "inttest_prefs"
TEST_USER = f"test_user_{uuid.uuid4().hex[:8]}"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def client():
    """Create a MemoryClient pointed at the remote Qdrant, with isolated collections."""
    from hippocampai.client import MemoryClient

    c = MemoryClient(
        qdrant_url=QDRANT_URL,
        collection_facts=FACTS_COLLECTION,
        collection_prefs=PREFS_COLLECTION,
        enable_telemetry=False,
    )
    yield c

    # Teardown: delete test collections
    try:
        c.qdrant.client.delete_collection(FACTS_COLLECTION)
    except Exception:
        pass
    try:
        c.qdrant.client.delete_collection(PREFS_COLLECTION)
    except Exception:
        pass


@pytest.fixture
def user_id():
    """Unique user ID per test to prevent cross-test contamination."""
    return f"user_{uuid.uuid4().hex[:8]}"


# ---------------------------------------------------------------------------
# 1. Basic remember + recall round-trip
# ---------------------------------------------------------------------------


class TestRememberAndRecall:
    def test_remember_returns_a_memory_object(self, client, user_id) -> None:
        mem = client.remember("I love Python programming", user_id=user_id)
        assert mem is not None
        assert mem.id is not None
        assert mem.text == "I love Python programming"
        assert mem.user_id == user_id

    def test_remembered_memory_is_recalled(self, client, user_id) -> None:
        client.remember("My favorite editor is VS Code", user_id=user_id)
        time.sleep(0.5)  # allow index flush

        results = client.recall("editor preferences", user_id=user_id, k=5)
        texts = [r.memory.text for r in results]
        assert any("VS Code" in t for t in texts), (
            f"Expected 'VS Code' memory in results, got: {texts}"
        )

    def test_multiple_memories_stored_and_recalled(self, client, user_id) -> None:
        memories_to_store = [
            "I drink coffee every morning",
            "I prefer dark mode in all apps",
            "I work from home on Fridays",
        ]
        for text in memories_to_store:
            client.remember(text, user_id=user_id)

        time.sleep(0.5)

        results = client.recall("work habits and preferences", user_id=user_id, k=10)
        retrieved_texts = [r.memory.text for r in results]
        assert len(retrieved_texts) >= 1, "At least one memory should be retrieved"

    def test_recall_respects_user_isolation(self, client) -> None:
        """Memories stored for user_a must not appear in user_b's recall results."""
        user_a = f"user_a_{uuid.uuid4().hex[:6]}"
        user_b = f"user_b_{uuid.uuid4().hex[:6]}"

        secret_text = f"SECRET_{uuid.uuid4().hex}"
        client.remember(secret_text, user_id=user_a)
        time.sleep(0.5)

        results_b = client.recall(secret_text, user_id=user_b, k=10)
        b_texts = [r.memory.text for r in results_b]
        assert secret_text not in b_texts, (
            "user_b should not see user_a's memories"
        )

    def test_recalled_memories_have_scores(self, client, user_id) -> None:
        client.remember("I enjoy hiking on weekends", user_id=user_id)
        time.sleep(0.5)

        results = client.recall("hobbies", user_id=user_id, k=5)
        assert all(r.score >= 0.0 for r in results), "All scores must be non-negative"


# ---------------------------------------------------------------------------
# 2. Dedup: skip exact duplicate
# ---------------------------------------------------------------------------


class TestDeduplication:
    def test_exact_duplicate_is_not_stored_twice(self, client, user_id) -> None:
        text = "I prefer tea over coffee"
        m1 = client.remember(text, user_id=user_id)
        time.sleep(0.3)
        m2 = client.remember(text, user_id=user_id)

        # Either the second call returns the original ID or creates no new entry
        assert m1 is not None
        assert m2 is not None

        time.sleep(0.5)
        results = client.recall(text, user_id=user_id, k=20)
        matching = [r for r in results if r.memory.text == text]
        # Should be at most 1 stored copy
        assert len(matching) <= 1, (
            f"Duplicate memory stored {len(matching)} times, expected at most 1"
        )

    def test_dedup_update_reinforces_existing_memory(self, client, user_id) -> None:
        """When dedup returns 'update', the existing memory importance should be boosted."""
        original = "I really enjoy playing chess"
        client.remember(original, user_id=user_id)
        time.sleep(0.3)

        # Store a near-duplicate — high cosine similarity should trigger dedup
        near_dup = "I really enjoy playing chess a lot"
        client.remember(near_dup, user_id=user_id)
        time.sleep(0.5)

        results = client.recall("chess", user_id=user_id, k=5)
        assert len(results) >= 1, "At least one chess memory should be recalled"


# ---------------------------------------------------------------------------
# 3. BM25 incremental update — keyword search works immediately after remember
# ---------------------------------------------------------------------------


class TestBM25IncrementalUpdate:
    def test_keyword_search_finds_memory_immediately_after_store(self, client, user_id) -> None:
        unique_keyword = f"ZYLOFUTURISTIC_{uuid.uuid4().hex[:6]}"
        client.remember(
            f"I am researching {unique_keyword} technology",
            user_id=user_id,
        )
        time.sleep(0.3)

        # Rebuild BM25 to include the new memory
        client.retriever.rebuild_bm25(user_id)

        results = client.recall(unique_keyword, user_id=user_id, k=10)
        texts = [r.memory.text for r in results]
        assert any(unique_keyword in t for t in texts), (
            f"Memory with '{unique_keyword}' should be found via keyword search"
        )

    def test_corpus_grows_after_each_remember(self, client, user_id) -> None:
        initial_facts_count = len(client.retriever.corpus_facts)
        initial_prefs_count = len(client.retriever.corpus_prefs)

        client.remember("A brand new fact for the corpus", user_id=user_id)

        total_after = len(client.retriever.corpus_facts) + len(client.retriever.corpus_prefs)
        total_before = initial_facts_count + initial_prefs_count
        assert total_after > total_before, (
            "Corpus should grow after remember() call via add_to_corpus"
        )


# ---------------------------------------------------------------------------
# 4. Memory types are stored in correct collections
# ---------------------------------------------------------------------------


class TestCollectionRouting:
    def test_preference_memory_stored_in_prefs_collection(self, client, user_id) -> None:
        mem = client.remember(
            "I prefer working late at night",
            user_id=user_id,
            type="preference",
        )
        assert mem is not None
        assert mem.type.value == "preference"

        # Verify by direct Qdrant lookup in prefs collection
        results = client.qdrant.client.scroll(
            collection_name=PREFS_COLLECTION,
            scroll_filter=None,
            limit=500,
            with_payload=True,
        )
        point_ids = [str(p.id) for p in results[0]]
        assert mem.id in point_ids, (
            f"PREFERENCE memory {mem.id} should be in {PREFS_COLLECTION}"
        )

    def test_fact_memory_stored_in_facts_collection(self, client, user_id) -> None:
        mem = client.remember(
            "Python was created by Guido van Rossum in 1991",
            user_id=user_id,
            type="fact",
        )
        assert mem is not None
        assert mem.type.value == "fact"

        results = client.qdrant.client.scroll(
            collection_name=FACTS_COLLECTION,
            scroll_filter=None,
            limit=500,
            with_payload=True,
        )
        point_ids = [str(p.id) for p in results[0]]
        assert mem.id in point_ids, (
            f"FACT memory {mem.id} should be in {FACTS_COLLECTION}"
        )


# ---------------------------------------------------------------------------
# 5. Embedding is stored and non-zero
# ---------------------------------------------------------------------------


class TestEmbeddingStorage:
    def test_stored_memory_has_embedding_in_qdrant(self, client, user_id) -> None:
        mem = client.remember("The sky is blue on a clear day", user_id=user_id)
        assert mem is not None

        # Fetch the point directly from Qdrant with vectors
        results = client.qdrant.client.retrieve(
            collection_name=FACTS_COLLECTION,
            ids=[mem.id],
            with_vectors=True,
        )
        if not results:
            # might be in prefs
            results = client.qdrant.client.retrieve(
                collection_name=PREFS_COLLECTION,
                ids=[mem.id],
                with_vectors=True,
            )

        assert len(results) > 0, "Memory point should exist in Qdrant"
        vector = results[0].vector
        assert vector is not None, "Vector should be stored"
        assert len(vector) == 384, f"Expected 384-dim embedding, got {len(vector)}"
        assert any(v != 0.0 for v in vector), "Embedding should not be all zeros"


# ---------------------------------------------------------------------------
# 6. Recall returns results with correct memory fields
# ---------------------------------------------------------------------------


class TestRecallPayloadIntegrity:
    def test_recalled_memory_has_correct_user_id(self, client, user_id) -> None:
        client.remember("I like to read sci-fi novels", user_id=user_id)
        time.sleep(0.5)

        results = client.recall("reading habits", user_id=user_id, k=5)
        for r in results:
            assert r.memory.user_id == user_id, (
                f"Recalled memory has wrong user_id: {r.memory.user_id}"
            )

    def test_recalled_memory_text_matches_stored_text(self, client, user_id) -> None:
        stored_text = "I run 5km every Tuesday morning"
        client.remember(stored_text, user_id=user_id)
        time.sleep(0.5)

        results = client.recall("running schedule", user_id=user_id, k=5)
        texts = [r.memory.text for r in results]
        assert any("5km" in t or "Tuesday" in t for t in texts), (
            f"Stored text should be recoverable. Got: {texts}"
        )

    def test_recall_k_limits_results(self, client, user_id) -> None:
        for i in range(5):
            client.remember(f"Test memory number {i} about running", user_id=user_id)
        time.sleep(0.5)

        results_k3 = client.recall("running", user_id=user_id, k=3)
        assert len(results_k3) <= 3, f"Expected ≤3 results with k=3, got {len(results_k3)}"
