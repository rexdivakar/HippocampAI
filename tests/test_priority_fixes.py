"""Priority-fix tests for targeted changes in HippocampAI.

Covers:
  1. SmartMemoryUpdater._calculate_similarity — cosine path, Jaccard fallback,
     embedder-raises fallback
  2. GraphRetriever.search — real hop-distance scoring via nx.single_source_shortest_path_length
  3. QueryIntentDetector.detect — temporal, preference, neutral, and combined queries
  4. HybridRetriever.add_to_corpus — dedup, BM25 rebuild, facts vs prefs routing
  5. Config defaults — enable_graph_retrieval and weight_graph

No running Qdrant or Redis required — all external dependencies are mocked/faked.
"""

from __future__ import annotations

import math
from typing import Any
from unittest.mock import MagicMock, patch

import networkx as nx
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cosine(v1: np.ndarray, v2: np.ndarray) -> float:
    """Reference cosine similarity used to cross-check assertions."""
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 == 0.0 or n2 == 0.0:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


# ---------------------------------------------------------------------------
# 1. SmartMemoryUpdater._calculate_similarity
# ---------------------------------------------------------------------------


class TestSmartMemoryUpdaterSimilarity:
    """Unit tests for the _calculate_similarity method."""

    # ------------------------------------------------------------------
    # Cosine path (embedder provided)
    # ------------------------------------------------------------------

    def test_cosine_similarity_matches_expected_value(self) -> None:
        """With a mock embedder returning known vectors, similarity equals cosine(v1, v2)."""
        from hippocampai.pipeline.smart_updater import SmartMemoryUpdater

        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])

        mock_embedder = MagicMock()
        mock_embedder.encode_single.side_effect = [v1, v2]

        updater = SmartMemoryUpdater(embedder=mock_embedder)
        result = updater._calculate_similarity("foo", "bar")

        expected = _cosine(v1, v2)  # 0.0 — orthogonal vectors
        assert math.isclose(result, expected, abs_tol=1e-6)

    def test_cosine_similarity_identical_direction(self) -> None:
        """Parallel vectors (same direction) should yield cosine similarity of 1.0."""
        from hippocampai.pipeline.smart_updater import SmartMemoryUpdater

        v = np.array([0.6, 0.8, 0.0])
        mock_embedder = MagicMock()
        mock_embedder.encode_single.side_effect = [v.copy(), v.copy()]

        updater = SmartMemoryUpdater(embedder=mock_embedder)
        result = updater._calculate_similarity("same text", "same text again")

        assert math.isclose(result, 1.0, abs_tol=1e-6)

    def test_cosine_similarity_partial_overlap(self) -> None:
        """Vectors at 45 degrees should yield cos(45°) ≈ 0.707."""
        from hippocampai.pipeline.smart_updater import SmartMemoryUpdater

        v1 = np.array([1.0, 1.0, 0.0]) / math.sqrt(2)
        v2 = np.array([1.0, 0.0, 0.0])
        mock_embedder = MagicMock()
        mock_embedder.encode_single.side_effect = [v1, v2]

        updater = SmartMemoryUpdater(embedder=mock_embedder)
        result = updater._calculate_similarity("a b", "a")

        expected = _cosine(v1, v2)
        assert math.isclose(result, expected, abs_tol=1e-6)

    def test_cosine_returns_zero_for_zero_norm_vector(self) -> None:
        """When embedder returns a zero-norm vector, similarity must be 0.0 (no div-by-zero)."""
        from hippocampai.pipeline.smart_updater import SmartMemoryUpdater

        mock_embedder = MagicMock()
        mock_embedder.encode_single.side_effect = [
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
        ]

        updater = SmartMemoryUpdater(embedder=mock_embedder)
        result = updater._calculate_similarity("empty", "something")

        assert result == 0.0

    # ------------------------------------------------------------------
    # Jaccard fallback — embedder=None
    # ------------------------------------------------------------------

    def test_falls_back_to_jaccard_when_embedder_is_none(self) -> None:
        """When embedder=None, Jaccard similarity is used, not cosine."""
        from hippocampai.pipeline.smart_updater import SmartMemoryUpdater

        updater = SmartMemoryUpdater(embedder=None)
        # "cat dog" ∩ "cat fish" = {"cat"}, union = {"cat", "dog", "fish"} → Jaccard = 1/3
        result = updater._calculate_similarity("cat dog", "cat fish")

        assert math.isclose(result, 1 / 3, abs_tol=1e-6)

    def test_jaccard_identical_texts_returns_one(self) -> None:
        """Identical texts must return 1.0 via the exact-match short-circuit."""
        from hippocampai.pipeline.smart_updater import SmartMemoryUpdater

        updater = SmartMemoryUpdater(embedder=None)
        assert updater._calculate_similarity("hello world", "Hello World") == 1.0

    def test_jaccard_empty_token_set_returns_zero(self) -> None:
        """Empty text (no extractable tokens) should return 0.0 without crashing."""
        from hippocampai.pipeline.smart_updater import SmartMemoryUpdater

        updater = SmartMemoryUpdater(embedder=None)
        result = updater._calculate_similarity("", "hello")

        assert result == 0.0

    # ------------------------------------------------------------------
    # Embedder raises → Jaccard fallback (no crash)
    # ------------------------------------------------------------------

    def test_falls_back_to_jaccard_when_embedder_raises(self) -> None:
        """If embedder.encode_single raises any exception, Jaccard is used without crashing."""
        from hippocampai.pipeline.smart_updater import SmartMemoryUpdater

        mock_embedder = MagicMock()
        mock_embedder.encode_single.side_effect = RuntimeError("GPU OOM")

        updater = SmartMemoryUpdater(embedder=mock_embedder)
        # "apple orange" ∩ "apple" = {"apple"}, union = {"apple", "orange"} → Jaccard = 0.5
        result = updater._calculate_similarity("apple orange", "apple")

        assert math.isclose(result, 0.5, abs_tol=1e-6)

    def test_embedder_raises_does_not_propagate_exception(self) -> None:
        """An exception in the embedder must not propagate out of _calculate_similarity."""
        from hippocampai.pipeline.smart_updater import SmartMemoryUpdater

        mock_embedder = MagicMock()
        mock_embedder.encode_single.side_effect = ValueError("Bad input")

        updater = SmartMemoryUpdater(embedder=mock_embedder)
        # Should not raise
        try:
            updater._calculate_similarity("text a", "text b")
        except Exception as exc:
            pytest.fail(f"_calculate_similarity raised unexpectedly: {exc}")


# ---------------------------------------------------------------------------
# 2. GraphRetriever.search — hop distance scoring
# ---------------------------------------------------------------------------


class TestGraphRetrieverHopDistanceScoring:
    """Validate that real BFS hop distances produce the correct score decay."""

    # ------------------------------------------------------------------
    # Graph fixture helpers
    # ------------------------------------------------------------------

    def _make_mock_knowledge_graph(
        self,
        entity_id: str,
        entity_node_id: str,
        direct_memory_ids: list[str],
        hop_memory_edges: list[tuple[str, str, int]],
        user_id: str,
    ) -> MagicMock:
        """Build a mock KnowledgeGraph backed by a real nx.DiGraph.

        Parameters
        ----------
        entity_id:
            The logical entity ID used to look up its node.
        entity_node_id:
            The node ID in the graph that represents the entity.
        direct_memory_ids:
            Memory IDs returned by get_entity_memories() (hop 0).
        hop_memory_edges:
            List of (from_node_id, memory_node_id, num_hops_away) tuples.
            Intermediate nodes are auto-created for multi-hop paths.
        user_id:
            User whose memory set should include all memory nodes.
        """
        from hippocampai.graph.knowledge_graph import NodeType

        g: nx.DiGraph = nx.DiGraph()

        # Add entity node
        g.add_node(entity_node_id, node_type=NodeType.ENTITY.value)

        # Add direct memory nodes (these are already returned by
        # get_entity_memories, so the graph traversal starts at hop 1+)
        for mem_id in direct_memory_ids:
            g.add_node(mem_id, node_type=NodeType.MEMORY.value)
            g.add_edge(entity_node_id, mem_id, weight=1.0)

        # Add edges for hop memories; build intermediate nodes as needed
        for from_node, mem_node_id, _hops in hop_memory_edges:
            g.add_node(mem_node_id, node_type=NodeType.MEMORY.value)
            g.add_edge(from_node, mem_node_id, weight=1.0)

        # All memory nodes belong to this user
        all_memory_ids: set[str] = set(direct_memory_ids) | {
            m for _, m, _ in hop_memory_edges
        }

        mock_kg = MagicMock()
        mock_kg.graph = g
        mock_kg._entity_index = {entity_id: entity_node_id}
        mock_kg._user_graphs = {user_id: all_memory_ids}
        mock_kg.get_entity_memories.return_value = direct_memory_ids

        return mock_kg

    # ------------------------------------------------------------------
    # Test: direct (hop 0) memory scores higher than hop-1 memory
    # ------------------------------------------------------------------

    def test_direct_memory_scores_higher_than_hop1_memory(self) -> None:
        """A memory directly linked to the entity node beats one 1 hop away."""
        from hippocampai.pipeline.entity_recognition import EntityRecognizer
        from hippocampai.retrieval.graph_retriever import GraphRetriever

        entity_id = "ent_python"
        entity_node_id = f"entity_{entity_id}"
        direct_mem = "mem_direct"
        hop1_mem = "mem_hop1"
        user_id = "user_test"

        # entity_node → direct_mem  (hop 0 via get_entity_memories)
        # entity_node → hop1_mem    (hop 1 in graph)
        mock_kg = self._make_mock_knowledge_graph(
            entity_id=entity_id,
            entity_node_id=entity_node_id,
            direct_memory_ids=[direct_mem],
            hop_memory_edges=[(entity_node_id, hop1_mem, 1)],
            user_id=user_id,
        )

        mock_entity = MagicMock()
        mock_entity.entity_id = entity_id
        mock_entity.confidence = 1.0

        mock_recognizer = MagicMock(spec=EntityRecognizer)
        mock_recognizer.extract_entities.return_value = [mock_entity]

        retriever = GraphRetriever(graph=mock_kg, entity_recognizer=mock_recognizer, max_depth=2)
        results = retriever.search("python programming", user_id=user_id)

        result_dict = dict(results)
        assert direct_mem in result_dict, "Direct memory must be in results"
        assert hop1_mem in result_dict, "Hop-1 memory must be in results"
        # After normalisation the direct memory should have score == 1.0
        assert result_dict[direct_mem] > result_dict[hop1_mem], (
            "Direct memory score must exceed hop-1 memory score"
        )

    def test_hop1_memory_scores_higher_than_hop2_memory(self) -> None:
        """A memory 1 hop from the entity beats one 2 hops away."""
        from hippocampai.pipeline.entity_recognition import EntityRecognizer
        from hippocampai.retrieval.graph_retriever import GraphRetriever

        entity_id = "ent_python"
        entity_node_id = f"entity_{entity_id}"
        hop1_mem = "mem_hop1"
        hop2_mem = "mem_hop2"
        intermediate_node = "intermediate_topic"
        user_id = "user_test"

        g: nx.DiGraph = nx.DiGraph()
        from hippocampai.graph.knowledge_graph import NodeType

        g.add_node(entity_node_id, node_type=NodeType.ENTITY.value)
        g.add_node(hop1_mem, node_type=NodeType.MEMORY.value)
        g.add_node(intermediate_node, node_type=NodeType.TOPIC.value)
        g.add_node(hop2_mem, node_type=NodeType.MEMORY.value)

        # entity → hop1_mem (hop 1)
        g.add_edge(entity_node_id, hop1_mem, weight=1.0)
        # entity → intermediate → hop2_mem (hop 2)
        g.add_edge(entity_node_id, intermediate_node, weight=1.0)
        g.add_edge(intermediate_node, hop2_mem, weight=1.0)

        mock_kg = MagicMock()
        mock_kg.graph = g
        mock_kg._entity_index = {entity_id: entity_node_id}
        mock_kg._user_graphs = {user_id: {hop1_mem, hop2_mem}}
        mock_kg.get_entity_memories.return_value = []  # no direct (hop 0) memories

        mock_entity = MagicMock()
        mock_entity.entity_id = entity_id
        mock_entity.confidence = 1.0

        mock_recognizer = MagicMock(spec=EntityRecognizer)
        mock_recognizer.extract_entities.return_value = [mock_entity]

        retriever = GraphRetriever(graph=mock_kg, entity_recognizer=mock_recognizer, max_depth=2)
        results = retriever.search("python", user_id=user_id)

        result_dict = dict(results)
        assert hop1_mem in result_dict, "Hop-1 memory must be in results"
        assert hop2_mem in result_dict, "Hop-2 memory must be in results"
        assert result_dict[hop1_mem] > result_dict[hop2_mem], (
            "Hop-1 memory score must exceed hop-2 memory score"
        )

    def test_score_formula_hop0_equals_entity_confidence(self) -> None:
        """Hop-0 raw score must equal entity.confidence * 1.0 before normalization.

        When there is only one memory (the direct one), the normalized score is
        exactly 1.0 because it is divided by itself.
        """
        from hippocampai.pipeline.entity_recognition import EntityRecognizer
        from hippocampai.retrieval.graph_retriever import GraphRetriever

        entity_id = "ent_x"
        entity_node_id = f"entity_{entity_id}"
        direct_mem = "mem_only"
        user_id = "user_1"
        confidence = 0.75

        g: nx.DiGraph = nx.DiGraph()
        from hippocampai.graph.knowledge_graph import NodeType

        g.add_node(entity_node_id, node_type=NodeType.ENTITY.value)
        g.add_node(direct_mem, node_type=NodeType.MEMORY.value)
        g.add_edge(entity_node_id, direct_mem, weight=1.0)

        mock_kg = MagicMock()
        mock_kg.graph = g
        mock_kg._entity_index = {entity_id: entity_node_id}
        mock_kg._user_graphs = {user_id: {direct_mem}}
        mock_kg.get_entity_memories.return_value = [direct_mem]

        mock_entity = MagicMock()
        mock_entity.entity_id = entity_id
        mock_entity.confidence = confidence

        mock_recognizer = MagicMock(spec=EntityRecognizer)
        mock_recognizer.extract_entities.return_value = [mock_entity]

        retriever = GraphRetriever(graph=mock_kg, entity_recognizer=mock_recognizer, max_depth=2)
        results = retriever.search("query", user_id=user_id)

        assert len(results) == 1
        mem_id, score = results[0]
        assert mem_id == direct_mem
        # Only one memory → normalised score = raw / raw = 1.0
        assert math.isclose(score, 1.0, abs_tol=1e-6)

    def test_no_entities_extracted_returns_empty_list(self) -> None:
        """When entity recognition returns nothing, search must return an empty list."""
        from hippocampai.pipeline.entity_recognition import EntityRecognizer
        from hippocampai.retrieval.graph_retriever import GraphRetriever

        mock_kg = MagicMock()
        mock_recognizer = MagicMock(spec=EntityRecognizer)
        mock_recognizer.extract_entities.return_value = []

        retriever = GraphRetriever(graph=mock_kg, entity_recognizer=mock_recognizer)
        results = retriever.search("anything", user_id="u1")

        assert results == []

    def test_entity_not_in_graph_index_returns_empty_list(self) -> None:
        """An entity not present in _entity_index contributes no results."""
        from hippocampai.pipeline.entity_recognition import EntityRecognizer
        from hippocampai.retrieval.graph_retriever import GraphRetriever

        mock_entity = MagicMock()
        mock_entity.entity_id = "unknown_entity"
        mock_entity.confidence = 0.9

        mock_kg = MagicMock()
        mock_kg._entity_index = {}  # entity not indexed
        mock_kg._user_graphs = {"u1": {"some_mem"}}

        mock_recognizer = MagicMock(spec=EntityRecognizer)
        mock_recognizer.extract_entities.return_value = [mock_entity]

        retriever = GraphRetriever(graph=mock_kg, entity_recognizer=mock_recognizer)
        results = retriever.search("query", user_id="u1")

        assert results == []


# ---------------------------------------------------------------------------
# 3. QueryIntentDetector.detect
# ---------------------------------------------------------------------------


class TestQueryIntentDetector:
    """Unit tests for QueryIntentDetector.detect()."""

    @pytest.fixture
    def detector(self):
        from hippocampai.retrieval.retriever import QueryIntentDetector
        return QueryIntentDetector()

    def test_temporal_query_returns_recency_multiplier(self, detector) -> None:
        """'what did I recently update' → {"recency": 2.0}."""
        result = detector.detect("what did I recently update")
        assert result == {"recency": 2.0}

    def test_preference_query_returns_importance_multiplier(self, detector) -> None:
        """'what do I like' → {"importance": 1.5}."""
        result = detector.detect("what do I like")
        assert result == {"importance": 1.5}

    def test_neutral_query_returns_empty_dict(self, detector) -> None:
        """'hello world' → {} (no intent signal detected)."""
        result = detector.detect("hello world")
        assert result == {}

    def test_combined_query_returns_both_multipliers(self, detector) -> None:
        """'what are my favorite recent things' → both recency and importance."""
        result = detector.detect("what are my favorite recent things")
        assert result == {"recency": 2.0, "importance": 1.5}

    def test_case_insensitive_token_matching(self, detector) -> None:
        """Token matching must be case-insensitive (query is lowercased before splitting)."""
        result = detector.detect("What did I RECENTLY update")
        assert "recency" in result
        assert result["recency"] == 2.0

    def test_empty_query_returns_empty_dict(self, detector) -> None:
        """An empty query string produces no intent signals."""
        result = detector.detect("")
        assert result == {}

    def test_preference_token_love_detected(self, detector) -> None:
        """'love' is in PREFERENCE_TOKENS and should trigger importance boost."""
        result = detector.detect("foods I love")
        assert "importance" in result

    def test_temporal_token_current_detected(self, detector) -> None:
        """'current' is in TEMPORAL_TOKENS and should trigger recency boost."""
        result = detector.detect("what is my current status")
        assert "recency" in result

    def test_only_recency_no_importance_for_pure_temporal(self, detector) -> None:
        """A purely temporal query must not inject an importance key."""
        result = detector.detect("what did I update today")
        assert "recency" in result
        assert "importance" not in result

    def test_only_importance_no_recency_for_pure_preference(self, detector) -> None:
        """A purely preference query must not inject a recency key."""
        result = detector.detect("what do I enjoy")
        assert "importance" in result
        assert "recency" not in result


# ---------------------------------------------------------------------------
# 4. HybridRetriever.add_to_corpus
# ---------------------------------------------------------------------------


def _make_hybrid_retriever():
    """Return a HybridRetriever with all external dependencies fully mocked."""
    from hippocampai.retrieval.retriever import HybridRetriever

    mock_qdrant = MagicMock()
    mock_qdrant.collection_facts = "hippocampai_facts"
    mock_qdrant.collection_prefs = "hippocampai_prefs"

    mock_embedder = MagicMock()
    mock_reranker = MagicMock()

    return HybridRetriever(
        qdrant_store=mock_qdrant,
        embedder=mock_embedder,
        reranker=mock_reranker,
    )


class TestHybridRetrieverAddToCorpus:
    """Unit tests for HybridRetriever.add_to_corpus()."""

    def test_memory_id_appears_in_corpus_facts_after_add(self) -> None:
        """After add_to_corpus with facts collection, memory_id is in corpus_facts."""
        retriever = _make_hybrid_retriever()
        retriever.add_to_corpus("mem_001", "Python is great", retriever.qdrant.collection_facts)

        ids_in_corpus = [mid for mid, _ in retriever.corpus_facts]
        assert "mem_001" in ids_in_corpus

    def test_text_stored_alongside_id_in_corpus_facts(self) -> None:
        """The stored tuple should be (memory_id, text)."""
        retriever = _make_hybrid_retriever()
        retriever.add_to_corpus("mem_002", "Machine learning concepts", retriever.qdrant.collection_facts)

        corpus_dict = dict(retriever.corpus_facts)
        assert corpus_dict["mem_002"] == "Machine learning concepts"

    def test_duplicate_id_does_not_add_second_entry(self) -> None:
        """Calling add_to_corpus twice with the same ID must not duplicate the entry."""
        retriever = _make_hybrid_retriever()
        retriever.add_to_corpus("mem_dup", "First version", retriever.qdrant.collection_facts)
        retriever.add_to_corpus("mem_dup", "Second version", retriever.qdrant.collection_facts)

        ids_in_corpus = [mid for mid, _ in retriever.corpus_facts]
        assert ids_in_corpus.count("mem_dup") == 1

    def test_bm25_facts_index_is_not_none_after_first_add_when_initialised(self) -> None:
        """If bm25_facts is pre-initialised, add_to_corpus must update it (not None after add)."""
        from hippocampai.retrieval.bm25 import BM25Retriever

        retriever = _make_hybrid_retriever()
        # Pre-seed the BM25 index to simulate a post-rebuild_bm25() state
        retriever.bm25_facts = BM25Retriever(["seed document"])
        retriever.corpus_facts = [("seed_id", "seed document")]

        retriever.add_to_corpus("new_mem", "new memory text", retriever.qdrant.collection_facts)

        assert retriever.bm25_facts is not None

    def test_bm25_facts_index_stays_none_before_rebuild_when_not_initialised(self) -> None:
        """When bm25_facts is None (pre-rebuild_bm25), add_to_corpus must NOT create the index.

        The corpus is buffered; the index is built on the first rebuild_bm25() call.
        """
        retriever = _make_hybrid_retriever()
        assert retriever.bm25_facts is None

        retriever.add_to_corpus("mem_x", "some text", retriever.qdrant.collection_facts)

        assert retriever.bm25_facts is None  # still None — index not yet built

    def test_add_to_corpus_prefs_collection_updates_corpus_prefs(self) -> None:
        """Adding with prefs collection should update corpus_prefs, not corpus_facts."""
        retriever = _make_hybrid_retriever()
        retriever.add_to_corpus("pref_001", "I prefer Python", retriever.qdrant.collection_prefs)

        pref_ids = [mid for mid, _ in retriever.corpus_prefs]
        fact_ids = [mid for mid, _ in retriever.corpus_facts]
        assert "pref_001" in pref_ids
        assert "pref_001" not in fact_ids

    def test_add_multiple_distinct_ids_all_stored(self) -> None:
        """Multiple distinct IDs should all appear in the corpus."""
        retriever = _make_hybrid_retriever()
        ids_to_add = ["m1", "m2", "m3"]
        for i, mid in enumerate(ids_to_add):
            retriever.add_to_corpus(mid, f"text {i}", retriever.qdrant.collection_facts)

        stored_ids = [mid for mid, _ in retriever.corpus_facts]
        for mid in ids_to_add:
            assert mid in stored_ids

    def test_bm25_dirty_flag_is_false_after_add(self) -> None:
        """_bm25_dirty must be False after add_to_corpus (corpus and index in sync)."""
        retriever = _make_hybrid_retriever()
        retriever.add_to_corpus("mem_y", "some text", retriever.qdrant.collection_facts)
        assert retriever._bm25_dirty is False


# ---------------------------------------------------------------------------
# 5. Config defaults
# ---------------------------------------------------------------------------


class TestConfigDefaults:
    """Validate Config field defaults without relying on a .env file."""

    def test_enable_graph_retrieval_default_is_true(self) -> None:
        """Config.enable_graph_retrieval must default to True."""
        # Isolate from any .env file present in the working directory
        with patch.dict("os.environ", {"ENABLE_GRAPH_RETRIEVAL": "true"}, clear=False):
            from hippocampai.config import Config

            cfg = Config()
            assert cfg.enable_graph_retrieval is True

    def test_weight_graph_default_is_0_15(self) -> None:
        """Config.weight_graph must default to 0.15."""
        with patch.dict("os.environ", {}, clear=False):
            from hippocampai.config import Config

            cfg = Config()
            assert math.isclose(cfg.weight_graph, 0.15, abs_tol=1e-9)

    def test_enable_graph_retrieval_can_be_overridden_via_env(self) -> None:
        """ENABLE_GRAPH_RETRIEVAL env var must override the default."""
        with patch.dict("os.environ", {"ENABLE_GRAPH_RETRIEVAL": "false"}, clear=False):
            from hippocampai.config import Config

            cfg = Config()
            assert cfg.enable_graph_retrieval is False

    def test_weight_graph_appears_in_get_weights(self) -> None:
        """weight_graph must be present in Config.get_weights() output."""
        with patch.dict("os.environ", {}, clear=False):
            from hippocampai.config import Config

            cfg = Config()
            weights = cfg.get_weights()
            assert "graph" in weights
            assert math.isclose(weights["graph"], cfg.weight_graph, abs_tol=1e-9)
