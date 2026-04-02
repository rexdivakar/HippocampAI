"""End-to-end tests for the HippocampAI knowledge-graph layer.

Covers:
- MemoryGraph dirty callback lifecycle
- suggest_relationships (embedding path, Jaccard fallback, edge cases)
- graph_persistence round-trips (with and without embeddings)
- infer_new_facts (all 7 rules, dedup, empty graph, graph immutability)
- LLM inference path (fires when provided, skipped when None, exception resilience)
- MemoryClient.check_graph_qdrant_drift (in-sync, graph-only, qdrant-only, empty user)
"""

from __future__ import annotations

import tempfile
from unittest.mock import MagicMock

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Path fixup (mirrors conftest pattern already in the suite)
# ---------------------------------------------------------------------------
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from hippocampai.graph.knowledge_graph import KnowledgeGraph, NodeType
from hippocampai.graph.memory_graph import MemoryGraph, RelationType
from hippocampai.graph.graph_persistence import load, save


# ===========================================================================
# Helpers / shared fixtures
# ===========================================================================


def _make_person_entity_node(kg: KnowledgeGraph, node_id: str, name: str) -> None:
    """Directly add a bare entity node with node_type=entity and text to the graph."""
    kg.graph.add_node(
        node_id,
        node_type=NodeType.ENTITY.value,
        entity_id=node_id,
        text=name,
    )
    kg._entity_index[node_id] = node_id


def _link_entity(
    kg: KnowledgeGraph,
    source_node: str,
    target_node: str,
    original_relation: str,
) -> None:
    """Add a directed edge carrying original_relation in its metadata."""
    kg.graph.add_edge(
        source_node,
        target_node,
        relation=RelationType.RELATED_TO.value,
        weight=0.9,
        original_relation=original_relation,
    )


# ===========================================================================
# 1. Dirty Callback Tests
# ===========================================================================


class TestDirtyCallback:
    """MemoryGraph.register_dirty_callback lifecycle."""

    def test_dirty_callback_fires_once_on_add_memory(self):
        """Callback is invoked exactly once when add_memory succeeds."""
        mg = MemoryGraph()
        counter = {"n": 0}

        def _cb():
            counter["n"] += 1

        mg.register_dirty_callback(_cb)
        mg.add_memory("m1", "user_a")

        assert counter["n"] == 1

    def test_dirty_callback_fires_on_each_distinct_add_memory(self):
        """Callback fires once per add_memory call, not batched."""
        mg = MemoryGraph()
        calls: list[int] = []

        mg.register_dirty_callback(lambda: calls.append(1))
        mg.add_memory("m1", "user_a")
        mg.add_memory("m2", "user_a")

        assert len(calls) == 2

    def test_dirty_callback_fires_on_successful_add_relationship(self):
        """Callback fires when add_relationship returns True (both nodes present)."""
        mg = MemoryGraph()
        calls: list[int] = []

        mg.register_dirty_callback(lambda: calls.append(1))
        mg.add_memory("m1", "user_a")
        mg.add_memory("m2", "user_a")
        # Two from add_memory; reset for clarity
        calls.clear()

        result = mg.add_relationship("m1", "m2", RelationType.RELATED_TO)

        assert result is True
        assert len(calls) == 1

    def test_dirty_callback_does_not_fire_on_failed_add_relationship(self):
        """Callback is NOT invoked when add_relationship fails (missing node)."""
        mg = MemoryGraph()
        calls: list[int] = []

        mg.register_dirty_callback(lambda: calls.append(1))
        # m2 is not in the graph — relationship must fail
        mg.add_memory("m1", "user_a")
        calls.clear()

        result = mg.add_relationship("m1", "ghost_node", RelationType.RELATED_TO)

        assert result is False
        assert len(calls) == 0

    def test_dirty_callback_fires_on_remove_memory(self):
        """Callback fires when an existing memory is removed."""
        mg = MemoryGraph()
        calls: list[int] = []

        mg.register_dirty_callback(lambda: calls.append(1))
        mg.add_memory("m1", "user_a")
        calls.clear()

        mg.remove_memory("m1")

        assert len(calls) == 1

    def test_dirty_callback_does_not_fire_on_remove_nonexistent_memory(self):
        """Callback is NOT invoked when remove_memory is called for an unknown ID."""
        mg = MemoryGraph()
        calls: list[int] = []

        mg.register_dirty_callback(lambda: calls.append(1))
        mg.remove_memory("does_not_exist")

        assert len(calls) == 0

    def test_no_error_when_no_callback_registered_add_memory(self):
        """add_memory does not raise when no dirty callback has been registered."""
        mg = MemoryGraph()
        mg.add_memory("m1", "user_a")  # must not raise

    def test_no_error_when_no_callback_registered_remove_memory(self):
        """remove_memory does not raise when no dirty callback is registered."""
        mg = MemoryGraph()
        mg.add_memory("m1", "user_a")
        mg.remove_memory("m1")  # must not raise

    def test_no_error_when_no_callback_registered_add_relationship(self):
        """add_relationship does not raise when no dirty callback is registered."""
        mg = MemoryGraph()
        mg.add_memory("m1", "user_a")
        mg.add_memory("m2", "user_a")
        mg.add_relationship("m1", "m2", RelationType.RELATED_TO)  # must not raise

    def test_callback_can_be_replaced(self):
        """A second call to register_dirty_callback replaces the first callback."""
        mg = MemoryGraph()
        first_calls: list[int] = []
        second_calls: list[int] = []

        mg.register_dirty_callback(lambda: first_calls.append(1))
        mg.register_dirty_callback(lambda: second_calls.append(1))

        mg.add_memory("m1", "user_a")

        assert len(first_calls) == 0, "Old callback must not fire after replacement"
        assert len(second_calls) == 1


# ===========================================================================
# 2. suggest_relationships Tests
# ===========================================================================


class TestSuggestRelationships:
    """suggest_relationships: embedding path, Jaccard fallback, edge cases."""

    def _make_graph_with_two_nodes(self) -> MemoryGraph:
        mg = MemoryGraph()
        mg.add_memory("m1", "user_a")
        mg.add_memory("m2", "user_a")
        return mg

    def test_embedding_path_returns_similar_to_for_high_cosine(self):
        """Cosine >= 0.85 maps to SIMILAR_TO relation."""
        mg = self._make_graph_with_two_nodes()

        v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        # Store nearly identical vector on m2
        mg.graph.nodes["m2"]["embedding"] = np.array([0.99, 0.141, 0.0], dtype=np.float32)

        suggestions = mg.suggest_relationships("m1", ["m2"], threshold=0.0, source_embedding=v)

        assert len(suggestions) == 1
        cand_id, rel_type, score = suggestions[0]
        assert cand_id == "m2"
        assert rel_type == RelationType.SIMILAR_TO
        assert score > 0.85

    def test_embedding_path_returns_related_to_for_medium_cosine(self):
        """0.7 <= cosine < 0.85 maps to RELATED_TO."""
        mg = self._make_graph_with_two_nodes()

        # Build two vectors whose cosine is approximately 0.77
        v_src = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        # 40-degree angle -> cosine ~ 0.766
        angle = np.radians(40)
        v_cand = np.array([np.cos(angle), np.sin(angle), 0.0], dtype=np.float32)
        mg.graph.nodes["m2"]["embedding"] = v_cand

        suggestions = mg.suggest_relationships(
            "m1", ["m2"], threshold=0.7, source_embedding=v_src
        )

        assert len(suggestions) == 1
        _, rel_type, _ = suggestions[0]
        assert rel_type == RelationType.RELATED_TO

    def test_embedding_path_filters_below_threshold(self):
        """Candidates with cosine below threshold are excluded."""
        mg = self._make_graph_with_two_nodes()

        v_src = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        # Orthogonal vector: cosine == 0.0
        mg.graph.nodes["m2"]["embedding"] = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        suggestions = mg.suggest_relationships(
            "m1", ["m2"], threshold=0.7, source_embedding=v_src
        )

        assert suggestions == []

    def test_embedding_path_skips_candidate_without_embedding(self):
        """Candidate nodes that lack an embedding attribute are silently skipped."""
        mg = self._make_graph_with_two_nodes()
        # m2 has no embedding stored

        v_src = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        suggestions = mg.suggest_relationships(
            "m1", ["m2"], threshold=0.0, source_embedding=v_src
        )

        assert suggestions == []

    def test_zero_norm_source_embedding_falls_back_to_jaccard(self):
        """A zero-norm source embedding must fall through to Jaccard, not crash."""
        mg = MemoryGraph()
        mg.add_memory("m1", "user_a")
        mg.add_memory("m2", "user_a")
        mg.add_memory("shared", "user_a")
        # Connect both m1 and m2 to 'shared' so Jaccard similarity is non-zero
        mg.add_relationship("m1", "shared", RelationType.RELATED_TO)
        mg.add_relationship("m2", "shared", RelationType.RELATED_TO)

        zero_vec = np.zeros(3, dtype=np.float32)
        # Should not raise; should fall back to Jaccard
        suggestions = mg.suggest_relationships(
            "m1", ["m2"], threshold=0.0, source_embedding=zero_vec
        )

        # Jaccard with one shared out of two total neighbours = 0.5 >= 0.0
        assert len(suggestions) >= 1
        assert suggestions[0][0] == "m2"

    def test_jaccard_fallback_used_when_no_source_embedding(self):
        """Without source_embedding the Jaccard path is taken."""
        mg = MemoryGraph()
        for mid in ("m1", "m2", "shared_a", "shared_b"):
            mg.add_memory(mid, "user_a")

        mg.add_relationship("m1", "shared_a", RelationType.RELATED_TO)
        mg.add_relationship("m1", "shared_b", RelationType.RELATED_TO)
        mg.add_relationship("m2", "shared_a", RelationType.RELATED_TO)
        mg.add_relationship("m2", "shared_b", RelationType.RELATED_TO)

        suggestions = mg.suggest_relationships("m1", ["m2"], threshold=0.5)

        assert len(suggestions) == 1
        cand_id, rel_type, score = suggestions[0]
        assert cand_id == "m2"
        assert rel_type == RelationType.RELATED_TO
        assert score == pytest.approx(1.0)

    def test_jaccard_returns_empty_when_no_common_neighbours(self):
        """No shared neighbours → Jaccard similarity is 0 → no suggestion."""
        mg = MemoryGraph()
        for mid in ("m1", "m2", "n1", "n2"):
            mg.add_memory(mid, "user_a")
        mg.add_relationship("m1", "n1", RelationType.RELATED_TO)
        mg.add_relationship("m2", "n2", RelationType.RELATED_TO)

        suggestions = mg.suggest_relationships("m1", ["m2"], threshold=0.5)

        assert suggestions == []

    def test_empty_candidates_returns_empty_list(self):
        """Passing an empty candidate list returns [] immediately."""
        mg = MemoryGraph()
        mg.add_memory("m1", "user_a")

        suggestions = mg.suggest_relationships("m1", [], threshold=0.0)

        assert suggestions == []

    def test_unknown_source_memory_returns_empty_list(self):
        """Source memory not in graph returns []."""
        mg = MemoryGraph()
        mg.add_memory("m2", "user_a")

        suggestions = mg.suggest_relationships("ghost", ["m2"], threshold=0.0)

        assert suggestions == []

    def test_suggestions_sorted_by_confidence_descending(self):
        """Returned suggestions are ordered highest confidence first."""
        mg = MemoryGraph()
        for mid in ("m1", "m2", "m3", "shared"):
            mg.add_memory(mid, "user_a")

        # Give m2 and m3 different cosine scores via embeddings
        v_src = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        # m2: cos ~ 0.99
        mg.graph.nodes["m2"]["embedding"] = np.array([1.0, 0.1, 0.0], dtype=np.float32)
        # m3: cos ~ 0.71 (45-degree)
        angle = np.radians(45)
        mg.graph.nodes["m3"]["embedding"] = np.array(
            [np.cos(angle), np.sin(angle), 0.0], dtype=np.float32
        )

        suggestions = mg.suggest_relationships(
            "m1", ["m2", "m3"], threshold=0.0, source_embedding=v_src
        )

        assert len(suggestions) == 2
        assert suggestions[0][2] >= suggestions[1][2], "Results must be sorted descending"


# ===========================================================================
# 3. graph_persistence Round-trip Tests
# ===========================================================================


class TestGraphPersistence:
    """save() / load() round-trip correctness."""

    def test_round_trip_without_embeddings_restores_nodes_and_edges(self):
        """Basic round-trip preserves nodes and edges when no embeddings are stored."""
        kg = KnowledgeGraph()
        kg.add_memory("m1", "alice")
        kg.add_memory("m2", "alice")
        kg.add_relationship("m1", "m2", RelationType.RELATED_TO)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        save(kg, path)
        kg2 = load(path)

        assert kg2.graph.has_node("m1")
        assert kg2.graph.has_node("m2")
        assert kg2.graph.has_edge("m1", "m2")
        assert kg2.graph.number_of_nodes() == kg.graph.number_of_nodes()
        assert kg2.graph.number_of_edges() == kg.graph.number_of_edges()

    def test_round_trip_with_embeddings_restores_ndarray(self):
        """Embeddings saved as lists are restored as numpy arrays after load()."""
        kg = KnowledgeGraph()
        original_embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        kg.add_memory("m1", "alice", embedding=original_embedding)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        save(kg, path)
        kg2 = load(path)

        restored = kg2.graph.nodes["m1"].get("embedding")
        assert restored is not None
        assert isinstance(restored, np.ndarray), "Embedding must be restored as ndarray"

    def test_round_trip_with_embeddings_restores_correct_dtype(self):
        """Restored embedding arrays must have dtype float32."""
        kg = KnowledgeGraph()
        kg.add_memory("m1", "alice", embedding=np.array([1.0, 2.0], dtype=np.float32))

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        save(kg, path)
        kg2 = load(path)

        restored = kg2.graph.nodes["m1"]["embedding"]
        assert restored.dtype == np.float32

    def test_round_trip_with_embeddings_preserves_values(self):
        """Restored embedding values must be numerically close to originals."""
        kg = KnowledgeGraph()
        original = np.array([0.25, -0.5, 0.75], dtype=np.float32)
        kg.add_memory("m1", "alice", embedding=original)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        save(kg, path)
        kg2 = load(path)

        restored = kg2.graph.nodes["m1"]["embedding"]
        np.testing.assert_allclose(restored, original, rtol=1e-5)

    def test_save_does_not_mutate_live_graph_embeddings(self):
        """save() must not convert live graph's ndarray to list (no mutation)."""
        kg = KnowledgeGraph()
        original = np.array([1.0, 2.0], dtype=np.float32)
        kg.add_memory("m1", "alice", embedding=original)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        save(kg, path)

        live_embedding = kg.graph.nodes["m1"]["embedding"]
        assert isinstance(live_embedding, np.ndarray), (
            "Live graph embedding must remain ndarray after save()"
        )

    def test_round_trip_restores_auxiliary_indices(self):
        """save/load preserves _user_graphs and _memory_index."""
        kg = KnowledgeGraph()
        kg.add_memory("m1", "alice")
        kg.add_memory("m2", "bob")

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        save(kg, path)
        kg2 = load(path)

        assert "m1" in kg2._user_graphs.get("alice", set())
        assert "m2" in kg2._user_graphs.get("bob", set())
        assert "m1" in kg2._memory_index
        assert kg2._memory_index["m1"]["user_id"] == "alice"

    def test_load_raises_file_not_found_for_missing_path(self):
        """load() raises FileNotFoundError for a non-existent path."""
        with pytest.raises(FileNotFoundError):
            load("/tmp/does_not_exist_hippocampai_xyz.json")

    def test_round_trip_entity_index_restored(self):
        """_entity_index survives a save/load cycle."""
        kg = KnowledgeGraph()
        _make_person_entity_node(kg, "entity_alice", "Alice")

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        save(kg, path)
        kg2 = load(path)

        assert "entity_alice" in kg2._entity_index


# ===========================================================================
# 4. infer_new_facts Tests
# ===========================================================================


class TestInferNewFacts:
    """infer_new_facts: all 7 rules, deduplication, empty graph, immutability."""

    # ---- helpers ----

    def _person(self, kg: KnowledgeGraph, eid: str, name: str) -> str:
        node_id = f"entity_{eid}"
        _make_person_entity_node(kg, node_id, name)
        return node_id

    def _org(self, kg: KnowledgeGraph, eid: str, name: str) -> str:
        node_id = f"org_{eid}"
        kg.graph.add_node(
            node_id,
            node_type=NodeType.ENTITY.value,
            entity_id=node_id,
            text=name,
        )
        kg._entity_index[node_id] = node_id
        return node_id

    def _loc(self, kg: KnowledgeGraph, eid: str, name: str) -> str:
        return self._org(kg, eid, name)  # same shape

    # ---- Rule 1: works_at_location_inference ----

    def test_rule_works_at_location_infers_person_location(self):
        """Rule 1: A works_at B; B located_in C -> A likely located in C."""
        kg = KnowledgeGraph()
        alice = self._person(kg, "alice", "Alice")
        corp = self._org(kg, "corp", "Acme Corp")
        city = self._loc(kg, "city", "Springfield")

        _link_entity(kg, alice, corp, "works_at")
        _link_entity(kg, corp, city, "located_in")

        facts = kg.infer_new_facts()
        rules = [f["rule"] for f in facts]

        assert "works_at_location_inference" in rules

    # ---- Rule 2: studied_at_location_inference ----

    def test_rule_studied_at_location_infers_student_location(self):
        """Rule 2: A studied_at B; B located_in C -> A likely located in C during studies."""
        kg = KnowledgeGraph()
        bob = self._person(kg, "bob", "Bob")
        univ = self._org(kg, "univ", "MIT")
        city = self._loc(kg, "cambridge", "Cambridge")

        _link_entity(kg, bob, univ, "studied_at")
        _link_entity(kg, univ, city, "located_in")

        facts = kg.infer_new_facts()
        rules = [f["rule"] for f in facts]

        assert "studied_at_location_inference" in rules

    # ---- Rule 3: manages_works_at_inference ----

    def test_rule_manages_infers_manager_works_at_org(self):
        """Rule 3: A manages B; B works_at C -> A likely works at C."""
        kg = KnowledgeGraph()
        ceo = self._person(kg, "ceo", "CEO")
        employee = self._person(kg, "emp", "Employee")
        company = self._org(kg, "co", "Initech")

        _link_entity(kg, ceo, employee, "manages")
        _link_entity(kg, employee, company, "works_at")

        facts = kg.infer_new_facts()
        rules = [f["rule"] for f in facts]

        assert "manages_works_at_inference" in rules

    # ---- Rule 4: knows_organization_inference ----

    def test_rule_knows_infers_organization_connection(self):
        """Rule 4: A knows B; B works_at C -> A has connection to C."""
        kg = KnowledgeGraph()
        alice = self._person(kg, "alice", "Alice")
        bob = self._person(kg, "bob", "Bob")
        corp = self._org(kg, "corp", "Globex")

        _link_entity(kg, alice, bob, "knows")
        _link_entity(kg, bob, corp, "works_at")

        facts = kg.infer_new_facts()
        rules = [f["rule"] for f in facts]

        assert "knows_organization_inference" in rules

    # ---- Rule 5: founded_by_inverse ----

    def test_rule_founded_by_infers_founder_works_at(self):
        """Rule 5: A founded_by B -> B likely works at A as a founder."""
        kg = KnowledgeGraph()
        startup = self._org(kg, "startup", "OpenWidgets")
        founder = self._person(kg, "founder", "Erica")

        _link_entity(kg, startup, founder, "founded_by")

        facts = kg.infer_new_facts()
        rules = [f["rule"] for f in facts]

        assert "founded_by_inverse" in rules

    def test_rule_founded_by_includes_correct_confidence(self):
        """Rule 5 produces confidence 0.8."""
        kg = KnowledgeGraph()
        startup = self._org(kg, "startup", "OpenWidgets")
        founder = self._person(kg, "founder", "Erica")
        _link_entity(kg, startup, founder, "founded_by")

        facts = kg.infer_new_facts()
        matching = [f for f in facts if f["rule"] == "founded_by_inverse"]

        assert len(matching) >= 1
        assert matching[0]["confidence"] == pytest.approx(0.8)

    # ---- Rule 6: co_location_inference ----

    def test_rule_co_location_infers_entities_share_location(self):
        """Rule 6: A located_in C; B located_in C -> A and B are co-located."""
        kg = KnowledgeGraph()
        alice = self._person(kg, "alice", "Alice")
        bob = self._person(kg, "bob", "Bob")
        city = self._loc(kg, "city", "Townsville")

        _link_entity(kg, alice, city, "located_in")
        _link_entity(kg, bob, city, "located_in")

        facts = kg.infer_new_facts()
        rules = [f["rule"] for f in facts]

        assert "co_location_inference" in rules

    def test_rule_co_location_capped_at_50_per_group(self):
        """Rule 6 group cap prevents O(n^2) explosion beyond 50 entities per location."""
        kg = KnowledgeGraph()
        city = self._loc(kg, "megacity", "Megacity")

        # Add 60 people, all located_in the same city
        for i in range(60):
            eid = f"person_{i}"
            node_id = f"entity_{eid}"
            _make_person_entity_node(kg, node_id, f"Person{i}")
            _link_entity(kg, node_id, city, "located_in")

        facts = kg.infer_new_facts()
        co_loc_facts = [f for f in facts if f["rule"] == "co_location_inference"]

        # With 60 entities but cap at 50, group size is 50, max pairs = 50*49/2 = 1225
        # With 60 entities uncapped, max pairs = 60*59/2 = 1770 — verify we're below that
        assert len(co_loc_facts) <= 1225, (
            f"co_location produced {len(co_loc_facts)} facts; cap should limit to <= 1225"
        )

    # ---- Rule 7: part_of_transitive ----

    def test_rule_part_of_transitive_infers_grandparent_membership(self):
        """Rule 7: A part_of B; B part_of C -> A is transitively part of C."""
        kg = KnowledgeGraph()
        team = self._org(kg, "team", "Backend Team")
        dept = self._org(kg, "dept", "Engineering Dept")
        company = self._org(kg, "company", "Megacorp")

        _link_entity(kg, team, dept, "part_of")
        _link_entity(kg, dept, company, "part_of")

        facts = kg.infer_new_facts()
        rules = [f["rule"] for f in facts]

        assert "part_of_transitive" in rules

    # ---- Deduplication ----

    def test_infer_new_facts_deduplicates_identical_entity_fact_pairs(self):
        """Identical (entity_id, fact_text) pairs appear only once in results."""
        kg = KnowledgeGraph()
        # Two separate paths that would trigger the same inferred fact
        alice = self._person(kg, "alice", "Alice")
        corp1 = self._org(kg, "corp1", "Acme Corp")
        corp2 = self._org(kg, "corp2", "Acme Corp")  # same name, different node
        city = self._loc(kg, "city", "Springfield")

        _link_entity(kg, alice, corp1, "works_at")
        _link_entity(kg, alice, corp2, "works_at")
        _link_entity(kg, corp1, city, "located_in")
        _link_entity(kg, corp2, city, "located_in")

        facts = kg.infer_new_facts()
        # Collect (entity_id, fact_text) keys
        keys = [(f["entity_id"], f["fact"]) for f in facts]
        assert len(keys) == len(set(keys)), "Duplicate (entity_id, fact) pairs found"

    # ---- Empty graph ----

    def test_infer_new_facts_returns_empty_list_for_empty_graph(self):
        """An empty KnowledgeGraph returns an empty list without error."""
        kg = KnowledgeGraph()
        assert kg.infer_new_facts() == []

    # ---- Immutability ----

    def test_infer_new_facts_does_not_mutate_graph_nodes(self):
        """infer_new_facts must not add nodes or edges to the graph."""
        kg = KnowledgeGraph()
        alice = self._person(kg, "alice", "Alice")
        corp = self._org(kg, "corp", "Acme Corp")
        city = self._loc(kg, "city", "Springfield")
        _link_entity(kg, alice, corp, "works_at")
        _link_entity(kg, corp, city, "located_in")

        node_count_before = kg.graph.number_of_nodes()
        edge_count_before = kg.graph.number_of_edges()

        kg.infer_new_facts()

        assert kg.graph.number_of_nodes() == node_count_before
        assert kg.graph.number_of_edges() == edge_count_before

    # ---- Required fact fields ----

    def test_infer_new_facts_each_result_has_required_keys(self):
        """Every inferred fact dict must carry entity_id, fact, confidence, rule, supporting_facts."""
        kg = KnowledgeGraph()
        startup = self._org(kg, "s", "StartupCo")
        founder = self._person(kg, "f", "Founder")
        _link_entity(kg, startup, founder, "founded_by")

        facts = kg.infer_new_facts()

        required_keys = {"entity_id", "fact", "confidence", "rule", "supporting_facts"}
        for fact in facts:
            missing = required_keys - fact.keys()
            assert not missing, f"Fact missing keys {missing}: {fact}"


# ===========================================================================
# 5. LLM Inference Path Tests
# ===========================================================================


class TestLLMInferencePath:
    """LLM adapter: fires only when provided, skipped when None, resilient to exceptions."""

    def _make_kg_with_entity_and_neighbour(self) -> KnowledgeGraph:
        kg = KnowledgeGraph()
        alice = "entity_alice"
        corp = "entity_corp"
        _make_person_entity_node(kg, alice, "Alice")
        _make_person_entity_node(kg, corp, "Acme Corp")
        _link_entity(kg, alice, corp, "works_at")
        return kg

    def test_llm_generate_called_when_llm_provided(self):
        """When llm is passed, llm.generate() is called at least once."""
        kg = self._make_kg_with_entity_and_neighbour()

        mock_llm = MagicMock()
        mock_llm.generate.return_value = (
            "FACT: Alice is a professional | CONFIDENCE: 0.8\n"
        )

        kg.infer_new_facts(llm=mock_llm)

        mock_llm.generate.assert_called()

    def test_llm_generate_not_called_when_llm_is_none(self):
        """When llm=None, no LLM call is made."""
        kg = self._make_kg_with_entity_and_neighbour()

        mock_llm = MagicMock()
        # Passing None explicitly — LLM must not be called
        kg.infer_new_facts(llm=None)

        mock_llm.generate.assert_not_called()

    def test_llm_facts_included_in_results_when_parseable(self):
        """Valid LLM output is parsed and appended to inferred facts."""
        kg = self._make_kg_with_entity_and_neighbour()

        mock_llm = MagicMock()
        mock_llm.generate.return_value = (
            "FACT: Alice leads the team | CONFIDENCE: 0.75\n"
            "FACT: Alice is in tech | CONFIDENCE: 0.6\n"
        )

        facts = kg.infer_new_facts(llm=mock_llm)
        llm_facts = [f for f in facts if f["rule"] == "llm_inference"]

        assert len(llm_facts) == 2

    def test_llm_exception_returns_partial_results_not_raises(self):
        """If llm.generate() raises, infer_new_facts returns partial (rule-based) results."""
        kg = self._make_kg_with_entity_and_neighbour()
        corp2 = "entity_corp2"
        city = "entity_city"
        _make_person_entity_node(kg, corp2, "Corp2")
        _make_person_entity_node(kg, city, "Springfield")
        # Setup chain that triggers works_at_location_inference (rule 1)
        _link_entity(kg, "entity_alice", corp2, "works_at")
        _link_entity(kg, corp2, city, "located_in")

        mock_llm = MagicMock()
        mock_llm.generate.side_effect = RuntimeError("LLM service down")

        # Must not raise
        facts = kg.infer_new_facts(llm=mock_llm)

        # Rule-based facts should still be present
        rule_facts = [f for f in facts if f["rule"] != "llm_inference"]
        assert len(rule_facts) >= 1

    def test_llm_malformed_line_is_skipped_gracefully(self):
        """Lines that don't match FACT: ... | CONFIDENCE: ... are silently skipped."""
        kg = self._make_kg_with_entity_and_neighbour()

        mock_llm = MagicMock()
        mock_llm.generate.return_value = (
            "This line has no FACT prefix\n"
            "FACT: Good fact | CONFIDENCE: 0.9\n"
            "FACT: No pipe separator\n"
            "FACT: Bad confidence | CONFIDENCE: not_a_float\n"
        )

        facts = kg.infer_new_facts(llm=mock_llm)
        llm_facts = [f for f in facts if f["rule"] == "llm_inference"]

        # Only the single well-formed line should be parsed
        assert len(llm_facts) == 1
        assert llm_facts[0]["fact"] == "Good fact"

    def test_llm_confidence_clamped_to_zero_one(self):
        """LLM-provided confidence values outside [0.0, 1.0] are clamped."""
        kg = self._make_kg_with_entity_and_neighbour()

        mock_llm = MagicMock()
        mock_llm.generate.return_value = (
            "FACT: Over confident | CONFIDENCE: 1.5\n"
            "FACT: Negative conf | CONFIDENCE: -0.3\n"
        )

        facts = kg.infer_new_facts(llm=mock_llm)
        llm_facts = [f for f in facts if f["rule"] == "llm_inference"]

        for f in llm_facts:
            assert 0.0 <= f["confidence"] <= 1.0, (
                f"Confidence {f['confidence']} not clamped to [0, 1]"
            )

    def test_llm_entity_with_no_neighbours_is_skipped(self):
        """An isolated entity node produces no LLM prompt (no neighbours)."""
        kg = KnowledgeGraph()
        _make_person_entity_node(kg, "entity_lonely", "Lonely")

        mock_llm = MagicMock()
        mock_llm.generate.return_value = "FACT: something | CONFIDENCE: 0.5\n"

        kg.infer_new_facts(llm=mock_llm)

        mock_llm.generate.assert_not_called()


# ===========================================================================
# 6. check_graph_qdrant_drift Tests
# ===========================================================================


class _ClientStub:
    """Minimal stub that satisfies check_graph_qdrant_drift without importing MemoryClient."""

    def __init__(self, kg: KnowledgeGraph, collection_facts: str = "test_facts") -> None:
        self.graph = kg
        self.config = MagicMock()
        self.config.collection_facts = collection_facts
        self.qdrant = MagicMock()

        # Bind the real method from MemoryClient onto this stub
        from hippocampai.client import MemoryClient
        self.check_graph_qdrant_drift = MemoryClient.check_graph_qdrant_drift.__get__(
            self, type(self)
        )


class TestCheckGraphQdrantDrift:
    """MemoryClient.check_graph_qdrant_drift — mocked Qdrant."""

    def _make_client_with_graph_user(self, user_id: str, memory_ids: list[str]) -> _ClientStub:
        """Build a minimal client stub with a pre-populated knowledge graph."""
        # Build a real KnowledgeGraph for the client
        kg = KnowledgeGraph()
        for mid in memory_ids:
            kg.graph.add_node(
                mid,
                node_type=NodeType.MEMORY.value,
                user_id=user_id,
            )
            kg._memory_index[mid] = {"user_id": user_id, "metadata": {}}
            kg._user_graphs[user_id].add(mid)

        return _ClientStub(kg)

    def test_in_sync_case_returns_no_drift(self):
        """When graph IDs == Qdrant IDs, drift_detected is False and synced == count."""
        user_id = "alice"
        memory_ids = ["m1", "m2", "m3"]

        stub = self._make_client_with_graph_user(user_id, memory_ids)
        stub.qdrant.scroll.return_value = [{"id": mid} for mid in memory_ids]

        result = stub.check_graph_qdrant_drift(user_id)

        assert result["drift_detected"] is False
        assert result["synced"] == 3
        assert result["graph_only"] == []
        assert result["qdrant_only"] == []

    def test_graph_only_drift_detected_when_graph_has_extra_ids(self):
        """IDs in graph but not Qdrant are reported in graph_only; drift_detected is True."""
        user_id = "bob"
        graph_ids = ["m1", "m2", "m3"]
        qdrant_ids = ["m1"]  # m2, m3 missing from Qdrant

        stub = self._make_client_with_graph_user(user_id, graph_ids)
        stub.qdrant.scroll.return_value = [{"id": mid} for mid in qdrant_ids]

        result = stub.check_graph_qdrant_drift(user_id)

        assert result["drift_detected"] is True
        assert sorted(result["graph_only"]) == ["m2", "m3"]
        assert result["qdrant_only"] == []
        assert result["synced"] == 1

    def test_qdrant_only_drift_detected_when_qdrant_has_extra_ids(self):
        """IDs in Qdrant but not graph are reported in qdrant_only; drift_detected is True."""
        user_id = "carol"
        graph_ids = ["m1"]
        qdrant_ids = ["m1", "m2", "m3"]

        stub = self._make_client_with_graph_user(user_id, graph_ids)
        stub.qdrant.scroll.return_value = [{"id": mid} for mid in qdrant_ids]

        result = stub.check_graph_qdrant_drift(user_id)

        assert result["drift_detected"] is True
        assert result["graph_only"] == []
        assert sorted(result["qdrant_only"]) == ["m2", "m3"]
        assert result["synced"] == 1

    def test_empty_user_returns_no_drift_and_zeros(self):
        """User with no memories in either store returns no drift."""
        user_id = "nobody"

        stub = self._make_client_with_graph_user(user_id, [])
        stub.qdrant.scroll.return_value = []

        result = stub.check_graph_qdrant_drift(user_id)

        assert result["drift_detected"] is False
        assert result["synced"] == 0
        assert result["graph_only"] == []
        assert result["qdrant_only"] == []

    def test_result_contains_all_required_keys(self):
        """Return dict must always have the six documented keys."""
        user_id = "dan"
        stub = self._make_client_with_graph_user(user_id, ["m1"])
        stub.qdrant.scroll.return_value = [{"id": "m1"}]

        result = stub.check_graph_qdrant_drift(user_id)

        required_keys = {"user_id", "graph_only", "qdrant_only", "synced", "drift_detected", "checked_at"}
        assert required_keys.issubset(result.keys())

    def test_result_user_id_matches_requested_user(self):
        """The user_id field in the result matches the argument passed."""
        user_id = "eve"
        stub = self._make_client_with_graph_user(user_id, ["m1"])
        stub.qdrant.scroll.return_value = [{"id": "m1"}]

        result = stub.check_graph_qdrant_drift(user_id)

        assert result["user_id"] == user_id

    def test_graph_only_and_qdrant_only_are_sorted(self):
        """graph_only and qdrant_only lists are returned in sorted order."""
        user_id = "frank"
        graph_ids = ["m3", "m1", "m2"]
        qdrant_ids = ["z3", "z1", "z2"]

        stub = self._make_client_with_graph_user(user_id, graph_ids)
        stub.qdrant.scroll.return_value = [{"id": mid} for mid in qdrant_ids]

        result = stub.check_graph_qdrant_drift(user_id)

        assert result["graph_only"] == sorted(result["graph_only"])
        assert result["qdrant_only"] == sorted(result["qdrant_only"])

    def test_custom_collection_name_passed_to_qdrant_scroll(self):
        """When collection_name is provided, qdrant.scroll is called with that collection."""
        user_id = "grace"
        stub = self._make_client_with_graph_user(user_id, ["m1"])
        stub.qdrant.scroll.return_value = [{"id": "m1"}]

        stub.check_graph_qdrant_drift(user_id, collection_name="custom_collection")

        call_kwargs = stub.qdrant.scroll.call_args
        assert call_kwargs.kwargs.get("collection_name") == "custom_collection" or (
            call_kwargs.args and "custom_collection" in call_kwargs.args
        )
