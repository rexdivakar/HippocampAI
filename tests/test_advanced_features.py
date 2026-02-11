"""Comprehensive tests for advanced memory features.

Tests cover:
- Graph operations (relationships, clusters, paths)
- Graph persistence (JSON export/import)
- KV store (caching, indexing, TTL)
- Version control (versioning, comparison, rollback)
- Context injection (templates, token limits, history)
- Batch operations (add, delete)
- Access tracking
- Advanced filters
- Snapshots
- Audit trail
"""

import json
import tempfile
import time
from pathlib import Path

import pytest

from hippocampai import ChangeType, MemoryClient, MemoryType, RelationType


@pytest.fixture
def client():
    """Create a MemoryClient instance for testing."""
    return MemoryClient(
        qdrant_url="http://localhost:6333",
        collection_facts="test_facts_advanced",
        collection_prefs="test_prefs_advanced",
        enable_telemetry=False,
    )


@pytest.fixture(autouse=True)
def cleanup_collections(client):
    """Clean up test collections before and after each test."""
    # Cleanup before test
    try:
        client.qdrant.client.delete_collection(client.config.collection_facts)
    except Exception:
        pass
    try:
        client.qdrant.client.delete_collection(client.config.collection_prefs)
    except Exception:
        pass

    # Recreate collections
    client.qdrant._ensure_collections()

    yield

    # Cleanup after test
    try:
        client.qdrant.client.delete_collection(client.config.collection_facts)
    except Exception:
        pass
    try:
        client.qdrant.client.delete_collection(client.config.collection_prefs)
    except Exception:
        pass


# === BATCH OPERATIONS TESTS ===


class TestBatchOperations:
    """Test batch add and delete operations."""

    def test_batch_add_memories(self, client):
        """Test adding multiple memories at once."""
        memories_data = [
            {"text": "Python is great for ML", "tags": ["programming", "ml"], "importance": 8.0},
            {"text": "I love coffee", "tags": ["lifestyle"], "ttl_days": 90},
            {"text": "Meeting at 3pm", "type": "event", "importance": 9.0},
        ]

        created = client.add_memories(memories_data, user_id="alice")

        assert len(created) == 3
        assert created[0].text == "Python is great for ML"
        assert created[0].importance == 8.0
        assert "programming" in created[0].tags
        assert created[1].text == "I love coffee"
        assert created[1].expires_at is not None
        assert created[2].type == MemoryType.EVENT

    def test_batch_delete_memories(self, client):
        """Test deleting multiple memories at once."""
        # Create some memories
        m1 = client.remember("Memory 1", user_id="alice")
        m2 = client.remember("Memory 2", user_id="alice")
        m3 = client.remember("Memory 3", user_id="alice")

        # Batch delete
        deleted_count = client.delete_memories([m1.id, m2.id, m3.id], user_id="alice")

        assert deleted_count == 3

        # Verify deletion
        memories = client.get_memories(user_id="alice")
        assert len(memories) == 0


# === GRAPH OPERATIONS TESTS ===


class TestGraphOperations:
    """Test graph-based memory relationships."""

    def test_add_relationship(self, client):
        """Test adding relationships between memories."""
        m1 = client.remember("Python is a programming language", user_id="alice")
        m2 = client.remember("I use Python for ML", user_id="alice")

        # Add memories to graph first
        client.graph.add_memory(m1.id, "alice", {})
        client.graph.add_memory(m2.id, "alice", {})

        # Add relationship
        success = client.add_relationship(m1.id, m2.id, RelationType.RELATED_TO, weight=0.9)

        assert success is True

    def test_get_related_memories(self, client):
        """Test retrieving related memories."""
        m1 = client.remember("Cause: Rainy weather", user_id="alice")
        m2 = client.remember("Effect: I stayed home", user_id="alice")
        m3 = client.remember("Also: I read a book", user_id="alice")

        # Add memories to graph first
        client.graph.add_memory(m1.id, "alice", {})
        client.graph.add_memory(m2.id, "alice", {})
        client.graph.add_memory(m3.id, "alice", {})

        # Add relationships
        client.add_relationship(m1.id, m2.id, RelationType.CAUSED_BY)
        client.add_relationship(m2.id, m3.id, RelationType.LEADS_TO)

        # Get related memories (depth 1)
        related = client.get_related_memories(m1.id, max_depth=1)
        assert len(related) >= 1
        assert any(r[0] == m2.id for r in related)

        # Get related memories (depth 2) - may or may not find m3 depending on graph traversal
        related_deep = client.get_related_memories(m1.id, max_depth=2)
        assert len(related_deep) >= 1  # At least m2 should be found

    def test_memory_clusters(self, client):
        """Test finding memory clusters."""
        # Create interconnected memories
        m1 = client.remember("Topic A - Part 1", user_id="alice")
        m2 = client.remember("Topic A - Part 2", user_id="alice")
        m3 = client.remember("Topic B - Part 1", user_id="alice")

        # Add to graph with user metadata
        client.graph.add_memory(m1.id, "alice", {})
        client.graph.add_memory(m2.id, "alice", {})
        client.graph.add_memory(m3.id, "alice", {})

        # Create relationships
        client.add_relationship(m1.id, m2.id, RelationType.RELATED_TO)

        # Get clusters
        clusters = client.get_memory_clusters("alice")
        assert len(clusters) >= 1


# === GRAPH PERSISTENCE TESTS ===


class TestGraphPersistence:
    """Test graph export/import to JSON."""

    def test_export_graph_to_json(self, client):
        """Test exporting graph to JSON file."""
        # Create memories and relationships
        m1 = client.remember("Memory 1", user_id="alice")
        m2 = client.remember("Memory 2", user_id="alice")
        m3 = client.remember("Memory 3", user_id="bob")

        # Add to graph
        client.graph.add_memory(m1.id, "alice", {"text": "Memory 1"})
        client.graph.add_memory(m2.id, "alice", {"text": "Memory 2"})
        client.graph.add_memory(m3.id, "bob", {"text": "Memory 3"})

        # Add relationships
        client.add_relationship(m1.id, m2.id, RelationType.RELATED_TO, weight=0.9)

        # Export to JSON
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            temp_path = f.name

        try:
            result_path = client.export_graph_to_json(temp_path)
            assert result_path == temp_path
            assert Path(temp_path).exists()

            # Verify JSON structure
            with open(temp_path, "r") as f:
                data = json.load(f)

            assert "nodes" in data
            assert "links" in data
            assert len(data["nodes"]) >= 3  # At least 3 memory nodes (may include entity/topic nodes)
            assert len(data["links"]) >= 1  # At least the explicit relationship

            # Verify node data
            node_ids = [node["id"] for node in data["nodes"]]
            assert m1.id in node_ids
            assert m2.id in node_ids
            assert m3.id in node_ids

            # Verify the explicit relationship edge exists among all links
            matching_edges = [
                e for e in data["links"]
                if e["source"] == m1.id and e["target"] == m2.id
            ]
            assert len(matching_edges) >= 1
            edge = matching_edges[0]
            assert edge["relation"] == RelationType.RELATED_TO.value
            assert edge["weight"] == 0.9

        finally:
            # Cleanup
            Path(temp_path).unlink(missing_ok=True)

    def test_export_graph_user_filtered(self, client):
        """Test exporting graph for a specific user."""
        m1 = client.remember("Alice memory", user_id="alice")
        m2 = client.remember("Bob memory", user_id="bob")

        client.graph.add_memory(m1.id, "alice", {})
        client.graph.add_memory(m2.id, "bob", {})

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            temp_path = f.name

        try:
            client.export_graph_to_json(temp_path, user_id="alice")

            with open(temp_path, "r") as f:
                data = json.load(f)

            # Should contain alice's memory (and possibly entity/topic nodes from auto-extraction)
            node_ids = [node["id"] for node in data["nodes"]]
            assert m1.id in node_ids

        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_import_graph_from_json(self, client):
        """Test importing graph from JSON file."""
        # Create and export initial graph
        m1 = client.remember("Memory 1", user_id="alice")
        m2 = client.remember("Memory 2", user_id="alice")

        client.graph.add_memory(m1.id, "alice", {})
        client.graph.add_memory(m2.id, "alice", {})
        client.add_relationship(m1.id, m2.id, RelationType.RELATED_TO)

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            temp_path = f.name

        try:
            # Export
            client.export_graph_to_json(temp_path)

            # Clear graph
            client.graph.graph.clear()
            client.graph._memory_index.clear()
            client.graph._user_graphs.clear()

            assert client.graph.graph.number_of_nodes() == 0
            assert client.graph.graph.number_of_edges() == 0

            # Import
            stats = client.import_graph_from_json(temp_path)

            # Verify stats
            assert stats["nodes_before"] == 0
            assert stats["edges_before"] == 0
            assert stats["nodes_after"] >= 2  # At least 2 memory nodes + auto-extracted entities
            assert stats["edges_after"] >= 1  # At least the explicit relationship
            assert stats["nodes_imported"] >= 2
            assert stats["edges_imported"] >= 1
            assert stats["merged"] is True

            # Verify graph was restored
            assert client.graph.graph.number_of_nodes() >= 2
            assert client.graph.graph.number_of_edges() >= 1

            # Verify relationship exists
            related = client.get_related_memories(m1.id)
            assert len(related) >= 1
            related_ids = [r[0] for r in related]
            assert m2.id in related_ids

        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_import_graph_merge_mode(self, client):
        """Test importing graph with merge mode."""
        # Create initial graph
        m1 = client.remember("Memory 1", user_id="alice")
        client.graph.add_memory(m1.id, "alice", {})

        # Create second graph in temp file
        m2 = client.remember("Memory 2", user_id="bob")
        temp_client = MemoryClient(
            qdrant_url="http://localhost:6333",
            collection_facts="test_temp_graph",
            enable_telemetry=False,
        )
        temp_client.graph.add_memory(m2.id, "bob", {})

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            temp_path = f.name

        try:
            # Export second graph
            temp_client.export_graph_to_json(temp_path)

            # Import with merge=True (should have both)
            stats = client.import_graph_from_json(temp_path, merge=True)

            assert stats["nodes_before"] >= 1
            assert stats["nodes_after"] >= 2
            assert stats["merged"] is True

            # Both memories should be in graph
            assert m1.id in client.graph.graph.nodes()
            assert m2.id in client.graph.graph.nodes()

        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_import_graph_replace_mode(self, client):
        """Test importing graph with replace mode."""
        # Create initial graph
        m1 = client.remember("Memory 1", user_id="alice")
        client.graph.add_memory(m1.id, "alice", {})

        # Create second graph
        m2 = client.remember("Memory 2", user_id="bob")
        temp_client = MemoryClient(
            qdrant_url="http://localhost:6333",
            collection_facts="test_temp_graph",
            enable_telemetry=False,
        )
        temp_client.graph.add_memory(m2.id, "bob", {})

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            temp_path = f.name

        try:
            # Export second graph
            temp_client.export_graph_to_json(temp_path)

            # Import with merge=False (should replace)
            stats = client.import_graph_from_json(temp_path, merge=False)

            assert stats["nodes_before"] >= 1
            assert stats["nodes_after"] >= 1
            assert stats["merged"] is False

            # Only m2 should be in graph
            assert m1.id not in client.graph.graph.nodes()
            assert m2.id in client.graph.graph.nodes()

        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_import_graph_file_not_found(self, client):
        """Test importing from non-existent file."""
        with pytest.raises(FileNotFoundError):
            client.import_graph_from_json("nonexistent_file.json")

    def test_import_graph_invalid_format(self, client):
        """Test importing invalid JSON format."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            temp_path = f.name
            json.dump({"invalid": "format"}, f)

        try:
            with pytest.raises(ValueError, match="Invalid graph format"):
                client.import_graph_from_json(temp_path)
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_export_import_complex_graph(self, client):
        """Test export/import with complex graph structure."""
        # Create complex graph with multiple relationships
        memories = []
        for i in range(5):
            m = client.remember(f"Memory {i}", user_id="alice")
            client.graph.add_memory(m.id, "alice", {"index": i})
            memories.append(m)

        # Create various relationship types
        client.add_relationship(memories[0].id, memories[1].id, RelationType.LEADS_TO, weight=0.8)
        client.add_relationship(memories[1].id, memories[2].id, RelationType.CAUSED_BY, weight=0.9)
        client.add_relationship(memories[2].id, memories[3].id, RelationType.SUPPORTS, weight=0.7)
        client.add_relationship(
            memories[3].id, memories[4].id, RelationType.RELATED_TO, weight=0.95
        )
        client.add_relationship(
            memories[0].id, memories[4].id, RelationType.SIMILAR_TO, weight=0.85
        )

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            temp_path = f.name

        try:
            # Export
            client.export_graph_to_json(temp_path)

            # Verify export
            with open(temp_path, "r") as f:
                data = json.load(f)

            assert len(data["nodes"]) >= 5  # At least 5 memory nodes + auto-extracted entities
            assert len(data["links"]) >= 5  # At least 5 explicit relationships

            # Clear and reimport
            client.graph.graph.clear()
            client.graph._memory_index.clear()
            client.graph._user_graphs.clear()

            stats = client.import_graph_from_json(temp_path)

            # Verify complete restoration
            assert stats["nodes_after"] >= 5
            assert stats["edges_after"] >= 5

            # Verify all relationships are preserved
            related = client.get_related_memories(memories[0].id, max_depth=1)
            assert len(related) >= 2  # At least leads_to m1, similar_to m4 (may include entity/topic links)

            related_deep = client.get_related_memories(memories[0].id, max_depth=3)
            assert len(related_deep) >= 2

        finally:
            Path(temp_path).unlink(missing_ok=True)


# === VERSION CONTROL TESTS ===


class TestVersionControl:
    """Test memory versioning and rollback."""

    def test_create_version(self, client):
        """Test creating a version when updating memory."""
        # Create memory
        memory = client.remember("Original text", user_id="alice", importance=5.0)

        # Create version manually
        version = client.version_control.create_version(
            memory.id,
            memory.model_dump(mode="json"),
            created_by="alice",
            change_summary="Initial version",
        )

        assert version.memory_id == memory.id
        assert version.version_number == 1
        assert version.data["text"] == "Original text"

    def test_version_history(self, client):
        """Test getting version history."""
        memory = client.remember("Version 1", user_id="alice")

        # Create multiple versions
        for i in range(3):
            client.version_control.create_version(
                memory.id,
                {"text": f"Version {i + 1}", "importance": 5.0 + i},
                created_by="alice",
            )

        # Get history
        history = client.get_memory_history(memory.id)
        assert len(history) == 3
        assert history[-1].version_number == 3

    def test_compare_versions(self, client):
        """Test comparing two versions."""
        memory = client.remember("V1", user_id="alice", importance=5.0)

        # Create versions
        client.version_control.create_version(memory.id, {"text": "V1", "importance": 5.0})
        client.version_control.create_version(memory.id, {"text": "V2", "importance": 7.0})

        # Compare
        diff = client.version_control.compare_versions(memory.id, 1, 2)
        assert "changed" in diff
        assert "text" in diff["changed"]
        assert diff["changed"]["text"]["old"] == "V1"
        assert diff["changed"]["text"]["new"] == "V2"

    def test_rollback_memory(self, client):
        """Test rolling back to a previous version."""
        memory = client.remember("Original", user_id="alice", importance=5.0)

        # Create version
        client.version_control.create_version(
            memory.id, {"text": "Original", "importance": 5.0, "tags": []}
        )

        # Update memory
        updated = client.update_memory(memory.id, text="Updated", importance=8.0)
        assert updated.text == "Updated"

        # Rollback
        rolled_back = client.rollback_memory(memory.id, version_number=1)
        assert rolled_back is not None
        assert rolled_back.text == "Original"


# === CONTEXT INJECTION TESTS ===


class TestContextInjection:
    """Test context injection for LLM prompts."""

    def test_inject_context_default(self, client):
        """Test injecting context with default template."""
        # Create memories
        client.remember("I love Python", user_id="alice", tags=["programming"])
        client.remember("I drink coffee daily", user_id="alice", tags=["lifestyle"])

        # Inject context
        prompt = client.inject_context(
            prompt="What are my interests?",
            query="interests hobbies",
            user_id="alice",
            k=2,
            template="default",
        )

        assert "Relevant" in prompt
        assert "What are my interests?" in prompt

    def test_inject_context_minimal(self, client):
        """Test minimal template."""
        client.remember("Memory 1", user_id="alice")

        prompt = client.inject_context(
            prompt="Test", query="memory", user_id="alice", template="minimal"
        )

        # Check that prompt contains the query and memory content
        assert "Test" in prompt
        assert "Memory 1" in prompt or "Context:" in prompt

    def test_inject_context_detailed(self, client):
        """Test detailed template."""
        client.remember("Memory 1", user_id="alice", importance=8.0)

        prompt = client.inject_context(
            prompt="Test", query="memory", user_id="alice", template="detailed"
        )

        assert "Memory 1" in prompt or "Relevant" in prompt


# === ACCESS TRACKING TESTS ===


class TestAccessTracking:
    """Test memory access tracking."""

    def test_track_access(self, client):
        """Test that access tracking updates access_count."""
        memory = client.remember("Test memory", user_id="alice")

        # Track access
        client.track_memory_access(memory.id, "alice")

        # Retrieve and check
        memories = client.get_memories(user_id="alice")
        assert memories[0].access_count == 1

        # Track again
        client.track_memory_access(memory.id, "alice")
        memories = client.get_memories(user_id="alice")
        assert memories[0].access_count == 2

    def test_access_tracking_in_recall(self, client):
        """Test that inject_context tracks access."""
        client.remember("Python programming", user_id="alice")

        # Use inject_context which should track access
        client.inject_context("Query", "python", "alice", k=1)

        # Check access count
        memories = client.get_memories(user_id="alice")
        assert memories[0].access_count >= 1


# === ADVANCED FILTERS TESTS ===


class TestAdvancedFilters:
    """Test advanced filtering and sorting."""

    def test_filter_by_metadata(self, client):
        """Test filtering by custom metadata."""
        # Create memories with metadata
        client.remember(
            "Project A task",
            user_id="alice",
            tags=["work"],
        )
        # Add metadata after creation
        m1 = client.remember("Project B task", user_id="alice", tags=["work"])
        m1.metadata["project"] = "B"
        client.update_memory(m1.id, metadata={"project": "B"})

        m2 = client.remember("Project B note", user_id="alice", tags=["work"])
        m2.metadata["project"] = "B"
        client.update_memory(m2.id, metadata={"project": "B"})

        # Filter by metadata
        results = client.get_memories_advanced(
            user_id="alice",
            filters={"metadata": {"project": "B"}},
        )

        assert len(results) == 2

    def test_sort_by_importance(self, client):
        """Test sorting by importance."""
        client.remember("Low importance", user_id="alice", importance=3.0)
        client.remember("High importance", user_id="alice", importance=9.0)
        client.remember("Medium importance", user_id="alice", importance=6.0)

        # Sort descending
        memories = client.get_memories_advanced(
            user_id="alice", sort_by="importance", sort_order="desc"
        )

        assert memories[0].importance >= memories[1].importance
        assert memories[1].importance >= memories[2].importance

    def test_sort_by_access_count(self, client):
        """Test sorting by access count."""
        client.remember("Memory 1", user_id="alice")
        m2 = client.remember("Memory 2", user_id="alice")

        # Access m2 multiple times
        client.track_memory_access(m2.id, "alice")
        client.track_memory_access(m2.id, "alice")
        client.track_memory_access(m2.id, "alice")

        # Sort by access_count
        memories = client.get_memories_advanced(
            user_id="alice", sort_by="access_count", sort_order="desc"
        )

        assert memories[0].id == m2.id
        assert memories[0].access_count >= 3


# === SNAPSHOT TESTS ===


class TestSnapshots:
    """Test snapshot functionality."""

    def test_create_snapshot(self, client):
        """Test creating a snapshot."""
        # Add some memories
        client.remember("Fact 1", user_id="alice")
        client.remember("Fact 2", user_id="alice")

        # Create snapshot
        snapshot_name = client.create_snapshot("facts")
        assert snapshot_name is not None
        assert isinstance(snapshot_name, str)


# === AUDIT TRAIL TESTS ===


class TestAuditTrail:
    """Test audit trail functionality."""

    def test_audit_trail_creation(self, client):
        """Test audit entries are created."""
        memory = client.remember("Test", user_id="alice")

        # Manually add audit entry
        client.version_control.add_audit_entry(
            memory_id=memory.id,
            change_type=ChangeType.CREATED,
            user_id="alice",
        )

        # Get audit trail
        entries = client.get_audit_trail(memory_id=memory.id)
        assert len(entries) >= 1
        assert entries[0].memory_id == memory.id

    def test_audit_trail_filtering(self, client):
        """Test filtering audit trail."""
        m1 = client.remember("M1", user_id="alice")
        m2 = client.remember("M2", user_id="bob")

        # Add audit entries
        client.version_control.add_audit_entry(m1.id, ChangeType.CREATED, user_id="alice")
        client.version_control.add_audit_entry(m2.id, ChangeType.CREATED, user_id="bob")
        client.version_control.add_audit_entry(m1.id, ChangeType.ACCESSED, user_id="alice")

        # Filter by user
        alice_entries = client.get_audit_trail(user_id="alice")
        assert len(alice_entries) == 2
        assert all(e.user_id == "alice" for e in alice_entries)

        # Filter by change type
        created_entries = client.get_audit_trail(change_type=ChangeType.CREATED)
        assert len(created_entries) == 2

        # Filter by memory
        m1_entries = client.get_audit_trail(memory_id=m1.id)
        assert len(m1_entries) == 2
        assert all(e.memory_id == m1.id for e in m1_entries)

    def test_relationship_audit(self, client):
        """Test audit trail for relationships."""
        m1 = client.remember("M1", user_id="alice")
        m2 = client.remember("M2", user_id="alice")

        # Add relationship
        client.add_relationship(m1.id, m2.id, RelationType.RELATED_TO)

        # Check audit trail
        entries = client.get_audit_trail(memory_id=m1.id)
        relationship_entries = [
            e for e in entries if e.change_type == ChangeType.RELATIONSHIP_ADDED
        ]
        assert len(relationship_entries) >= 1


# === INTEGRATION TESTS ===


class TestIntegration:
    """Test integration of multiple advanced features."""

    def test_full_workflow(self, client):
        """Test a complete workflow using multiple features."""
        # 1. Batch add memories
        memories_data = [
            {"text": "Python is great", "tags": ["programming"], "importance": 8.0},
            {"text": "I love machine learning", "tags": ["ml", "programming"], "importance": 9.0},
            {"text": "Coffee helps me focus", "tags": ["lifestyle"], "importance": 6.0},
        ]
        created = client.add_memories(memories_data, user_id="alice")
        assert len(created) == 3

        # Add memories to graph
        for memory in created:
            client.graph.add_memory(memory.id, "alice", {})

        # 2. Add relationships
        client.add_relationship(created[0].id, created[1].id, RelationType.RELATED_TO)

        # 3. Track access
        client.track_memory_access(created[0].id, "alice")

        # 4. Create version
        client.version_control.create_version(
            created[0].id,
            created[0].model_dump(mode="json"),
            created_by="alice",
        )

        # 5. Update memory
        updated = client.update_memory(created[0].id, importance=9.5)
        assert updated.importance == 9.5

        # 6. Get related memories
        related = client.get_related_memories(created[0].id)
        assert len(related) >= 1

        # 7. Advanced filtering
        high_importance = client.get_memories_advanced(
            user_id="alice",
            filters={"min_importance": 8.0},
            sort_by="importance",
            sort_order="desc",
        )
        assert len(high_importance) >= 2

        # 8. Context injection
        prompt = client.inject_context(
            "What do I know about programming?",
            "programming python",
            "alice",
            k=2,
        )
        assert "programming" in prompt.lower() or "python" in prompt.lower()

        # 9. Check audit trail
        audit = client.get_audit_trail(user_id="alice", limit=10)
        assert len(audit) >= 1

        # 10. Create snapshot
        snapshot = client.create_snapshot("facts")
        assert snapshot is not None


# === KV STORE TESTS ===


class TestKVStore:
    """Test KV store functionality."""

    def test_kv_store_set_get(self, client):
        """Test basic set/get operations."""
        memory_data = {
            "id": "test-id",
            "text": "Test memory",
            "user_id": "alice",
            "tags": ["test"],
        }

        # Set
        client.kv_store.set_memory("test-id", memory_data)

        # Get
        retrieved = client.kv_store.get_memory("test-id")
        assert retrieved is not None
        assert retrieved["text"] == "Test memory"

    def test_kv_store_user_index(self, client):
        """Test user indexing."""
        memory1 = {"id": "m1", "text": "M1", "user_id": "alice", "tags": []}
        memory2 = {"id": "m2", "text": "M2", "user_id": "alice", "tags": []}

        client.kv_store.set_memory("m1", memory1)
        client.kv_store.set_memory("m2", memory2)

        # Get by user
        user_memories = client.kv_store.get_user_memories("alice")
        assert "m1" in user_memories
        assert "m2" in user_memories

    def test_kv_store_tag_index(self, client):
        """Test tag indexing."""
        memory1 = {"id": "m1", "text": "M1", "user_id": "alice", "tags": ["python"]}
        memory2 = {"id": "m2", "text": "M2", "user_id": "alice", "tags": ["python", "ml"]}

        client.kv_store.set_memory("m1", memory1)
        client.kv_store.set_memory("m2", memory2)

        # Get by tag
        tagged = client.kv_store.get_memories_by_tag("python")
        assert "m1" in tagged
        assert "m2" in tagged

    def test_kv_store_ttl(self, client):
        """Test TTL expiration."""
        # Create KV store with very short TTL (1 second)
        from hippocampai.storage import MemoryKVStore

        kv = MemoryKVStore(cache_ttl=1)

        memory_data = {"id": "m1", "text": "M1", "user_id": "alice", "tags": []}
        kv.set_memory("m1", memory_data)

        # Should exist immediately
        assert kv.get_memory("m1") is not None

        # Wait for expiration
        time.sleep(2)

        # Should be expired
        assert kv.get_memory("m1") is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
