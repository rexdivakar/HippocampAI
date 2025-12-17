"""Tests for new HippocampAI features: plugins, namespaces, portability, offline, tiered storage."""

import os
import tempfile

import pytest


class TestPluginSystem:
    """Tests for the plugin system."""

    def test_plugin_registry_creation(self):
        """Test creating a plugin registry."""
        from hippocampai.plugins import PluginRegistry

        registry = PluginRegistry()
        assert registry is not None
        plugins = registry.list_plugins()
        assert "processors" in plugins
        assert "scorers" in plugins
        assert "retrievers" in plugins
        assert "filters" in plugins

    def test_memory_processor_registration(self):
        """Test registering a custom processor."""
        from hippocampai.models.memory import Memory, MemoryType
        from hippocampai.plugins import MemoryProcessor, PluginRegistry

        class TestProcessor(MemoryProcessor):
            name = "test_processor"

            def process(self, memory, context=None):
                memory.metadata["processed"] = True
                return memory

        registry = PluginRegistry()
        processor = TestProcessor()
        registry.register_processor(processor)

        plugins = registry.list_plugins()
        assert "test_processor" in plugins["processors"]

        # Test processing
        memory = Memory(
            text="Test memory",
            user_id="test_user",
            type=MemoryType.FACT,
        )
        processed = registry.run_processors(memory)
        assert processed.metadata.get("processed") is True

    def test_memory_scorer_registration(self):
        """Test registering a custom scorer."""
        from hippocampai.models.memory import Memory, MemoryType
        from hippocampai.plugins import MemoryScorer, PluginRegistry

        class TestScorer(MemoryScorer):
            name = "test_scorer"
            weight = 0.5

            def score(self, memory, query, context=None):
                return 0.8

        registry = PluginRegistry()
        scorer = TestScorer()
        registry.register_scorer(scorer)

        plugins = registry.list_plugins()
        assert "test_scorer" in plugins["scorers"]

        # Test scoring
        memory = Memory(
            text="Test memory",
            user_id="test_user",
            type=MemoryType.FACT,
        )
        combined, breakdown = registry.run_scorers(memory, "test query", base_score=0.5)
        assert "test_scorer" in breakdown
        assert breakdown["test_scorer"] == 0.8

    def test_memory_filter_registration(self):
        """Test registering a custom filter."""
        from hippocampai.models.memory import Memory, MemoryType
        from hippocampai.plugins import MemoryFilter, PluginRegistry

        class TestFilter(MemoryFilter):
            name = "test_filter"

            def filter(self, memory, context=None):
                return "exclude" not in memory.text.lower()

        registry = PluginRegistry()
        filter_plugin = TestFilter()
        registry.register_filter(filter_plugin)

        memories = [
            Memory(text="Keep this", user_id="test", type=MemoryType.FACT),
            Memory(text="Exclude this", user_id="test", type=MemoryType.FACT),
            Memory(text="Also keep", user_id="test", type=MemoryType.FACT),
        ]

        filtered = registry.run_filters(memories)
        assert len(filtered) == 2
        assert all("exclude" not in m.text.lower() for m in filtered)

    def test_plugin_unregister(self):
        """Test unregistering a plugin."""
        from hippocampai.plugins import MemoryProcessor, PluginRegistry

        class TestProcessor(MemoryProcessor):
            name = "removable"

            def process(self, memory, context=None):
                return memory

        registry = PluginRegistry()
        registry.register_processor(TestProcessor())
        assert "removable" in registry.list_plugins()["processors"]

        registry.unregister("removable")
        assert "removable" not in registry.list_plugins()["processors"]


class TestNamespaces:
    """Tests for the namespace system."""

    def test_namespace_creation(self):
        """Test creating a namespace."""
        from hippocampai.namespaces import NamespaceManager

        manager = NamespaceManager()
        ns = manager.create("work", user_id="alice", description="Work memories")

        assert ns.name == "work"
        assert ns.path == "work"
        assert ns.owner_id == "alice"
        assert ns.description == "Work memories"

    def test_nested_namespace(self):
        """Test creating nested namespaces."""
        from hippocampai.namespaces import NamespaceManager

        manager = NamespaceManager()
        manager.create("work", user_id="alice")
        child = manager.create("work/project-x", user_id="alice")

        assert child.parent_path == "work"
        assert child.get_depth() == 1

    def test_namespace_permissions(self):
        """Test namespace permission checking."""
        from hippocampai.namespaces import NamespaceManager, NamespacePermission

        manager = NamespaceManager()
        ns = manager.create("private", user_id="alice")

        # Owner has full access
        assert ns.can_read("alice")
        assert ns.can_write("alice")
        assert ns.can_admin("alice")

        # Others have no access by default
        assert not ns.can_read("bob")
        assert not ns.can_write("bob")

        # Grant read permission
        manager.grant_permission("private", "bob", NamespacePermission.READ, "alice")
        ns = manager.get("private")
        assert ns.can_read("bob")
        assert not ns.can_write("bob")

    def test_namespace_quota(self):
        """Test namespace quota checking."""
        from hippocampai.namespaces import NamespaceManager, NamespaceQuota

        manager = NamespaceManager()
        quota = NamespaceQuota(max_memories=10, max_storage_bytes=1000)
        ns = manager.create("limited", user_id="alice", quota=quota)

        assert ns.is_within_quota(additional_memories=5)
        assert not ns.is_within_quota(additional_memories=15)

    def test_namespace_list(self):
        """Test listing namespaces."""
        from hippocampai.namespaces import NamespaceManager

        manager = NamespaceManager()
        manager.create("ns1", user_id="alice")
        manager.create("ns2", user_id="alice")
        manager.create("ns3", user_id="bob")

        alice_ns = manager.list("alice")
        assert len(alice_ns) == 2

    def test_namespace_delete(self):
        """Test deleting a namespace."""
        from hippocampai.namespaces import NamespaceManager, NamespaceNotFoundError

        manager = NamespaceManager()
        manager.create("deleteme", user_id="alice")
        manager.delete("deleteme", user_id="alice")

        with pytest.raises(NamespaceNotFoundError):
            manager.get("deleteme")


class TestPortability:
    """Tests for export/import functionality."""

    def test_export_formats(self):
        """Test export format definitions."""
        from hippocampai.portability import ExportFormat, ExportOptions

        options = ExportOptions(format=ExportFormat.JSON)
        assert options.format == ExportFormat.JSON
        assert options.include_metadata is True
        assert options.compress is True

    def test_memory_record_serialization(self):
        """Test MemoryRecord to/from dict."""
        from hippocampai.portability.formats import MemoryRecord

        record = MemoryRecord(
            id="test-id",
            text="Test memory",
            user_id="alice",
            type="fact",
            importance=7.5,
            confidence=0.9,
            tags=["test", "example"],
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
        )

        data = record.to_dict()
        assert data["id"] == "test-id"
        assert data["text"] == "Test memory"

        restored = MemoryRecord.from_dict(data)
        assert restored.id == record.id
        assert restored.text == record.text

    def test_import_stats(self):
        """Test import statistics."""
        from hippocampai.portability import ImportStats

        stats = ImportStats()
        stats.imported_memories = 10
        stats.skipped_memories = 2
        stats.failed_memories = 1

        assert stats.total_records == 0  # Not set yet
        assert stats.imported_memories == 10


class TestOfflineMode:
    """Tests for offline queue functionality."""

    def test_offline_queue_creation(self):
        """Test creating an offline queue."""
        from hippocampai.offline import OfflineQueue

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "queue.db")
            queue = OfflineQueue(db_path)
            assert queue is not None

    def test_queue_enqueue_dequeue(self):
        """Test enqueueing and retrieving operations."""
        from hippocampai.offline import OfflineQueue, OperationType

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "queue.db")
            queue = OfflineQueue(db_path)

            # Enqueue
            op_id = queue.enqueue(
                operation=OperationType.REMEMBER,
                user_id="alice",
                data={"text": "Test memory", "type": "fact"},
            )
            assert op_id is not None

            # Get pending
            pending = queue.get_pending()
            assert len(pending) == 1
            assert pending[0].user_id == "alice"
            assert pending[0].data["text"] == "Test memory"

    def test_queue_mark_completed(self):
        """Test marking operations as completed."""
        from hippocampai.offline import OfflineQueue, OperationType

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "queue.db")
            queue = OfflineQueue(db_path)

            op_id = queue.enqueue(
                operation=OperationType.REMEMBER,
                user_id="alice",
                data={"text": "Test"},
            )

            queue.mark_completed(op_id)

            pending = queue.get_pending()
            assert len(pending) == 0

            stats = queue.get_stats()
            assert stats["completed"] == 1

    def test_queue_retry_logic(self):
        """Test retry logic for failed operations."""
        from hippocampai.offline import OfflineQueue, OperationType

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "queue.db")
            queue = OfflineQueue(db_path)

            op_id = queue.enqueue(
                operation=OperationType.REMEMBER,
                user_id="alice",
                data={"text": "Test"},
            )

            # First failure - should retry
            should_retry = queue.mark_failed(op_id, "Connection error")
            assert should_retry is True

            # Second failure - should retry
            should_retry = queue.mark_failed(op_id, "Connection error")
            assert should_retry is True

            # Third failure - max retries reached
            should_retry = queue.mark_failed(op_id, "Connection error")
            assert should_retry is False


class TestTieredStorage:
    """Tests for tiered storage functionality."""

    def test_storage_tier_enum(self):
        """Test storage tier enumeration."""
        from hippocampai.tiered import StorageTier

        assert StorageTier.HOT.value == "hot"
        assert StorageTier.WARM.value == "warm"
        assert StorageTier.COLD.value == "cold"
        assert StorageTier.FROZEN.value == "frozen"

    def test_tier_config(self):
        """Test tier configuration."""
        from hippocampai.tiered import TierConfig

        config = TierConfig(
            hot_to_warm_days=15,
            warm_to_cold_days=60,
        )
        assert config.hot_to_warm_days == 15
        assert config.warm_to_cold_days == 60

    def test_migration_stats(self):
        """Test migration statistics."""
        from hippocampai.tiered import MigrationStats

        stats = MigrationStats()
        stats.hot_to_warm = 5
        stats.warm_to_cold = 3
        stats.promoted = 1

        assert stats.hot_to_warm == 5
        assert stats.warm_to_cold == 3


class TestIntegrations:
    """Tests for framework integrations."""

    def test_langchain_import_check(self):
        """Test LangChain integration import handling."""
        from hippocampai.integrations import langchain

        # Should not raise even if langchain not installed
        assert hasattr(langchain, "LANGCHAIN_AVAILABLE")

    def test_llamaindex_import_check(self):
        """Test LlamaIndex integration import handling."""
        from hippocampai.integrations import llamaindex

        # Should not raise even if llama_index not installed
        assert hasattr(llamaindex, "LLAMAINDEX_AVAILABLE")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
