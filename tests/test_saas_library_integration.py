"""Integration tests for SaaS and Library components working together."""

import os
import tempfile

import pytest


class TestCoreLibraryIntegration:
    """Test core library components work together."""

    def test_core_imports(self):
        """Test all core imports work."""
        from hippocampai.core import Config, Memory, MemoryClient, MemoryType, RetrievalResult

        assert MemoryClient is not None
        assert Memory is not None
        assert MemoryType is not None
        assert RetrievalResult is not None
        assert Config is not None

    def test_core_memory_creation(self):
        """Test creating memories with core library."""
        from hippocampai.core import Memory, MemoryType

        memory = Memory(
            text="Test memory from core",
            user_id="test_user",
            type=MemoryType.FACT,
        )
        assert memory.text == "Test memory from core"
        assert memory.user_id == "test_user"
        assert memory.type == MemoryType.FACT

    def test_core_config(self):
        """Test core configuration."""
        from hippocampai.core import Config

        config = Config()
        assert config is not None


class TestPlatformIntegration:
    """Test platform (SaaS) components work together."""

    def test_platform_imports(self):
        """Test all platform imports work."""
        from hippocampai.platform import AutomationController, run_api_server

        assert run_api_server is not None
        assert AutomationController is not None

    def test_automation_controller(self):
        """Test automation controller initialization."""
        from unittest.mock import MagicMock

        from hippocampai.saas.automation import AutomationController, AutomationPolicy, PolicyType

        # Mock memory service
        mock_memory_service = MagicMock()
        mock_memory_service.get_memory_statistics.return_value = {"total_memories": 100}

        controller = AutomationController(memory_service=mock_memory_service)
        assert controller is not None

        # Create a policy
        policy = AutomationPolicy(
            user_id="test_user",
            policy_type=PolicyType.THRESHOLD,
            enabled=True,
        )
        assert policy.user_id == "test_user"
        assert policy.policy_type == PolicyType.THRESHOLD

        # Test policy management
        controller.create_policy(policy)
        retrieved = controller.get_policy("test_user")
        assert retrieved is not None
        assert retrieved.user_id == "test_user"

    def test_task_manager(self):
        """Test task manager."""
        from unittest.mock import MagicMock

        from hippocampai.saas.automation import AutomationController
        from hippocampai.saas.tasks import BackgroundTask, TaskManager, TaskPriority, TaskStatus

        # Mock dependencies
        mock_memory_service = MagicMock()
        controller = AutomationController(memory_service=mock_memory_service)
        manager = TaskManager(automation_controller=controller)
        assert manager is not None

        # Test BackgroundTask model
        task = BackgroundTask(
            user_id="test_user",
            task_type="summarization",
            priority=TaskPriority.NORMAL,
            status=TaskStatus.PENDING,
        )
        assert task.user_id == "test_user"
        assert task.task_type == "summarization"

        # Test task creation
        created_task = manager.create_task(
            user_id="alice",
            task_type="consolidation",
            priority=TaskPriority.HIGH,
        )
        assert created_task.user_id == "alice"
        assert created_task.task_type == "consolidation"
        assert created_task.priority == TaskPriority.HIGH


class TestNewFeaturesWithCore:
    """Test new features integrate with core library."""

    def test_plugins_with_core_memory(self):
        """Test plugin system works with core Memory model."""
        from hippocampai.core import Memory, MemoryType
        from hippocampai.plugins import MemoryProcessor, PluginRegistry

        class TagProcessor(MemoryProcessor):
            name = "tag_processor"

            def process(self, memory, context=None):
                memory.metadata["tagged"] = True
                return memory

        registry = PluginRegistry()
        registry.register_processor(TagProcessor())

        memory = Memory(
            text="Test memory",
            user_id="test_user",
            type=MemoryType.FACT,
        )

        processed = registry.run_processors(memory)
        assert processed.metadata.get("tagged") is True

    def test_namespaces_standalone(self):
        """Test namespace manager works standalone."""
        from hippocampai.namespaces import NamespaceManager

        manager = NamespaceManager()
        ns = manager.create("test_ns", user_id="alice")

        assert ns.name == "test_ns"
        assert ns.owner_id == "alice"
        assert ns.can_write("alice")

    def test_portability_formats(self):
        """Test portability formats work."""
        from hippocampai.portability import ExportFormat, ExportOptions, ImportOptions

        export_opts = ExportOptions(format=ExportFormat.JSON)
        assert export_opts.format == ExportFormat.JSON

        import_opts = ImportOptions(merge_strategy="skip")
        assert import_opts.merge_strategy == "skip"

    def test_offline_queue(self):
        """Test offline queue works."""
        from hippocampai.offline import OfflineQueue, OperationType

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_queue.db")
            queue = OfflineQueue(db_path)

            queue.enqueue(
                operation=OperationType.REMEMBER,
                user_id="alice",
                data={"text": "Test memory"},
            )

            pending = queue.get_pending()
            assert len(pending) == 1
            assert pending[0].user_id == "alice"

    def test_tiered_storage_models(self):
        """Test tiered storage models work."""
        from hippocampai.tiered import MigrationStats, StorageTier, TierConfig

        config = TierConfig(hot_to_warm_days=30, warm_to_cold_days=90)
        assert config.hot_to_warm_days == 30

        stats = MigrationStats()
        stats.hot_to_warm = 5
        assert stats.hot_to_warm == 5

        assert StorageTier.HOT.value == "hot"


class TestBackwardCompatibility:
    """Test backward compatibility with main hippocampai package."""

    def test_main_package_imports(self):
        """Test main package exports everything."""
        from hippocampai import (
            Memory,
            MemoryType,
            RetrievalResult,
            Session,
            SessionStatus,
        )

        assert Memory is not None
        assert MemoryType is not None
        assert RetrievalResult is not None
        assert Session is not None
        assert SessionStatus is not None

    def test_main_package_submodules(self):
        """Test submodules accessible from main package."""
        from hippocampai import (
            core,
            integrations,
            namespaces,
            offline,
            platform,
            plugins,
            portability,
            tiered,
        )

        assert core is not None
        assert platform is not None
        assert plugins is not None
        assert namespaces is not None
        assert portability is not None
        assert offline is not None
        assert tiered is not None
        assert integrations is not None

    def test_saas_automation_from_main(self):
        """Test SaaS automation accessible from main package."""
        from hippocampai import AutomationController, AutomationPolicy, PolicyType

        assert AutomationController is not None
        assert AutomationPolicy is not None
        assert PolicyType is not None

    def test_task_management_from_main(self):
        """Test task management accessible from main package."""
        from hippocampai import BackgroundTask, TaskManager, TaskPriority, TaskStatus

        assert TaskManager is not None
        assert TaskPriority is not None
        assert TaskStatus is not None
        assert BackgroundTask is not None


class TestIntegrationsModule:
    """Test framework integrations module."""

    def test_langchain_module_loads(self):
        """Test LangChain integration module loads."""
        from hippocampai.integrations import langchain

        assert hasattr(langchain, "LANGCHAIN_AVAILABLE")
        # Should have the classes even if langchain not installed
        assert hasattr(langchain, "HippocampMemory")
        assert hasattr(langchain, "HippocampRetriever")

    def test_llamaindex_module_loads(self):
        """Test LlamaIndex integration module loads."""
        from hippocampai.integrations import llamaindex

        assert hasattr(llamaindex, "LLAMAINDEX_AVAILABLE")
        assert hasattr(llamaindex, "HippocampRetriever")
        assert hasattr(llamaindex, "HippocampMemoryStore")

    def test_integrations_helper_functions(self):
        """Test integration helper functions."""
        from hippocampai.integrations import (
            get_langchain_memory,
            get_langchain_retriever,
            get_llamaindex_retriever,
        )

        # These should return None or raise if dependencies not installed
        # but should not crash on import
        assert callable(get_langchain_memory)
        assert callable(get_langchain_retriever)
        assert callable(get_llamaindex_retriever)


class TestCrossCuttingConcerns:
    """Test cross-cutting concerns between library and SaaS."""

    def test_memory_model_consistency(self):
        """Test Memory model is consistent across packages."""
        from hippocampai import Memory as MainMemory
        from hippocampai.core import Memory as CoreMemory
        from hippocampai.models.memory import Memory as ModelsMemory

        # All should be the same class
        assert MainMemory is CoreMemory
        assert CoreMemory is ModelsMemory

    def test_config_consistency(self):
        """Test Config is consistent across packages."""
        from hippocampai import Config as MainConfig
        from hippocampai.core import Config as CoreConfig

        assert MainConfig is CoreConfig

    def test_plugin_with_saas_memory(self):
        """Test plugins work with memories that could be from SaaS."""
        from hippocampai import Memory, MemoryType
        from hippocampai.plugins import MemoryScorer, PluginRegistry

        class ImportanceScorer(MemoryScorer):
            name = "importance"
            weight = 0.3

            def score(self, memory, query, context=None):
                # Score based on importance
                return memory.importance / 10.0 if memory.importance else 0.5

        registry = PluginRegistry()
        registry.register_scorer(ImportanceScorer())

        memory = Memory(
            text="Important memory",
            user_id="test",
            type=MemoryType.FACT,
            importance=8,
        )

        combined, breakdown = registry.run_scorers(memory, "test query", base_score=0.5)
        assert "importance" in breakdown
        assert breakdown["importance"] == 0.8  # 8/10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
