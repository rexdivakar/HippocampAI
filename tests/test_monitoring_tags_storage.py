"""Tests for monitoring tags and Qdrant storage integration."""

import time
from datetime import datetime, timedelta, timezone

import pytest

from hippocampai.embed.embedder import Embedder
from hippocampai.models.memory import Memory, MemoryType
from hippocampai.monitoring import (
    MemoryHealthMonitor,
    MetricsCollector,
    MonitoringStorage,
    OperationType,
)
from hippocampai.vector.qdrant_store import QdrantStore


@pytest.fixture
def qdrant_store():
    """Create Qdrant store instance."""
    return QdrantStore(
        url="http://localhost:6333",
        collection_facts="test_facts",
        collection_prefs="test_prefs",
    )


@pytest.fixture
def monitoring_storage(qdrant_store):
    """Create monitoring storage instance."""
    return MonitoringStorage(
        qdrant_store=qdrant_store,
        collection_health="test_health_reports",
        collection_traces="test_traces",
    )


@pytest.fixture
def collector():
    """Create metrics collector."""
    return MetricsCollector(enable_tracing=True)


@pytest.fixture
def embedder():
    """Create embedder."""
    return Embedder(model_name="BAAI/bge-small-en-v1.5")


@pytest.fixture
def monitor(embedder):
    """Create health monitor."""
    return MemoryHealthMonitor(embedder=embedder)


class TestTraceTags:
    """Test trace tagging functionality."""

    def test_trace_with_tags(self, collector):
        """Test creating trace with tags."""
        tags = {"environment": "production", "version": "1.0", "region": "us-west"}

        with collector.trace_operation(
            OperationType.CREATE,
            tags=tags,
            user_id="user123",
            memory_type="preference",
        ) as trace:
            time.sleep(0.01)

        assert trace.tags == tags
        assert trace.user_id == "user123"
        assert trace.memory_type == "preference"

    def test_trace_with_metadata_and_tags(self, collector):
        """Test trace with both metadata and tags."""
        tags = {"environment": "staging"}
        metadata = {"batch_size": 10, "model": "gpt-4"}

        with collector.trace_operation(
            OperationType.SEARCH,
            tags=tags,
            metadata=metadata,
            user_id="user456",
            session_id="session789",
        ) as trace:
            pass

        assert trace.tags == tags
        assert trace.metadata == metadata
        assert trace.user_id == "user456"
        assert trace.session_id == "session789"

    def test_query_traces_by_tags(self, collector):
        """Test querying traces by tags."""
        # Create traces with different tags
        with collector.trace_operation(
            OperationType.CREATE, tags={"environment": "prod", "version": "1.0"}
        ):
            pass

        with collector.trace_operation(
            OperationType.CREATE, tags={"environment": "staging", "version": "1.0"}
        ):
            pass

        with collector.trace_operation(
            OperationType.CREATE, tags={"environment": "prod", "version": "2.0"}
        ):
            pass

        # Query by single tag
        prod_traces = collector.query_traces(tags={"environment": "prod"})
        assert len(prod_traces) == 2

        # Query by multiple tags (all must match)
        prod_v1_traces = collector.query_traces(tags={"environment": "prod", "version": "1.0"})
        assert len(prod_v1_traces) == 1

    def test_query_traces_by_user_id(self, collector):
        """Test querying traces by user ID."""
        with collector.trace_operation(OperationType.CREATE, user_id="user1"):
            pass

        with collector.trace_operation(OperationType.CREATE, user_id="user2"):
            pass

        with collector.trace_operation(OperationType.UPDATE, user_id="user1"):
            pass

        user1_traces = collector.query_traces(user_id="user1")
        assert len(user1_traces) == 2

    def test_query_traces_by_memory_type(self, collector):
        """Test querying traces by memory type."""
        with collector.trace_operation(OperationType.CREATE, memory_type="preference"):
            pass

        with collector.trace_operation(OperationType.CREATE, memory_type="fact"):
            pass

        with collector.trace_operation(OperationType.UPDATE, memory_type="preference"):
            pass

        pref_traces = collector.query_traces(memory_type="preference")
        assert len(pref_traces) == 2

    def test_query_traces_by_duration(self, collector):
        """Test querying traces by duration."""
        with collector.trace_operation(OperationType.CREATE):
            time.sleep(0.01)  # ~10ms

        with collector.trace_operation(OperationType.CREATE):
            time.sleep(0.05)  # ~50ms

        with collector.trace_operation(OperationType.CREATE):
            time.sleep(0.1)  # ~100ms

        # Query slow traces (>40ms)
        slow_traces = collector.query_traces(min_duration_ms=40.0)
        assert len(slow_traces) >= 2

        # Query fast traces (<30ms)
        fast_traces = collector.query_traces(max_duration_ms=30.0)
        assert len(fast_traces) >= 1

    def test_get_trace_statistics(self, collector):
        """Test getting trace statistics."""
        # Create successful traces
        for _ in range(5):
            with collector.trace_operation(
                OperationType.CREATE,
                tags={"environment": "prod"},
                user_id="user1",
            ):
                time.sleep(0.01)

        # Create failed trace
        try:
            with collector.trace_operation(
                OperationType.CREATE,
                tags={"environment": "prod"},
                user_id="user1",
            ):
                raise ValueError("Test error")
        except ValueError:
            pass

        stats = collector.get_trace_statistics(
            operation=OperationType.CREATE, user_id="user1", tags={"environment": "prod"}
        )

        assert stats["count"] == 6
        assert stats["success_count"] == 5
        assert stats["error_count"] == 1
        assert 80.0 < stats["success_rate"] < 85.0
        assert "duration_stats" in stats


class TestMonitoringStorage:
    """Test Qdrant storage integration."""

    def test_store_health_report(self, monitoring_storage, monitor):
        """Test storing health report in Qdrant."""
        # Create sample memories
        now = datetime.now(timezone.utc)
        memories = [
            Memory(
                text=f"Memory {i}",
                user_id="user1",
                type=MemoryType.FACT,
                confidence=0.9,
                created_at=now - timedelta(days=i),
            )
            for i in range(10)
        ]

        # Generate report
        report = monitor.generate_quality_report(memories, user_id="user1")

        # Store with tags
        tags = {"environment": "test", "version": "1.0"}
        report_id = monitoring_storage.store_health_report(report, tags=tags)

        # Should return a valid UUID string
        assert report_id is not None
        assert len(report_id) > 0
        # Verify it's a valid UUID
        import uuid as uuid_lib

        uuid_lib.UUID(report_id)  # Will raise ValueError if not valid UUID

    def test_store_trace(self, monitoring_storage, collector):
        """Test storing trace in Qdrant."""
        with collector.trace_operation(
            OperationType.CREATE,
            tags={"environment": "test"},
            user_id="user1",
            memory_type="preference",
        ) as trace:
            time.sleep(0.01)

        # Store trace
        additional_tags = {"version": "2.0"}
        stored_trace_id = monitoring_storage.store_trace(trace, additional_tags=additional_tags)

        # Should return a valid UUID string
        assert stored_trace_id is not None
        assert len(stored_trace_id) > 0
        # Verify it's a valid UUID
        import uuid as uuid_lib

        uuid_lib.UUID(stored_trace_id)  # Will raise ValueError if not valid UUID

    def test_query_health_reports(self, monitoring_storage, monitor):
        """Test querying health reports."""
        datetime.now(timezone.utc)

        # Create and store multiple reports
        for i in range(3):
            memories = [
                Memory(
                    text=f"Memory {j}",
                    user_id=f"user{i}",
                    type=MemoryType.FACT,
                    confidence=0.9,
                )
                for j in range(5)
            ]

            report = monitor.generate_quality_report(memories, user_id=f"user{i}")
            monitoring_storage.store_health_report(report, tags={"test": "true", "batch": str(i)})

        # Query all reports
        reports = monitoring_storage.query_health_reports(limit=10)
        assert len(reports) >= 3

        # Query by user
        user1_reports = monitoring_storage.query_health_reports(user_id="user1")
        assert len(user1_reports) >= 1
        assert user1_reports[0]["user_id"] == "user1"

        # Query by tags
        tagged_reports = monitoring_storage.query_health_reports(tags={"test": "true"})
        assert len(tagged_reports) >= 3

    def test_query_traces(self, monitoring_storage, collector):
        """Test querying traces from Qdrant."""
        # Create and store traces
        for i in range(3):
            with collector.trace_operation(
                OperationType.CREATE,
                tags={"test": "true", "batch": str(i)},
                user_id=f"user{i}",
                memory_type="fact",
            ) as trace:
                time.sleep(0.01)

            monitoring_storage.store_trace(trace)

        # Query all traces
        traces = monitoring_storage.query_traces(limit=10)
        assert len(traces) >= 3

        # Query by user
        user0_traces = monitoring_storage.query_traces(user_id="user0")
        assert len(user0_traces) >= 1

        # Query by tags
        tagged_traces = monitoring_storage.query_traces(tags={"test": "true"})
        assert len(tagged_traces) >= 3

        # Query by operation
        create_traces = monitoring_storage.query_traces(operation="create")
        assert len(create_traces) >= 3

    def test_get_health_history(self, monitoring_storage, monitor):
        """Test getting health history."""
        # Create reports over time
        memories = [
            Memory(
                text=f"Memory {i}",
                user_id="user1",
                type=MemoryType.FACT,
                confidence=0.9,
            )
            for i in range(10)
        ]

        report = monitor.generate_quality_report(memories, user_id="user1")
        monitoring_storage.store_health_report(report)

        # Get history
        history = monitoring_storage.get_health_history(user_id="user1", days=30)

        assert len(history) >= 1
        assert "timestamp" in history[0]
        assert "health_score" in history[0]
        assert "status" in history[0]

    def test_get_trace_statistics(self, monitoring_storage, collector):
        """Test getting trace statistics from storage."""
        # Create and store traces
        for _ in range(5):
            with collector.trace_operation(
                OperationType.SEARCH,
                tags={"environment": "test"},
                user_id="user1",
            ) as trace:
                time.sleep(0.01)

            monitoring_storage.store_trace(trace)

        # Get statistics
        stats = monitoring_storage.get_trace_statistics(operation="search", user_id="user1", days=1)

        assert stats["count"] >= 5
        assert stats["success_count"] >= 5
        assert "duration_stats" in stats


class TestIntegration:
    """Test end-to-end integration."""

    def test_full_monitoring_workflow(self, monitoring_storage, collector, monitor, embedder):
        """Test complete monitoring workflow with tags and storage."""
        user_id = "test_user"
        tags = {"environment": "integration_test", "version": "1.0"}

        # Step 1: Create memories with traced operations
        memories = []
        for i in range(5):
            with collector.trace_operation(
                OperationType.CREATE,
                tags=tags,
                user_id=user_id,
                memory_type="fact",
            ) as trace:
                memory = Memory(
                    text=f"Test memory {i}",
                    user_id=user_id,
                    type=MemoryType.FACT,
                    confidence=0.9,
                )
                memories.append(memory)
                time.sleep(0.005)

            # Store trace
            monitoring_storage.store_trace(trace, additional_tags={"step": "create"})

        # Step 2: Generate and store health report
        report = monitor.generate_quality_report(memories, user_id=user_id)
        monitoring_storage.store_health_report(report, tags=tags)

        # Step 3: Query by tags
        tagged_traces = monitoring_storage.query_traces(tags={"environment": "integration_test"})
        assert len(tagged_traces) >= 5

        tagged_reports = monitoring_storage.query_health_reports(
            tags={"environment": "integration_test"}
        )
        assert len(tagged_reports) >= 1

        # Step 4: Get statistics
        stats = collector.get_trace_statistics(
            user_id=user_id, tags={"environment": "integration_test"}
        )
        assert stats["count"] >= 5
        assert stats["success_rate"] == 100.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
