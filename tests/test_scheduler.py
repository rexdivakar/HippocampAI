"""Tests for memory consolidation scheduler."""

import time

import pytest

from hippocampai import MemoryClient
from hippocampai.config import Config


@pytest.fixture
def client_with_scheduler():
    """Create a client with scheduler enabled."""
    config = Config(
        enable_scheduler=True,
        consolidate_cron="0 3 * * 0",  # Weekly on Sunday at 3am
        decay_cron="0 2 * * *",  # Daily at 2am
        snapshot_cron="0 * * * *",  # Hourly
    )
    client = MemoryClient(config=config, enable_telemetry=True)
    yield client
    # Cleanup
    if client.scheduler:
        client.stop_scheduler()


@pytest.fixture
def test_user_id():
    """Generate unique test user ID."""
    return f"test_sched_{int(time.time() * 1000)}"


class TestSchedulerInitialization:
    """Test scheduler initialization and lifecycle."""

    def test_scheduler_initialized(self, client_with_scheduler):
        """Test that scheduler is initialized when enabled."""
        assert client_with_scheduler.scheduler is not None

    def test_scheduler_disabled_by_default(self):
        """Test that scheduler is disabled when enable_scheduler=False."""
        config = Config(enable_scheduler=False)
        client = MemoryClient(config=config, enable_telemetry=False)
        assert client.scheduler is None

    def test_start_scheduler(self, client_with_scheduler):
        """Test starting the scheduler."""
        client_with_scheduler.start_scheduler()

        status = client_with_scheduler.get_scheduler_status()
        assert status["status"] == "running"
        assert len(status["jobs"]) > 0

        client_with_scheduler.stop_scheduler()

    def test_stop_scheduler(self, client_with_scheduler):
        """Test stopping the scheduler."""
        client_with_scheduler.start_scheduler()
        client_with_scheduler.stop_scheduler()

        status = client_with_scheduler.get_scheduler_status()
        assert status["status"] == "stopped"

    def test_scheduler_status_not_initialized(self):
        """Test scheduler status when not initialized."""
        config = Config(enable_scheduler=False)
        client = MemoryClient(config=config, enable_telemetry=False)

        status = client.get_scheduler_status()
        assert status["status"] == "not_initialized"
        assert status["jobs"] == []


class TestConsolidation:
    """Test memory consolidation functionality."""

    def test_consolidate_all_memories_empty(self, client_with_scheduler, test_user_id):
        """Test consolidation with no memories."""
        count = client_with_scheduler.consolidate_all_memories()
        assert count == 0

    def test_consolidate_all_memories_single_user(self, client_with_scheduler, test_user_id):
        """Test consolidation with similar memories."""
        # Create similar memories
        client_with_scheduler.remember(
            text="I love Python programming",
            user_id=test_user_id,
            type="preference",
        )
        client_with_scheduler.remember(
            text="Python is my favorite programming language",
            user_id=test_user_id,
            type="preference",
        )
        client_with_scheduler.remember(
            text="I enjoy coding in Python",
            user_id=test_user_id,
            type="preference",
        )

        # Run consolidation (may or may not consolidate depending on similarity)
        count = client_with_scheduler.consolidate_all_memories(similarity_threshold=0.70)
        # Just check it runs without error
        assert count >= 0

    def test_find_similar_clusters(self, client_with_scheduler, test_user_id):
        """Test finding similar memory clusters."""
        # Create memories
        mem1 = client_with_scheduler.remember(
            text="I like coffee",
            user_id=test_user_id,
            type="preference",
        )
        mem2 = client_with_scheduler.remember(
            text="Coffee is great",
            user_id=test_user_id,
            type="preference",
        )
        mem3 = client_with_scheduler.remember(
            text="I prefer tea over coffee",
            user_id=test_user_id,
            type="preference",
        )

        memories = [mem1, mem2, mem3]
        clusters = client_with_scheduler._find_similar_clusters(memories, threshold=0.60)

        # Should find at least some relationship
        assert isinstance(clusters, list)


class TestImportanceDecay:
    """Test importance decay functionality."""

    def test_apply_importance_decay_empty(self, client_with_scheduler):
        """Test decay with no memories."""
        count = client_with_scheduler.apply_importance_decay()
        assert count == 0

    def test_apply_importance_decay(self, client_with_scheduler, test_user_id):
        """Test importance decay on memories."""
        # Create a memory with high importance
        memory = client_with_scheduler.remember(
            text="Test memory for decay",
            user_id=test_user_id,
            importance=9.0,
        )

        initial_importance = memory.importance

        # Apply decay (won't decay much for brand new memory)
        client_with_scheduler.apply_importance_decay()

        # Fetch updated memory
        memories = client_with_scheduler.get_memories(user_id=test_user_id)
        updated_memory = next(m for m in memories if m.id == memory.id)

        # New memory shouldn't decay much
        assert updated_memory.importance <= initial_importance


class TestSchedulerJobs:
    """Test scheduler job execution."""

    def test_scheduler_job_registration(self, client_with_scheduler):
        """Test that jobs are registered correctly."""
        client_with_scheduler.start_scheduler()

        status = client_with_scheduler.get_scheduler_status()

        # Should have at least consolidation, decay, and snapshot jobs
        job_ids = [job["id"] for job in status["jobs"]]
        assert "consolidate_memories" in job_ids
        assert "decay_importance" in job_ids
        assert "create_snapshots" in job_ids

        client_with_scheduler.stop_scheduler()

    def test_manual_trigger_consolidation(self, client_with_scheduler, test_user_id):
        """Test manually triggering consolidation."""
        client_with_scheduler.start_scheduler()

        # Create some memories
        client_with_scheduler.remember(
            text="Manual trigger test",
            user_id=test_user_id,
        )

        # Manually trigger consolidation
        client_with_scheduler.scheduler.trigger_consolidation_now()

        # Should complete without error
        client_with_scheduler.stop_scheduler()

    def test_manual_trigger_decay(self, client_with_scheduler, test_user_id):
        """Test manually triggering decay."""
        client_with_scheduler.start_scheduler()

        # Create a memory
        client_with_scheduler.remember(
            text="Decay trigger test",
            user_id=test_user_id,
            importance=8.0,
        )

        # Manually trigger decay
        client_with_scheduler.scheduler.trigger_decay_now()

        # Should complete without error
        client_with_scheduler.stop_scheduler()

    def test_manual_trigger_snapshot(self, client_with_scheduler):
        """Test manually triggering snapshot."""
        client_with_scheduler.start_scheduler()

        # Manually trigger snapshot
        client_with_scheduler.scheduler.trigger_snapshot_now()

        # Should complete without error
        client_with_scheduler.stop_scheduler()


class TestSchedulerConfiguration:
    """Test scheduler configuration."""

    def test_custom_cron_schedules(self):
        """Test custom cron schedules."""
        config = Config(
            enable_scheduler=True,
            consolidate_cron="0 4 * * 1",  # Monday at 4am
            decay_cron="0 3 * * *",  # Daily at 3am
            snapshot_cron="0 */2 * * *",  # Every 2 hours
        )
        client = MemoryClient(config=config, enable_telemetry=True)

        assert client.scheduler is not None
        client.start_scheduler()

        status = client.get_scheduler_status()
        assert status["status"] == "running"

        # Check job triggers contain expected patterns
        jobs = {job["id"]: job for job in status["jobs"]}
        assert "consolidate_memories" in jobs
        assert "decay_importance" in jobs
        assert "create_snapshots" in jobs

        client.stop_scheduler()


class TestSchedulerIntegration:
    """Integration tests for scheduler."""

    def test_full_workflow_with_scheduler(self, client_with_scheduler, test_user_id):
        """Test complete workflow with scheduler."""
        # Start scheduler
        client_with_scheduler.start_scheduler()

        # Create memories
        for i in range(5):
            client_with_scheduler.remember(
                text=f"Memory {i} about Python programming",
                user_id=test_user_id,
                type="fact",
                importance=7.0 + i * 0.5,
            )

        # Check scheduler status
        status = client_with_scheduler.get_scheduler_status()
        assert status["status"] == "running"

        # Manually trigger consolidation
        client_with_scheduler.scheduler.trigger_consolidation_now()

        # Manually trigger decay
        client_with_scheduler.scheduler.trigger_decay_now()

        # Get final memory count
        memories = client_with_scheduler.get_memories(user_id=test_user_id)
        assert len(memories) > 0  # Should have at least some memories

        # Stop scheduler
        client_with_scheduler.stop_scheduler()

        status = client_with_scheduler.get_scheduler_status()
        assert status["status"] == "stopped"
