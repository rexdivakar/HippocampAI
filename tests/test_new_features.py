"""Tests for new high-priority features."""

import time
from datetime import datetime, timedelta

import pytest

from hippocampai import MemoryClient
from hippocampai.models.memory import MemoryType


@pytest.fixture
def client():
    """Create a test client."""
    return MemoryClient()


@pytest.fixture
def test_user_id():
    """Generate a unique test user ID."""
    return f"test_user_{int(time.time())}"


class TestUpdateMemory:
    """Test update_memory functionality."""

    def test_update_text(self, client, test_user_id):
        """Test updating memory text."""
        # Create a memory
        memory = client.remember(
            text="I like coffee",
            user_id=test_user_id,
            type="preference",
            tags=["beverages"],
        )

        # Update the text
        updated = client.update_memory(
            memory_id=memory.id,
            text="I love coffee",
        )

        assert updated is not None
        assert updated.text == "I love coffee"
        assert updated.tags == ["beverages"]
        assert updated.id == memory.id

    def test_update_importance(self, client, test_user_id):
        """Test updating importance score."""
        memory = client.remember(
            text="I like tea",
            user_id=test_user_id,
            importance=5.0,
        )

        updated = client.update_memory(
            memory_id=memory.id,
            importance=9.0,
        )

        assert updated is not None
        assert updated.importance == 9.0
        assert updated.text == "I like tea"

    def test_update_tags(self, client, test_user_id):
        """Test updating tags."""
        memory = client.remember(
            text="I exercise daily",
            user_id=test_user_id,
            tags=["health"],
        )

        updated = client.update_memory(
            memory_id=memory.id,
            tags=["health", "fitness", "routine"],
        )

        assert updated is not None
        assert set(updated.tags) == {"health", "fitness", "routine"}

    def test_update_metadata(self, client, test_user_id):
        """Test updating metadata."""
        memory = client.remember(
            text="Meeting scheduled",
            user_id=test_user_id,
        )

        updated = client.update_memory(
            memory_id=memory.id,
            metadata={"location": "zoom", "duration": "30min"},
        )

        assert updated is not None
        assert updated.metadata["location"] == "zoom"
        assert updated.metadata["duration"] == "30min"

    def test_update_nonexistent(self, client):
        """Test updating non-existent memory."""
        updated = client.update_memory(
            memory_id="nonexistent-id",
            text="Should fail",
        )

        assert updated is None


class TestDeleteMemory:
    """Test delete_memory functionality."""

    def test_delete_memory(self, client, test_user_id):
        """Test deleting a memory."""
        memory = client.remember(
            text="Temporary note",
            user_id=test_user_id,
        )

        # Delete it
        result = client.delete_memory(memory_id=memory.id)
        assert result is True

        # Verify it's gone
        memories = client.get_memories(user_id=test_user_id)
        assert not any(m.id == memory.id for m in memories)

    def test_delete_with_user_check(self, client, test_user_id):
        """Test deleting with user ownership check."""
        memory = client.remember(
            text="User-specific note",
            user_id=test_user_id,
        )

        # Try to delete with wrong user
        result = client.delete_memory(memory_id=memory.id, user_id="wrong_user")
        assert result is False

        # Delete with correct user
        result = client.delete_memory(memory_id=memory.id, user_id=test_user_id)
        assert result is True

    def test_delete_nonexistent(self, client):
        """Test deleting non-existent memory."""
        result = client.delete_memory(memory_id="nonexistent-id")
        assert result is False


class TestGetMemories:
    """Test get_memories functionality."""

    def test_get_all_memories(self, client, test_user_id):
        """Test getting all memories for a user."""
        # Create some memories
        client.remember(text="Fact 1", user_id=test_user_id, type="fact")
        client.remember(text="Fact 2", user_id=test_user_id, type="fact")
        client.remember(text="Pref 1", user_id=test_user_id, type="preference")

        memories = client.get_memories(user_id=test_user_id)
        assert len(memories) >= 3

    def test_filter_by_type(self, client, test_user_id):
        """Test filtering by memory type."""
        client.remember(text="Fact", user_id=test_user_id, type="fact")
        client.remember(text="Goal", user_id=test_user_id, type="goal")

        facts = client.get_memories(user_id=test_user_id, filters={"type": "fact"})
        assert all(m.type == MemoryType.FACT for m in facts)

    def test_filter_by_tags(self, client, test_user_id):
        """Test filtering by tags."""
        client.remember(
            text="Coffee habit",
            user_id=test_user_id,
            tags=["beverages", "morning"],
        )
        client.remember(
            text="Tea habit",
            user_id=test_user_id,
            tags=["beverages", "afternoon"],
        )
        client.remember(
            text="Workout habit",
            user_id=test_user_id,
            tags=["fitness"],
        )

        beverage_memories = client.get_memories(
            user_id=test_user_id,
            filters={"tags": "beverages"},
        )
        assert len(beverage_memories) >= 2
        assert all("beverages" in m.tags for m in beverage_memories)

    def test_filter_by_importance(self, client, test_user_id):
        """Test filtering by importance range."""
        client.remember(
            text="Low importance",
            user_id=test_user_id,
            importance=2.0,
        )
        client.remember(
            text="High importance",
            user_id=test_user_id,
            importance=9.0,
        )

        high_importance = client.get_memories(
            user_id=test_user_id,
            filters={"min_importance": 7.0},
        )
        assert all(m.importance >= 7.0 for m in high_importance)


class TestTagFiltering:
    """Test tag-based filtering in recall."""

    def test_recall_with_tag_filter(self, client, test_user_id):
        """Test semantic recall with tag filtering."""
        client.remember(
            text="I drink coffee every morning",
            user_id=test_user_id,
            tags=["beverages", "routine"],
        )
        client.remember(
            text="I go to the gym at 6am",
            user_id=test_user_id,
            tags=["fitness", "routine"],
        )

        # Recall with tag filter
        results = client.recall(
            query="What are my morning habits?",
            user_id=test_user_id,
            filters={"tags": "routine"},
        )

        assert len(results) >= 2
        assert all("routine" in r.memory.tags for r in results)

    def test_recall_multiple_tags(self, client, test_user_id):
        """Test recall filtering by multiple tags."""
        client.remember(
            text="Coffee in the morning",
            user_id=test_user_id,
            tags=["beverages", "morning"],
        )
        client.remember(
            text="Tea in the afternoon",
            user_id=test_user_id,
            tags=["beverages", "afternoon"],
        )

        morning_results = client.recall(
            query="morning drink",
            user_id=test_user_id,
            filters={"tags": ["beverages", "morning"]},
        )

        # Should match memories with ANY of the tags
        assert len(morning_results) > 0


class TestMemoryTTL:
    """Test memory TTL functionality."""

    def test_create_with_ttl(self, client, test_user_id):
        """Test creating memory with TTL."""
        memory = client.remember(
            text="Expires in 7 days",
            user_id=test_user_id,
            ttl_days=7,
        )

        assert memory.expires_at is not None
        # Check that expires_at is approximately 7 days from now
        expected_expiry = datetime.utcnow() + timedelta(days=7)
        time_diff = abs((memory.expires_at - expected_expiry).total_seconds())
        assert time_diff < 10  # Within 10 seconds

    def test_memory_expiration_check(self, client, test_user_id):
        """Test that expired memories are excluded by default."""
        # Create an already-expired memory
        past_date = datetime.utcnow() - timedelta(days=1)
        memory = client.remember(
            text="Already expired",
            user_id=test_user_id,
        )

        # Manually update to be expired
        client.update_memory(
            memory_id=memory.id,
            expires_at=past_date,
        )

        # Get memories - should exclude expired by default
        memories = client.get_memories(user_id=test_user_id)
        assert not any(m.id == memory.id for m in memories)

        # Get with include_expired flag
        memories_with_expired = client.get_memories(
            user_id=test_user_id,
            filters={"include_expired": True},
        )
        assert any(m.id == memory.id for m in memories_with_expired)

    def test_expire_memories_function(self, client, test_user_id):
        """Test the expire_memories cleanup function."""
        # Create some expired memories
        past_date = datetime.utcnow() - timedelta(days=1)

        for i in range(3):
            memory = client.remember(
                text=f"Expired memory {i}",
                user_id=test_user_id,
            )
            client.update_memory(memory_id=memory.id, expires_at=past_date)

        # Create a non-expired memory
        client.remember(
            text="Active memory",
            user_id=test_user_id,
            ttl_days=30,
        )

        # Run expiration
        expired_count = client.expire_memories(user_id=test_user_id)

        assert expired_count >= 3

        # Verify expired memories are gone
        memories = client.get_memories(
            user_id=test_user_id,
            filters={"include_expired": True},
        )
        assert all(not m.is_expired() for m in memories)


class TestTelemetry:
    """Test telemetry for new operations."""

    def test_update_telemetry(self, client, test_user_id):
        """Test that update operations are tracked."""
        memory = client.remember(
            text="Test memory",
            user_id=test_user_id,
        )

        client.update_memory(memory_id=memory.id, text="Updated text")

        operations = client.get_recent_operations(limit=5, operation="update")
        assert len(operations) > 0
        assert operations[0].operation.value == "update"

    def test_delete_telemetry(self, client, test_user_id):
        """Test that delete operations are tracked."""
        memory = client.remember(
            text="To be deleted",
            user_id=test_user_id,
        )

        client.delete_memory(memory_id=memory.id)

        operations = client.get_recent_operations(limit=5, operation="delete")
        assert len(operations) > 0
        assert operations[0].operation.value == "delete"

    def test_get_telemetry(self, client, test_user_id):
        """Test that get operations are tracked."""
        client.get_memories(user_id=test_user_id)

        operations = client.get_recent_operations(limit=5, operation="get")
        assert len(operations) > 0
        assert operations[0].operation.value == "get"


class TestMemorySizeTracking:
    """Test memory size tracking functionality."""

    def test_size_calculated_on_creation(self, client, test_user_id):
        """Test that text_length and token_count are calculated on creation."""
        text = "This is a test memory for size tracking"
        memory = client.remember(
            text=text,
            user_id=test_user_id,
        )

        assert memory.text_length == len(text)
        assert memory.token_count == len(text) // 4  # 4 chars ≈ 1 token

    def test_size_recalculated_on_update(self, client, test_user_id):
        """Test that sizes are recalculated when text is updated."""
        memory = client.remember(
            text="Short",
            user_id=test_user_id,
        )

        old_length = memory.text_length
        old_tokens = memory.token_count

        # Update with longer text
        new_text = "This is a much longer piece of text for testing size updates"
        updated = client.update_memory(
            memory_id=memory.id,
            text=new_text,
        )

        assert updated.text_length == len(new_text)
        assert updated.token_count == len(new_text) // 4
        assert updated.text_length > old_length
        assert updated.token_count > old_tokens

    def test_get_memory_statistics(self, client, test_user_id):
        """Test getting memory statistics."""
        # Create memories with different sizes
        client.remember(text="Short memory", user_id=test_user_id)
        client.remember(
            text="This is a medium length memory for testing",
            user_id=test_user_id,
        )
        client.remember(
            text="This is a much longer memory that contains significantly more text to test the statistics calculation properly",
            user_id=test_user_id,
        )

        stats = client.get_memory_statistics(user_id=test_user_id)

        assert stats["total_memories"] >= 3
        assert stats["total_characters"] > 0
        assert stats["total_tokens"] > 0
        assert stats["avg_memory_size_chars"] > 0
        assert stats["avg_memory_size_tokens"] > 0
        assert stats["largest_memory_chars"] > stats["smallest_memory_chars"]

    def test_statistics_by_type(self, client, test_user_id):
        """Test statistics grouped by memory type."""
        client.remember(
            text="I like programming",
            user_id=test_user_id,
            type="preference",
        )
        client.remember(
            text="Python is a versatile language",
            user_id=test_user_id,
            type="fact",
        )
        client.remember(
            text="Learn advanced Python patterns",
            user_id=test_user_id,
            type="goal",
        )

        stats = client.get_memory_statistics(user_id=test_user_id)

        assert "by_type" in stats
        by_type = stats["by_type"]

        # Should have at least preference, fact, and goal types
        assert "preference" in by_type or "fact" in by_type or "goal" in by_type

        # Each type should have count, total_chars, total_tokens, avg_chars, avg_tokens
        for type_stats in by_type.values():
            assert "count" in type_stats
            assert "total_chars" in type_stats
            assert "total_tokens" in type_stats
            assert "avg_chars" in type_stats
            assert "avg_tokens" in type_stats

    def test_empty_statistics(self, client):
        """Test statistics for user with no memories."""
        stats = client.get_memory_statistics(user_id="nonexistent_user_123")

        assert stats["total_memories"] == 0
        assert stats["total_characters"] == 0
        assert stats["total_tokens"] == 0


class TestIntegration:
    """Integration tests combining multiple features."""

    def test_full_workflow(self, client, test_user_id):
        """Test complete CRUD workflow with all new features."""
        # Create memory with tags and TTL
        memory = client.remember(
            text="I prefer dark roast coffee",
            user_id=test_user_id,
            type="preference",
            tags=["beverages", "coffee"],
            ttl_days=30,
            importance=8.0,
        )

        assert memory.id is not None
        assert "beverages" in memory.tags
        assert memory.expires_at is not None
        # Verify size tracking
        assert memory.text_length > 0
        assert memory.token_count > 0

        # Update it
        updated = client.update_memory(
            memory_id=memory.id,
            text="I prefer dark roast coffee with oat milk",
            tags=["beverages", "coffee", "milk"],
        )

        assert "milk" in updated.tags
        # Verify size was recalculated
        assert updated.text_length > memory.text_length

        # Retrieve with filters
        results = client.recall(
            query="coffee preferences",
            user_id=test_user_id,
            filters={"tags": "coffee"},
        )

        assert len(results) > 0
        assert any(r.memory.id == memory.id for r in results)

        # Get with advanced filters
        memories = client.get_memories(
            user_id=test_user_id,
            filters={"tags": ["beverages"], "min_importance": 7.0},
        )

        assert len(memories) > 0
        assert any(m.id == memory.id for m in memories)

        # Check statistics
        stats = client.get_memory_statistics(user_id=test_user_id)
        assert stats["total_memories"] >= 1
        assert stats["total_characters"] > 0

        # Delete
        deleted = client.delete_memory(memory_id=memory.id, user_id=test_user_id)
        assert deleted is True

        # Verify deletion
        memories_after = client.get_memories(user_id=test_user_id)
        assert not any(m.id == memory.id for m in memories_after)
