"""Tests for async client operations."""

import asyncio

import pytest

from hippocampai import AsyncMemoryClient


@pytest.fixture
def client():
    """Create a test async client."""
    return AsyncMemoryClient()


@pytest.fixture
def test_user_id():
    """Generate a unique test user ID."""
    import time

    return f"test_async_user_{int(time.time() * 1000)}"


class TestAsyncOperations:
    """Test async variants of core operations."""

    @pytest.mark.asyncio
    async def test_remember_async(self, client, test_user_id):
        """Test async remember operation."""
        memory = await client.remember_async(
            text="This is an async test memory",
            user_id=test_user_id,
            type="fact",
            tags=["async", "test"],
        )

        assert memory.id is not None
        assert memory.text == "This is an async test memory"
        assert "async" in memory.tags
        assert memory.text_length > 0
        assert memory.token_count > 0

    @pytest.mark.asyncio
    async def test_recall_async(self, client, test_user_id):
        """Test async recall operation."""
        # First, add some memories
        await client.remember_async(
            text="Python is great for async programming",
            user_id=test_user_id,
            tags=["python", "async"],
        )
        await client.remember_async(
            text="JavaScript has async/await too",
            user_id=test_user_id,
            tags=["javascript", "async"],
        )

        # Recall them
        results = await client.recall_async(
            query="async programming",
            user_id=test_user_id,
            k=5,
        )

        assert len(results) >= 2
        assert all(hasattr(r, "memory") for r in results)

    @pytest.mark.asyncio
    async def test_update_memory_async(self, client, test_user_id):
        """Test async update operation."""
        memory = await client.remember_async(
            text="Original text",
            user_id=test_user_id,
        )

        updated = await client.update_memory_async(
            memory_id=memory.id,
            text="Updated text",
            importance=9.0,
        )

        assert updated is not None
        assert updated.text == "Updated text"
        assert updated.importance == 9.0

    @pytest.mark.asyncio
    async def test_delete_memory_async(self, client, test_user_id):
        """Test async delete operation."""
        memory = await client.remember_async(
            text="To be deleted",
            user_id=test_user_id,
        )

        result = await client.delete_memory_async(memory_id=memory.id)
        assert result is True

        # Verify deletion
        memories = await client.get_memories_async(user_id=test_user_id)
        assert not any(m.id == memory.id for m in memories)

    @pytest.mark.asyncio
    async def test_get_memories_async(self, client, test_user_id):
        """Test async get_memories operation."""
        await client.remember_async(text="Memory 1", user_id=test_user_id)
        await client.remember_async(text="Memory 2", user_id=test_user_id)

        memories = await client.get_memories_async(user_id=test_user_id)
        assert len(memories) >= 2

    @pytest.mark.asyncio
    async def test_batch_operations_async(self, client, test_user_id):
        """Test async batch operations."""
        memories_data = [
            {"text": "Async batch memory 1", "tags": ["batch"]},
            {"text": "Async batch memory 2", "tags": ["batch"]},
            {"text": "Async batch memory 3", "tags": ["batch"]},
        ]

        created = await client.add_memories_async(
            memories=memories_data,
            user_id=test_user_id,
        )

        assert len(created) == 3
        assert all(m.text_length > 0 for m in created)

        # Delete them
        memory_ids = [m.id for m in created]
        deleted_count = await client.delete_memories_async(memory_ids, test_user_id)
        assert deleted_count == 3

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, client, test_user_id):
        """Test running multiple async operations concurrently."""
        # Create multiple memories concurrently
        tasks = [
            client.remember_async(
                text=f"Concurrent memory {i}",
                user_id=test_user_id,
                tags=["concurrent"],
            )
            for i in range(5)
        ]

        memories = await asyncio.gather(*tasks)
        assert len(memories) == 5
        assert all(m.id is not None for m in memories)

    @pytest.mark.asyncio
    async def test_memory_statistics_async(self, client, test_user_id):
        """Test async statistics retrieval."""
        await client.remember_async(text="Short", user_id=test_user_id)
        await client.remember_async(
            text="This is a longer memory for testing statistics",
            user_id=test_user_id,
        )

        stats = await client.get_memory_statistics_async(user_id=test_user_id)

        assert stats["total_memories"] >= 2
        assert stats["total_characters"] > 0
        assert stats["avg_memory_size_chars"] > 0

    @pytest.mark.asyncio
    async def test_context_injection_async(self, client, test_user_id):
        """Test async context injection."""
        await client.remember_async(
            text="I know Python programming",
            user_id=test_user_id,
            tags=["skills"],
        )

        prompt = await client.inject_context_async(
            prompt="What do I know?",
            query="programming skills",
            user_id=test_user_id,
            k=3,
        )

        assert isinstance(prompt, str)
        assert len(prompt) > 0
