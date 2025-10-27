"""Comprehensive tests for Memory Management APIs."""

from datetime import datetime, timedelta, timezone

import pytest

from hippocampai.config import Config
from hippocampai.embed.embedder import Embedder
from hippocampai.models.memory import MemoryType
from hippocampai.retrieval.rerank import Reranker
from hippocampai.services.memory_service import MemoryManagementService
from hippocampai.storage.redis_store import AsyncMemoryKVStore
from hippocampai.vector.qdrant_store import QdrantStore


@pytest.fixture
async def redis_store():
    """Create Redis store for testing."""
    store = AsyncMemoryKVStore(redis_url="redis://localhost:6379", cache_ttl=60)
    await store.connect()
    yield store
    # Cleanup
    await store.backend.clear()
    await store.close()


@pytest.fixture
def qdrant_store():
    """Create Qdrant store for testing."""
    config = Config()
    store = QdrantStore(
        url=config.qdrant_url,
        collection_facts="test_facts",
        collection_prefs="test_prefs",
    )
    # Ensure collections exist
    store.ensure_collection(
        collection_name="test_facts",
        vector_size=384,
        distance="Cosine",
    )
    store.ensure_collection(
        collection_name="test_prefs",
        vector_size=384,
        distance="Cosine",
    )
    yield store
    # Cleanup
    try:
        store.client.delete_collection("test_facts")
        store.client.delete_collection("test_prefs")
    except Exception:
        pass


@pytest.fixture
def embedder():
    """Create embedder for testing."""
    return Embedder(model_name="BAAI/bge-small-en-v1.5", quantized=True, batch_size=32)


@pytest.fixture
def reranker():
    """Create reranker for testing."""
    return Reranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")


@pytest.fixture
async def service(qdrant_store, embedder, reranker, redis_store):
    """Create memory management service for testing."""
    service = MemoryManagementService(
        qdrant_store=qdrant_store,
        embedder=embedder,
        reranker=reranker,
        redis_store=redis_store,
    )
    return service


# ============================================================================
# CRUD Tests
# ============================================================================


@pytest.mark.asyncio
async def test_create_memory(service):
    """Test creating a memory."""
    memory = await service.create_memory(
        text="Paris is the capital of France",
        user_id="user123",
        memory_type="fact",
        importance=8.0,
        tags=["geography", "france"],
    )

    assert memory.id is not None
    assert memory.text == "Paris is the capital of France"
    assert memory.user_id == "user123"
    assert memory.type == MemoryType.FACT
    assert memory.importance == 8.0
    assert "geography" in memory.tags


@pytest.mark.asyncio
async def test_get_memory(service):
    """Test retrieving a memory."""
    # Create memory
    created = await service.create_memory(
        text="Python is a programming language",
        user_id="user123",
        memory_type="fact",
    )

    # Retrieve memory
    retrieved = await service.get_memory(created.id)

    assert retrieved is not None
    assert retrieved.id == created.id
    assert retrieved.text == created.text


@pytest.mark.asyncio
async def test_update_memory(service):
    """Test updating a memory."""
    # Create memory
    memory = await service.create_memory(
        text="Original text",
        user_id="user123",
        importance=5.0,
    )

    # Update memory
    updated = await service.update_memory(
        memory_id=memory.id,
        text="Updated text",
        importance=8.0,
        tags=["updated"],
    )

    assert updated is not None
    assert updated.text == "Updated text"
    assert updated.importance == 8.0
    assert "updated" in updated.tags


@pytest.mark.asyncio
async def test_delete_memory(service):
    """Test deleting a memory."""
    # Create memory
    memory = await service.create_memory(
        text="To be deleted",
        user_id="user123",
    )

    # Delete memory
    deleted = await service.delete_memory(memory.id)
    assert deleted is True

    # Verify deletion
    retrieved = await service.get_memory(memory.id)
    assert retrieved is None


@pytest.mark.asyncio
async def test_get_memories(service):
    """Test querying memories with filters."""
    # Create multiple memories
    await service.create_memory(
        text="Memory 1",
        user_id="user123",
        memory_type="fact",
        tags=["tag1"],
    )
    await service.create_memory(
        text="Memory 2",
        user_id="user123",
        memory_type="preference",
        tags=["tag2"],
    )
    await service.create_memory(
        text="Memory 3",
        user_id="user456",
        memory_type="fact",
    )

    # Query user123's memories
    memories = await service.get_memories(user_id="user123")
    assert len(memories) >= 2

    # Query with type filter
    fact_memories = await service.get_memories(
        user_id="user123",
        filters={"type": "fact"},
    )
    assert all(m.type == MemoryType.FACT for m in fact_memories)


# ============================================================================
# Batch Operations Tests
# ============================================================================


@pytest.mark.asyncio
async def test_batch_create_memories(service):
    """Test batch creating memories."""
    memories_data = [
        {
            "text": "Batch memory 1",
            "user_id": "user123",
            "type": "fact",
        },
        {
            "text": "Batch memory 2",
            "user_id": "user123",
            "type": "fact",
        },
        {
            "text": "Batch memory 3",
            "user_id": "user123",
            "type": "preference",
        },
    ]

    created_memories = await service.batch_create_memories(memories_data)

    assert len(created_memories) == 3
    assert all(m.user_id == "user123" for m in created_memories)


@pytest.mark.asyncio
async def test_batch_update_memories(service):
    """Test batch updating memories."""
    # Create memories
    mem1 = await service.create_memory(text="Memory 1", user_id="user123")
    mem2 = await service.create_memory(text="Memory 2", user_id="user123")

    # Batch update
    updates = [
        {"memory_id": mem1.id, "importance": 9.0},
        {"memory_id": mem2.id, "importance": 8.0},
    ]

    updated_memories = await service.batch_update_memories(updates)

    assert len(updated_memories) == 2
    assert updated_memories[0].importance == 9.0
    assert updated_memories[1].importance == 8.0


@pytest.mark.asyncio
async def test_batch_delete_memories(service):
    """Test batch deleting memories."""
    # Create memories
    mem1 = await service.create_memory(text="Memory 1", user_id="user123")
    mem2 = await service.create_memory(text="Memory 2", user_id="user123")
    mem3 = await service.create_memory(text="Memory 3", user_id="user123")

    # Batch delete
    results = await service.batch_delete_memories([mem1.id, mem2.id])

    assert results[mem1.id] is True
    assert results[mem2.id] is True

    # Verify deletions
    assert await service.get_memory(mem1.id) is None
    assert await service.get_memory(mem2.id) is None
    assert await service.get_memory(mem3.id) is not None


# ============================================================================
# Hybrid Search Tests
# ============================================================================


@pytest.mark.asyncio
async def test_recall_memories(service):
    """Test hybrid search recall."""
    # Create test memories
    await service.create_memory(
        text="Python is a high-level programming language",
        user_id="user123",
        importance=8.0,
    )
    await service.create_memory(
        text="JavaScript is used for web development",
        user_id="user123",
        importance=7.0,
    )
    await service.create_memory(
        text="Machine learning uses Python extensively",
        user_id="user123",
        importance=9.0,
    )

    # Rebuild BM25 index
    service.retriever.rebuild_bm25("user123")

    # Recall with query
    results = await service.recall_memories(
        query="Python programming",
        user_id="user123",
        k=3,
    )

    assert len(results) > 0
    assert all(hasattr(r, "memory") for r in results)
    assert all(hasattr(r, "score") for r in results)


@pytest.mark.asyncio
async def test_recall_with_custom_weights(service):
    """Test recall with custom scoring weights."""
    # Create memory
    await service.create_memory(
        text="Test memory for custom weights",
        user_id="user123",
        importance=9.0,
    )

    # Rebuild BM25
    service.retriever.rebuild_bm25("user123")

    # Recall with custom weights
    custom_weights = {
        "sim": 0.3,
        "rerank": 0.2,
        "recency": 0.2,
        "importance": 0.3,  # Emphasize importance
    }

    results = await service.recall_memories(
        query="custom weights",
        user_id="user123",
        k=5,
        custom_weights=custom_weights,
    )

    assert len(results) > 0


# ============================================================================
# Deduplication Tests
# ============================================================================


@pytest.mark.asyncio
async def test_create_with_deduplication(service):
    """Test automatic deduplication during creation."""
    # Create first memory
    await service.create_memory(
        text="The Eiffel Tower is in Paris",
        user_id="user123",
        check_duplicate=True,
    )

    # Try to create similar memory
    await service.create_memory(
        text="The Eiffel Tower is located in Paris",
        user_id="user123",
        check_duplicate=True,
    )

    # Should either skip or update, not create new
    # In update case, IDs will be the same
    # In skip case, the returned memory will be the existing one


@pytest.mark.asyncio
async def test_deduplicate_user_memories(service):
    """Test batch deduplication."""
    # Create duplicate memories
    await service.create_memory(
        text="Rome is the capital of Italy",
        user_id="user123",
        check_duplicate=False,
    )
    await service.create_memory(
        text="Rome is the capital city of Italy",
        user_id="user123",
        check_duplicate=False,
    )

    # Run deduplication in dry run mode
    result = await service.deduplicate_user_memories(user_id="user123", dry_run=True)

    assert result["user_id"] == "user123"
    assert "duplicates_found" in result
    assert result["dry_run"] is True


# ============================================================================
# Consolidation Tests
# ============================================================================


@pytest.mark.asyncio
async def test_consolidate_memories(service):
    """Test memory consolidation."""
    # Create similar memories
    await service.create_memory(
        text="I like coffee in the morning",
        user_id="user123",
        memory_type="preference",
    )
    await service.create_memory(
        text="I enjoy drinking coffee in the morning",
        user_id="user123",
        memory_type="preference",
    )
    await service.create_memory(
        text="Morning coffee is my favorite",
        user_id="user123",
        memory_type="preference",
    )

    # Run consolidation in dry run mode
    result = await service.consolidate_memories(
        user_id="user123",
        similarity_threshold=0.75,
        dry_run=True,
    )

    assert result["user_id"] == "user123"
    assert "groups_found" in result
    assert result["dry_run"] is True


# ============================================================================
# TTL & Expiration Tests
# ============================================================================


@pytest.mark.asyncio
async def test_memory_expiration(service):
    """Test memory expiration based on TTL."""
    # Create memory with 1-day TTL
    memory = await service.create_memory(
        text="Expiring memory",
        user_id="user123",
        ttl_days=1,
    )

    # Verify it has expiration
    assert memory.expires_at is not None

    # Manually set to expired (for testing)
    await service.update_memory(
        memory_id=memory.id,
        expires_at=datetime.now(timezone.utc) - timedelta(days=1),
    )

    # Run expiration
    expired_count = await service.expire_memories(user_id="user123")

    assert expired_count > 0

    # Verify memory is deleted
    retrieved = await service.get_memory(memory.id)
    assert retrieved is None


# ============================================================================
# Redis Cache Tests
# ============================================================================


@pytest.mark.asyncio
async def test_redis_caching(service):
    """Test Redis caching functionality."""
    # Create memory
    memory = await service.create_memory(
        text="Cached memory",
        user_id="user123",
    )

    # Get from cache
    cached = await service.redis.get_memory(memory.id)
    assert cached is not None
    assert cached["text"] == "Cached memory"

    # Get stats
    stats = await service.redis.get_stats()
    assert "total_keys" in stats
    assert stats["memory_keys"] > 0


@pytest.mark.asyncio
async def test_redis_batch_operations(service):
    """Test Redis batch operations."""
    # Create memories
    memories = [(f"mem_{i}", {"text": f"Memory {i}", "user_id": "user123"}) for i in range(5)]

    # Batch set
    await service.redis.batch_set_memories(memories)

    # Verify
    for mem_id, _ in memories:
        cached = await service.redis.get_memory(mem_id)
        assert cached is not None

    # Batch delete
    await service.redis.batch_delete_memories([mem_id for mem_id, _ in memories])

    # Verify deletion
    for mem_id, _ in memories:
        cached = await service.redis.get_memory(mem_id)
        assert cached is None


# ============================================================================
# Integration Tests
# ============================================================================


@pytest.mark.asyncio
async def test_end_to_end_workflow(service):
    """Test complete workflow from creation to retrieval."""
    # 1. Create memories
    memories = await service.batch_create_memories(
        [
            {
                "text": "Python is great for data science",
                "user_id": "user123",
                "type": "fact",
                "importance": 8.0,
            },
            {
                "text": "I prefer Python over Java",
                "user_id": "user123",
                "type": "preference",
                "importance": 7.0,
            },
            {
                "text": "Learn machine learning with Python",
                "user_id": "user123",
                "type": "goal",
                "importance": 9.0,
            },
        ]
    )

    assert len(memories) == 3

    # 2. Query memories
    all_memories = await service.get_memories(user_id="user123")
    assert len(all_memories) >= 3

    # 3. Rebuild BM25 and recall
    service.retriever.rebuild_bm25("user123")
    results = await service.recall_memories(
        query="Python programming",
        user_id="user123",
        k=3,
    )
    assert len(results) > 0

    # 4. Update a memory
    updated = await service.update_memory(
        memory_id=memories[0].id,
        importance=10.0,
    )
    assert updated.importance == 10.0

    # 5. Delete a memory
    deleted = await service.delete_memory(memories[2].id)
    assert deleted is True

    # 6. Verify final state
    final_memories = await service.get_memories(user_id="user123")
    assert len(final_memories) >= 2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
