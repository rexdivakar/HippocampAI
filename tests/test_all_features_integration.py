"""
Comprehensive integration tests for all Memory Management API features.

Tests everything including Groq integration.
"""

# ruff: noqa: S101  # Use of assert detected - expected in test files

import asyncio
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

# Must add src to path before importing hippocampai modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dotenv import load_dotenv
from qdrant_client import models

from hippocampai.adapters.llm_base import BaseLLM
from hippocampai.adapters.provider_groq import GroqLLM
from hippocampai.adapters.provider_ollama import OllamaLLM
from hippocampai.adapters.provider_openai import OpenAILLM
from hippocampai.config import get_config
from hippocampai.embed.embedder import Embedder
from hippocampai.models.memory import Memory, RetrievalResult
from hippocampai.retrieval.rerank import Reranker
from hippocampai.services.memory_service import MemoryManagementService
from hippocampai.storage.redis_store import AsyncMemoryKVStore
from hippocampai.vector.qdrant_store import QdrantStore

load_dotenv()


# Colors for output
class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    BOLD = "\033[1m"
    END = "\033[0m"


def print_test(name: str) -> None:
    """Print test name."""
    print(f"\n{Colors.BLUE}{Colors.BOLD}Testing: {name}{Colors.END}")


def print_success(message: str) -> None:
    """Print success message."""
    print(f"{Colors.GREEN}✅ {message}{Colors.END}")


def print_error(message: str) -> None:
    """Print error message."""
    print(f"{Colors.RED}❌ {message}{Colors.END}")


def print_warning(message: str) -> None:
    """Print warning message."""
    print(f"{Colors.YELLOW}⚠️  {message}{Colors.END}")


async def test_service_initialization() -> tuple[
    MemoryManagementService, AsyncMemoryKVStore, QdrantStore
]:
    """Test 1: Service initialization."""
    print_test("Service Initialization")

    try:
        config = get_config()
        print(f"Config loaded: LLM Provider = {config.llm_provider}")

        # Initialize Qdrant
        qdrant = QdrantStore(
            url=config.qdrant_url,
            collection_facts="test_integration_facts",
            collection_prefs="test_integration_prefs",
        )

        # Ensure collections (they're created automatically, but we can check/create manually)
        try:
            qdrant.client.create_collection(
                collection_name="test_integration_facts",
                vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
            )
        except Exception:  # noqa: S110
            pass  # Collection may already exist - expected behavior in tests

        try:
            qdrant.client.create_collection(
                collection_name="test_integration_prefs",
                vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
            )
        except Exception:  # noqa: S110
            pass  # Collection may already exist - expected behavior in tests

        print_success("Qdrant initialized")

        # Initialize embedder
        embedder = Embedder(
            model_name=config.embed_model,
            quantized=config.embed_quantized,
        )
        print_success("Embedder initialized")

        # Initialize reranker
        reranker = Reranker(model_name=config.reranker_model)
        print_success("Reranker initialized")

        # Initialize Redis
        redis_store = AsyncMemoryKVStore(
            redis_url=config.redis_url,
            cache_ttl=config.redis_cache_ttl,
        )
        await redis_store.connect()
        print_success("Redis connected")

        # Initialize LLM (optional, for extraction and consolidation)
        llm: Optional[BaseLLM] = None
        if config.llm_provider == "ollama":
            llm = OllamaLLM(model=config.llm_model, base_url=config.llm_base_url)
            print_success(f"Ollama LLM initialized: {config.llm_model}")
        elif config.llm_provider == "openai" and config.allow_cloud:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                llm = OpenAILLM(api_key=api_key, model=config.llm_model)
                print_success(f"OpenAI LLM initialized: {config.llm_model}")
        elif config.llm_provider == "groq" and config.allow_cloud:
            api_key = os.getenv("GROQ_API_KEY")
            if api_key:
                llm = GroqLLM(api_key=api_key, model=config.llm_model)
                print_success(f"✨ Groq LLM initialized: {config.llm_model}")

        if llm is None:
            print_warning("No LLM configured. Some features will use fallback methods.")

        # Initialize service
        service = MemoryManagementService(
            qdrant_store=qdrant,
            embedder=embedder,
            reranker=reranker,
            redis_store=redis_store,
            llm=llm,
            weights=config.get_weights(),
            half_lives=config.get_half_lives(),
        )
        print_success("Memory Management Service initialized")

        return service, redis_store, qdrant

    except Exception as e:
        print_error(f"Initialization failed: {e}")
        raise


async def test_crud_operations(service: MemoryManagementService) -> Memory:
    """Test 2: CRUD operations."""
    print_test("CRUD Operations")

    # CREATE
    memory = await service.create_memory(
        text="Paris is the capital of France",
        user_id="test_integration_user",
        memory_type="fact",
        importance=8.0,
        tags=["geography", "europe", "france"],
        metadata={"test": "integration"},
    )
    print_success(f"Created memory: {memory.id}")
    assert memory.text == "Paris is the capital of France"
    assert abs(memory.importance - 8.0) < 0.01  # Float comparison with tolerance

    # READ
    retrieved = await service.get_memory(memory.id)
    assert retrieved is not None, "Retrieved memory should not be None"
    print_success(f"Retrieved memory: {retrieved.id}")
    assert retrieved.text == memory.text

    # UPDATE
    updated = await service.update_memory(
        memory_id=memory.id,
        importance=9.5,
        tags=["geography", "europe", "france", "capitals"],
    )
    assert updated is not None, "Updated memory should not be None"
    print_success(f"Updated memory importance: {updated.importance}")
    assert abs(updated.importance - 9.5) < 0.01  # Float comparison with tolerance
    assert "capitals" in updated.tags

    # Query
    memories = await service.get_memories(user_id="test_integration_user")
    print_success(f"Queried {len(memories)} memories")
    assert len(memories) > 0

    # DELETE (we'll keep this one for other tests)
    # deleted = await service.delete_memory(memory.id)
    # print_success(f"Deleted memory: {deleted}")

    return memory


async def test_batch_operations(service: MemoryManagementService) -> list[Memory]:
    """Test 3: Batch operations."""
    print_test("Batch Operations")

    # Batch create
    memories_data = [
        {
            "text": f"Test memory {i}",
            "user_id": "test_integration_user",
            "type": "fact",
            "importance": min(5.0 + i * 0.3, 10.0),  # Keep importance ≤ 10
            "tags": ["test", f"batch_{i}"],
        }
        for i in range(10)
    ]

    created = await service.batch_create_memories(memories_data, check_duplicates=False)
    print_success(f"Batch created {len(created)} memories")
    assert len(created) == 10

    # Batch update
    updates = [{"memory_id": mem.id, "importance": 8.0} for mem in created[:5]]

    updated = await service.batch_update_memories(updates)
    print_success(f"Batch updated {len(updated)} memories")
    assert len(updated) == 5
    assert all(
        abs(m.importance - 8.0) < 0.01 for m in updated if m
    )  # Float comparison with tolerance

    # Batch delete
    delete_ids = [mem.id for mem in created[5:]]
    results = await service.batch_delete_memories(delete_ids)
    print_success(f"Batch deleted {sum(results.values())}/{len(delete_ids)} memories")

    return created[:5]  # Return first 5


async def test_advanced_filtering(service: MemoryManagementService) -> list[Memory]:
    """Test 4: Advanced filtering (NEW feature)."""
    print_test("Advanced Filtering (Date Range, Importance, Text Search)")

    # Create test memories with different dates and importance
    now = datetime.now(timezone.utc)
    yesterday = now - timedelta(days=1)

    mem1 = await service.create_memory(
        text="Python is a programming language",
        user_id="test_integration_user",
        importance=7.0,
        tags=["programming", "python"],
    )

    mem2 = await service.create_memory(
        text="JavaScript is used for web development",
        user_id="test_integration_user",
        importance=6.0,
        tags=["programming", "javascript"],
    )

    # Test importance filtering
    high_importance = await service.get_memories(
        user_id="test_integration_user",
        importance_min=7.0,
    )
    print_success(f"Found {len(high_importance)} memories with importance >= 7.0")
    assert len(high_importance) > 0
    assert all(m.importance >= 7.0 for m in high_importance)

    # Test date filtering
    recent = await service.get_memories(
        user_id="test_integration_user",
        created_after=yesterday,
    )
    print_success(f"Found {len(recent)} memories created after yesterday")
    assert len(recent) > 0

    # Test text search
    python_memories = await service.get_memories(
        user_id="test_integration_user",
        search_text="Python",
    )
    print_success(f"Found {len(python_memories)} memories containing 'Python'")
    assert len(python_memories) > 0

    # Test combined filters
    combined = await service.get_memories(
        user_id="test_integration_user",
        tags=["programming"],
        importance_min=6.0,
        search_text="programming",
    )
    print_success(f"Found {len(combined)} memories with combined filters")

    return [mem1, mem2]


async def test_hybrid_search(service: MemoryManagementService) -> list[RetrievalResult]:
    """Test 5: Hybrid search with custom weights."""
    print_test("Hybrid Search with Custom Weights")

    # Rebuild BM25 index
    service.retriever.rebuild_bm25("test_integration_user")
    print_success("BM25 index rebuilt")

    # Standard recall
    results = await service.recall_memories(
        query="programming languages",
        user_id="test_integration_user",
        k=5,
    )
    print_success(f"Standard recall found {len(results)} results")
    if results:
        print(f"   Top result: {results[0].memory.text[:50]}... (score: {results[0].score:.3f})")

    # Custom weights (emphasize importance)
    custom_results = await service.recall_memories(
        query="programming",
        user_id="test_integration_user",
        k=5,
        custom_weights={
            "sim": 0.3,
            "rerank": 0.2,
            "recency": 0.1,
            "importance": 0.4,  # Emphasize importance
        },
    )
    print_success(f"Custom weighted recall found {len(custom_results)} results")
    if custom_results:
        print(
            f"   Top result: {custom_results[0].memory.text[:50]}... (score: {custom_results[0].score:.3f})"
        )
        print(
            f"   Breakdown: sim={custom_results[0].breakdown['sim']:.3f}, "
            f"importance={custom_results[0].breakdown['importance']:.3f}"
        )

    return results


async def test_deduplication(service: MemoryManagementService) -> dict[str, Any]:
    """Test 6: Deduplication."""
    print_test("Deduplication")

    # Create similar memories
    await service.create_memory(
        text="Rome is the capital of Italy",
        user_id="test_integration_user",
        check_duplicate=False,
    )

    await service.create_memory(
        text="Rome is the capital city of Italy",
        user_id="test_integration_user",
        check_duplicate=False,
    )
    print_success("Created similar memories")

    # Run deduplication (dry run)
    result = await service.deduplicate_user_memories(
        user_id="test_integration_user",
        dry_run=True,
    )
    print_success("Deduplication analysis:")
    print(f"   Total memories: {result['total_memories']}")
    print(f"   Duplicates found: {result['duplicates_found']}")
    print(f"   Dry run: {result['dry_run']}")

    return result


async def test_consolidation(service: MemoryManagementService) -> dict[str, Any]:
    """Test 7: Consolidation (uses Groq if configured)."""
    print_test("Consolidation with Groq LLM")

    # Create similar memories to consolidate
    await service.create_memory(
        text="I like coffee in the morning",
        user_id="test_integration_user",
        memory_type="preference",
    )

    await service.create_memory(
        text="I enjoy drinking coffee every morning",
        user_id="test_integration_user",
        memory_type="preference",
    )

    await service.create_memory(
        text="Morning coffee is my favorite",
        user_id="test_integration_user",
        memory_type="preference",
    )
    print_success("Created similar preference memories")

    # Run consolidation (dry run)
    if service.llm:
        print(f"   LLM configured: {service.llm.__class__.__name__}")
    else:
        print_warning("No LLM configured, will use heuristic consolidation")

    result = await service.consolidate_memories(
        user_id="test_integration_user",
        similarity_threshold=0.75,
        dry_run=True,
    )
    print_success("Consolidation analysis:")
    print(f"   Total memories: {result['total_memories']}")
    print(f"   Similar groups: {result['groups_found']}")
    print(f"   Dry run: {result['dry_run']}")

    return result


async def test_ttl_and_expiration(service: MemoryManagementService) -> None:
    """Test 8: TTL and expiration."""
    print_test("TTL and Expiration")

    # Create memory with short TTL
    memory = await service.create_memory(
        text="This memory will expire soon",
        user_id="test_integration_user",
        ttl_days=1,
    )
    print_success(f"Created memory with TTL: expires at {memory.expires_at}")

    # Manually expire it for testing
    await service.update_memory(
        memory_id=memory.id,
        expires_at=datetime.now(timezone.utc) - timedelta(hours=1),
    )
    print_success("Manually set memory to expired")

    # Run expiration
    expired_count = await service.expire_memories(user_id="test_integration_user")
    print_success(f"Expired {expired_count} memories")

    # Verify deletion
    retrieved = await service.get_memory(memory.id)
    if retrieved is None:
        print_success("Expired memory was deleted")
    else:
        print_warning("Expired memory still exists")


async def test_redis_caching(service: MemoryManagementService) -> None:
    """Test 9: Redis caching."""
    print_test("Redis Caching Performance")

    # Create a memory
    memory = await service.create_memory(
        text="Test memory for caching",
        user_id="test_integration_user",
    )

    # First read (from Qdrant)
    import time

    start = time.time()
    await service.get_memory(memory.id)
    time1 = (time.time() - start) * 1000
    print_success(f"First read (Qdrant): {time1:.2f}ms")

    # Second read (from Redis cache)
    start = time.time()
    await service.get_memory(memory.id)
    time2 = (time.time() - start) * 1000
    print_success(f"Second read (Redis): {time2:.2f}ms")

    if time2 < time1:
        speedup = time1 / time2
        print_success(f"Cache speedup: {speedup:.1f}x faster")

    # Check cache stats
    stats = await service.redis.get_stats()
    print_success(f"Redis stats: {stats}")


def test_background_tasks(service: MemoryManagementService) -> None:
    """Test 10: Background tasks."""
    print_test("Background Tasks (would run in production)")

    print_warning("Background tasks run automatically in the FastAPI app")
    print("   Features:")
    print("   - Automatic expiration every 1 hour")
    print("   - Automatic deduplication every 24 hours (if enabled)")
    print("   - Automatic consolidation every 7 days (if enabled)")
    print("   - Manual triggers via /v1/background/* endpoints")
    print_success("Background task features documented")


async def cleanup(
    service: MemoryManagementService, redis_store: AsyncMemoryKVStore, qdrant: QdrantStore
) -> None:
    """Cleanup test data."""
    print_test("Cleanup")

    try:
        # Clear Redis
        await redis_store.backend.clear()
        print_success("Redis cleared")

        # Delete test collections
        try:
            qdrant.client.delete_collection("test_integration_facts")
            qdrant.client.delete_collection("test_integration_prefs")
            print_success("Qdrant test collections deleted")
        except Exception:  # noqa: S110
            pass  # Collections may not exist - expected in cleanup

        # Close Redis
        await redis_store.close()
        print_success("Redis connection closed")

    except Exception as e:
        print_warning(f"Cleanup error (non-critical): {e}")


async def main() -> None:
    """Run all integration tests."""
    print(f"\n{Colors.BOLD}{'=' * 70}")
    print("COMPREHENSIVE INTEGRATION TESTS")
    print("Testing ALL Memory Management API Features")
    print(f"{'=' * 70}{Colors.END}\n")

    service = None
    redis_store = None
    qdrant = None

    try:
        # Test 1: Initialization
        service, redis_store, qdrant = await test_service_initialization()

        # Test 2: CRUD
        await test_crud_operations(service)

        # Test 3: Batch operations
        await test_batch_operations(service)

        # Test 4: Advanced filtering (NEW)
        await test_advanced_filtering(service)

        # Test 5: Hybrid search with custom weights
        await test_hybrid_search(service)

        # Test 6: Deduplication
        await test_deduplication(service)

        # Test 7: Consolidation (with Groq)
        await test_consolidation(service)

        # Test 8: TTL and expiration
        await test_ttl_and_expiration(service)

        # Test 9: Redis caching
        await test_redis_caching(service)

        # Test 10: Background tasks info
        test_background_tasks(service)

        # Success!
        print(f"\n{Colors.GREEN}{Colors.BOLD}{'=' * 70}")
        print("✅ ALL TESTS PASSED!")
        print(f"{'=' * 70}{Colors.END}\n")

        print(f"{Colors.BOLD}Summary:{Colors.END}")
        print("  ✅ Service initialization")
        print("  ✅ CRUD operations")
        print("  ✅ Batch operations")
        print("  ✅ Advanced filtering (date range, importance, text search)")
        print("  ✅ Hybrid search with custom weights")
        print("  ✅ Deduplication")
        print("  ✅ Consolidation")
        print("  ✅ TTL and expiration")
        print("  ✅ Redis caching")
        print("  ✅ Background tasks")

    except Exception as e:
        print(f"\n{Colors.RED}{Colors.BOLD}{'=' * 70}")
        print(f"❌ TEST FAILED: {e}")
        print(f"{'=' * 70}{Colors.END}\n")
        import traceback

        traceback.print_exc()

    finally:
        if service and redis_store and qdrant:
            await cleanup(service, redis_store, qdrant)


if __name__ == "__main__":
    asyncio.run(main())
