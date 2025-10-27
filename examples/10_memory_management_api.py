"""
Example: Comprehensive Memory Management API Demo

Demonstrates all memory management features:
- CRUD operations
- Batch operations
- Hybrid search with customizable weights
- Automatic extraction from conversations
- Deduplication service
- Consolidation service
"""

import asyncio
import logging

from hippocampai.config import get_config
from hippocampai.embed.embedder import Embedder
from hippocampai.retrieval.rerank import Reranker
from hippocampai.services.memory_service import MemoryManagementService
from hippocampai.storage.redis_store import AsyncMemoryKVStore
from hippocampai.vector.qdrant_store import QdrantStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    """Run comprehensive memory management demo."""

    # Initialize configuration
    config = get_config()

    # Initialize components
    logger.info("Initializing components...")

    qdrant = QdrantStore(
        url=config.qdrant_url,
        collection_facts=config.collection_facts,
        collection_prefs=config.collection_prefs,
    )

    embedder = Embedder(
        model_name=config.embed_model,
        quantized=config.embed_quantized,
    )

    reranker = Reranker(model_name=config.reranker_model)

    redis_store = AsyncMemoryKVStore(
        redis_url=config.redis_url,
        cache_ttl=config.redis_cache_ttl,
    )
    await redis_store.connect()

    # Create service
    service = MemoryManagementService(
        qdrant_store=qdrant,
        embedder=embedder,
        reranker=reranker,
        redis_store=redis_store,
        weights=config.get_weights(),
        half_lives=config.get_half_lives(),
    )

    logger.info("✓ Components initialized")

    # ========================================================================
    # 1. CRUD Operations
    # ========================================================================

    logger.info("\n" + "=" * 70)
    logger.info("1. CRUD OPERATIONS")
    logger.info("=" * 70)

    # Create memory
    memory = await service.create_memory(
        text="Paris is the capital of France",
        user_id="demo_user",
        memory_type="fact",
        importance=8.0,
        tags=["geography", "europe", "france"],
        metadata={"source": "demo"},
    )
    logger.info(f"✓ Created memory: {memory.id}")
    logger.info(f"  Text: {memory.text}")
    logger.info(f"  Type: {memory.type}")
    logger.info(f"  Importance: {memory.importance}")

    # Read memory
    retrieved = await service.get_memory(memory.id)
    logger.info(f"✓ Retrieved memory: {retrieved.id}")

    # Update memory
    updated = await service.update_memory(
        memory_id=memory.id,
        importance=9.0,
        tags=["geography", "europe", "france", "capitals"],
    )
    logger.info(f"✓ Updated memory importance: {updated.importance}")

    # Query memories
    memories = await service.get_memories(user_id="demo_user", limit=10)
    logger.info(f"✓ Found {len(memories)} memories for user")

    # ========================================================================
    # 2. Batch Operations
    # ========================================================================

    logger.info("\n" + "=" * 70)
    logger.info("2. BATCH OPERATIONS")
    logger.info("=" * 70)

    # Batch create
    batch_data = [
        {
            "text": "Python is a programming language",
            "user_id": "demo_user",
            "type": "fact",
            "importance": 7.0,
            "tags": ["programming", "python"],
        },
        {
            "text": "I prefer dark mode in IDEs",
            "user_id": "demo_user",
            "type": "preference",
            "importance": 6.0,
            "tags": ["preferences", "ide"],
        },
        {
            "text": "Learn FastAPI and async programming",
            "user_id": "demo_user",
            "type": "goal",
            "importance": 9.0,
            "tags": ["goals", "learning"],
        },
    ]

    created_memories = await service.batch_create_memories(batch_data)
    logger.info(f"✓ Batch created {len(created_memories)} memories")
    for mem in created_memories:
        logger.info(f"  - {mem.text} ({mem.type})")

    # Batch update
    updates = [
        {"memory_id": created_memories[0].id, "importance": 8.0},
        {"memory_id": created_memories[1].id, "importance": 7.0},
    ]
    updated_memories = await service.batch_update_memories(updates)
    logger.info(f"✓ Batch updated {len(updated_memories)} memories")

    # ========================================================================
    # 3. Hybrid Search with Customizable Weights
    # ========================================================================

    logger.info("\n" + "=" * 70)
    logger.info("3. HYBRID SEARCH")
    logger.info("=" * 70)

    # Rebuild BM25 index
    service.retriever.rebuild_bm25("demo_user")

    # Standard recall
    results = await service.recall_memories(
        query="programming languages",
        user_id="demo_user",
        k=5,
    )
    logger.info(f"✓ Standard recall found {len(results)} results:")
    for i, result in enumerate(results[:3], 1):
        logger.info(f"  {i}. {result.memory.text}")
        logger.info(
            f"     Score: {result.score:.3f} (sim={result.breakdown['sim']:.3f}, "
            f"rerank={result.breakdown['rerank']:.3f})"
        )

    # Recall with custom weights (emphasize recency)
    custom_weights = {
        "sim": 0.3,
        "rerank": 0.2,
        "recency": 0.4,  # Emphasize recent memories
        "importance": 0.1,
    }

    results_custom = await service.recall_memories(
        query="programming",
        user_id="demo_user",
        k=5,
        custom_weights=custom_weights,
    )
    logger.info(f"✓ Custom weighted recall found {len(results_custom)} results")
    logger.info("  Weights: sim=0.3, rerank=0.2, recency=0.4, importance=0.1")

    # ========================================================================
    # 4. Deduplication
    # ========================================================================

    logger.info("\n" + "=" * 70)
    logger.info("4. DEDUPLICATION")
    logger.info("=" * 70)

    # Create similar memories (duplicates)
    await service.create_memory(
        text="Rome is the capital of Italy",
        user_id="demo_user",
        check_duplicate=False,
    )
    await service.create_memory(
        text="Rome is the capital city of Italy",
        user_id="demo_user",
        check_duplicate=False,
    )

    # Run deduplication (dry run)
    dedup_result = await service.deduplicate_user_memories(
        user_id="demo_user",
        dry_run=True,
    )
    logger.info("✓ Deduplication analysis (dry run):")
    logger.info(f"  Total memories: {dedup_result['total_memories']}")
    logger.info(f"  Duplicates found: {dedup_result['duplicates_found']}")
    logger.info(f"  Would remove: {dedup_result['duplicates_found']} duplicates")

    # ========================================================================
    # 5. Consolidation
    # ========================================================================

    logger.info("\n" + "=" * 70)
    logger.info("5. CONSOLIDATION")
    logger.info("=" * 70)

    # Create similar memories to consolidate
    await service.create_memory(
        text="I like coffee in the morning",
        user_id="demo_user",
        memory_type="preference",
    )
    await service.create_memory(
        text="I enjoy drinking coffee in the morning",
        user_id="demo_user",
        memory_type="preference",
    )
    await service.create_memory(
        text="Morning coffee is my favorite",
        user_id="demo_user",
        memory_type="preference",
    )

    # Run consolidation (dry run)
    consolidate_result = await service.consolidate_memories(
        user_id="demo_user",
        similarity_threshold=0.75,
        dry_run=True,
    )
    logger.info("✓ Consolidation analysis (dry run):")
    logger.info(f"  Total memories: {consolidate_result['total_memories']}")
    logger.info(f"  Similar groups found: {consolidate_result['groups_found']}")
    logger.info(f"  Would consolidate: {consolidate_result['groups_found']} groups")

    # ========================================================================
    # 6. TTL and Expiration
    # ========================================================================

    logger.info("\n" + "=" * 70)
    logger.info("6. TTL & EXPIRATION")
    logger.info("=" * 70)

    # Create memory with TTL
    expiring_memory = await service.create_memory(
        text="This memory will expire in 7 days",
        user_id="demo_user",
        ttl_days=7,
    )
    logger.info("✓ Created memory with TTL:")
    logger.info(f"  Text: {expiring_memory.text}")
    logger.info(f"  Expires at: {expiring_memory.expires_at}")

    # Check expiration
    expired_count = await service.expire_memories(user_id="demo_user")
    logger.info(f"✓ Expired {expired_count} memories")

    # ========================================================================
    # 7. Redis Cache Statistics
    # ========================================================================

    logger.info("\n" + "=" * 70)
    logger.info("7. REDIS CACHE STATISTICS")
    logger.info("=" * 70)

    stats = await redis_store.get_stats()
    logger.info("✓ Redis cache stats:")
    logger.info(f"  Total keys: {stats['total_keys']}")
    logger.info(f"  Memory keys: {stats['memory_keys']}")
    logger.info(f"  User indices: {stats['user_indices']}")
    logger.info(f"  Tag indices: {stats['tag_indices']}")

    # ========================================================================
    # Final Summary
    # ========================================================================

    logger.info("\n" + "=" * 70)
    logger.info("DEMO COMPLETE")
    logger.info("=" * 70)

    final_memories = await service.get_memories(user_id="demo_user")
    logger.info(f"✓ Total memories for demo_user: {len(final_memories)}")
    logger.info("✓ All memory management features demonstrated successfully!")

    # Cleanup
    await redis_store.close()
    logger.info("\n✓ Resources cleaned up")


if __name__ == "__main__":
    asyncio.run(main())
