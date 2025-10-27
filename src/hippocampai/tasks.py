"""Celery tasks for distributed background processing."""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from hippocampai.celery_app import celery_app
from hippocampai.config import Config
from hippocampai.embed.embedder import Embedder
from hippocampai.retrieval.rerank import Reranker
from hippocampai.services.memory_service import MemoryManagementService
from hippocampai.storage.redis_store import AsyncMemoryKVStore
from hippocampai.vector.qdrant_store import QdrantStore

logger = logging.getLogger(__name__)

# Lazy initialization of services
_services: Dict[str, Any] = {}


def get_service() -> MemoryManagementService:
    """Get or create the MemoryManagementService instance (singleton pattern)."""
    if "memory_service" not in _services:
        config = Config()

        # Initialize components
        qdrant_store = QdrantStore(
            url=config.qdrant_url,
            collection_facts=config.collection_facts,
            collection_prefs=config.collection_prefs,
            dimension=config.embed_dimension,
            hnsw_m=config.hnsw_m,
            ef_construction=config.ef_construction,
            ef_search=config.ef_search,
        )

        embedder = Embedder(
            model_name=config.embed_model,
            quantized=config.embed_quantized,
            batch_size=config.embed_batch_size,
        )

        reranker = Reranker(model_name=config.reranker_model)

        # Note: Redis store is optional for background tasks
        redis_store = None
        try:
            redis_store = AsyncMemoryKVStore(
                redis_url=config.redis_url,
                cache_ttl=config.redis_cache_ttl,
            )
        except Exception as e:
            logger.warning(f"Redis store not available for background tasks: {e}")

        _services["memory_service"] = MemoryManagementService(
            qdrant_store=qdrant_store,
            embedder=embedder,
            reranker=reranker,
            redis_store=redis_store,
        )

    return _services["memory_service"]


# ============================================================================
# Memory Operation Tasks
# ============================================================================

@celery_app.task(name="hippocampai.tasks.create_memory_task", bind=True, max_retries=3)
def create_memory_task(
    self,
    text: str,
    user_id: str,
    memory_type: str = "fact",
    importance: Optional[float] = None,
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Create a memory asynchronously via Celery.

    Args:
        text: Memory text content
        user_id: User ID
        memory_type: Type of memory (fact, preference, goal, etc.)
        importance: Importance score (0-10)
        tags: List of tags
        metadata: Additional metadata

    Returns:
        Created memory as dict
    """
    try:
        service = get_service()

        # Since service methods are async, we need to handle this properly
        # For now, we'll use the synchronous wrapper
        import asyncio

        async def _create():
            return await service.create_memory(
                text=text,
                user_id=user_id,
                memory_type=memory_type,
                importance=importance,
                tags=tags or [],
                metadata=metadata or {},
            )

        memory = asyncio.run(_create())

        logger.info(f"Task {self.request.id}: Created memory {memory.id} for user {user_id}")
        return {
            "id": memory.id,
            "text": memory.text,
            "user_id": memory.user_id,
            "type": memory.type.value,
            "importance": memory.importance,
        }

    except Exception as exc:
        logger.error(f"Task {self.request.id} failed: {exc}")
        raise self.retry(exc=exc, countdown=60)


@celery_app.task(name="hippocampai.tasks.batch_create_memories_task", bind=True, max_retries=3)
def batch_create_memories_task(
    self,
    memories: List[Dict[str, Any]],
    check_duplicates: bool = True,
) -> List[Dict[str, Any]]:
    """
    Batch create memories asynchronously.

    Args:
        memories: List of memory data dictionaries
        check_duplicates: Whether to check for duplicates

    Returns:
        List of created memories
    """
    try:
        service = get_service()

        import asyncio

        async def _batch_create():
            return await service.batch_create_memories(
                memories=memories,
                check_duplicates=check_duplicates,
            )

        created_memories = asyncio.run(_batch_create())

        logger.info(f"Task {self.request.id}: Created {len(created_memories)} memories in batch")
        return [
            {
                "id": m.id,
                "text": m.text,
                "user_id": m.user_id,
                "type": m.type.value,
            }
            for m in created_memories
        ]

    except Exception as exc:
        logger.error(f"Task {self.request.id} failed: {exc}")
        raise self.retry(exc=exc, countdown=60)


@celery_app.task(name="hippocampai.tasks.recall_memories_task", bind=True)
def recall_memories_task(
    self,
    query: str,
    user_id: str,
    k: int = 5,
    filters: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Recall memories asynchronously.

    Args:
        query: Search query
        user_id: User ID
        k: Number of results
        filters: Optional filters

    Returns:
        List of recalled memories with scores
    """
    try:
        service = get_service()

        import asyncio

        async def _recall():
            return await service.recall_memories(
                query=query,
                user_id=user_id,
                k=k,
                filters=filters,
            )

        results = asyncio.run(_recall())

        logger.info(f"Task {self.request.id}: Recalled {len(results)} memories for user {user_id}")
        return [
            {
                "memory": {
                    "id": r.memory.id,
                    "text": r.memory.text,
                    "type": r.memory.type.value,
                },
                "score": r.score,
            }
            for r in results
        ]

    except Exception as exc:
        logger.error(f"Task {self.request.id} failed: {exc}")
        raise


@celery_app.task(name="hippocampai.tasks.update_memory_task", bind=True, max_retries=3)
def update_memory_task(
    self,
    memory_id: str,
    user_id: str,
    updates: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Update a memory asynchronously.

    Args:
        memory_id: Memory ID
        user_id: User ID
        updates: Dictionary of fields to update

    Returns:
        Updated memory
    """
    try:
        service = get_service()

        import asyncio

        async def _update():
            return await service.update_memory(
                memory_id=memory_id,
                user_id=user_id,
                **updates,
            )

        updated_memory = asyncio.run(_update())

        logger.info(f"Task {self.request.id}: Updated memory {memory_id}")
        return {
            "id": updated_memory.id,
            "text": updated_memory.text,
            "importance": updated_memory.importance,
        }

    except Exception as exc:
        logger.error(f"Task {self.request.id} failed: {exc}")
        raise self.retry(exc=exc, countdown=60)


@celery_app.task(name="hippocampai.tasks.delete_memory_task", bind=True, max_retries=3)
def delete_memory_task(
    self,
    memory_id: str,
    user_id: str,
) -> bool:
    """
    Delete a memory asynchronously.

    Args:
        memory_id: Memory ID
        user_id: User ID

    Returns:
        Success status
    """
    try:
        service = get_service()

        import asyncio

        async def _delete():
            return await service.delete_memory(
                memory_id=memory_id,
                user_id=user_id,
            )

        success = asyncio.run(_delete())

        logger.info(f"Task {self.request.id}: Deleted memory {memory_id}")
        return success

    except Exception as exc:
        logger.error(f"Task {self.request.id} failed: {exc}")
        raise self.retry(exc=exc, countdown=60)


# ============================================================================
# Scheduled Maintenance Tasks
# ============================================================================

@celery_app.task(name="hippocampai.tasks.deduplicate_all_memories", bind=True)
def deduplicate_all_memories(self) -> Dict[str, int]:
    """
    Run deduplication across all users' memories.

    Returns:
        Statistics about deduplicated memories
    """
    try:
        import asyncio

        async def _deduplicate():
            # Get all unique user IDs
            # This would need a method to list all users
            # For now, this is a placeholder
            # TODO: Use get_service() when implementing bulk deduplication logic
            logger.info("Starting global deduplication task")
            return {"deduplicated": 0, "kept": 0}

        stats = asyncio.run(_deduplicate())

        logger.info(f"Task {self.request.id}: Deduplication completed - {stats}")
        return stats

    except Exception as exc:
        logger.error(f"Task {self.request.id} failed: {exc}")
        raise


@celery_app.task(name="hippocampai.tasks.consolidate_all_memories", bind=True)
def consolidate_all_memories(self) -> Dict[str, int]:
    """
    Run memory consolidation across all users.

    Returns:
        Statistics about consolidated memories
    """
    try:
        import asyncio

        async def _consolidate():
            # TODO: Use get_service() when implementing bulk consolidation logic
            logger.info("Starting global consolidation task")
            return {"consolidated": 0, "original": 0}

        stats = asyncio.run(_consolidate())

        logger.info(f"Task {self.request.id}: Consolidation completed - {stats}")
        return stats

    except Exception as exc:
        logger.error(f"Task {self.request.id} failed: {exc}")
        raise


@celery_app.task(name="hippocampai.tasks.cleanup_expired_memories", bind=True)
def cleanup_expired_memories(self) -> Dict[str, int]:
    """
    Clean up expired memories across all collections.

    Returns:
        Number of expired memories cleaned up
    """
    try:
        service = get_service()

        import asyncio

        async def _cleanup():
            logger.info("Starting expired memories cleanup task")
            # Get all memories and check expiration
            # This is a simplified version
            count = 0
            for collection in [service.qdrant.collection_facts, service.qdrant.collection_prefs]:
                try:
                    # Scroll through all memories
                    results = service.qdrant.scroll(
                        collection_name=collection,
                        limit=1000,
                    )

                    expired_ids = []
                    for result in results:
                        expires_at = result.get("payload", {}).get("expires_at")
                        if expires_at and datetime.fromisoformat(expires_at) < datetime.now(timezone.utc):
                            expired_ids.append(result["id"])

                    if expired_ids:
                        service.qdrant.delete(collection, expired_ids)
                        count += len(expired_ids)
                        logger.info(f"Deleted {len(expired_ids)} expired memories from {collection}")

                except Exception as e:
                    logger.error(f"Error cleaning up {collection}: {e}")

            return {"deleted": count}

        stats = asyncio.run(_cleanup())

        logger.info(f"Task {self.request.id}: Cleanup completed - {stats['deleted']} memories deleted")
        return stats

    except Exception as exc:
        logger.error(f"Task {self.request.id} failed: {exc}")
        raise


@celery_app.task(name="hippocampai.tasks.decay_memory_importance", bind=True)
def decay_memory_importance(self) -> Dict[str, int]:
    """
    Apply importance decay to all memories based on their age.

    Returns:
        Statistics about decayed memories
    """
    try:
        import asyncio

        async def _decay():
            # TODO: Use get_service() when implementing importance decay logic
            # This would iterate through all memories and apply decay formula
            logger.info("Starting importance decay task")
            return {"decayed": 0}

        stats = asyncio.run(_decay())

        logger.info(f"Task {self.request.id}: Decay completed - {stats}")
        return stats

    except Exception as exc:
        logger.error(f"Task {self.request.id} failed: {exc}")
        raise


@celery_app.task(name="hippocampai.tasks.create_collection_snapshots", bind=True)
def create_collection_snapshots(self) -> Dict[str, str]:
    """
    Create snapshots of all Qdrant collections.

    Returns:
        Snapshot names created
    """
    try:
        service = get_service()

        snapshots = {}

        for collection in [service.qdrant.collection_facts, service.qdrant.collection_prefs]:
            try:
                snapshot_name = service.qdrant.create_snapshot(collection)
                snapshots[collection] = snapshot_name
                logger.info(f"Created snapshot {snapshot_name} for {collection}")
            except Exception as e:
                logger.error(f"Error creating snapshot for {collection}: {e}")

        logger.info(f"Task {self.request.id}: Snapshots created - {snapshots}")
        return snapshots

    except Exception as exc:
        logger.error(f"Task {self.request.id} failed: {exc}")
        raise


# ============================================================================
# Utility Tasks
# ============================================================================

@celery_app.task(name="hippocampai.tasks.health_check_task", bind=True)
def health_check_task(self) -> Dict[str, bool]:
    """
    Perform health checks on all services.

    Returns:
        Health status of each service
    """
    try:
        service = get_service()

        health = {
            "qdrant": False,
            "redis": False,
            "embedder": False,
        }

        # Check Qdrant
        try:
            service.qdrant.client.get_collections()
            health["qdrant"] = True
        except Exception as e:
            logger.error(f"Qdrant health check failed: {e}")

        # Check Redis
        if service.redis:
            try:
                import asyncio
                asyncio.run(service.redis.backend.ping())
                health["redis"] = True
            except Exception as e:
                logger.error(f"Redis health check failed: {e}")

        # Check Embedder
        try:
            service.embedder.encode(["test"])
            health["embedder"] = True
        except Exception as e:
            logger.error(f"Embedder health check failed: {e}")

        logger.info(f"Task {self.request.id}: Health check completed - {health}")
        return health

    except Exception as exc:
        logger.error(f"Task {self.request.id} failed: {exc}")
        return {"error": str(exc)}
