"""Celery tasks for distributed background processing."""

import logging
from datetime import datetime, timezone
from typing import Any, Optional, cast

from hippocampai.celery_app import celery_app
from hippocampai.config import Config
from hippocampai.embed.embedder import Embedder
from hippocampai.retrieval.rerank import Reranker
from hippocampai.services.memory_service import MemoryManagementService
from hippocampai.storage.redis_store import AsyncMemoryKVStore
from hippocampai.vector.qdrant_store import QdrantStore

logger = logging.getLogger(__name__)

# Lazy initialization of services
_services: dict[str, Any] = {}


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

        # Note: Redis store is required for memory service
        redis_store: Optional[AsyncMemoryKVStore] = None
        try:
            redis_store = AsyncMemoryKVStore(
                redis_url=config.redis_url,
                cache_ttl=config.redis_cache_ttl,
            )
        except Exception as e:
            logger.error(f"Redis store required but not available: {e}")
            raise RuntimeError("Redis store is required for memory service") from e

        _services["memory_service"] = MemoryManagementService(
            qdrant_store=qdrant_store,
            embedder=embedder,
            reranker=reranker,
            redis_store=redis_store,
        )

    return cast(MemoryManagementService, _services["memory_service"])


# ============================================================================
# Memory Operation Tasks
# ============================================================================


@celery_app.task(name="hippocampai.tasks.create_memory_task", bind=True, max_retries=3)
def create_memory_task(
    self: Any,
    text: str,
    user_id: str,
    memory_type: str = "fact",
    importance: Optional[float] = None,
    tags: Optional[list[str]] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
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

        async def _create() -> Any:
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
    self: Any,
    memories: list[dict[str, Any]],
    check_duplicates: bool = True,
) -> list[dict[str, Any]]:
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

        async def _batch_create() -> Any:
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
    self: Any,
    query: str,
    user_id: str,
    k: int = 5,
    filters: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
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

        async def _recall() -> Any:
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
    self: Any,
    memory_id: str,
    user_id: str,
    updates: dict[str, Any],
) -> dict[str, Any]:
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

        async def _update() -> Any:
            return await service.update_memory(
                memory_id=memory_id,
                **updates,
            )

        updated_memory = asyncio.run(_update())

        if not updated_memory:
            raise Exception(f"Memory {memory_id} not found or could not be updated")

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
    self: Any,
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

        async def _delete() -> Any:
            return await service.delete_memory(
                memory_id=memory_id,
                user_id=user_id,
            )

        success = asyncio.run(_delete())

        logger.info(f"Task {self.request.id}: Deleted memory {memory_id}")
        return cast(bool, success)

    except Exception as exc:
        logger.error(f"Task {self.request.id} failed: {exc}")
        raise self.retry(exc=exc, countdown=60)


# ============================================================================
# Scheduled Maintenance Tasks
# ============================================================================


@celery_app.task(name="hippocampai.tasks.deduplicate_all_memories", bind=True)
def deduplicate_all_memories(self: Any) -> dict[str, int]:
    """
    Run deduplication across all users' memories.

    Returns:
        Statistics about deduplicated memories
    """
    try:
        # Placeholder implementation for bulk deduplication
        # Future: Implement user listing and per-user deduplication
        logger.info("Starting global deduplication task")
        stats = {"deduplicated": 0, "kept": 0}

        logger.info(f"Task {self.request.id}: Deduplication completed - {stats}")
        return stats

    except Exception as exc:
        logger.error(f"Task {self.request.id} failed: {exc}")
        raise


@celery_app.task(name="hippocampai.tasks.consolidate_all_memories", bind=True)
def consolidate_all_memories(self: Any) -> dict[str, int]:
    """
    Run memory consolidation across all users.

    Returns:
        Statistics about consolidated memories
    """
    try:
        # Placeholder implementation for bulk consolidation
        # Future: Implement user listing and per-user consolidation
        logger.info("Starting global consolidation task")
        stats = {"consolidated": 0, "original": 0}

        logger.info(f"Task {self.request.id}: Consolidation completed - {stats}")
        return stats

    except Exception as exc:
        logger.error(f"Task {self.request.id} failed: {exc}")
        raise


@celery_app.task(name="hippocampai.tasks.cleanup_expired_memories", bind=True)
def cleanup_expired_memories(self: Any) -> dict[str, int]:
    """
    Clean up expired memories across all collections.

    Returns:
        Number of expired memories cleaned up
    """
    try:
        service = get_service()

        logger.info("Starting expired memories cleanup task")
        # Get all memories and check expiration
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
                    if expires_at and datetime.fromisoformat(expires_at) < datetime.now(
                        timezone.utc
                    ):
                        expired_ids.append(result["id"])

                if expired_ids:
                    service.qdrant.delete(collection, expired_ids)
                    count += len(expired_ids)
                    logger.info(f"Deleted {len(expired_ids)} expired memories from {collection}")

            except Exception as e:
                logger.error(f"Error cleaning up {collection}: {e}")

        stats = {"deleted": count}

        logger.info(
            f"Task {self.request.id}: Cleanup completed - {stats['deleted']} memories deleted"
        )
        return stats

    except Exception as exc:
        logger.error(f"Task {self.request.id} failed: {exc}")
        raise


@celery_app.task(name="hippocampai.tasks.decay_memory_importance", bind=True)
def decay_memory_importance(self: Any) -> dict[str, int]:
    """
    Apply importance decay to all memories based on their age.

    Returns:
        Statistics about decayed memories
    """
    try:
        # Placeholder implementation for importance decay
        # Future: Iterate through all memories and apply decay formula based on age
        logger.info("Starting importance decay task")
        stats = {"decayed": 0}

        logger.info(f"Task {self.request.id}: Decay completed - {stats}")
        return stats

    except Exception as exc:
        logger.error(f"Task {self.request.id} failed: {exc}")
        raise


@celery_app.task(name="hippocampai.tasks.create_collection_snapshots", bind=True)
def create_collection_snapshots(self: Any) -> dict[str, str]:
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


@celery_app.task(
    name="hippocampai.tasks.migrate_embeddings_task",
    bind=True,
    soft_time_limit=3600,
)
def migrate_embeddings_task(
    self: Any,
    migration_id: str,
) -> dict[str, Any]:
    """
    Re-embed all memories with a new embedding model.

    Runs as a long-lived Celery task with a 1-hour soft time limit.

    Args:
        migration_id: ID of the EmbeddingMigration to execute.

    Returns:
        Final migration status and counts.
    """
    try:
        from hippocampai.api.deps import get_memory_client

        client = get_memory_client()
        migration = client.migration_manager.run_migration(migration_id)

        logger.info(
            f"Task {self.request.id}: Migration {migration_id} finished "
            f"status={migration.status.value}, "
            f"migrated={migration.migrated_count}, failed={migration.failed_count}"
        )
        return {
            "migration_id": migration.id,
            "status": migration.status.value,
            "migrated_count": migration.migrated_count,
            "failed_count": migration.failed_count,
        }
    except Exception as exc:
        logger.error(f"Task {self.request.id} migration failed: {exc}")
        raise


@celery_app.task(name="hippocampai.tasks.consolidate_procedural_rules", bind=True)
def consolidate_procedural_rules(
    self: Any,
    user_id: str,
) -> dict[str, Any]:
    """
    Consolidate procedural rules for a user (merge redundant/contradicting rules).

    Args:
        user_id: User whose rules to consolidate.

    Returns:
        Statistics about consolidation.
    """
    try:
        from hippocampai.config import get_config
        from hippocampai.procedural.procedural_memory import ProceduralMemoryManager

        config = get_config()
        if not config.enable_procedural_memory:
            return {"status": "disabled"}

        manager = ProceduralMemoryManager(
            max_rules=config.procedural_rule_max_count,
        )
        rules = manager.consolidate_rules(user_id)

        logger.info(
            f"Task {self.request.id}: Consolidated procedural rules for {user_id}, "
            f"result={len(rules)} rules"
        )
        return {
            "user_id": user_id,
            "consolidated_count": len(rules),
        }
    except Exception as exc:
        logger.error(f"Task {self.request.id} failed: {exc}")
        raise


@celery_app.task(name="hippocampai.tasks.evaluate_prospective_triggers", bind=True)
def evaluate_prospective_triggers(self: Any) -> dict[str, Any]:
    """
    Evaluate time-based prospective memory intents.

    Returns:
        Statistics about triggered intents.
    """
    try:
        from hippocampai.config import get_config
        from hippocampai.prospective.prospective_memory import ProspectiveMemoryManager

        config = get_config()
        if not config.enable_prospective_memory:
            return {"status": "disabled"}

        manager = ProspectiveMemoryManager(
            max_intents_per_user=config.prospective_max_intents_per_user,
            eval_interval_seconds=config.prospective_eval_interval_seconds,
        )
        triggered = manager.evaluate_time_triggers()

        logger.info(
            f"Task {self.request.id}: Evaluated prospective triggers, "
            f"{len(triggered)} intent(s) fired"
        )
        return {
            "triggered_count": len(triggered),
            "triggered_ids": [i.id for i in triggered],
        }
    except Exception as exc:
        logger.error(f"Task {self.request.id} failed: {exc}")
        raise


@celery_app.task(name="hippocampai.tasks.expire_prospective_intents", bind=True)
def expire_prospective_intents(self: Any, user_id: Optional[str] = None) -> dict[str, Any]:
    """
    Expire stale prospective memory intents.

    Args:
        user_id: Optional user to scope expiration to.

    Returns:
        Number of expired intents.
    """
    try:
        from hippocampai.config import get_config
        from hippocampai.prospective.prospective_memory import ProspectiveMemoryManager

        config = get_config()
        if not config.enable_prospective_memory:
            return {"status": "disabled"}

        manager = ProspectiveMemoryManager(
            max_intents_per_user=config.prospective_max_intents_per_user,
        )
        expired_count = manager.expire_stale_intents(user_id=user_id)

        logger.info(
            f"Task {self.request.id}: Expired {expired_count} prospective intent(s)"
        )
        return {"expired_count": expired_count}
    except Exception as exc:
        logger.error(f"Task {self.request.id} failed: {exc}")
        raise


@celery_app.task(name="hippocampai.tasks.health_check_task", bind=True)
def health_check_task(self: Any) -> dict[str, Any]:
    """
    Perform health checks on all services.

    Returns:
        Health status of each service (dict with bool values or error string)
    """
    try:
        service = get_service()

        health: dict[str, Any] = {
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

                async def check_redis() -> Any:
                    if service.redis and service.redis.store._client:
                        return await service.redis.store._client.ping()
                    return False

                result = asyncio.run(check_redis())
                health["redis"] = bool(result)
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
