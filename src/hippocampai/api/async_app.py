"""Async FastAPI application with comprehensive memory management APIs."""

import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Optional

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from hippocampai.adapters.llm_base import BaseLLM
from hippocampai.adapters.provider_anthropic import AnthropicLLM
from hippocampai.adapters.provider_groq import GroqLLM
from hippocampai.adapters.provider_ollama import OllamaLLM
from hippocampai.adapters.provider_openai import OpenAILLM
from hippocampai.config import get_config
from hippocampai.embed.embedder import Embedder
from hippocampai.models.memory import Memory, RetrievalResult
from hippocampai.retrieval.rerank import Reranker
from hippocampai.services.background_tasks import BackgroundTaskManager
from hippocampai.services.memory_service import MemoryManagementService
from hippocampai.storage.redis_store import AsyncMemoryKVStore
from hippocampai.vector.qdrant_store import QdrantStore

logger = logging.getLogger(__name__)

# Global service instances
_service: Optional[MemoryManagementService] = None
_redis_store: Optional[AsyncMemoryKVStore] = None
_background_tasks: Optional[BackgroundTaskManager] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    global _service, _redis_store, _background_tasks

    # Startup
    config = get_config()
    logger.info("Initializing HippocampAI services...")

    # Initialize components
    qdrant = QdrantStore(
        url=config.qdrant_url,
        collection_facts=config.collection_facts,
        collection_prefs=config.collection_prefs,
    )
    embedder = Embedder(
        model_name=config.embed_model,
        quantized=config.embed_quantized,
        batch_size=config.embed_batch_size,
    )
    reranker = Reranker(model_name=config.reranker_model)

    _redis_store = AsyncMemoryKVStore(
        redis_url=config.redis_url,
        cache_ttl=config.redis_cache_ttl,
        max_connections=config.redis_max_connections,
        min_idle=config.redis_min_idle,
    )
    await _redis_store.connect()
    logger.info(
        f"Redis connected with connection pool (max={config.redis_max_connections}, min_idle={config.redis_min_idle})"
    )

    # Initialize LLM (optional, for extraction and consolidation)
    llm: Optional[BaseLLM] = None
    if config.llm_provider == "ollama":
        llm = OllamaLLM(model=config.llm_model, base_url=config.llm_base_url)
        logger.info(f"Initialized Ollama LLM: {config.llm_model}")
    elif config.llm_provider == "openai" and config.allow_cloud:
        import os

        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            llm = OpenAILLM(api_key=api_key, model=config.llm_model)
            logger.info(f"Initialized OpenAI LLM: {config.llm_model}")
    elif config.llm_provider == "groq" and config.allow_cloud:
        import os

        api_key = os.getenv("GROQ_API_KEY")
        if api_key:
            llm = GroqLLM(api_key=api_key, model=config.llm_model)
            logger.info(f"Initialized Groq LLM: {config.llm_model}")
    elif config.llm_provider == "anthropic" and config.allow_cloud:
        import os

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if api_key:
            llm = AnthropicLLM(api_key=api_key, model=config.llm_model)
            logger.info(f"Initialized Anthropic LLM: {config.llm_model}")

    if llm is None:
        logger.warning(
            "No LLM configured. Conversation extraction and LLM-based consolidation will not be available."
        )

    # Initialize service
    _service = MemoryManagementService(
        qdrant_store=qdrant,
        embedder=embedder,
        reranker=reranker,
        redis_store=_redis_store,
        llm=llm,
        weights=config.get_weights(),
        half_lives=config.get_half_lives(),
    )

    # Initialize background tasks (if enabled)
    if config.enable_background_tasks:
        _background_tasks = BackgroundTaskManager(
            service=_service,
            dedup_interval_hours=config.dedup_interval_hours,
            consolidation_interval_hours=config.consolidation_interval_hours,
            expiration_interval_hours=config.expiration_interval_hours,
            auto_dedup_enabled=config.auto_dedup_enabled,
            auto_consolidation_enabled=config.auto_consolidation_enabled,
            dedup_threshold=config.dedup_threshold,
            consolidation_threshold=config.consolidation_threshold,
        )
        await _background_tasks.start()
        logger.info("Background tasks enabled and started")
    else:
        logger.info("Background tasks disabled by configuration")

    logger.info("HippocampAI services initialized successfully")
    yield

    # Shutdown
    logger.info("Shutting down HippocampAI services...")
    if _background_tasks:
        await _background_tasks.stop()
    if _redis_store:
        await _redis_store.close()


app = FastAPI(
    title="HippocampAI API",
    description="Autonomous memory engine with hybrid retrieval, batch operations, deduplication, and consolidation",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include intelligence routes
try:
    from hippocampai.api.intelligence_routes import router as intelligence_router

    app.include_router(intelligence_router)
    logger.info("Intelligence routes registered successfully")
except ImportError as e:
    logger.warning(f"Could not load intelligence routes: {e}")

# Include Celery routes for background task processing
try:
    from hippocampai.api.celery_routes import router as celery_router

    app.include_router(celery_router, prefix="/celery", tags=["celery"])
    logger.info("Celery routes registered successfully")
except ImportError as e:
    logger.warning(f"Could not load celery routes: {e}")


# Dependency to get service
def get_service() -> MemoryManagementService:
    """Get memory management service."""
    if _service is None:
        raise HTTPException(status_code=500, detail="Service not initialized")
    return _service


# ============================================================================
# Request/Response Models
# ============================================================================


class MemoryCreate(BaseModel):
    text: str
    user_id: str
    session_id: Optional[str] = None
    type: str = "fact"
    importance: Optional[float] = Field(default=5.0, ge=0.0, le=10.0)
    tags: Optional[list[str]] = None
    ttl_days: Optional[int] = None
    metadata: Optional[dict[str, Any]] = None
    check_duplicate: bool = True


class MemoryUpdate(BaseModel):
    memory_id: str
    text: Optional[str] = None
    importance: Optional[float] = Field(default=None, ge=0.0, le=10.0)
    tags: Optional[list[str]] = None
    metadata: Optional[dict[str, Any]] = None
    expires_at: Optional[datetime] = None


class MemoryDelete(BaseModel):
    memory_id: str
    user_id: Optional[str] = None


class MemoryQuery(BaseModel):
    user_id: str
    filters: Optional[dict[str, Any]] = None
    limit: int = Field(default=100, le=1000)
    # Advanced filtering
    memory_type: Optional[str] = None
    tags: Optional[list[str]] = None
    importance_min: Optional[float] = Field(default=None, ge=0.0, le=10.0)
    importance_max: Optional[float] = Field(default=None, ge=0.0, le=10.0)
    created_after: Optional[datetime] = None
    created_before: Optional[datetime] = None
    updated_after: Optional[datetime] = None
    updated_before: Optional[datetime] = None
    search_text: Optional[str] = None  # Text search in memory content


class RecallRequest(BaseModel):
    query: str
    user_id: str
    session_id: Optional[str] = None
    k: int = Field(default=5, ge=1, le=100)
    filters: Optional[dict[str, Any]] = None
    custom_weights: Optional[dict[str, float]] = None


class BatchCreateRequest(BaseModel):
    memories: list[MemoryCreate]
    check_duplicates: bool = True


class BatchUpdateRequest(BaseModel):
    updates: list[MemoryUpdate]


class BatchDeleteRequest(BaseModel):
    memory_ids: list[str]
    user_id: Optional[str] = None


class BatchGetRequest(BaseModel):
    memory_ids: list[str]


class AnalyticsRequest(BaseModel):
    user_id: str


class ExtractRequest(BaseModel):
    conversation: str
    user_id: str
    session_id: Optional[str] = None


class DeduplicateRequest(BaseModel):
    user_id: str
    dry_run: bool = True


class ConsolidateRequest(BaseModel):
    user_id: str
    similarity_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    dry_run: bool = True


class ExpireRequest(BaseModel):
    user_id: Optional[str] = None


# ============================================================================
# Health & Info
# ============================================================================


@app.get("/healthz")
@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok", "service": "hippocampai", "version": "2.0.0"}


@app.get("/stats")
async def get_stats(service: MemoryManagementService = Depends(get_service)) -> dict[str, Any]:
    """Get Redis cache statistics."""
    stats = await service.redis.get_stats()
    return stats


# ============================================================================
# CRUD Operations
# ============================================================================


@app.post("/v1/memories", response_model=Memory, status_code=status.HTTP_201_CREATED)
async def create_memory(
    request: MemoryCreate, service: MemoryManagementService = Depends(get_service)
) -> Memory:
    """Create a new memory with optional deduplication check."""
    try:
        memory = await service.create_memory(
            text=request.text,
            user_id=request.user_id,
            session_id=request.session_id,
            memory_type=request.type,
            importance=request.importance,
            tags=request.tags,
            ttl_days=request.ttl_days,
            metadata=request.metadata,
            check_duplicate=request.check_duplicate,
        )
        return memory
    except Exception as e:
        logger.error(f"Create memory failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/memories/{memory_id}", response_model=Memory)
async def get_memory(
    memory_id: str, service: MemoryManagementService = Depends(get_service)
) -> Memory:
    """Get a memory by ID."""
    try:
        memory = await service.get_memory(memory_id)
        if not memory:
            raise HTTPException(status_code=404, detail="Memory not found")
        return memory
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get memory failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.patch("/v1/memories/{memory_id}", response_model=Memory)
async def update_memory(
    memory_id: str,
    request: MemoryUpdate,
    service: MemoryManagementService = Depends(get_service),
) -> Memory:
    """Update an existing memory."""
    try:
        memory = await service.update_memory(
            memory_id=memory_id,
            text=request.text,
            importance=request.importance,
            tags=request.tags,
            metadata=request.metadata,
            expires_at=request.expires_at,
        )
        if not memory:
            raise HTTPException(status_code=404, detail="Memory not found")
        return memory
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Update memory failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/v1/memories/{memory_id}")
async def delete_memory(
    memory_id: str,
    user_id: Optional[str] = None,
    service: MemoryManagementService = Depends(get_service),
) -> dict[str, Any]:
    """Delete a memory."""
    try:
        deleted = await service.delete_memory(memory_id, user_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Memory not found or unauthorized")
        return {"success": True, "memory_id": memory_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete memory failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/memories/query", response_model=list[Memory])
async def query_memories(
    request: MemoryQuery, service: MemoryManagementService = Depends(get_service)
):
    """Query memories with advanced filters (type, tags, date range, importance threshold, text search)."""
    try:
        memories = await service.get_memories(
            user_id=request.user_id,
            filters=request.filters,
            limit=request.limit,
            memory_type=request.memory_type,
            tags=request.tags,
            importance_min=request.importance_min,
            importance_max=request.importance_max,
            created_after=request.created_after,
            created_before=request.created_before,
            updated_after=request.updated_after,
            updated_before=request.updated_before,
            search_text=request.search_text,
        )
        return memories
    except Exception as e:
        logger.error(f"Query memories failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Batch Operations
# ============================================================================


@app.post("/v1/memories/batch", response_model=list[Memory], status_code=status.HTTP_201_CREATED)
async def batch_create_memories(
    request: BatchCreateRequest, service: MemoryManagementService = Depends(get_service)
):
    """Batch create multiple memories."""
    try:
        memories_data = [mem.model_dump() for mem in request.memories]
        memories = await service.batch_create_memories(
            memories_data, check_duplicates=request.check_duplicates
        )
        return memories
    except Exception as e:
        logger.error(f"Batch create failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.patch("/v1/memories/batch", response_model=list[Memory])
async def batch_update_memories(
    request: BatchUpdateRequest, service: MemoryManagementService = Depends(get_service)
):
    """Batch update multiple memories."""
    try:
        updates_data = [upd.model_dump() for upd in request.updates]
        memories = await service.batch_update_memories(updates_data)
        return [m for m in memories if m is not None]
    except Exception as e:
        logger.error(f"Batch update failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/v1/memories/batch")
async def batch_delete_memories(
    request: BatchDeleteRequest, service: MemoryManagementService = Depends(get_service)
):
    """Batch delete multiple memories."""
    try:
        results = await service.batch_delete_memories(request.memory_ids, request.user_id)
        return {
            "success": True,
            "deleted_count": sum(results.values()),
            "total": len(request.memory_ids),
            "results": results,
        }
    except Exception as e:
        logger.error(f"Batch delete failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/memories/batch/get", response_model=list[Memory])
async def batch_get_memories(
    request: BatchGetRequest, service: MemoryManagementService = Depends(get_service)
):
    """Batch get multiple memories by IDs."""
    try:
        memories = []
        for memory_id in request.memory_ids:
            memory = await service.get_memory(memory_id)
            if memory:
                memories.append(memory)
        return memories
    except Exception as e:
        logger.error(f"Batch get failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Retrieval & Search
# ============================================================================


@app.post("/v1/memories/recall", response_model=list[RetrievalResult])
async def recall_memories(
    request: RecallRequest, service: MemoryManagementService = Depends(get_service)
):
    """Recall memories using hybrid search with customizable weights."""
    try:
        results = await service.recall_memories(
            query=request.query,
            user_id=request.user_id,
            session_id=request.session_id,
            k=request.k,
            filters=request.filters,
            custom_weights=request.custom_weights,
        )
        return results
    except Exception as e:
        logger.error(f"Recall failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Extraction
# ============================================================================


@app.post("/v1/memories/extract", response_model=list[Memory])
async def extract_from_conversation(
    request: ExtractRequest, service: MemoryManagementService = Depends(get_service)
):
    """Extract memories from conversation logs."""
    try:
        memories = await service.extract_from_conversation(
            conversation=request.conversation,
            user_id=request.user_id,
            session_id=request.session_id,
        )
        return memories
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Extract failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Deduplication & Consolidation
# ============================================================================


@app.post("/v1/memories/deduplicate")
async def deduplicate_memories(
    request: DeduplicateRequest, service: MemoryManagementService = Depends(get_service)
):
    """Deduplicate memories for a user."""
    try:
        result = await service.deduplicate_user_memories(
            user_id=request.user_id, dry_run=request.dry_run
        )
        return result
    except Exception as e:
        logger.error(f"Deduplication failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/memories/consolidate")
async def consolidate_memories(
    request: ConsolidateRequest, service: MemoryManagementService = Depends(get_service)
):
    """Consolidate similar memories for a user."""
    try:
        result = await service.consolidate_memories(
            user_id=request.user_id,
            similarity_threshold=request.similarity_threshold,
            dry_run=request.dry_run,
        )
        return result
    except Exception as e:
        logger.error(f"Consolidation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Maintenance
# ============================================================================


@app.post("/v1/memories/expire")
async def expire_memories(
    request: ExpireRequest, service: MemoryManagementService = Depends(get_service)
):
    """Expire memories based on TTL."""
    try:
        expired_count = await service.expire_memories(user_id=request.user_id)
        return {"success": True, "expired_count": expired_count}
    except Exception as e:
        logger.error(f"Expire failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/memories/cleanup")
async def cleanup_expired_memories(service: MemoryManagementService = Depends(get_service)):
    """Cleanup expired memories (alias for expire endpoint)."""
    try:
        expired_count = await service.expire_memories(user_id=None)
        return {"success": True, "deleted_count": expired_count}
    except Exception as e:
        logger.error(f"Cleanup failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/memories/analytics")
async def get_memory_analytics(
    user_id: str, service: MemoryManagementService = Depends(get_service)
):
    """Get analytics for user's memories."""
    try:
        # Get all memories for user
        memories = await service.get_memories(user_id=user_id, filters={}, limit=10000)

        if not memories:
            return {
                "total_memories": 0,
                "avg_importance": 0.0,
                "top_tags": [],
                "top_entities": [],
                "memory_types": {},
            }

        # Calculate analytics
        total = len(memories)
        avg_importance = sum(m.importance for m in memories) / total if total > 0 else 0.0

        # Count tags
        tag_counts: dict[str, int] = {}
        for memory in memories:
            for tag in memory.tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        top_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10]

        # Count entities
        entity_counts: dict[str, int] = {}
        for memory in memories:
            if memory.entities:
                for entity_type, entities in memory.entities.items():
                    for entity in entities:
                        entity_counts[entity] = entity_counts.get(entity, 0) + 1
        top_entities = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)[:10]

        # Count memory types
        type_counts: dict[str, int] = {}
        for memory in memories:
            mem_type = memory.memory_type.value if memory.memory_type else "unknown"
            type_counts[mem_type] = type_counts.get(mem_type, 0) + 1

        return {
            "total_memories": total,
            "avg_importance": round(avg_importance, 2),
            "top_tags": [{"tag": tag, "count": count} for tag, count in top_tags],
            "top_entities": [{"entity": entity, "count": count} for entity, count in top_entities],
            "memory_types": type_counts,
        }
    except Exception as e:
        logger.error(f"Analytics failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Background Tasks Management
# ============================================================================


# Dependency to get background task manager
def get_background_tasks() -> BackgroundTaskManager:
    """Get background task manager."""
    if _background_tasks is None:
        raise HTTPException(status_code=500, detail="Background tasks not initialized")
    return _background_tasks


@app.get("/v1/background/status")
async def get_background_status(tasks: BackgroundTaskManager = Depends(get_background_tasks)):
    """Get status of background tasks."""
    return tasks.get_status()


@app.post("/v1/background/dedup/trigger")
async def trigger_background_dedup(
    user_id: str,
    dry_run: bool = True,
    tasks: BackgroundTaskManager = Depends(get_background_tasks),
):
    """Manually trigger deduplication for a user via background task manager."""
    try:
        result = await tasks.trigger_deduplication(user_id=user_id, dry_run=dry_run)
        return result
    except Exception as e:
        logger.error(f"Background dedup trigger failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/background/consolidate/trigger")
async def trigger_background_consolidate(
    user_id: str,
    dry_run: bool = True,
    threshold: Optional[float] = None,
    tasks: BackgroundTaskManager = Depends(get_background_tasks),
):
    """Manually trigger consolidation for a user via background task manager."""
    try:
        result = await tasks.trigger_consolidation(
            user_id=user_id, dry_run=dry_run, threshold=threshold
        )
        return result
    except Exception as e:
        logger.error(f"Background consolidate trigger failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Legacy Endpoints (for backward compatibility)
# ============================================================================


@app.post("/v1/memories:remember", response_model=Memory)
async def remember(request: MemoryCreate, service: MemoryManagementService = Depends(get_service)):
    """Legacy endpoint: Create a memory."""
    return await create_memory(request, service)


@app.post("/v1/memories:recall", response_model=list[RetrievalResult])
async def recall(request: RecallRequest, service: MemoryManagementService = Depends(get_service)):
    """Legacy endpoint: Recall memories."""
    return await recall_memories(request, service)


@app.post("/v1/memories:extract", response_model=list[Memory])
async def extract(request: ExtractRequest, service: MemoryManagementService = Depends(get_service)):
    """Legacy endpoint: Extract from conversation."""
    return await extract_from_conversation(request, service)


# ============================================================================
# Server
# ============================================================================


def run_server(host: str = "127.0.0.1", port: int = 8000):
    """Run FastAPI server."""
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
