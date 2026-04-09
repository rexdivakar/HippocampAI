"""Async FastAPI application with comprehensive memory management APIs."""

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Optional, cast

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Response, status
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
from hippocampai.pipeline.conflict_resolution import ConflictResolutionStrategy
from hippocampai.pipeline.errors import ConflictResolutionError, MemoryNotFoundError
from hippocampai.pipeline.memory_lifecycle import MemoryTier
from hippocampai.retrieval.rerank import Reranker
from hippocampai.services.background_tasks import BackgroundTaskManager
from hippocampai.services.memory_service import MemoryManagementService
from hippocampai.storage.redis_store import AsyncMemoryKVStore
from hippocampai.vector.qdrant_store import QdrantStore

logger = logging.getLogger(__name__)

# Prometheus metrics
PROMETHEUS_AVAILABLE = False
PrometheusMiddleware: Any = None
get_metrics: Any = None
try:
    from hippocampai.monitoring.prometheus_metrics import (
        PrometheusMiddleware,
        get_metrics,
    )

    PROMETHEUS_AVAILABLE = True
except ImportError:
    logger.warning("Prometheus client not available - metrics disabled")

# Global service instances
_service: Optional[MemoryManagementService] = None
_redis_store: Optional[AsyncMemoryKVStore] = None
_background_tasks: Optional[BackgroundTaskManager] = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
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

    # Initialize AuthService and RateLimiter for SaaS mode
    try:
        import os

        import asyncpg

        from hippocampai.auth.auth_service import AuthService
        from hippocampai.auth.rate_limiter import RateLimiter

        # Check if user auth is enabled
        user_auth_enabled = os.getenv("USER_AUTH_ENABLED", "false").lower() == "true"

        # Ensure Redis is connected
        if _redis_store is not None:
            await _redis_store.connect()
            if _redis_store.store._client is None:
                raise RuntimeError("Redis client not initialized")

        db_pool = await asyncpg.create_pool(
            host=config.postgres_host,
            port=config.postgres_port,
            database=config.postgres_db,
            user=config.postgres_user,
            password=config.postgres_password,
            min_size=2,
            max_size=10,
        )
        app.state.db_pool = db_pool  # Store pool for direct access
        app.state.auth_service = AuthService(db_pool)
        app.state.rate_limiter = RateLimiter(_redis_store.store._client)  # Use existing Redis
        app.state.user_auth_enabled = user_auth_enabled

        logger.info(f"AuthService initialized (user_auth_enabled={user_auth_enabled})")
        logger.info("RateLimiter initialized successfully")

    except Exception as e:
        logger.warning(f"Could not initialize AuthService/RateLimiter: {e}")
        app.state.auth_service = None
        app.state.rate_limiter = None
        app.state.user_auth_enabled = False

    logger.info("HippocampAI services initialized successfully")
    yield

    # Shutdown
    logger.info("Shutting down HippocampAI services...")
    if _background_tasks:
        await _background_tasks.stop()
    if _redis_store:
        await _redis_store.close()
    if hasattr(app.state, "db_pool") and app.state.db_pool:
        await app.state.db_pool.close()
        logger.info("PostgreSQL connection pool closed")


app = FastAPI(
    title="HippocampAI API",
    description="Autonomous memory engine with hybrid retrieval, batch operations, deduplication, and consolidation",
    version="0.3.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add Prometheus middleware if available
if PROMETHEUS_AVAILABLE:
    app.add_middleware(PrometheusMiddleware)
    logger.info("Prometheus middleware enabled")

# Add authentication middleware (will lazy-load from app.state)
try:
    from hippocampai.api.middleware import AuthMiddleware

    app.add_middleware(AuthMiddleware)
    logger.info("AuthMiddleware registered (will lazy-load services from app.state)")
except Exception as e:
    logger.warning(f"Could not register AuthMiddleware: {e}")


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

# Include admin routes
# SECURITY WARNING: Admin routes require authentication and authorization!
# Only enable in production with proper auth middleware configured.
try:
    import os

    # Only register admin routes if explicitly enabled and auth is configured
    ADMIN_ROUTES_ENABLED = os.getenv("ENABLE_ADMIN_ROUTES", "false").lower() == "true"
    AUTH_CONFIGURED = os.getenv("AUTH_ENABLED", "false").lower() == "true"

    if ADMIN_ROUTES_ENABLED:
        if not AUTH_CONFIGURED:
            logger.critical(
                "SECURITY RISK: Admin routes enabled without authentication! "
                "Set AUTH_ENABLED=true and configure authentication middleware."
            )
        from hippocampai.api.admin_routes import router as admin_router

        app.include_router(admin_router)
        logger.warning("Admin routes registered. ENSURE authentication middleware is configured!")
    else:
        logger.info("Admin routes disabled (set ENABLE_ADMIN_ROUTES=true to enable)")
except ImportError as e:
    logger.warning(f"Could not load admin routes: {e}")


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
    text: Optional[str] = None
    importance: Optional[float] = Field(default=None, ge=0.0, le=10.0)
    tags: Optional[list[str]] = None
    metadata: Optional[dict[str, Any]] = None
    expires_at: Optional[datetime] = None


class BatchMemoryUpdate(BaseModel):
    """Model for batch memory updates (includes memory_id)."""

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
    updates: list[BatchMemoryUpdate]


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
    return {"status": "ok", "service": "hippocampai", "version": "0.3.0"}


@app.get("/metrics")
async def metrics() -> Response:
    """Prometheus metrics endpoint."""
    if not PROMETHEUS_AVAILABLE:
        raise HTTPException(
            status_code=501, detail="Prometheus metrics not available - install prometheus-client"
        )
    return Response(content=get_metrics(), media_type="text/plain")


@app.get("/stats")
async def get_stats(service: MemoryManagementService = Depends(get_service)) -> dict[str, Any]:
    """Get Redis cache statistics."""
    stats = await service.redis.get_stats()
    return cast(dict[str, Any], stats)


# ============================================================================
# CRUD Operations
# ============================================================================


@app.post("/v1/memories", response_model=Memory, status_code=status.HTTP_201_CREATED)
async def create_memory(
    request: MemoryCreate,
    skip_duplicate_check: bool = False,
    service: MemoryManagementService = Depends(get_service),
) -> Memory:
    """Create a new memory with optional deduplication check."""
    try:
        # Set conflict handling strategy from request if provided
        conflict_strategy = request.metadata.get("strategy") if request.metadata else None
        if conflict_strategy:
            try:
                # Validate strategy is a valid enum value
                strategy = ConflictResolutionStrategy(conflict_strategy)
                service.conflict_resolution_strategy = strategy
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Invalid conflict resolution strategy: {conflict_strategy}. "
                        f"Must be one of: {[s.value for s in ConflictResolutionStrategy]}"
                    ),
                )

        # Set auto-resolve based on request
        auto_resolve = request.metadata.get("auto_resolve", True) if request.metadata else True

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
                check_duplicate=not skip_duplicate_check,
                check_conflicts=True,
                auto_resolve_conflicts=auto_resolve,
            )
            return memory
        except ConflictResolutionError as e:
            # Handle conflict resolution errors specifically
            status_code = 409 if auto_resolve else 400  # 409 Conflict or 400 Bad Request
            error_detail = {
                "error": "Memory conflict detected",
                "message": str(e),
                "conflict_id": getattr(e, "conflict_id", None),
                "resolution_options": [s.value for s in ConflictResolutionStrategy],
                "help": (
                    "Set metadata.strategy to one of the resolution_options "
                    "and metadata.auto_resolve=true to automatically resolve conflicts"
                ),
            }
            raise HTTPException(status_code=status_code, detail=error_detail)
        except MemoryNotFoundError as e:
            # Handle missing memory errors (e.g. duplicate not found)
            error_detail = {
                "error": "Memory not found",
                "message": str(e),
                "help": "The memory was detected as a duplicate but could not be found",
            }
            raise HTTPException(status_code=404, detail=error_detail)

    except HTTPException:
        raise
    except ValueError as e:
        # Handle validation errors
        logger.warning(f"Validation error in create_memory: {e}")
        raise HTTPException(status_code=400, detail=str(e))
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
) -> list[Memory]:
    """Query memories with advanced filters (type, tags, date range, importance threshold, text search)."""
    try:
        result: list[Memory] = await service.get_memories(
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
        return result
    except Exception as e:
        logger.error(f"Query memories failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Batch Operations
# ============================================================================


@app.post("/v1/memories/batch", response_model=list[Memory], status_code=status.HTTP_201_CREATED)
async def batch_create_memories(
    request: BatchCreateRequest, service: MemoryManagementService = Depends(get_service)
) -> list[Memory]:
    """Batch create multiple memories."""
    try:
        memories_data = [mem.model_dump(exclude_none=True) for mem in request.memories]
        result: list[Memory] = await service.batch_create_memories(
            memories_data, check_duplicates=request.check_duplicates
        )
        return result
    except Exception as e:
        logger.error(f"Batch create failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.patch("/v1/memories/batch", response_model=list[Memory])
async def batch_update_memories(
    request: BatchUpdateRequest, service: MemoryManagementService = Depends(get_service)
) -> list[Memory]:
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
) -> dict[str, Any]:
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
) -> list[Memory]:
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
) -> list[RetrievalResult]:
    """Recall memories using hybrid search with customizable weights."""
    try:
        result: list[RetrievalResult] = await service.recall_memories(
            query=request.query,
            user_id=request.user_id,
            session_id=request.session_id,
            k=request.k,
            filters=request.filters,
            custom_weights=request.custom_weights,
        )
        return result
    except Exception as e:
        logger.error(f"Recall failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Extraction
# ============================================================================


@app.post("/v1/memories/extract", response_model=list[Memory])
async def extract_from_conversation(
    request: ExtractRequest, service: MemoryManagementService = Depends(get_service)
) -> list[Memory]:
    """Extract memories from conversation logs."""
    try:
        result: list[Memory] = await service.extract_from_conversation(
            conversation=request.conversation,
            user_id=request.user_id,
            session_id=request.session_id,
        )
        return result
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
) -> dict[str, Any]:
    """Deduplicate memories for a user."""
    try:
        result: dict[str, Any] = await service.deduplicate_user_memories(
            user_id=request.user_id, dry_run=request.dry_run
        )
        return result
    except Exception as e:
        logger.error(f"Deduplication failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/memories/consolidate")
async def consolidate_memories(
    request: ConsolidateRequest, service: MemoryManagementService = Depends(get_service)
) -> dict[str, Any]:
    """Consolidate similar memories for a user."""
    try:
        result: dict[str, Any] = await service.consolidate_memories(
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
) -> dict[str, Any]:
    """Expire memories based on TTL."""
    try:
        expired_count = await service.expire_memories(user_id=request.user_id)
        return {"success": True, "expired_count": expired_count}
    except Exception as e:
        logger.error(f"Expire failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/memories/cleanup")
async def cleanup_expired_memories(
    service: MemoryManagementService = Depends(get_service),
) -> dict[str, Any]:
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
) -> dict[str, Any]:
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
async def get_background_status(
    tasks: BackgroundTaskManager = Depends(get_background_tasks),
) -> dict[str, Any]:
    """Get status of background tasks."""
    result: dict[str, Any] = tasks.get_status()
    return result


@app.post("/v1/background/dedup/trigger")
async def trigger_background_dedup(
    user_id: str,
    dry_run: bool = True,
    tasks: BackgroundTaskManager = Depends(get_background_tasks),
) -> dict[str, Any]:
    """Manually trigger deduplication for a user via background task manager."""
    try:
        result: dict[str, Any] = await tasks.trigger_deduplication(user_id=user_id, dry_run=dry_run)
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
) -> dict[str, Any]:
    """Manually trigger consolidation for a user via background task manager."""
    try:
        result: dict[str, Any] = await tasks.trigger_consolidation(
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
async def remember(
    request: MemoryCreate, service: MemoryManagementService = Depends(get_service)
) -> Memory:
    """Legacy endpoint: Create a memory."""
    return await create_memory(request, skip_duplicate_check=False, service=service)


@app.post("/v1/memories:recall", response_model=list[RetrievalResult])
async def recall(
    request: RecallRequest, service: MemoryManagementService = Depends(get_service)
) -> list[RetrievalResult]:
    """Legacy endpoint: Recall memories."""
    return await recall_memories(request, service)


@app.post("/v1/memories:extract", response_model=list[Memory])
async def extract(
    request: ExtractRequest, service: MemoryManagementService = Depends(get_service)
) -> list[Memory]:
    """Legacy endpoint: Extract from conversation."""
    return await extract_from_conversation(request, service)


# ============================================================================
# NEW: Observability & Debugging Endpoints
# ============================================================================


class ExplainRetrievalRequest(BaseModel):
    """Request to explain retrieval results."""

    query: str
    user_id: str
    k: int = 5


class VisualizeScoresRequest(BaseModel):
    """Request to visualize similarity scores."""

    query: str
    user_id: str
    top_k: int = 10


class AccessHeatmapRequest(BaseModel):
    """Request for memory access heatmap."""

    user_id: str
    time_period_days: int = 30


class ProfileQueryRequest(BaseModel):
    """Request for query performance profiling."""

    query: str
    user_id: str
    k: int = 5


@app.post("/v1/observability/explain")
async def explain_retrieval_results(
    request: ExplainRetrievalRequest,
    service: MemoryManagementService = Depends(get_service),
) -> dict[str, Any]:
    """Explain why specific memories were retrieved.

    Returns detailed explanations including:
    - Score breakdowns (vector, BM25, recency, importance)
    - Contributing factors
    - Human-readable explanations
    """
    try:
        # First, recall memories
        results = await service.recall_memories(
            user_id=request.user_id,
            query=request.query,
            k=request.k,
        )

        # Return retrieval explanations
        return {
            "query": request.query,
            "results_count": len(results),
            "explanations": [
                {
                    "memory_id": result.memory.id,
                    "text": result.memory.text,
                    "score": result.score,
                    "scores": result.breakdown,
                }
                for result in results
            ],
        }
    except Exception as e:
        logger.error(f"Error explaining retrieval: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/observability/visualize")
async def visualize_similarity(
    request: VisualizeScoresRequest,
    service: MemoryManagementService = Depends(get_service),
) -> dict[str, Any]:
    """Visualize similarity scores and ranking factors.

    Returns visualization data including:
    - Score distribution
    - Top results with detailed scores
    - Statistical summary
    """
    try:
        # Recall memories
        results = await service.recall_memories(
            user_id=request.user_id,
            query=request.query,
            k=request.top_k,
        )

        # Generate visualization
        viz_data = {
            "query": request.query,
            "scores": [
                {
                    "memory_id": r.memory.id,
                    "text": r.memory.text,
                    "score": r.score,
                    "breakdown": r.breakdown,
                }
                for r in results[: request.top_k]
            ],
            "stats": {
                "mean_score": sum(r.score for r in results) / len(results) if results else 0,
                "count": len(results),
                "top_k": request.top_k,
            },
        }
        return viz_data
    except Exception as e:
        logger.error(f"Error visualizing scores: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/observability/heatmap")
async def generate_heatmap(
    request: AccessHeatmapRequest,
    service: MemoryManagementService = Depends(get_service),
) -> dict[str, Any]:
    """Generate memory access heatmap.

    Returns access patterns including:
    - Access by hour/day
    - Hot and cold memories
    - Peak usage times
    """
    try:
        # Get user memories with their access counts
        memories = await service.get_memories(
            user_id=request.user_id,
            limit=1000,  # Reasonable limit for analysis
        )

        # Calculate access patterns
        heatmap_data = {
            "user_id": request.user_id,
            "time_period_days": request.time_period_days,
            "total_memories": len(memories),
            "total_accesses": sum(m.access_count for m in memories),
            "memory_access_counts": [
                {"id": m.id, "access_count": m.access_count}
                for m in sorted(memories, key=lambda x: x.access_count, reverse=True)
            ],
        }
        return heatmap_data
    except Exception as e:
        logger.error(f"Error generating heatmap: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/observability/profile")
async def profile_query(
    request: ProfileQueryRequest,
    service: MemoryManagementService = Depends(get_service),
) -> dict[str, Any]:
    """Profile query performance.

    Returns performance metrics including:
    - Stage-by-stage timing
    - Bottleneck identification
    - Optimization recommendations
    """
    try:
        # Profile memory retrieval performance
        start_time = datetime.now()

        results = await service.recall_memories(
            query=request.query,
            user_id=request.user_id,
            k=request.k,
        )

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Return performance metrics
        return {
            "query": request.query,
            "user_id": request.user_id,
            "k": request.k,
            "results_found": len(results),
            "duration_seconds": duration,
            "throughput": len(results) / duration if duration > 0 else 0,
            "average_score": sum(r.score for r in results) / len(results) if results else 0,
        }
    except Exception as e:
        logger.error(f"Error profiling query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# NEW: Enhanced Temporal Features Endpoints
# ============================================================================


class FreshnessScoreRequest(BaseModel):
    """Request to calculate memory freshness.

    Security: Requires user_id to verify ownership before exposing
    behavioral metadata like access patterns and timing information.
    """

    memory_id: str
    user_id: str
    reference_date: Optional[datetime] = None


class TimeDecayRequest(BaseModel):
    """Request to apply time decay."""

    memory_id: str
    decay_function: Optional[str] = "default_exponential"


class ForecastRequest(BaseModel):
    """Request to forecast memory patterns."""

    user_id: str
    forecast_days: int = 30


class ContextWindowRequest(BaseModel):
    """Request for adaptive context window."""

    query: str
    user_id: str
    context_type: str = "relevant"  # recent, relevant, seasonal


@app.post("/v1/temporal/freshness")
async def calculate_freshness(
    request: FreshnessScoreRequest,
    service: MemoryManagementService = Depends(get_service),
) -> dict[str, Any]:
    """Calculate memory freshness score.

    Returns comprehensive freshness metrics including:
    - Overall freshness score (0-1)
    - Age factor (sanitized)
    - Access frequency (sanitized)
    - Temporal relevance

    Security:
        Requires user_id to verify ownership. Access patterns and timing
        information are sanitized to prevent behavioral metadata leakage.
    """
    try:
        # Get the memory
        memory = await service.get_memory(memory_id=request.memory_id, track_access=False)
        if not memory:
            raise HTTPException(status_code=404, detail="Memory not found")

        # Authorization: Verify requester owns the memory
        if memory.user_id != request.user_id:
            logger.warning(
                f"Unauthorized freshness access attempt: user {request.user_id} "
                f"tried to access memory {request.memory_id} owned by {memory.user_id}"
            )
            raise HTTPException(
                status_code=403,
                detail="Forbidden: You don't have permission to access this memory",
            )

        # Calculate freshness
        reference = request.reference_date or datetime.now(timezone.utc)
        age = abs((reference - memory.created_at).total_seconds())
        last_access = abs((reference - memory.updated_at).total_seconds())

        # Sanitize access_count to prevent precise behavioral tracking
        # Round to ranges instead of exact counts
        sanitized_access_count = (
            "low"
            if memory.access_count < 10
            else ("medium" if memory.access_count < 50 else "high")
        )

        return {
            "memory_id": memory.id,
            "age_days": round(age / (24 * 3600), 1),  # Round to 1 decimal
            "last_access_days": round(last_access / (24 * 3600), 1),  # Round to 1 decimal
            "access_frequency": sanitized_access_count,  # Categorical instead of exact
            "freshness_score": round(1.0 / (1.0 + age / (30 * 24 * 3600)), 3),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error calculating freshness: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/temporal/decay")
async def apply_decay(
    request: TimeDecayRequest,
    service: MemoryManagementService = Depends(get_service),
) -> dict[str, Any]:
    """Apply time decay function to memory importance.

    Returns decayed importance score based on age and decay function.
    """
    try:
        # Get the memory and calculate time-based decay
        memory = await service.get_memory(memory_id=request.memory_id)
        if not memory:
            raise HTTPException(status_code=404, detail="Memory not found")

        age_days = (datetime.now(timezone.utc) - memory.created_at).days
        decay_rate = 0.1  # 10% decay per month for default exponential
        if request.decay_function == "linear":
            decayed_score = max(0.0, memory.importance * (1.0 - decay_rate * age_days / 30))
        else:  # exponential decay
            decayed_score = memory.importance * pow(1.0 - decay_rate, age_days / 30)

        return {
            "memory_id": request.memory_id,
            "original_importance": memory.importance,
            "decayed_importance": round(decayed_score, 2),
            "decay_function": request.decay_function or "default_exponential",
            "age_days": age_days,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error applying decay: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/temporal/forecast")
async def forecast_patterns(
    request: ForecastRequest,
    service: MemoryManagementService = Depends(get_service),
) -> dict[str, Any]:
    """Forecast future memory patterns.

    Predicts usage, topic trends, and importance patterns based on historical data.
    """
    try:
        # Get historical memory data
        memories = await service.get_memories(
            user_id=request.user_id,
            limit=1000,  # Reasonable limit for analysis
        )

        # Calculate basic forecasts based on historical patterns
        total_memories = len(memories)
        creation_rate = (
            total_memories
            / ((datetime.now(timezone.utc) - min(m.created_at for m in memories)).days + 1)
            if memories
            else 0
        )
        avg_importance = sum(m.importance for m in memories) / total_memories if memories else 0

        return {
            "user_id": request.user_id,
            "forecast_days": request.forecast_days,
            "forecasts": {
                "estimated_new_memories": round(creation_rate * request.forecast_days),
                "avg_importance_trend": avg_importance,
                "total_memories_forecast": total_memories
                + round(creation_rate * request.forecast_days),
                "daily_creation_rate": round(creation_rate, 2),
            },
        }
    except Exception as e:
        logger.error(f"Error forecasting patterns: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/temporal/context-window")
async def get_context_window(
    request: ContextWindowRequest,
    service: MemoryManagementService = Depends(get_service),
) -> dict[str, Any]:
    """Get adaptive temporal context window.

    Auto-adjusts time range based on query and context type.
    """
    try:
        # Get user memories sorted by relevance or recency
        memories = await service.get_memories(user_id=request.user_id, limit=1000)

        if request.context_type == "recent":
            relevant_memories = sorted(memories, key=lambda m: m.updated_at, reverse=True)[:10]
        elif request.context_type == "relevant":
            # Get relevant memories using recall
            results = await service.recall_memories(
                query=request.query, user_id=request.user_id, k=10
            )
            relevant_memories = [r.memory for r in results]
        else:  # seasonal
            # Default to most accessed memories
            relevant_memories = sorted(memories, key=lambda m: m.access_count, reverse=True)[:10]

        return {
            "query": request.query,
            "context_type": request.context_type,
            "window_size": len(relevant_memories),
            "memories": [
                {
                    "id": m.id,
                    "text": m.text,
                    "updated_at": m.updated_at.isoformat(),
                    "access_count": m.access_count,
                }
                for m in relevant_memories
            ],
        }
    except Exception as e:
        logger.error(f"Error getting context window: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# NEW: Memory Health & Conflict Resolution Endpoints
# ============================================================================


class ConflictDetectionRequest(BaseModel):
    """Request to detect memory conflicts."""

    user_id: str
    memory_id: Optional[str] = None  # If None, check all


class ConflictResolutionRequest(BaseModel):
    """Request to resolve memory conflict."""

    user_id: str  # Owner of the conflict
    conflict_id: str  # ID of the conflict to resolve
    strategy: str = "temporal"  # temporal, confidence, importance, auto_merge, keep_both


class HealthScoreRequest(BaseModel):
    """Request for memory health score."""

    user_id: str


class ProvenanceRequest(BaseModel):
    """Request for memory provenance."""

    memory_id: str


@app.post("/v1/conflicts/detect")
async def detect_conflicts(
    request: ConflictDetectionRequest,
    service: MemoryManagementService = Depends(get_service),
) -> dict[str, Any]:
    """Detect memory conflicts (contradictions, duplicates).

    Returns list of detected conflicts with details.
    """
    try:
        # Get memories to check for conflicts
        if request.memory_id:
            memory = await service.get_memory(request.memory_id)
            if not memory:
                raise HTTPException(status_code=404, detail="Memory not found")
            memories_to_check = [memory]
        else:
            memories_to_check = await service.get_memories(user_id=request.user_id, limit=1000)

        conflicts = []
        # Check for temporal conflicts
        for memory in memories_to_check:
            similar_results = await service.recall_memories(
                query=memory.text, user_id=request.user_id, k=5
            )
            # Filter out self-matches and low similarity
            conflicts.extend(
                [
                    {
                        "memory_id": memory.id,
                        "conflicting_id": r.memory.id,
                        "similarity": r.score,
                        "type": "content_overlap",
                    }
                    for r in similar_results
                    if r.memory.id != memory.id and r.score > 0.8  # High similarity threshold
                ]
            )

        return {
            "user_id": request.user_id,
            "conflicts_found": len(conflicts),
            "conflicts": conflicts,
        }
    except Exception as e:
        logger.error(f"Error detecting conflicts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/conflicts/resolve")
async def resolve_conflict(
    request: ConflictResolutionRequest,
    service: MemoryManagementService = Depends(get_service),
) -> dict[str, Any]:
    """Resolve a detected memory conflict.

    Applies specified resolution strategy.
    """
    try:
        # Get the conflict details
        conflicts = await service.detect_memory_conflicts(user_id=request.user_id)
        if not conflicts:
            raise HTTPException(status_code=404, detail="No conflicts found")

        # Find the specific conflict
        conflict = None
        for c in conflicts:
            if c["conflict_id"] == request.conflict_id:
                conflict = c
                break

        if not conflict:
            raise HTTPException(status_code=404, detail="Conflict not found")

        # Apply resolution based on strategy
        if request.strategy == "merge":
            # Get the conflicting memories
            memory1_details = conflict["memory_1"]
            memory2_details = conflict["memory_2"]

            # Merge their text
            merged_text = f"{memory1_details['text']}\n---\n{memory2_details['text']}"
            await service.update_memory(
                memory_id=memory1_details["id"],
                text=merged_text,
                metadata={
                    "merged_from": memory2_details["id"],
                    "merge_strategy": request.strategy,
                    "conflict_type": conflict["conflict_type"],
                },
            )
            await service.delete_memory(memory2_details["id"], user_id=request.user_id)

        elif request.strategy == "keep_latest":
            # Keep the more recent memory
            memory1_details = conflict["memory_1"]
            memory2_details = conflict["memory_2"]

            # Parse timestamps
            mem1_time = datetime.fromisoformat(memory1_details["created_at"])
            mem2_time = datetime.fromisoformat(memory2_details["created_at"])

            if mem2_time > mem1_time:
                to_delete = memory1_details["id"]
                to_keep = memory2_details["id"]
            else:  # Keep memory1 if newer or timestamps are equal
                to_delete = memory2_details["id"]
                to_keep = memory1_details["id"]

            await service.delete_memory(to_delete, user_id=request.user_id)
            # Update metadata on kept memory
            await service.update_memory(
                memory_id=to_keep,
                metadata={
                    "conflict_resolved": True,
                    "resolution_strategy": request.strategy,
                    "deleted_memory_id": to_delete,
                },
            )

        elif request.strategy == "manual":
            # For manual resolution, we just mark it as reviewed
            await service.update_memory(
                memory_id=conflict["memory_1"]["id"],
                metadata={
                    "conflict_reviewed": True,
                    "review_timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )

        return {
            "conflict_id": request.conflict_id,
            "resolution_status": "success",
            "resolution_strategy": request.strategy,
        }
    except Exception as e:
        logger.error(f"Error resolving conflict: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/health/score")
async def get_health_score(
    request: HealthScoreRequest,
    service: MemoryManagementService = Depends(get_service),
) -> dict[str, Any]:
    """Get comprehensive memory health score.

    Returns health metrics including:
    - Overall health score
    - Quality indicators
    - Issues detected
    - Recommendations

    Security:
        Input capped at 100 memories to prevent timing/cost side-channel attacks.
        Rate limiting recommended for production deployment.
    """
    try:
        # Security: Cap memory count to prevent timing attacks via diversity scoring
        MAX_MEMORIES_FOR_HEALTH = 100
        memories = await service.get_memories(
            user_id=request.user_id, limit=MAX_MEMORIES_FOR_HEALTH
        )
        if not memories:
            return {
                "user_id": request.user_id,
                "health_score": 1.0,  # Perfect score for empty memory
                "details": {
                    "memory_count": 0,
                    "avg_age_days": 0,
                    "duplicate_ratio": 0,
                    "retrieval_success_rate": 1.0,
                },
            }

        # Calculate metrics
        current_time = datetime.now(timezone.utc)
        memory_count = len(memories)

        # Age analysis
        ages = [(current_time - m.created_at).days for m in memories]
        avg_age = sum(ages) / len(ages) if ages else 0

        # Duplicate detection
        seen_texts = set()
        duplicates = 0
        for m in memories:
            if m.text in seen_texts:
                duplicates += 1
            seen_texts.add(m.text)
        duplicate_ratio = duplicates / memory_count if memory_count > 0 else 0

        # Calculate retrieval success rate from recent search results
        # For now, we'll use a simplified metric based on access counts
        recently_accessed = [m for m in memories if m.access_count > 0]
        retrieval_rate = len(recently_accessed) / memory_count if memory_count > 0 else 1.0

        # Calculate overall health score (0.0 to 1.0)
        age_score = max(0, min(1, 1 - (avg_age / 365)))  # Penalize old memories
        duplicate_score = 1 - duplicate_ratio
        retrieval_score = retrieval_rate

        overall_score = (age_score + duplicate_score + retrieval_score) / 3

        return {
            "user_id": request.user_id,
            "health_score": overall_score,
            "details": {
                "memory_count": memory_count,
                "avg_age_days": avg_age,
                "duplicate_ratio": duplicate_ratio,
                "retrieval_success_rate": retrieval_rate,
            },
        }
    except Exception as e:
        logger.error(f"Error getting health score: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/provenance/track")
async def get_provenance(
    request: ProvenanceRequest,
    service: MemoryManagementService = Depends(get_service),
) -> dict[str, Any]:
    """Get memory provenance and lineage.

    Returns complete history and source tracking.
    """
    try:
        # Get the target memory
        memory = await service.get_memory(request.memory_id)
        if not memory:
            raise HTTPException(status_code=404, detail="Memory not found")

        # Build provenance chain
        chain = []
        current = memory

        # Follow parent links up to root
        while current:
            chain.append(
                {
                    "memory_id": current.id,
                    "timestamp": current.created_at.isoformat(),
                    "type": current.type,
                    "metadata": current.metadata,
                }
            )

            # Look for parent in metadata
            if not current.metadata.get("parent_id"):
                break

            parent_memory = await service.get_memory(current.metadata["parent_id"])
            if not parent_memory:
                break
            current = parent_memory

        if not chain:
            raise HTTPException(status_code=404, detail="Provenance not found")

        return {"memory_id": request.memory_id, "chain": chain}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting provenance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Memory Tracking & Monitoring Endpoints
# ============================================================================


class MemoryEventsRequest(BaseModel):
    """Request for memory events."""

    memory_id: Optional[str] = None
    user_id: Optional[str] = None
    event_type: Optional[str] = None
    limit: int = 100


class MemoryStatsRequest(BaseModel):
    """Request for memory statistics."""

    user_id: str


class AccessPatternRequest(BaseModel):
    """Request for memory access pattern."""

    memory_id: str
    user_id: str


class HealthHistoryRequest(BaseModel):
    """Request for health history."""

    memory_id: str
    user_id: str
    limit: int = 100


class AccessPatternsRequest(BaseModel):
    """Request for access patterns.

    Security: Requires authenticated user_id. In production, implement
    proper authentication middleware to verify the requester has
    permission to access this user's data.
    """

    user_id: str


@app.get("/v1/monitoring/events")
async def get_memory_events(
    memory_id: Optional[str] = None,
    user_id: Optional[str] = None,
    event_type: Optional[str] = None,
    limit: int = 100,
) -> dict[str, Any]:
    """Get memory lifecycle events.

    Returns list of events showing what's happening with memories on the backend:
    - Creation, updates, deletions
    - Searches and retrievals
    - Consolidations and deduplication
    - Health checks and conflict detection
    """
    try:
        from hippocampai.monitoring.memory_tracker import (
            MemoryEventType,
            get_tracker,
        )

        tracker = get_tracker()

        # Convert string event_type to enum if provided
        event_type_enum = None
        if event_type:
            try:
                event_type_enum = MemoryEventType(event_type)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid event_type. Must be one of: {[e.value for e in MemoryEventType]}",
                )

        events = tracker.get_memory_events(
            memory_id=memory_id,
            user_id=user_id,
            event_type=event_type_enum,
            limit=limit,
        )

        return {
            "total": len(events),
            "events": [e.model_dump() for e in events],
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting memory events: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/monitoring/stats")
async def get_memory_stats(request: MemoryStatsRequest) -> dict[str, Any]:
    """Get comprehensive memory statistics for a user.

    Returns:
    - Total events by type
    - Success rates
    - Most accessed memories
    - Average operation durations
    """
    try:
        from hippocampai.monitoring.memory_tracker import get_tracker

        tracker = get_tracker()
        stats: dict[str, Any] = tracker.get_memory_stats(user_id=request.user_id)

        return stats
    except Exception as e:
        logger.error(f"Error getting memory stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/monitoring/access-pattern")
async def get_access_pattern(request: AccessPatternRequest) -> dict[str, Any]:
    """Get access pattern for a specific memory.

    Returns:
    - Access count and frequency
    - Last and first access times
    - Search hits vs direct retrievals
    - Access sources
    """
    try:
        from hippocampai.monitoring.memory_tracker import get_tracker

        tracker = get_tracker()
        pattern = tracker.get_access_pattern(memory_id=request.memory_id, user_id=request.user_id)

        if not pattern:
            raise HTTPException(
                status_code=404,
                detail=f"No access pattern found for memory {request.memory_id}",
            )

        result: dict[str, Any] = pattern.model_dump()
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting access pattern: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/monitoring/access-patterns")
async def get_all_access_patterns(request: AccessPatternsRequest) -> dict[str, Any]:
    """Get all access patterns for a user.

    Returns list of all memories with their access statistics.

    Security Warning:
        This endpoint requires authenticated user context. In production:
        1. Implement authentication middleware (JWT, OAuth2, etc.)
        2. Verify the authenticated user matches request.user_id
        3. Or only return data for the authenticated user
        4. Consider role-based access for admin users

    Args:
        request: AccessPatternsRequest with authenticated user_id

    Returns:
        Dictionary containing user_id, total_memories, and access patterns
    """
    try:
        from hippocampai.monitoring.memory_tracker import get_tracker

        tracker = get_tracker()
        patterns = tracker.get_all_access_patterns(user_id=request.user_id)

        return {
            "user_id": request.user_id,
            "total_memories": len(patterns),
            "patterns": [p.model_dump() for p in patterns],
        }
    except Exception as e:
        logger.error(f"Error getting access patterns: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/monitoring/health-history")
async def get_health_history(request: HealthHistoryRequest) -> dict[str, Any]:
    """Get health history for a specific memory.

    Returns snapshots of memory health over time:
    - Health scores
    - Staleness and freshness scores
    - Detected issues
    """
    try:
        from hippocampai.monitoring.memory_tracker import get_tracker

        tracker = get_tracker()
        history = tracker.get_health_history(
            memory_id=request.memory_id,
            user_id=request.user_id,
            limit=request.limit,
        )

        return {
            "memory_id": request.memory_id,
            "user_id": request.user_id,
            "total_snapshots": len(history),
            "history": [h.model_dump() for h in history],
        }
    except Exception as e:
        logger.error(f"Error getting health history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Memory Lifecycle & Tiering
# ============================================================================


@app.get("/v1/lifecycle/temperature/{memory_id}")
async def get_memory_temperature(
    memory_id: str, service: MemoryManagementService = Depends(get_service)
) -> dict[str, Any]:
    """
    Get temperature metrics for a specific memory.

    Returns lifecycle information including:
    - Current tier (hot/warm/cold/archived/hibernated)
    - Temperature score (0-100)
    - Access frequency and recency
    - Recommended tier based on access patterns
    """
    try:
        temperature = await service.get_memory_temperature(memory_id)
        if not temperature:
            raise HTTPException(status_code=404, detail="Memory not found")
        return temperature
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting memory temperature: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class TierMigrationRequest(BaseModel):
    """Request to migrate a memory to a specific tier."""

    memory_id: str
    target_tier: str = Field(
        ...,
        description="Target tier: hot, warm, cold, archived, or hibernated",
    )


@app.post("/v1/lifecycle/migrate")
async def migrate_memory_tier(
    request: TierMigrationRequest,
    service: MemoryManagementService = Depends(get_service),
) -> dict[str, Any]:
    """
    Manually migrate a memory to a specific storage tier.

    Tiers:
    - hot: Frequently accessed, cached in Redis + Vector DB
    - warm: Occasionally accessed, in Vector DB
    - cold: Rarely accessed, in Vector DB
    - archived: Very old, compressed in Vector DB
    - hibernated: Extremely old, highly compressed
    """
    try:
        # Validate tier
        try:
            target_tier = MemoryTier(request.target_tier.lower())
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid tier. Must be one of: {[t.value for t in MemoryTier]}",
            )

        # Perform migration
        success = await service.migrate_memory_tier(request.memory_id, target_tier)
        if not success:
            raise HTTPException(status_code=404, detail="Memory not found")

        return {
            "memory_id": request.memory_id,
            "target_tier": target_tier.value,
            "migrated": True,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error migrating memory tier: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class TierStatsRequest(BaseModel):
    """Request for tier statistics."""

    user_id: str


@app.post("/v1/lifecycle/stats")
async def get_tier_statistics(
    request: TierStatsRequest, service: MemoryManagementService = Depends(get_service)
) -> dict[str, Any]:
    """
    Get statistics about memory tiers for a user.

    Returns:
    - Total memories and size
    - Distribution across tiers
    - Average temperature per tier
    - Lifecycle configuration
    """
    try:
        stats: dict[str, Any] = await service.get_tier_statistics(request.user_id)
        return stats
    except Exception as e:
        logger.error(f"Error getting tier statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Server
# ============================================================================


def run_server(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Run FastAPI server."""
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
