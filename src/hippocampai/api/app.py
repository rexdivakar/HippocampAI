"""FastAPI application."""

import logging
from datetime import datetime
from typing import Any, Optional

import socketio
import uvicorn
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from hippocampai.api.deps import get_memory_client
from hippocampai.api.websocket import sio
from hippocampai.client import MemoryClient
from hippocampai.models.memory import Memory, RetrievalResult

logger = logging.getLogger(__name__)

app = FastAPI(
    title="HippocampAI API",
    description="Autonomous memory engine with hybrid retrieval",
    version="0.3.0",
)

# Mount Socket.IO app
socket_app = socketio.ASGIApp(sio, app)

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

# Include admin routes
try:
    from hippocampai.api.admin_routes import router as admin_router

    app.include_router(admin_router)
    logger.info("Admin routes registered successfully")
except ImportError as e:
    logger.warning(f"Could not load admin routes: {e}")

# Include collaboration routes
try:
    from hippocampai.api.collaboration_routes import router as collaboration_router

    app.include_router(collaboration_router)
    logger.info("Collaboration routes registered successfully")
except ImportError as e:
    logger.warning(f"Could not load collaboration routes: {e}")

# Include prediction routes
try:
    from hippocampai.api.prediction_routes import router as prediction_router

    app.include_router(prediction_router)
    logger.info("Prediction routes registered successfully")
except ImportError as e:
    logger.warning(f"Could not load prediction routes: {e}")

# Include healing routes
try:
    from hippocampai.api.healing_routes import router as healing_router

    app.include_router(healing_router)
    logger.info("Healing routes registered successfully")
except ImportError as e:
    logger.warning(f"Could not load healing routes: {e}")

# Include consolidation routes (Sleep Phase)
try:
    from hippocampai.api.consolidation_routes import router as consolidation_router

    app.include_router(consolidation_router)
    logger.info("Consolidation routes registered successfully")
except ImportError as e:
    logger.warning(f"Could not load consolidation routes: {e}")

# Include dashboard routes
try:
    from hippocampai.api.dashboard_routes import router as dashboard_router

    app.include_router(dashboard_router)
    logger.info("Dashboard routes registered successfully")
except ImportError as e:
    logger.warning(f"Could not load dashboard routes: {e}")


# Request/Response models
class RememberRequest(BaseModel):
    text: str
    user_id: str
    session_id: Optional[str] = None
    type: Optional[str] = None  # Auto-detect if not provided
    importance: Optional[float] = None
    tags: Optional[list[str]] = None
    ttl_days: Optional[int] = None


class RecallRequest(BaseModel):
    query: str
    user_id: str
    session_id: Optional[str] = None
    k: int = 5
    filters: Optional[dict[str, Any]] = None


class ExtractRequest(BaseModel):
    conversation: str
    user_id: str
    session_id: Optional[str] = None


class UpdateMemoryRequest(BaseModel):
    memory_id: str
    text: Optional[str] = None
    importance: Optional[float] = None
    tags: Optional[list[str]] = None
    metadata: Optional[dict[str, Any]] = None
    expires_at: Optional[datetime] = None


class DeleteMemoryRequest(BaseModel):
    memory_id: str
    user_id: Optional[str] = None


class GetMemoriesRequest(BaseModel):
    user_id: str
    filters: Optional[dict[str, Any]] = None
    limit: int = 100


class ExpireMemoriesRequest(BaseModel):
    user_id: Optional[str] = None


# Routes
@app.get("/healthz")
def health_check() -> dict[str, str]:
    """Health check."""
    return {"status": "ok"}


@app.post("/v1/memories:remember", response_model=Memory)
def remember(request: RememberRequest, client: MemoryClient = Depends(get_memory_client)) -> Memory:
    """
    Store a memory with automatic type detection.

    If type is not provided, it will be automatically detected based on content patterns:
    - fact: Personal information, identity statements
    - preference: Likes, dislikes, opinions
    - goal: Intentions, aspirations, plans
    - habit: Routines, regular activities
    - event: Specific occurrences, meetings
    - context: General conversation (default)
    """
    try:
        # Automatic type detection if not provided (LLM-based with fallback)
        memory_type = request.type
        if not memory_type:
            from hippocampai.utils.llm_classifier import get_llm_classifier

            llm_classifier = get_llm_classifier(use_cache=True)
            detected_type, confidence = llm_classifier.classify_with_confidence(request.text)
            memory_type = detected_type.value
            logger.info(
                f"Auto-detected memory type: {memory_type} (confidence: {confidence:.2f}) "
                f"for text: {request.text[:50]}..."
            )

        memory = client.remember(
            text=request.text,
            user_id=request.user_id,
            session_id=request.session_id,
            type=memory_type,
            importance=request.importance,
            tags=request.tags,
            ttl_days=request.ttl_days,
        )
        return memory
    except Exception as e:
        logger.error(f"Remember failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/memories:recall", response_model=list[RetrievalResult])
def recall(
    request: RecallRequest, client: MemoryClient = Depends(get_memory_client)
) -> list[RetrievalResult]:
    """Retrieve memories."""
    try:
        results: list[RetrievalResult] = client.recall(
            query=request.query,
            user_id=request.user_id,
            session_id=request.session_id,
            k=request.k,
            filters=request.filters,
        )
        return results
    except Exception as e:
        logger.error(f"Recall failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/memories:extract", response_model=list[Memory])
def extract(
    request: ExtractRequest, client: MemoryClient = Depends(get_memory_client)
) -> list[Memory]:
    """Extract memories from conversation."""
    try:
        memories: list[Memory] = client.extract_from_conversation(
            conversation=request.conversation,
            user_id=request.user_id,
            session_id=request.session_id,
        )
        return memories
    except Exception as e:
        logger.error(f"Extract failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.patch("/v1/memories:update", response_model=Memory)
def update_memory(
    request: UpdateMemoryRequest, client: MemoryClient = Depends(get_memory_client)
) -> Memory:
    """Update an existing memory."""
    try:
        memory = client.update_memory(
            memory_id=request.memory_id,
            text=request.text,
            importance=request.importance,
            tags=request.tags,
            metadata=request.metadata,
            expires_at=request.expires_at,
        )
        if memory is None:
            raise HTTPException(status_code=404, detail="Memory not found")
        return memory
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Update failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/v1/memories:delete")
def delete_memory(
    request: DeleteMemoryRequest, client: MemoryClient = Depends(get_memory_client)
) -> dict[str, Any]:
    """Delete a memory."""
    try:
        deleted = client.delete_memory(
            memory_id=request.memory_id,
            user_id=request.user_id,
        )
        if not deleted:
            raise HTTPException(status_code=404, detail="Memory not found or unauthorized")
        return {"success": True, "memory_id": request.memory_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/memories:get", response_model=list[Memory])
def get_memories(
    request: GetMemoriesRequest, client: MemoryClient = Depends(get_memory_client)
) -> list[Memory]:
    """Get memories with advanced filtering."""
    try:
        memories: list[Memory] = client.get_memories(
            user_id=request.user_id,
            filters=request.filters,
            limit=request.limit,
        )
        return memories
    except Exception as e:
        logger.error(f"Get memories failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/memories:expire")
def expire_memories(
    request: ExpireMemoriesRequest, client: MemoryClient = Depends(get_memory_client)
) -> dict[str, Any]:
    """Clean up expired memories."""
    try:
        expired_count = client.expire_memories(user_id=request.user_id)
        return {"success": True, "expired_count": expired_count}
    except Exception as e:
        logger.error(f"Expire memories failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def run_server(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Run FastAPI server with WebSocket support."""
    uvicorn.run(socket_app, host=host, port=port)


if __name__ == "__main__":
    run_server()
