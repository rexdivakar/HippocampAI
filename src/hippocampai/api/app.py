"""FastAPI application."""

import logging
from datetime import datetime
from typing import Any, Optional

import uvicorn
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from hippocampai.api.deps import get_memory_client
from hippocampai.client import MemoryClient
from hippocampai.models.memory import Memory, RetrievalResult

logger = logging.getLogger(__name__)

app = FastAPI(
    title="HippocampAI API",
    description="Autonomous memory engine with hybrid retrieval",
    version="1.0.0",
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


# Request/Response models
class RememberRequest(BaseModel):
    text: str
    user_id: str
    session_id: Optional[str] = None
    type: str = "fact"
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
def health_check():
    """Health check."""
    return {"status": "ok"}


@app.post("/v1/memories:remember", response_model=Memory)
def remember(request: RememberRequest, client: MemoryClient = Depends(get_memory_client)):
    """Store a memory."""
    try:
        memory = client.remember(
            text=request.text,
            user_id=request.user_id,
            session_id=request.session_id,
            type=request.type,
            importance=request.importance,
            tags=request.tags,
            ttl_days=request.ttl_days,
        )
        return memory
    except Exception as e:
        logger.error(f"Remember failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/memories:recall", response_model=list[RetrievalResult])
def recall(request: RecallRequest, client: MemoryClient = Depends(get_memory_client)):
    """Retrieve memories."""
    try:
        results = client.recall(
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
def extract(request: ExtractRequest, client: MemoryClient = Depends(get_memory_client)):
    """Extract memories from conversation."""
    try:
        memories = client.extract_from_conversation(
            conversation=request.conversation,
            user_id=request.user_id,
            session_id=request.session_id,
        )
        return memories
    except Exception as e:
        logger.error(f"Extract failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.patch("/v1/memories:update", response_model=Memory)
def update_memory(request: UpdateMemoryRequest, client: MemoryClient = Depends(get_memory_client)):
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
def delete_memory(request: DeleteMemoryRequest, client: MemoryClient = Depends(get_memory_client)):
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
def get_memories(request: GetMemoriesRequest, client: MemoryClient = Depends(get_memory_client)):
    """Get memories with advanced filtering."""
    try:
        memories = client.get_memories(
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
):
    """Clean up expired memories."""
    try:
        expired_count = client.expire_memories(user_id=request.user_id)
        return {"success": True, "expired_count": expired_count}
    except Exception as e:
        logger.error(f"Expire memories failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run FastAPI server."""
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
