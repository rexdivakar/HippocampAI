"""FastAPI application."""

import logging
from typing import Dict, List, Optional
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from hippocampai.client import MemoryClient
from hippocampai.api.deps import get_memory_client
from hippocampai.models.memory import Memory, RetrievalResult

logger = logging.getLogger(__name__)

app = FastAPI(
    title="HippocampAI API",
    description="Autonomous memory engine with hybrid retrieval",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class RememberRequest(BaseModel):
    text: str
    user_id: str
    session_id: Optional[str] = None
    type: str = "fact"
    importance: Optional[float] = None
    tags: Optional[List[str]] = None


class RecallRequest(BaseModel):
    query: str
    user_id: str
    session_id: Optional[str] = None
    k: int = 5


class ExtractRequest(BaseModel):
    conversation: str
    user_id: str
    session_id: Optional[str] = None


# Routes
@app.get("/healthz")
def health_check():
    """Health check."""
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    """Basic metrics."""
    return {"memories_stored": "N/A", "queries_served": "N/A"}


@app.post("/v1/memories:remember", response_model=Memory)
def remember(
    request: RememberRequest,
    client: MemoryClient = Depends(get_memory_client)
):
    """Store a memory."""
    try:
        memory = client.remember(
            text=request.text,
            user_id=request.user_id,
            session_id=request.session_id,
            type=request.type,
            importance=request.importance,
            tags=request.tags
        )
        return memory
    except Exception as e:
        logger.error(f"Remember failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/memories:recall", response_model=List[RetrievalResult])
def recall(
    request: RecallRequest,
    client: MemoryClient = Depends(get_memory_client)
):
    """Retrieve memories."""
    try:
        results = client.recall(
            query=request.query,
            user_id=request.user_id,
            session_id=request.session_id,
            k=request.k
        )
        return results
    except Exception as e:
        logger.error(f"Recall failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/memories:extract", response_model=List[Memory])
def extract(
    request: ExtractRequest,
    client: MemoryClient = Depends(get_memory_client)
):
    """Extract memories from conversation."""
    try:
        memories = client.extract_from_conversation(
            conversation=request.conversation,
            user_id=request.user_id,
            session_id=request.session_id
        )
        return memories
    except Exception as e:
        logger.error(f"Extract failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run FastAPI server."""
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
