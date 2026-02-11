"""REST endpoints for memory relevance feedback."""

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from hippocampai.api.deps import get_memory_client
from hippocampai.client import MemoryClient
from hippocampai.feedback.feedback_manager import FeedbackType

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["feedback"])


class FeedbackRequest(BaseModel):
    user_id: str
    query: str = ""
    feedback_type: str  # "relevant", "not_relevant", "partially_relevant", "outdated"


class FeedbackResponse(BaseModel):
    memory_id: str
    feedback_type: str
    score: float


class AggregatedScoreResponse(BaseModel):
    memory_id: str
    score: float
    event_count: int
    breakdown: dict[str, int]


class FeedbackStatsResponse(BaseModel):
    user_id: str
    stats: dict[str, int]


@router.post("/memories/{memory_id}/feedback", response_model=FeedbackResponse)
def submit_feedback(
    memory_id: str,
    request: FeedbackRequest,
    client: MemoryClient = Depends(get_memory_client),
) -> FeedbackResponse:
    """Submit feedback on a retrieved memory."""
    try:
        ft = FeedbackType(request.feedback_type)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid feedback_type: {request.feedback_type}. "
            f"Must be one of: {[ft.value for ft in FeedbackType]}",
        )

    event = client.feedback_manager.record_feedback(
        memory_id=memory_id,
        user_id=request.user_id,
        feedback_type=ft,
        query=request.query,
    )
    return FeedbackResponse(
        memory_id=memory_id,
        feedback_type=event.feedback_type.value,
        score=event.score,
    )


@router.get("/memories/{memory_id}/feedback")
def get_memory_feedback(
    memory_id: str,
    client: MemoryClient = Depends(get_memory_client),
) -> dict[str, Any]:
    """Get aggregated feedback score for a memory."""
    result = client.feedback_manager.get_aggregated_score(memory_id)
    if result is None:
        return {"memory_id": memory_id, "score": 0.5, "event_count": 0}
    return {"memory_id": memory_id, **result}


@router.get("/feedback/stats")
def get_feedback_stats(
    user_id: str = Query(...),
    client: MemoryClient = Depends(get_memory_client),
) -> FeedbackStatsResponse:
    """Get feedback statistics for a user."""
    stats = client.feedback_manager.get_user_feedback_stats(user_id)
    return FeedbackStatsResponse(user_id=user_id, stats=stats)
