"""REST endpoints for prospective memory (remembering to remember)."""

import logging
from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from hippocampai.api.deps import get_memory_client
from hippocampai.client import MemoryClient

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/prospective", tags=["prospective"])


# -- Request / Response models ------------------------------------------------


class IntentResponse(BaseModel):
    id: str
    user_id: str
    intent_text: str
    action_description: str
    trigger_type: str
    status: str
    priority: int
    trigger_count: int
    created_at: datetime
    triggered_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    context_keywords: list[str] = Field(default_factory=list)
    recurrence: str = "none"
    tags: list[str] = Field(default_factory=list)


class CreateIntentRequest(BaseModel):
    user_id: str
    intent_text: str
    trigger_type: str = "event_based"
    action_description: str = ""
    trigger_at: Optional[datetime] = None
    trigger_cron: Optional[str] = None
    time_window_start: Optional[datetime] = None
    time_window_end: Optional[datetime] = None
    context_keywords: Optional[list[str]] = None
    context_pattern: Optional[str] = None
    similarity_threshold: float = 0.75
    recurrence: str = "none"
    recurrence_cron: Optional[str] = None
    remaining_occurrences: Optional[int] = None
    priority: int = 5
    expires_at: Optional[datetime] = None
    tags: Optional[list[str]] = None
    source_conversation: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None


class ParseIntentRequest(BaseModel):
    user_id: str
    text: str


class EvaluateContextRequest(BaseModel):
    user_id: str
    context_text: str
    context_embedding: Optional[list[float]] = None


class ConsolidateRequest(BaseModel):
    user_id: str


class ExpireRequest(BaseModel):
    user_id: Optional[str] = None


# -- Helpers -------------------------------------------------------------------


def _intent_to_response(intent: Any) -> IntentResponse:
    return IntentResponse(
        id=intent.id,
        user_id=intent.user_id,
        intent_text=intent.intent_text,
        action_description=intent.action_description,
        trigger_type=intent.trigger_type.value,
        status=intent.status.value,
        priority=intent.priority,
        trigger_count=intent.trigger_count,
        created_at=intent.created_at,
        triggered_at=intent.triggered_at,
        expires_at=intent.expires_at,
        context_keywords=intent.context_keywords,
        recurrence=intent.recurrence.value,
        tags=intent.tags,
    )


def _check_enabled(client: MemoryClient) -> None:
    if not client.config.enable_prospective_memory:
        raise HTTPException(status_code=400, detail="Prospective memory is disabled")


# -- Routes --------------------------------------------------------------------


@router.post("/intents", response_model=IntentResponse)
def create_intent(
    request: CreateIntentRequest,
    client: MemoryClient = Depends(get_memory_client),
) -> IntentResponse:
    """Create a prospective intent with explicit fields."""
    _check_enabled(client)

    from hippocampai.prospective.prospective_memory import (
        ProspectiveTriggerType,
        RecurrencePattern,
    )

    intent = client.prospective.create_intent(
        user_id=request.user_id,
        intent_text=request.intent_text,
        trigger_type=ProspectiveTriggerType(request.trigger_type),
        action_description=request.action_description,
        trigger_at=request.trigger_at,
        trigger_cron=request.trigger_cron,
        time_window_start=request.time_window_start,
        time_window_end=request.time_window_end,
        context_keywords=request.context_keywords,
        context_pattern=request.context_pattern,
        similarity_threshold=request.similarity_threshold,
        recurrence=RecurrencePattern(request.recurrence),
        recurrence_cron=request.recurrence_cron,
        remaining_occurrences=request.remaining_occurrences,
        priority=request.priority,
        expires_at=request.expires_at,
        tags=request.tags,
        source_conversation=request.source_conversation,
        metadata=request.metadata,
    )
    return _intent_to_response(intent)


@router.post("/intents:parse", response_model=IntentResponse)
def parse_intent(
    request: ParseIntentRequest,
    client: MemoryClient = Depends(get_memory_client),
) -> IntentResponse:
    """Create a prospective intent from natural language."""
    _check_enabled(client)
    intent = client.prospective.create_intent_from_natural_language(
        user_id=request.user_id,
        text=request.text,
    )
    return _intent_to_response(intent)


@router.get("/intents", response_model=list[IntentResponse])
def list_intents(
    user_id: str = Query(...),
    status: Optional[str] = Query(default=None),
    client: MemoryClient = Depends(get_memory_client),
) -> list[IntentResponse]:
    """List prospective intents for a user."""
    _check_enabled(client)

    status_filter = None
    if status:
        from hippocampai.prospective.prospective_memory import ProspectiveStatus

        status_filter = ProspectiveStatus(status)

    intents = client.prospective.list_intents(user_id, status=status_filter)
    return [_intent_to_response(i) for i in intents]


@router.get("/intents/{intent_id}", response_model=IntentResponse)
def get_intent(
    intent_id: str,
    client: MemoryClient = Depends(get_memory_client),
) -> IntentResponse:
    """Get a single prospective intent."""
    _check_enabled(client)
    intent = client.prospective.get_intent(intent_id)
    if intent is None:
        raise HTTPException(status_code=404, detail="Intent not found")
    return _intent_to_response(intent)


@router.put("/intents/{intent_id}/cancel", response_model=IntentResponse)
def cancel_intent(
    intent_id: str,
    user_id: str = Query(...),
    client: MemoryClient = Depends(get_memory_client),
) -> IntentResponse:
    """Cancel a pending intent."""
    _check_enabled(client)
    intent = client.prospective.cancel_intent(intent_id, user_id)
    if intent is None:
        raise HTTPException(status_code=404, detail="Intent not found or cannot be cancelled")
    return _intent_to_response(intent)


@router.put("/intents/{intent_id}/complete", response_model=IntentResponse)
def complete_intent(
    intent_id: str,
    user_id: str = Query(...),
    client: MemoryClient = Depends(get_memory_client),
) -> IntentResponse:
    """Mark a triggered intent as completed."""
    _check_enabled(client)
    intent = client.prospective.complete_intent(intent_id, user_id)
    if intent is None:
        raise HTTPException(status_code=404, detail="Intent not found or cannot be completed")
    return _intent_to_response(intent)


@router.post("/evaluate", response_model=list[IntentResponse])
def evaluate_context(
    request: EvaluateContextRequest,
    client: MemoryClient = Depends(get_memory_client),
) -> list[IntentResponse]:
    """Evaluate context against pending event-based intents."""
    _check_enabled(client)
    triggered = client.prospective.evaluate_context(
        user_id=request.user_id,
        context_text=request.context_text,
        context_embedding=request.context_embedding,
    )
    return [_intent_to_response(i) for i in triggered]


@router.post("/consolidate")
def consolidate_intents(
    request: ConsolidateRequest,
    client: MemoryClient = Depends(get_memory_client),
) -> dict[str, Any]:
    """Consolidate redundant intents for a user."""
    _check_enabled(client)
    consolidated = client.prospective.consolidate_intents(request.user_id)
    return {
        "user_id": request.user_id,
        "consolidated_count": len(consolidated),
    }


@router.post("/expire")
def expire_intents(
    request: ExpireRequest,
    client: MemoryClient = Depends(get_memory_client),
) -> dict[str, Any]:
    """Force-expire stale intents."""
    _check_enabled(client)
    expired_count = client.prospective.expire_stale_intents(user_id=request.user_id)
    return {
        "expired_count": expired_count,
    }
