"""REST endpoints for memory triggers."""

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from hippocampai.api.deps import get_memory_client
from hippocampai.client import MemoryClient
from hippocampai.triggers.trigger_manager import (
    Trigger,
    TriggerAction,
    TriggerCondition,
    TriggerEvent,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["triggers"])


class TriggerCreateRequest(BaseModel):
    name: str
    user_id: str
    event: str  # TriggerEvent value
    conditions: list[dict[str, Any]] = Field(default_factory=list)
    action: str = "log"  # TriggerAction value
    action_config: dict[str, Any] = Field(default_factory=dict)


class TriggerResponse(BaseModel):
    id: str
    name: str
    user_id: str
    event: str
    action: str
    enabled: bool
    fired_count: int


@router.post("/triggers", response_model=TriggerResponse)
def register_trigger(
    request: TriggerCreateRequest,
    client: MemoryClient = Depends(get_memory_client),
) -> TriggerResponse:
    """Register a new trigger."""
    try:
        event = TriggerEvent(request.event)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid event: {request.event}. "
            f"Must be one of: {[e.value for e in TriggerEvent]}",
        )

    try:
        action = TriggerAction(request.action)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid action: {request.action}. "
            f"Must be one of: {[a.value for a in TriggerAction]}",
        )

    conditions = [
        TriggerCondition(
            field=c.get("field", ""),
            operator=c.get("operator", "eq"),
            value=c.get("value"),
        )
        for c in request.conditions
    ]

    trigger = Trigger(
        name=request.name,
        user_id=request.user_id,
        event=event,
        conditions=conditions,
        action=action,
        action_config=request.action_config,
    )

    registered = client.trigger_manager.register_trigger(trigger)

    return TriggerResponse(
        id=registered.id,
        name=registered.name,
        user_id=registered.user_id,
        event=registered.event.value,
        action=registered.action.value,
        enabled=registered.enabled,
        fired_count=registered.fired_count,
    )


@router.get("/triggers")
def list_triggers(
    user_id: str = Query(...),
    client: MemoryClient = Depends(get_memory_client),
) -> list[TriggerResponse]:
    """List all triggers for a user."""
    triggers = client.trigger_manager.list_triggers(user_id)
    return [
        TriggerResponse(
            id=t.id,
            name=t.name,
            user_id=t.user_id,
            event=t.event.value,
            action=t.action.value,
            enabled=t.enabled,
            fired_count=t.fired_count,
        )
        for t in triggers
    ]


@router.delete("/triggers/{trigger_id}")
def remove_trigger(
    trigger_id: str,
    user_id: str = Query(...),
    client: MemoryClient = Depends(get_memory_client),
) -> dict[str, Any]:
    """Remove a trigger."""
    removed = client.trigger_manager.remove_trigger(trigger_id, user_id)
    if not removed:
        raise HTTPException(status_code=404, detail="Trigger not found or unauthorized")
    return {"success": True, "trigger_id": trigger_id}


@router.get("/triggers/{trigger_id}/history")
def get_trigger_history(
    trigger_id: str,
    client: MemoryClient = Depends(get_memory_client),
) -> list[dict[str, Any]]:
    """Get fire history for a trigger."""
    trigger = client.trigger_manager.get_trigger(trigger_id)
    if trigger is None:
        raise HTTPException(status_code=404, detail="Trigger not found")

    fires = client.trigger_manager.get_fire_history(trigger_id)
    return [
        {
            "trigger_id": f.trigger_id,
            "memory_id": f.memory_id,
            "event": f.event.value,
            "fired_at": f.fired_at.isoformat(),
            "success": f.success,
            "error": f.error,
        }
        for f in fires
    ]
