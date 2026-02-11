"""REST endpoints for procedural memory (prompt self-optimization)."""

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from hippocampai.api.deps import get_memory_client
from hippocampai.client import MemoryClient

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/procedural", tags=["procedural"])


class RuleResponse(BaseModel):
    id: str
    user_id: str
    rule_text: str
    confidence: float
    success_rate: float
    active: bool


class ExtractRequest(BaseModel):
    user_id: str
    interactions: list[str] = Field(min_length=1)


class InjectRequest(BaseModel):
    user_id: str
    base_prompt: str
    max_rules: int = 5


class InjectResponse(BaseModel):
    prompt: str
    rules_injected: int


class RuleFeedbackRequest(BaseModel):
    was_successful: bool


@router.get("/rules", response_model=list[RuleResponse])
def list_rules(
    user_id: str = Query(...),
    client: MemoryClient = Depends(get_memory_client),
) -> list[RuleResponse]:
    """List active procedural rules for a user."""
    if not client.config.enable_procedural_memory:
        raise HTTPException(status_code=400, detail="Procedural memory is disabled")

    rules = client.procedural.get_active_rules(user_id)
    return [
        RuleResponse(
            id=r.id,
            user_id=r.user_id,
            rule_text=r.rule_text,
            confidence=r.confidence,
            success_rate=r.success_rate,
            active=r.active,
        )
        for r in rules
    ]


@router.post("/extract", response_model=list[RuleResponse])
def extract_rules(
    request: ExtractRequest,
    client: MemoryClient = Depends(get_memory_client),
) -> list[RuleResponse]:
    """Extract procedural rules from interactions."""
    if not client.config.enable_procedural_memory:
        raise HTTPException(status_code=400, detail="Procedural memory is disabled")

    rules = client.procedural.extract_rules(
        user_id=request.user_id,
        recent_interactions=request.interactions,
    )
    return [
        RuleResponse(
            id=r.id,
            user_id=r.user_id,
            rule_text=r.rule_text,
            confidence=r.confidence,
            success_rate=r.success_rate,
            active=r.active,
        )
        for r in rules
    ]


@router.post("/inject", response_model=InjectResponse)
def inject_rules(
    request: InjectRequest,
    client: MemoryClient = Depends(get_memory_client),
) -> InjectResponse:
    """Inject procedural rules into a prompt."""
    if not client.config.enable_procedural_memory:
        raise HTTPException(status_code=400, detail="Procedural memory is disabled")

    result = client.procedural.inject_rules_into_prompt(
        user_id=request.user_id,
        base_prompt=request.base_prompt,
        max_rules=request.max_rules,
    )
    rules_count = len(client.procedural.get_active_rules(request.user_id)[:request.max_rules])
    return InjectResponse(prompt=result, rules_injected=rules_count)


@router.put("/rules/{rule_id}/feedback")
def update_rule_effectiveness(
    rule_id: str,
    request: RuleFeedbackRequest,
    client: MemoryClient = Depends(get_memory_client),
) -> dict[str, Any]:
    """Update rule effectiveness based on feedback."""
    if not client.config.enable_procedural_memory:
        raise HTTPException(status_code=400, detail="Procedural memory is disabled")

    rule = client.procedural.update_rule_effectiveness(rule_id, request.was_successful)
    if rule is None:
        raise HTTPException(status_code=404, detail="Rule not found")
    return {
        "id": rule.id,
        "success_rate": rule.success_rate,
        "updated_at": rule.updated_at.isoformat(),
    }


@router.post("/consolidate")
def consolidate_rules(
    user_id: str = Query(...),
    client: MemoryClient = Depends(get_memory_client),
) -> dict[str, Any]:
    """Consolidate (merge redundant) rules for a user."""
    if not client.config.enable_procedural_memory:
        raise HTTPException(status_code=400, detail="Procedural memory is disabled")

    rules = client.procedural.consolidate_rules(user_id)
    return {
        "user_id": user_id,
        "consolidated_count": len(rules),
    }
