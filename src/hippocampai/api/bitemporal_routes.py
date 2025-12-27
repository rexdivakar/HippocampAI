"""Bi-temporal fact tracking API routes."""

import logging
from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from hippocampai.api.deps import get_memory_client
from hippocampai.client import MemoryClient
from hippocampai.models.bitemporal import (
    BiTemporalFact,
    BiTemporalQueryResult,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/bitemporal", tags=["bi-temporal"])


# Request/Response models
class StoreBiTemporalFactRequest(BaseModel):
    """Request to store a bi-temporal fact."""

    text: str
    user_id: str
    entity_id: Optional[str] = None
    property_name: Optional[str] = None
    event_time: Optional[datetime] = None
    valid_from: Optional[datetime] = None
    valid_to: Optional[datetime] = None
    confidence: float = Field(default=0.9, ge=0.0, le=1.0)
    source: str = "api"
    metadata: Optional[dict[str, Any]] = None


class ReviseFactRequest(BaseModel):
    """Request to revise an existing fact."""

    original_fact_id: str
    new_text: str
    user_id: str
    new_valid_from: Optional[datetime] = None
    new_valid_to: Optional[datetime] = None
    reason: str = "correction"
    confidence: float = Field(default=0.9, ge=0.0, le=1.0)
    metadata: Optional[dict[str, Any]] = None


class RetractFactRequest(BaseModel):
    """Request to retract a fact."""

    fact_id: str
    reason: str = "retracted"


class QueryBiTemporalFactsRequest(BaseModel):
    """Request to query bi-temporal facts."""

    user_id: str
    query: Optional[str] = None
    entity_id: Optional[str] = None
    property_name: Optional[str] = None
    as_of_system_time: Optional[datetime] = None
    valid_at: Optional[datetime] = None
    valid_from: Optional[datetime] = None
    valid_to: Optional[datetime] = None
    include_superseded: bool = False
    include_retracted: bool = False
    limit: int = 100


class GetFactHistoryRequest(BaseModel):
    """Request to get fact history."""

    fact_id: str


class GetLatestValidFactRequest(BaseModel):
    """Request to get latest valid fact."""

    user_id: str
    entity_id: Optional[str] = None
    property_name: Optional[str] = None


# Routes
@router.post("/facts:store", response_model=BiTemporalFact)
def store_bitemporal_fact(
    request: StoreBiTemporalFactRequest,
    client: MemoryClient = Depends(get_memory_client),
) -> BiTemporalFact:
    """Store a fact with bi-temporal tracking.

    Bi-temporal facts track two time dimensions:
    - event_time: When the fact occurred/was stated
    - valid_time: [valid_from, valid_to) interval when fact is true
    - system_time: When HippocampAI recorded it (automatic)
    """
    try:
        fact = client.store_bitemporal_fact(
            text=request.text,
            user_id=request.user_id,
            entity_id=request.entity_id,
            property_name=request.property_name,
            event_time=request.event_time,
            valid_from=request.valid_from,
            valid_to=request.valid_to,
            confidence=request.confidence,
            source=request.source,
            metadata=request.metadata,
        )
        return fact
    except Exception as e:
        logger.error(f"Failed to store bi-temporal fact: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/facts:revise", response_model=BiTemporalFact)
def revise_bitemporal_fact(
    request: ReviseFactRequest,
    client: MemoryClient = Depends(get_memory_client),
) -> BiTemporalFact:
    """Revise an existing bi-temporal fact.

    Creates a new version that supersedes the original without deleting history.
    """
    try:
        fact = client.revise_bitemporal_fact(
            original_fact_id=request.original_fact_id,
            new_text=request.new_text,
            user_id=request.user_id,
            new_valid_from=request.new_valid_from,
            new_valid_to=request.new_valid_to,
            reason=request.reason,
            confidence=request.confidence,
            metadata=request.metadata,
        )
        return fact
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to revise bi-temporal fact: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/facts:retract")
def retract_bitemporal_fact(
    request: RetractFactRequest,
    client: MemoryClient = Depends(get_memory_client),
) -> dict[str, Any]:
    """Retract a bi-temporal fact (mark as invalid without deleting)."""
    try:
        success = client.retract_bitemporal_fact(
            fact_id=request.fact_id,
            reason=request.reason,
        )
        if not success:
            raise HTTPException(status_code=404, detail="Fact not found")
        return {"success": True, "fact_id": request.fact_id, "status": "retracted"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retract bi-temporal fact: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/facts:query", response_model=BiTemporalQueryResult)
def query_bitemporal_facts(
    request: QueryBiTemporalFactsRequest,
    client: MemoryClient = Depends(get_memory_client),
) -> BiTemporalQueryResult:
    """Query bi-temporal facts with temporal filters.

    Supports three query modes:
    1. Current: Get currently valid facts (default)
    2. As-of system time: What did we believe at time T?
    3. Valid-time range: What was valid during [start, end]?
    """
    try:
        result = client.query_bitemporal_facts(
            user_id=request.user_id,
            query=request.query,
            entity_id=request.entity_id,
            property_name=request.property_name,
            as_of_system_time=request.as_of_system_time,
            valid_at=request.valid_at,
            valid_from=request.valid_from,
            valid_to=request.valid_to,
            include_superseded=request.include_superseded,
            include_retracted=request.include_retracted,
            limit=request.limit,
        )
        return result
    except Exception as e:
        logger.error(f"Failed to query bi-temporal facts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/facts:history", response_model=list[BiTemporalFact])
def get_fact_history(
    request: GetFactHistoryRequest,
    client: MemoryClient = Depends(get_memory_client),
) -> list[BiTemporalFact]:
    """Get all versions of a logical fact."""
    try:
        history = client.get_bitemporal_fact_history(request.fact_id)
        return history
    except Exception as e:
        logger.error(f"Failed to get fact history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/facts:latest")
def get_latest_valid_fact(
    request: GetLatestValidFactRequest,
    client: MemoryClient = Depends(get_memory_client),
) -> Optional[BiTemporalFact]:
    """Get the latest valid fact for an entity/property."""
    try:
        fact = client.get_latest_valid_fact(
            user_id=request.user_id,
            entity_id=request.entity_id,
            property_name=request.property_name,
        )
        return fact
    except Exception as e:
        logger.error(f"Failed to get latest valid fact: {e}")
        raise HTTPException(status_code=500, detail=str(e))
