"""API routes for Sleep Phase consolidation and dream reports."""

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from hippocampai.api.deps import get_current_user
from hippocampai.consolidation.models import ConsolidationRun, ConsolidationStatus
from hippocampai.consolidation.tasks import consolidate_user_memories

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/consolidation", tags=["consolidation"])


# ============================================
# RESPONSE MODELS
# ============================================


class ConsolidationRunSummary(BaseModel):
    """Summary of a consolidation run for dashboard display."""

    run_id: str
    user_id: str
    status: str
    started_at: datetime
    completed_at: Optional[datetime]
    duration_seconds: float

    memories_reviewed: int
    memories_deleted: int
    memories_archived: int
    memories_promoted: int
    memories_updated: int
    memories_synthesized: int

    dream_report: Optional[str]


class ConsolidationStats(BaseModel):
    """Overall consolidation statistics."""

    total_runs: int
    successful_runs: int
    failed_runs: int
    total_memories_reviewed: int
    total_memories_deleted: int
    total_memories_promoted: int
    total_memories_synthesized: int
    avg_duration_seconds: float
    last_run_at: Optional[datetime]
    next_run_at: Optional[datetime]


class TriggerConsolidationRequest(BaseModel):
    """Request to manually trigger consolidation."""

    dry_run: bool = True
    lookback_hours: int = 24


class ConsolidationConfig(BaseModel):
    """Current consolidation configuration."""

    enabled: bool
    schedule_hour: int
    lookback_hours: int
    min_importance: float
    max_memories_per_user: int
    dry_run: bool


# ============================================
# IN-MEMORY STORAGE (Replace with actual DB)
# ============================================

# TODO: Replace with actual database storage
# For now, using in-memory dict for demo purposes
consolidation_runs_db: dict[str, ConsolidationRun] = {}


def store_consolidation_run(run: ConsolidationRun) -> None:
    """Store consolidation run (replace with actual DB)."""
    consolidation_runs_db[run.id] = run


def get_consolidation_run(run_id: str) -> Optional[ConsolidationRun]:
    """Get consolidation run by ID (replace with actual DB)."""
    return consolidation_runs_db.get(run_id)


def get_user_consolidation_runs(user_id: str, limit: int = 50) -> list[ConsolidationRun]:
    """Get consolidation runs for a user (replace with actual DB)."""
    user_runs = [
        run for run in consolidation_runs_db.values()
        if run.user_id == user_id
    ]
    # Sort by started_at descending
    user_runs.sort(key=lambda x: x.started_at, reverse=True)
    return user_runs[:limit]


# ============================================
# API ENDPOINTS
# ============================================


@router.get("/status")
async def get_consolidation_status(user_id: str = Depends(get_current_user)) -> dict[str, Any]:
    """
    Get the current status of Sleep Phase consolidation.

    Returns whether consolidation is enabled, last run info, etc.
    """
    import os

    enabled = os.getenv("ACTIVE_CONSOLIDATION_ENABLED", "false").lower() == "true"
    dry_run = os.getenv("CONSOLIDATION_DRY_RUN", "false").lower() == "true"
    schedule_hour = int(os.getenv("CONSOLIDATION_SCHEDULE_HOUR", "3"))

    # Get last run for this user
    user_runs = get_user_consolidation_runs(user_id, limit=1)
    last_run = user_runs[0] if user_runs else None

    return {
        "enabled": enabled,
        "dry_run": dry_run,
        "schedule_hour": schedule_hour,
        "last_run": ConsolidationRunSummary(**last_run.dict()) if last_run else None,
        "total_runs": len(get_user_consolidation_runs(user_id)),
    }


@router.get("/runs", response_model=list[ConsolidationRunSummary])
async def get_consolidation_runs(
    user_id: str = Depends(get_current_user),
    limit: int = Query(50, ge=1, le=100),
) -> list[ConsolidationRunSummary]:
    """
    Get consolidation run history for the current user.

    Returns list of past sleep/dream cycles with statistics.
    """
    runs = get_user_consolidation_runs(user_id, limit=limit)
    return [ConsolidationRunSummary(**run.dict()) for run in runs]


@router.get("/runs/{run_id}", response_model=ConsolidationRunSummary)
async def get_consolidation_run_detail(
    run_id: str,
    user_id: str = Depends(get_current_user),
) -> ConsolidationRunSummary:
    """
    Get detailed information about a specific consolidation run.

    Includes full stats, dream report, and affected memories.
    """
    run = get_consolidation_run(run_id)

    if not run:
        raise HTTPException(status_code=404, detail=f"Consolidation run {run_id} not found")

    # Verify ownership
    if run.user_id != user_id:
        raise HTTPException(status_code=403, detail="Access denied")

    return ConsolidationRunSummary(**run.dict())


@router.get("/stats", response_model=ConsolidationStats)
async def get_consolidation_stats(
    user_id: str = Depends(get_current_user),
) -> ConsolidationStats:
    """
    Get aggregated consolidation statistics for the user.

    Shows overall trends and metrics across all sleep cycles.
    """
    runs = get_user_consolidation_runs(user_id, limit=1000)

    if not runs:
        return ConsolidationStats(
            total_runs=0,
            successful_runs=0,
            failed_runs=0,
            total_memories_reviewed=0,
            total_memories_deleted=0,
            total_memories_promoted=0,
            total_memories_synthesized=0,
            avg_duration_seconds=0.0,
            last_run_at=None,
            next_run_at=None,
        )

    successful_runs = [r for r in runs if r.status == ConsolidationStatus.COMPLETED]
    failed_runs = [r for r in runs if r.status == ConsolidationStatus.FAILED]

    total_memories_reviewed = sum(r.memories_reviewed for r in runs)
    total_memories_deleted = sum(r.memories_deleted for r in runs)
    total_memories_promoted = sum(r.memories_promoted for r in runs)
    total_memories_synthesized = sum(r.memories_synthesized for r in runs)

    avg_duration = sum(r.duration_seconds for r in successful_runs) / len(successful_runs) if successful_runs else 0.0

    last_run = runs[0] if runs else None

    # Calculate next run time (next 3 AM)
    import os
    from datetime import timedelta
    schedule_hour = int(os.getenv("CONSOLIDATION_SCHEDULE_HOUR", "3"))
    now = datetime.now(timezone.utc)
    next_run = now.replace(hour=schedule_hour, minute=0, second=0, microsecond=0)
    if next_run < now:
        next_run += timedelta(days=1)

    return ConsolidationStats(
        total_runs=len(runs),
        successful_runs=len(successful_runs),
        failed_runs=len(failed_runs),
        total_memories_reviewed=total_memories_reviewed,
        total_memories_deleted=total_memories_deleted,
        total_memories_promoted=total_memories_promoted,
        total_memories_synthesized=total_memories_synthesized,
        avg_duration_seconds=avg_duration,
        last_run_at=last_run.completed_at if last_run and last_run.completed_at else None,
        next_run_at=next_run,
    )


@router.post("/trigger", response_model=ConsolidationRunSummary)
async def trigger_consolidation(
    request: TriggerConsolidationRequest,
    user_id: str = Depends(get_current_user),
) -> ConsolidationRunSummary:
    """
    Manually trigger a consolidation run for the current user.

    Useful for testing or on-demand memory cleanup.
    By default runs in dry-run mode for safety.
    """
    logger.info(f"Manual consolidation triggered for user {user_id} (dry_run={request.dry_run})")

    try:
        # Trigger Celery task
        task = consolidate_user_memories.delay(
            user_id=user_id,
            lookback_hours=request.lookback_hours,
            dry_run=request.dry_run,
        )

        # Wait for result (with timeout)
        result = task.get(timeout=300)  # 5 minute timeout

        # Convert result dict to ConsolidationRun
        run = ConsolidationRun(**result)

        # Store the run
        store_consolidation_run(run)

        return ConsolidationRunSummary(**run.dict())

    except Exception as e:
        logger.exception(f"Failed to trigger consolidation for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Consolidation failed: {str(e)}")


@router.get("/config", response_model=ConsolidationConfig)
async def get_consolidation_config(
    user_id: str = Depends(get_current_user),
) -> ConsolidationConfig:
    """
    Get current consolidation configuration.

    Returns all configurable settings for the Sleep Phase.
    """
    import os

    return ConsolidationConfig(
        enabled=os.getenv("ACTIVE_CONSOLIDATION_ENABLED", "false").lower() == "true",
        schedule_hour=int(os.getenv("CONSOLIDATION_SCHEDULE_HOUR", "3")),
        lookback_hours=int(os.getenv("CONSOLIDATION_LOOKBACK_HOURS", "24")),
        min_importance=float(os.getenv("CONSOLIDATION_MIN_IMPORTANCE", "3.0")),
        max_memories_per_user=int(os.getenv("CONSOLIDATION_MAX_MEMORIES_PER_USER", "10000")),
        dry_run=os.getenv("CONSOLIDATION_DRY_RUN", "false").lower() == "true",
    )


@router.get("/latest", response_model=Optional[ConsolidationRunSummary])
async def get_latest_consolidation_run(
    user_id: str = Depends(get_current_user),
) -> Optional[ConsolidationRunSummary]:
    """
    Get the most recent consolidation run for dashboard display.

    Returns None if no runs found.
    """
    runs = get_user_consolidation_runs(user_id, limit=1)

    if not runs:
        return None

    return ConsolidationRunSummary(**runs[0].dict())


# ============================================
# MOCK DATA FOR DEMO
# ============================================

def create_mock_consolidation_runs(user_id: str) -> None:
    """Create mock consolidation runs for demo purposes."""
    from datetime import timedelta
    from uuid import uuid4

    # Create 5 mock runs
    for i in range(5):
        run = ConsolidationRun(
            id=str(uuid4()),
            user_id=user_id,
            status=ConsolidationStatus.COMPLETED if i < 4 else ConsolidationStatus.FAILED,
            started_at=datetime.now(timezone.utc) - timedelta(days=i),
            completed_at=datetime.now(timezone.utc) - timedelta(days=i, hours=-1),
            duration_seconds=12.5 + i * 2,
            lookback_hours=24,
            memories_reviewed=50 - i * 5,
            memories_deleted=3 + i,
            memories_archived=2,
            memories_promoted=8 - i,
            memories_updated=4,
            memories_synthesized=2,
            clusters_created=5,
            llm_calls_made=3,
            dream_report=f"Consolidation run #{i+1}: Productive session focusing on strategic planning and team dynamics. Promoted important decisions, archived routine events.",
        )
        store_consolidation_run(run)


# Initialize with mock data on first import (for demo)
# Remove this in production
if not consolidation_runs_db:
    logger.info("Initializing mock consolidation data for demo")
    # This will be populated when first user accesses the API
    pass
