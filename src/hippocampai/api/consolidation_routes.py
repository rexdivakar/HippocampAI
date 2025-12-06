"""API routes for Sleep Phase consolidation and dream reports."""

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from hippocampai.api.deps import get_current_user
from hippocampai.consolidation.db import get_db
from hippocampai.consolidation.models import ConsolidationRun
from hippocampai.consolidation.tasks import _run_consolidation_sync

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/consolidation", tags=["consolidation"])


# ============================================
# RESPONSE MODELS
# ============================================


class ConsolidationRunSummary(BaseModel):
    """Summary of a consolidation run for dashboard display."""

    id: str  # Changed from run_id to match ConsolidationRun model
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
# DATABASE STORAGE (SQLite/Postgres)
# ============================================


def store_consolidation_run(run: ConsolidationRun) -> None:
    """Store consolidation run in database."""
    db = get_db()
    db.create_run(run)
    logger.debug(f"Stored consolidation run {run.id} in database")


def get_consolidation_run(run_id: str) -> Optional[ConsolidationRun]:
    """Get consolidation run by ID from database."""
    db = get_db()
    run_data = db.get_run(run_id)
    if run_data:
        return ConsolidationRun(**run_data)
    return None


def get_user_consolidation_runs(user_id: str, limit: int = 50) -> list[ConsolidationRun]:
    """Get consolidation runs for a user from database."""
    db = get_db()
    runs_data = db.get_user_runs(user_id, limit=limit)
    return [ConsolidationRun(**run_data) for run_data in runs_data]


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
    db = get_db()
    stats = db.get_stats(user_id)

    # Calculate next run time (next 3 AM)
    import os
    from datetime import timedelta
    schedule_hour = int(os.getenv("CONSOLIDATION_SCHEDULE_HOUR", "3"))
    now = datetime.now(timezone.utc)
    next_run = now.replace(hour=schedule_hour, minute=0, second=0, microsecond=0)
    if next_run < now:
        next_run += timedelta(days=1)

    # Parse last_run_at from string if needed (SQLite returns strings)
    last_run_at = None
    if stats.get("last_run_at"):
        from dateutil import parser
        last_run_at = parser.parse(stats["last_run_at"]) if isinstance(stats["last_run_at"], str) else stats["last_run_at"]

    return ConsolidationStats(
        total_runs=stats.get("total_runs", 0),
        successful_runs=stats.get("successful_runs", 0),
        failed_runs=stats.get("failed_runs", 0),
        total_memories_reviewed=stats.get("total_memories_reviewed", 0),
        total_memories_deleted=stats.get("total_memories_deleted", 0),
        total_memories_promoted=stats.get("total_memories_promoted", 0),
        total_memories_synthesized=stats.get("total_memories_synthesized", 0),
        avg_duration_seconds=stats.get("avg_duration_seconds", 0.0) or 0.0,
        last_run_at=last_run_at,
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
        # Run consolidation synchronously (no Celery required for manual triggers)
        result = _run_consolidation_sync(
            user_id=user_id,
            lookback_hours=request.lookback_hours,
            dry_run=request.dry_run,
        )

        # Convert result dict to ConsolidationRun
        run = ConsolidationRun(**result)

        # Note: run is already stored in the database by _run_consolidation_sync

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
# DATABASE INITIALIZED
# ============================================

# Database is initialized automatically via get_db()
# SQLite database will be created at data/consolidation.db
# To use Postgres, set CONSOLIDATION_DB_TYPE=postgres
