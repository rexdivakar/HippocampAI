"""API routes for usage analytics and tenant statistics."""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from hippocampai.api.middleware import require_admin

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/usage", tags=["usage"])


# ============================================
# RESPONSE MODELS
# ============================================


class UsagePeriod(BaseModel):
    """Usage statistics for a time period."""

    period_start: datetime
    period_end: datetime
    api_calls: int = 0
    memories_created: int = 0
    memories_deleted: int = 0
    memories_retrieved: int = 0
    tokens_used: int = 0
    storage_bytes: int = 0
    avg_response_time_ms: float = 0.0


class TenantUsage(BaseModel):
    """Complete usage statistics for a tenant/user."""

    user_id: UUID
    email: Optional[str] = None
    tier: str = "free"

    # Current totals
    total_memories: int = 0
    total_storage_bytes: int = 0
    total_api_calls: int = 0

    # Limits based on tier
    memory_limit: int = 10000
    storage_limit_bytes: int = 1073741824  # 1GB
    api_calls_limit_day: int = 1000

    # Usage percentages
    memory_usage_percent: float = 0.0
    storage_usage_percent: float = 0.0
    api_usage_percent: float = 0.0

    # Time-based usage
    usage_today: Optional[UsagePeriod] = None
    usage_this_week: Optional[UsagePeriod] = None
    usage_this_month: Optional[UsagePeriod] = None

    # Trends
    api_calls_trend: list[dict[str, Any]] = Field(default_factory=list)
    storage_trend: list[dict[str, Any]] = Field(default_factory=list)


class PlatformUsage(BaseModel):
    """Platform-wide usage statistics (admin only)."""

    total_users: int = 0
    active_users_today: int = 0
    active_users_week: int = 0
    active_users_month: int = 0

    total_memories: int = 0
    total_storage_bytes: int = 0
    total_api_calls_today: int = 0

    # By tier breakdown
    users_by_tier: dict[str, int] = Field(default_factory=dict)
    usage_by_tier: dict[str, dict[str, Any]] = Field(default_factory=dict)

    # Top users
    top_users_by_memories: list[dict[str, Any]] = Field(default_factory=list)
    top_users_by_api_calls: list[dict[str, Any]] = Field(default_factory=list)

    # Trends
    daily_api_calls: list[dict[str, Any]] = Field(default_factory=list)
    daily_new_users: list[dict[str, Any]] = Field(default_factory=list)


class APIKeyUsageStats(BaseModel):
    """Usage statistics for an API key."""

    api_key_id: UUID
    key_name: str
    total_calls: int = 0
    calls_today: int = 0
    calls_this_week: int = 0
    last_used: Optional[datetime] = None
    avg_response_time_ms: float = 0.0
    error_rate: float = 0.0
    endpoints_used: dict[str, int] = Field(default_factory=dict)


# ============================================
# TIER LIMITS
# ============================================

TIER_LIMITS = {
    "free": {
        "memories": 1000,
        "storage_bytes": 104857600,  # 100MB
        "api_calls_day": 100,
        "api_calls_month": 1000,
    },
    "pro": {
        "memories": 100000,
        "storage_bytes": 10737418240,  # 10GB
        "api_calls_day": 10000,
        "api_calls_month": 100000,
    },
    "enterprise": {
        "memories": 10000000,
        "storage_bytes": 107374182400,  # 100GB
        "api_calls_day": 1000000,
        "api_calls_month": 10000000,
    },
}


# ============================================
# API ENDPOINTS
# ============================================


@router.get("/me")
async def get_my_usage(
    request: Request,
    user_id: Optional[UUID] = None,
) -> TenantUsage:
    """
    Get usage statistics for the current user.

    This endpoint returns memory counts, storage usage, API call counts,
    and usage trends for the authenticated user.
    """
    # In production, get user_id from auth token
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id required")

    try:
        db_pool = getattr(request.app.state, "db_pool", None)

        # Get user info
        tier = "free"
        email = None

        if db_pool:
            async with db_pool.acquire() as conn:
                user_row = await conn.fetchrow(
                    "SELECT email, tier FROM users WHERE id = $1",
                    user_id,
                )
                if user_row:
                    email = user_row["email"]
                    tier = user_row["tier"]

        limits = TIER_LIMITS.get(tier, TIER_LIMITS["free"])

        # Get memory client for counts
        memory_client = getattr(request.app.state, "memory_client", None)
        total_memories = 0
        total_storage = 0

        if memory_client:
            try:
                stats = memory_client.get_memory_statistics(user_id=str(user_id))
                total_memories = stats.get("total_memories", 0)
                total_storage = stats.get("total_storage_bytes", 0)
            except Exception as e:
                logger.warning(f"Could not get memory stats: {e}")

        # Get API call counts from database
        api_calls_today = 0
        api_calls_month = 0

        if db_pool:
            async with db_pool.acquire() as conn:
                now = datetime.now(timezone.utc)

                # Today's calls
                row = await conn.fetchrow(
                    """
                    SELECT COUNT(*) as count FROM api_key_usage aku
                    JOIN api_keys ak ON aku.api_key_id = ak.id
                    WHERE ak.user_id = $1 AND aku.created_at >= $2
                    """,
                    user_id,
                    now.replace(hour=0, minute=0, second=0, microsecond=0),
                )
                api_calls_today = row["count"] if row else 0

                # This week's calls
                week_start = now - timedelta(days=now.weekday())
                row = await conn.fetchrow(
                    """
                    SELECT COUNT(*) as count FROM api_key_usage aku
                    JOIN api_keys ak ON aku.api_key_id = ak.id
                    WHERE ak.user_id = $1 AND aku.created_at >= $2
                    """,
                    user_id,
                    week_start.replace(hour=0, minute=0, second=0, microsecond=0),
                )
                _ = row["count"] if row else 0  # Week count available if needed

                # This month's calls
                month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
                row = await conn.fetchrow(
                    """
                    SELECT COUNT(*) as count FROM api_key_usage aku
                    JOIN api_keys ak ON aku.api_key_id = ak.id
                    WHERE ak.user_id = $1 AND aku.created_at >= $2
                    """,
                    user_id,
                    month_start,
                )
                api_calls_month = row["count"] if row else 0

        # Calculate percentages
        memory_pct = (total_memories / limits["memories"] * 100) if limits["memories"] else 0
        storage_pct = (
            (total_storage / limits["storage_bytes"] * 100) if limits["storage_bytes"] else 0
        )
        api_pct = (
            (api_calls_today / limits["api_calls_day"] * 100) if limits["api_calls_day"] else 0
        )

        # Build usage periods
        now = datetime.now(timezone.utc)
        usage_today = UsagePeriod(
            period_start=now.replace(hour=0, minute=0, second=0, microsecond=0),
            period_end=now,
            api_calls=api_calls_today,
        )

        return TenantUsage(
            user_id=user_id,
            email=email,
            tier=tier,
            total_memories=total_memories,
            total_storage_bytes=total_storage,
            total_api_calls=api_calls_month,
            memory_limit=limits["memories"],
            storage_limit_bytes=limits["storage_bytes"],
            api_calls_limit_day=limits["api_calls_day"],
            memory_usage_percent=min(memory_pct, 100),
            storage_usage_percent=min(storage_pct, 100),
            api_usage_percent=min(api_pct, 100),
            usage_today=usage_today,
        )

    except Exception as e:
        logger.exception(f"Failed to get usage for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get usage: {str(e)}")


@router.get("/platform", dependencies=[Depends(require_admin)])
async def get_platform_usage(request: Request) -> PlatformUsage:
    """
    Get platform-wide usage statistics (admin only).

    Returns aggregate statistics across all users and tenants.
    """
    try:
        db_pool = getattr(request.app.state, "db_pool", None)

        if not db_pool:
            return PlatformUsage()

        now = datetime.now(timezone.utc)
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        week_start = now - timedelta(days=now.weekday())
        month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

        async with db_pool.acquire() as conn:
            # Total users
            row = await conn.fetchrow("SELECT COUNT(*) as count FROM users")
            total_users = row["count"]

            # Active users
            row = await conn.fetchrow(
                "SELECT COUNT(*) as count FROM users WHERE last_login_at >= $1",
                today_start,
            )
            active_today = row["count"]

            row = await conn.fetchrow(
                "SELECT COUNT(*) as count FROM users WHERE last_login_at >= $1",
                week_start,
            )
            active_week = row["count"]

            row = await conn.fetchrow(
                "SELECT COUNT(*) as count FROM users WHERE last_login_at >= $1",
                month_start,
            )
            active_month = row["count"]

            # Users by tier
            rows = await conn.fetch(
                "SELECT tier, COUNT(*) as count FROM users GROUP BY tier"
            )
            users_by_tier = {row["tier"]: row["count"] for row in rows}

            # API calls today
            row = await conn.fetchrow(
                "SELECT COUNT(*) as count FROM api_key_usage WHERE created_at >= $1",
                today_start,
            )
            api_calls_today = row["count"]

            # Top users by API calls (last 30 days)
            rows = await conn.fetch(
                """
                SELECT u.id, u.email, COUNT(*) as call_count
                FROM api_key_usage aku
                JOIN api_keys ak ON aku.api_key_id = ak.id
                JOIN users u ON ak.user_id = u.id
                WHERE aku.created_at >= $1
                GROUP BY u.id, u.email
                ORDER BY call_count DESC
                LIMIT 10
                """,
                now - timedelta(days=30),
            )
            top_by_calls = [
                {"user_id": str(r["id"]), "email": r["email"], "calls": r["call_count"]}
                for r in rows
            ]

            # Daily API calls trend (last 30 days)
            rows = await conn.fetch(
                """
                SELECT DATE(created_at) as date, COUNT(*) as count
                FROM api_key_usage
                WHERE created_at >= $1
                GROUP BY DATE(created_at)
                ORDER BY date
                """,
                now - timedelta(days=30),
            )
            daily_calls = [
                {"date": r["date"].isoformat(), "count": r["count"]} for r in rows
            ]

            # Daily new users (last 30 days)
            rows = await conn.fetch(
                """
                SELECT DATE(created_at) as date, COUNT(*) as count
                FROM users
                WHERE created_at >= $1
                GROUP BY DATE(created_at)
                ORDER BY date
                """,
                now - timedelta(days=30),
            )
            daily_users = [
                {"date": r["date"].isoformat(), "count": r["count"]} for r in rows
            ]

        return PlatformUsage(
            total_users=total_users,
            active_users_today=active_today,
            active_users_week=active_week,
            active_users_month=active_month,
            total_api_calls_today=api_calls_today,
            users_by_tier=users_by_tier,
            top_users_by_api_calls=top_by_calls,
            daily_api_calls=daily_calls,
            daily_new_users=daily_users,
        )

    except Exception as e:
        logger.exception(f"Failed to get platform usage: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get platform usage: {str(e)}")


@router.get("/api-keys/{key_id}")
async def get_api_key_usage(
    key_id: UUID,
    request: Request,
    days: int = 30,
) -> APIKeyUsageStats:
    """
    Get usage statistics for a specific API key.

    Args:
        key_id: API key ID
        days: Number of days to include in statistics
    """
    try:
        db_pool = getattr(request.app.state, "db_pool", None)

        if not db_pool:
            raise HTTPException(status_code=500, detail="Database not configured")

        now = datetime.now(timezone.utc)
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        week_start = now - timedelta(days=7)
        period_start = now - timedelta(days=days)

        async with db_pool.acquire() as conn:
            # Get key info
            key_row = await conn.fetchrow(
                "SELECT name, last_used_at FROM api_keys WHERE id = $1",
                key_id,
            )
            if not key_row:
                raise HTTPException(status_code=404, detail="API key not found")

            # Total calls in period
            row = await conn.fetchrow(
                """
                SELECT COUNT(*) as count,
                       AVG(response_time_ms) as avg_time,
                       SUM(CASE WHEN status_code >= 400 THEN 1 ELSE 0 END) as errors
                FROM api_key_usage
                WHERE api_key_id = $1 AND created_at >= $2
                """,
                key_id,
                period_start,
            )
            total_calls = row["count"]
            avg_time = row["avg_time"] or 0
            errors = row["errors"] or 0
            error_rate = (errors / total_calls * 100) if total_calls > 0 else 0

            # Today's calls
            row = await conn.fetchrow(
                "SELECT COUNT(*) as count FROM api_key_usage WHERE api_key_id = $1 AND created_at >= $2",
                key_id,
                today_start,
            )
            calls_today = row["count"]

            # This week's calls
            row = await conn.fetchrow(
                "SELECT COUNT(*) as count FROM api_key_usage WHERE api_key_id = $1 AND created_at >= $2",
                key_id,
                week_start,
            )
            calls_week = row["count"]

            # Endpoints used
            rows = await conn.fetch(
                """
                SELECT endpoint, COUNT(*) as count
                FROM api_key_usage
                WHERE api_key_id = $1 AND created_at >= $2
                GROUP BY endpoint
                ORDER BY count DESC
                LIMIT 20
                """,
                key_id,
                period_start,
            )
            endpoints = {r["endpoint"]: r["count"] for r in rows}

        return APIKeyUsageStats(
            api_key_id=key_id,
            key_name=key_row["name"],
            total_calls=total_calls,
            calls_today=calls_today,
            calls_this_week=calls_week,
            last_used=key_row["last_used_at"],
            avg_response_time_ms=avg_time,
            error_rate=error_rate,
            endpoints_used=endpoints,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to get API key usage: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get usage: {str(e)}")


@router.get("/quotas")
async def get_quota_status(
    request: Request,
    user_id: Optional[UUID] = None,
) -> dict[str, Any]:
    """
    Get quota status and warnings for a user.

    Returns current usage vs limits with warning thresholds.
    """
    usage = await get_my_usage(request, user_id)

    warnings = []
    if usage.memory_usage_percent >= 90:
        warnings.append({"type": "memory", "message": "Memory usage above 90%", "level": "critical"})
    elif usage.memory_usage_percent >= 75:
        warnings.append({"type": "memory", "message": "Memory usage above 75%", "level": "warning"})

    if usage.storage_usage_percent >= 90:
        warnings.append({"type": "storage", "message": "Storage usage above 90%", "level": "critical"})
    elif usage.storage_usage_percent >= 75:
        warnings.append({"type": "storage", "message": "Storage usage above 75%", "level": "warning"})

    if usage.api_usage_percent >= 90:
        warnings.append({"type": "api_calls", "message": "Daily API limit nearly reached", "level": "critical"})
    elif usage.api_usage_percent >= 75:
        warnings.append({"type": "api_calls", "message": "Daily API usage above 75%", "level": "warning"})

    return {
        "user_id": str(usage.user_id),
        "tier": usage.tier,
        "quotas": {
            "memories": {
                "used": usage.total_memories,
                "limit": usage.memory_limit,
                "percent": usage.memory_usage_percent,
            },
            "storage": {
                "used": usage.total_storage_bytes,
                "limit": usage.storage_limit_bytes,
                "percent": usage.storage_usage_percent,
            },
            "api_calls_daily": {
                "used": usage.usage_today.api_calls if usage.usage_today else 0,
                "limit": usage.api_calls_limit_day,
                "percent": usage.api_usage_percent,
            },
        },
        "warnings": warnings,
        "upgrade_available": usage.tier != "enterprise",
    }
