"""Redis-based rate limiter using sliding window algorithm."""

import time
from uuid import UUID

from redis.asyncio import Redis

from hippocampai.auth.models import RateLimitInfo


class RateLimiter:
    """Rate limiter using Redis sliding window algorithm."""

    # Rate limits per tier (requests per time window)
    TIER_LIMITS = {
        "free": {
            "minute": 10,
            "hour": 100,
            "day": 1000,
        },
        "pro": {
            "minute": 100,
            "hour": 10000,
            "day": 100000,
        },
        "enterprise": {
            "minute": 1000,
            "hour": 100000,
            "day": 1000000,
        },
        "admin": {
            "minute": 10000,
            "hour": 1000000,
            "day": 10000000,
        },
    }

    # Window durations in seconds
    WINDOW_SECONDS = {
        "minute": 60,
        "hour": 3600,
        "day": 86400,
    }

    def __init__(self, redis: Redis):
        """Initialize with Redis connection.

        Args:
            redis: Redis async client
        """
        self.redis = redis

    async def check_rate_limit(
        self, api_key_id: UUID, tier: str = "free"
    ) -> tuple[bool, RateLimitInfo]:
        """Check if request is allowed under rate limits.

        Uses sliding window algorithm with multiple time windows.
        All windows must be under their respective limits.

        Args:
            api_key_id: API key ID
            tier: User tier (free, pro, enterprise, admin)

        Returns:
            Tuple of (allowed, rate_limit_info)
            - allowed: True if request is allowed
            - rate_limit_info: RateLimitInfo with current limits
        """
        # Get limits for tier
        limits = self.TIER_LIMITS.get(tier, self.TIER_LIMITS["free"])

        # Check all windows
        for window, limit in limits.items():
            key = f"rate_limit:{api_key_id}:{window}"
            window_seconds = self.WINDOW_SECONDS[window]

            # Get current count
            count = await self.redis.get(key)
            current_count = int(count) if count else 0

            # Check if over limit
            if current_count >= limit:
                # Calculate reset time
                ttl = await self.redis.ttl(key)
                reset_at = int(time.time()) + (ttl if ttl > 0 else window_seconds)

                return False, RateLimitInfo(
                    limit=limit,
                    remaining=0,
                    reset_at=reset_at,
                    window=window,
                )

        # All windows OK - increment counters atomically
        pipe = self.redis.pipeline()
        for window, limit in limits.items():
            key = f"rate_limit:{api_key_id}:{window}"
            window_seconds = self.WINDOW_SECONDS[window]

            pipe.incr(key)
            pipe.expire(key, window_seconds)

        await pipe.execute()

        # Return info for most restrictive window (minute)
        minute_key = f"rate_limit:{api_key_id}:minute"
        minute_count = await self.redis.get(minute_key)
        current_count = int(minute_count) if minute_count else 1

        minute_limit = limits["minute"]
        ttl = await self.redis.ttl(minute_key)
        reset_at = int(time.time()) + (ttl if ttl > 0 else 60)

        return True, RateLimitInfo(
            limit=minute_limit,
            remaining=max(0, minute_limit - current_count),
            reset_at=reset_at,
            window="minute",
        )

    async def get_current_usage(self, api_key_id: UUID, tier: str = "free") -> dict[str, dict]:
        """Get current usage across all windows.

        Args:
            api_key_id: API key ID
            tier: User tier

        Returns:
            Dict mapping window -> {count, limit, remaining, reset_at}
        """
        limits = self.TIER_LIMITS.get(tier, self.TIER_LIMITS["free"])
        usage = {}

        for window, limit in limits.items():
            key = f"rate_limit:{api_key_id}:{window}"
            count = await self.redis.get(key)
            current_count = int(count) if count else 0

            ttl = await self.redis.ttl(key)
            window_seconds = self.WINDOW_SECONDS[window]
            reset_at = int(time.time()) + (ttl if ttl > 0 else window_seconds)

            usage[window] = {
                "count": current_count,
                "limit": limit,
                "remaining": max(0, limit - current_count),
                "reset_at": reset_at,
            }

        return usage

    async def reset_limits(self, api_key_id: UUID) -> None:
        """Reset all rate limit counters for an API key.

        Useful for testing or emergency rate limit resets.

        Args:
            api_key_id: API key ID
        """
        for window in self.WINDOW_SECONDS.keys():
            key = f"rate_limit:{api_key_id}:{window}"
            await self.redis.delete(key)

    async def set_custom_limit(
        self, api_key_id: UUID, window: str, limit: int, duration_seconds: int
    ) -> None:
        """Set a custom rate limit for a specific key.

        Args:
            api_key_id: API key ID
            window: Window name (e.g., 'custom_5min')
            limit: Request limit
            duration_seconds: Window duration in seconds
        """
        key = f"rate_limit:{api_key_id}:{window}"
        await self.redis.set(key, 0, ex=duration_seconds)

    def get_tier_limits(self, tier: str = "free") -> dict:
        """Get rate limits for a tier.

        Args:
            tier: User tier

        Returns:
            Dict mapping window -> limit
        """
        return self.TIER_LIMITS.get(tier, self.TIER_LIMITS["free"])
