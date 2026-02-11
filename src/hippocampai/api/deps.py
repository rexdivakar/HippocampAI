"""FastAPI dependencies with authentication and rate limiting."""

import os
import time
from collections import defaultdict
from typing import Optional

from fastapi import Header, HTTPException, Query

from hippocampai.client import MemoryClient
from hippocampai.config import get_config

_client: Optional[MemoryClient] = None


def get_memory_client() -> MemoryClient:
    """Get shared memory client instance (singleton)."""
    global _client
    if _client is None:
        config = get_config()
        _client = MemoryClient(config=config)
    return _client


# ============================================
# Rate Limiting
# ============================================


class RateLimiter:
    """Simple in-memory rate limiter to prevent brute force attacks."""

    def __init__(self):
        self.requests = defaultdict(list)  # {user_id: [timestamp1, timestamp2, ...]}
        self.rate_limit = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))

    def check_rate_limit(self, user_id: str) -> bool:
        """Check if user has exceeded rate limit. Returns True if allowed."""
        now = time.time()
        minute_ago = now - 60

        # Remove old requests
        self.requests[user_id] = [t for t in self.requests[user_id] if t > minute_ago]

        # Check if under limit
        if len(self.requests[user_id]) >= self.rate_limit:
            return False

        # Add current request
        self.requests[user_id].append(now)
        return True


# Global rate limiter instance
_rate_limiter = RateLimiter()


# ============================================
# Authentication
# ============================================


def get_current_user(
    user_id: Optional[str] = Header(None, alias="X-User-Id"),
    api_key: Optional[str] = Header(None, alias="X-API-Key"),
    user_id_query: Optional[str] = Query(None, alias="user_id"),
) -> str:
    """
    Validate and get current user ID with optional API key verification.

    Security features:
    - Validates user_id against whitelist (if configured)
    - Optional API key validation
    - Rate limiting per user_id to prevent brute force
    - Configurable via environment variables

    Headers:
        X-User-Id: User identifier (required)
        X-API-Key: API key (optional, based on AUTH_API_KEY_REQUIRED)

    Query params:
        user_id: Alternative to X-User-Id header

    Raises:
        HTTPException: 401 if authentication fails, 429 if rate limited
    """
    # Get user_id from header or query param
    effective_user_id = user_id or user_id_query

    # Check if auth is enabled
    auth_enabled = os.getenv("AUTH_ENABLED", "false").lower() == "true"

    if not auth_enabled:
        # Auth disabled - return user_id or default
        if not effective_user_id:
            return "demo_user_w1fgba"
        return effective_user_id

    # Auth enabled - validate
    if not effective_user_id:
        raise HTTPException(
            status_code=401,
            detail="Missing user_id. Provide via X-User-Id header or user_id query parameter.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Rate limiting check
    if not _rate_limiter.check_rate_limit(effective_user_id):
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Maximum {_rate_limiter.rate_limit} requests per minute per user.",
        )

    # Validate user_id against whitelist
    valid_user_ids = os.getenv("VALID_USER_IDS", "")
    if valid_user_ids:
        allowed_users = [uid.strip() for uid in valid_user_ids.split(",") if uid.strip()]
        if allowed_users and effective_user_id not in allowed_users:
            raise HTTPException(
                status_code=401,
                detail="Invalid user_id. User not authorized.",
                headers={"WWW-Authenticate": "Bearer"},
            )

    # Optional API key validation
    api_key_required = os.getenv("API_KEY_REQUIRED", "false").lower() == "true"
    if api_key_required:
        if not api_key:
            raise HTTPException(
                status_code=401,
                detail="Missing API key. Provide via X-API-Key header.",
                headers={"WWW-Authenticate": "Bearer"},
            )

        valid_api_keys = os.getenv("VALID_API_KEYS", "")
        if valid_api_keys:
            allowed_keys = [key.strip() for key in valid_api_keys.split(",") if key.strip()]
            if api_key not in allowed_keys:
                raise HTTPException(
                    status_code=401,
                    detail="Invalid API key.",
                    headers={"WWW-Authenticate": "Bearer"},
                )

    return effective_user_id
