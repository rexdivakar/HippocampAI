"""FastAPI dependencies."""

from functools import lru_cache
from typing import Optional

from fastapi import Header

from hippocampai.client import MemoryClient
from hippocampai.config import get_config


@lru_cache
def get_memory_client() -> MemoryClient:
    """Get shared memory client instance."""
    config = get_config()
    return MemoryClient(config=config)


def get_current_user(user_id: Optional[str] = Header(None, alias="X-User-Id")) -> str:
    """
    Get current user ID from header or query parameter.

    For demo/development purposes, returns a default user if not provided.
    In production, this should validate authentication tokens.
    """
    if not user_id:
        # Default demo user for development
        return "demo_user_w1fgba"
    return user_id
