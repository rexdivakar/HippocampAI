"""FastAPI dependencies."""

from functools import lru_cache
from hippocampai.client import MemoryClient
from hippocampai.config import get_config


@lru_cache()
def get_memory_client() -> MemoryClient:
    """Get shared memory client instance."""
    config = get_config()
    return MemoryClient(config=config)
