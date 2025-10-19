"""Simple TTL cache."""

from typing import Any, Optional

from cachetools import TTLCache

_cache_instance = None


class Cache:
    def __init__(self, maxsize: int = 10000, ttl: int = 86400):
        self.cache = TTLCache(maxsize=maxsize, ttl=ttl)

    def get(self, key: str) -> Optional[Any]:
        return self.cache.get(key)

    def set(self, key: str, value: Any):
        self.cache[key] = value

    def clear(self):
        self.cache.clear()


def get_cache(maxsize: int = 10000, ttl: int = 86400) -> Cache:
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = Cache(maxsize=maxsize, ttl=ttl)
    return _cache_instance
