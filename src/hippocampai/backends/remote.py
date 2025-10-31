"""Remote backend for connecting to HippocampAI SaaS API."""

import logging
from datetime import datetime
from typing import Any, Optional

import httpx

from hippocampai.backends.base import BaseBackend
from hippocampai.models.memory import Memory, MemoryType, RetrievalResult

logger = logging.getLogger(__name__)


class RemoteBackend(BaseBackend):
    """Remote backend that connects to HippocampAI SaaS API via HTTP."""

    def __init__(
        self,
        api_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        timeout: int = 30,
    ):
        """
        Initialize remote backend.

        Args:
            api_url: Base URL of the HippocampAI API server
            api_key: Optional API key for authentication
            timeout: Request timeout in seconds
        """
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

        logger.info(f"Initialized RemoteBackend: {self.api_url}")

    def _get(self, endpoint: str, params: Optional[dict[str, Any]] = None) -> Any:
        """Make GET request."""
        url = f"{self.api_url}{endpoint}"
        with httpx.Client(timeout=self.timeout) as client:
            response = client.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            return response.json()

    def _post(self, endpoint: str, data: dict[str, Any]) -> Any:
        """Make POST request."""
        url = f"{self.api_url}{endpoint}"
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(url, headers=self.headers, json=data)
            response.raise_for_status()
            return response.json()

    def _patch(self, endpoint: str, data: dict[str, Any]) -> Any:
        """Make PATCH request."""
        url = f"{self.api_url}{endpoint}"
        with httpx.Client(timeout=self.timeout) as client:
            response = client.patch(url, headers=self.headers, json=data)
            response.raise_for_status()
            return response.json()

    def _delete(self, endpoint: str) -> bool:
        """Make DELETE request."""
        url = f"{self.api_url}{endpoint}"
        with httpx.Client(timeout=self.timeout) as client:
            response = client.delete(url, headers=self.headers)
            response.raise_for_status()
            return response.status_code == 204

    def _dict_to_memory(self, data: dict[str, Any]) -> Memory:
        """Convert API response dict to Memory object."""
        # Handle both 'type' and 'memory_type' keys for backward compatibility
        mem_type_str = data.get("type") or data.get("memory_type", "fact")
        return Memory(
            id=data["id"],
            text=data["text"],
            user_id=data["user_id"],
            session_id=data.get("session_id"),
            type=MemoryType(mem_type_str),
            metadata=data.get("metadata", {}),
            tags=data.get("tags", []),
            importance=data.get("importance", 0.5),
            created_at=datetime.fromisoformat(data["created_at"])
            if "created_at" in data
            else datetime.now(),
            updated_at=datetime.fromisoformat(data["updated_at"])
            if "updated_at" in data
            else datetime.now(),
            expires_at=datetime.fromisoformat(data["expires_at"])
            if data.get("expires_at")
            else None,
            entities=data.get("entities"),
            facts=data.get("facts"),
            relationships=data.get("relationships"),
            embedding=data.get("embedding"),
            rank=data.get("rank"),
        )

    def _dict_to_retrieval_result(self, data: dict[str, Any]) -> RetrievalResult:
        """Convert API response dict to RetrievalResult object."""
        # Rank is stored in memory, not in RetrievalResult
        memory_data = data["memory"]
        if "rank" in data:
            memory_data["rank"] = data["rank"]
        return RetrievalResult(
            memory=self._dict_to_memory(memory_data),
            score=data["score"],
            breakdown=data.get("breakdown", {}),
        )

    def remember(
        self,
        text: str,
        user_id: str,
        session_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        tags: Optional[list[str]] = None,
        importance: Optional[float] = None,
        expires_at: Optional[datetime] = None,
        extract_entities: bool = False,
        extract_facts: bool = False,
        extract_relationships: bool = False,
    ) -> Memory:
        """Store a memory via API."""
        payload = {
            "text": text,
            "user_id": user_id,
            "session_id": session_id,
            "metadata": metadata or {},
            "tags": tags or [],
            "importance": importance,
            "expires_at": expires_at.isoformat() if expires_at else None,
            "extract_entities": extract_entities,
            "extract_facts": extract_facts,
            "extract_relationships": extract_relationships,
        }
        # Remove None values
        payload = {k: v for k, v in payload.items() if v is not None}

        data = self._post("/v1/memories", payload)
        return self._dict_to_memory(data)

    def recall(
        self,
        query: str,
        user_id: str,
        session_id: Optional[str] = None,
        limit: int = 10,
        filters: Optional[dict[str, Any]] = None,
        min_score: float = 0.0,
    ) -> list[RetrievalResult]:
        """Retrieve relevant memories via API."""
        payload = {
            "query": query,
            "user_id": user_id,
            "session_id": session_id,
            "limit": limit,
            "filters": filters,
            "min_score": min_score,
        }
        # Remove None values
        payload = {k: v for k, v in payload.items() if v is not None}

        data = self._post("/v1/memories/recall", payload)
        return [self._dict_to_retrieval_result(item) for item in data]

    def get_memory(self, memory_id: str) -> Optional[Memory]:
        """Get a memory by ID via API."""
        try:
            data = self._get(f"/v1/memories/{memory_id}")
            return self._dict_to_memory(data)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            raise

    def get_memories(
        self,
        user_id: str,
        session_id: Optional[str] = None,
        limit: int = 100,
        filters: Optional[dict[str, Any]] = None,
        min_importance: Optional[float] = None,
        after: Optional[datetime] = None,
        before: Optional[datetime] = None,
    ) -> list[Memory]:
        """Get all memories for a user via API."""
        params = {
            "user_id": user_id,
            "session_id": session_id,
            "limit": limit,
            "min_importance": min_importance,
            "after": after.isoformat() if after else None,
            "before": before.isoformat() if before else None,
        }
        # Add filters to params
        if filters:
            for key, value in filters.items():
                params[key] = value

        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}

        data = self._get("/v1/memories", params)
        return [self._dict_to_memory(item) for item in data]

    def update_memory(
        self,
        memory_id: str,
        text: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        tags: Optional[list[str]] = None,
        importance: Optional[float] = None,
        expires_at: Optional[datetime] = None,
    ) -> Optional[Memory]:
        """Update a memory via API."""
        payload = {
            "text": text,
            "metadata": metadata,
            "tags": tags,
            "importance": importance,
            "expires_at": expires_at.isoformat() if expires_at else None,
        }
        # Remove None values
        payload = {k: v for k, v in payload.items() if v is not None}

        try:
            data = self._patch(f"/v1/memories/{memory_id}", payload)
            return self._dict_to_memory(data)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            raise

    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory via API."""
        try:
            return self._delete(f"/v1/memories/{memory_id}")
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return False
            raise

    def batch_remember(self, memories: list[dict[str, Any]]) -> list[Memory]:
        """Store multiple memories via API."""
        payload = {"memories": memories}
        data = self._post("/v1/memories/batch", payload)
        return [self._dict_to_memory(item) for item in data]

    def batch_get_memories(self, memory_ids: list[str]) -> list[Memory]:
        """Get multiple memories by IDs via API."""
        payload = {"memory_ids": memory_ids}
        data = self._post("/v1/memories/batch/get", payload)
        return [self._dict_to_memory(item) for item in data]

    def batch_delete_memories(self, memory_ids: list[str]) -> bool:
        """Delete multiple memories via API."""
        payload = {"memory_ids": memory_ids}
        try:
            self._post("/v1/memories/batch/delete", payload)
            return True
        except Exception as e:
            logger.error(f"Batch delete failed: {e}")
            return False

    def consolidate_memories(
        self, user_id: str, session_id: Optional[str] = None
    ) -> list[dict[str, Any]]:
        """Consolidate related memories via API."""
        payload = {"user_id": user_id, "session_id": session_id}
        payload = {k: v for k, v in payload.items() if v is not None}
        return self._post("/v1/memories/consolidate", payload)

    def cleanup_expired_memories(self) -> int:
        """Remove expired memories via API."""
        data = self._post("/v1/memories/cleanup", {})
        return data.get("deleted_count", 0)

    def get_memory_analytics(self, user_id: str) -> dict[str, Any]:
        """Get analytics for user's memories via API."""
        params = {"user_id": user_id}
        return self._get("/v1/memories/analytics", params)

    def health_check(self) -> dict[str, Any]:
        """Check API server health."""
        return self._get("/health")
