"""Search-related models and enums."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class SearchMode(str, Enum):
    """Search mode for retrieval."""

    HYBRID = "hybrid"  # Vector + BM25 with RRF fusion
    VECTOR_ONLY = "vector_only"  # Vector search only
    KEYWORD_ONLY = "keyword_only"  # BM25 keyword search only


class SavedSearch(BaseModel):
    """Saved search query for quick retrieval."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    query: str
    user_id: str
    search_mode: SearchMode = SearchMode.HYBRID
    enable_reranking: bool = True
    filters: dict[str, Any] = Field(default_factory=dict)
    k: int = 5
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_used_at: Optional[datetime] = None
    use_count: int = 0
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def increment_usage(self) -> None:
        """Increment usage count and update last used timestamp."""
        self.use_count += 1
        self.last_used_at = datetime.now(timezone.utc)


class SearchSuggestion(BaseModel):
    """Auto-suggested search query based on history."""

    query: str
    confidence: float = Field(ge=0.0, le=1.0)
    frequency: int = 0  # How many times similar queries were used
    last_used: Optional[datetime] = None
    tags: list[str] = Field(default_factory=list)


class RetrievalMode(BaseModel):
    """Configuration for retrieval behavior."""

    search_mode: SearchMode = SearchMode.HYBRID
    enable_reranking: bool = True
    enable_score_breakdown: bool = True
    top_k_qdrant: int = 200
    top_k_final: int = 20
    weights: dict[str, float] = Field(
        default_factory=lambda: {"sim": 0.55, "rerank": 0.20, "recency": 0.15, "importance": 0.10}
    )


class RetentionPolicy(BaseModel):
    """Retention policy for automatic memory cleanup."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    user_id: Optional[str] = None  # None = global policy
    memory_type: Optional[str] = None  # None = all types
    retention_days: int = Field(ge=1)  # Days to keep
    min_importance: Optional[float] = None  # Keep if importance >= threshold
    min_access_count: Optional[int] = None  # Keep if accessed >= N times
    tags_to_preserve: list[str] = Field(default_factory=list)  # Keep memories with these tags
    enabled: bool = True
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_run_at: Optional[datetime] = None
    deleted_count: int = 0

    def should_delete(self, memory_data: dict[str, Any]) -> bool:
        """
        Check if a memory should be deleted based on this policy.

        Args:
            memory_data: Memory data dictionary with created_at, importance, access_count, tags

        Returns:
            True if memory should be deleted, False otherwise
        """
        if not self.enabled:
            return False

        # Check if memory is old enough
        created_at = memory_data.get("created_at")
        if not created_at:
            return False

        if isinstance(created_at, str):
            from hippocampai.utils.time import parse_iso_datetime

            created_at = parse_iso_datetime(created_at)

        age_days = (datetime.now(timezone.utc) - created_at).days
        if age_days < self.retention_days:
            return False

        # Preserve based on importance
        if self.min_importance is not None:
            importance = memory_data.get("importance", 5.0)
            if importance >= self.min_importance:
                return False

        # Preserve based on access count
        if self.min_access_count is not None:
            access_count = memory_data.get("access_count", 0)
            if access_count >= self.min_access_count:
                return False

        # Preserve based on tags
        if self.tags_to_preserve:
            memory_tags = set(memory_data.get("tags", []))
            if any(tag in memory_tags for tag in self.tags_to_preserve):
                return False

        return True
