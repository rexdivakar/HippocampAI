"""Memory models and enums."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class MemoryType(str, Enum):
    PREFERENCE = "preference"
    FACT = "fact"
    GOAL = "goal"
    HABIT = "habit"
    EVENT = "event"
    CONTEXT = "context"
    SUMMARY = "summary"  # Compacted/consolidated memories


class Memory(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    text: str
    user_id: str
    session_id: Optional[str] = None
    type: MemoryType
    importance: float = Field(default=5.0, ge=0.0, le=10.0)
    confidence: float = Field(default=0.9, ge=0.0, le=1.0)
    tags: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None  # TTL support
    access_count: int = 0
    text_length: int = 0  # Character count
    token_count: int = 0  # Approximate token count
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Multi-agent support (optional for backward compatibility)
    agent_id: Optional[str] = None
    run_id: Optional[str] = None
    visibility: Optional[str] = None  # "private", "shared", "public"

    # Graph features (optional)
    entities: Optional[dict[str, list[str]]] = None
    facts: Optional[list[str]] = None
    relationships: Optional[list[dict[str, Any]]] = None
    embedding: Optional[list[float]] = None
    rank: Optional[float] = None

    # Alias for backward compatibility
    @property
    def memory_type(self) -> MemoryType:
        """Alias for type field for backward compatibility."""
        return self.type

    def collection_name(self, facts_col: str, prefs_col: str) -> str:
        """Route to appropriate collection."""
        if self.type in {MemoryType.PREFERENCE, MemoryType.GOAL, MemoryType.HABIT}:
            return prefs_col
        return facts_col

    def is_expired(self) -> bool:
        """Check if memory has expired."""
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) > self.expires_at

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Estimate token count (rough approximation: 4 chars ≈ 1 token)."""
        return len(text) // 4

    def calculate_size_metrics(self) -> None:
        """Calculate and update size metrics."""
        self.text_length = len(self.text)
        self.token_count = self.estimate_tokens(self.text)


class RetrievalResult(BaseModel):
    memory: Memory
    score: float
    breakdown: dict[str, Any] = Field(default_factory=dict)


class RetrievalQuery(BaseModel):
    query: str
    user_id: str
    session_id: Optional[str] = None
    k: int = 5
    filters: Optional[dict[str, Any]] = None
