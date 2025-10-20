"""Memory models and enums."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class MemoryType(str, Enum):
    PREFERENCE = "preference"
    FACT = "fact"
    GOAL = "goal"
    HABIT = "habit"
    EVENT = "event"
    CONTEXT = "context"


class Memory(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    text: str
    user_id: str
    session_id: Optional[str] = None
    type: MemoryType
    importance: float = Field(default=5.0, ge=0.0, le=10.0)
    confidence: float = Field(default=0.9, ge=0.0, le=1.0)
    tags: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None  # TTL support
    access_count: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def collection_name(self, facts_col: str, prefs_col: str) -> str:
        """Route to appropriate collection."""
        if self.type in {MemoryType.PREFERENCE, MemoryType.GOAL, MemoryType.HABIT}:
            return prefs_col
        return facts_col

    def is_expired(self) -> bool:
        """Check if memory has expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at


class RetrievalResult(BaseModel):
    memory: Memory
    score: float
    breakdown: Dict[str, float] = Field(default_factory=dict)


class RetrievalQuery(BaseModel):
    query: str
    user_id: str
    session_id: Optional[str] = None
    k: int = 5
    filters: Optional[Dict[str, Any]] = None
