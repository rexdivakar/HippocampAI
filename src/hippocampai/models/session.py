"""Session models for conversation tracking and management."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class SessionStatus(str, Enum):
    """Session status enumeration."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    COMPLETED = "completed"
    ARCHIVED = "archived"


class Entity(BaseModel):
    """Extracted entity from session."""

    name: str
    type: str  # person, organization, location, event, etc.
    mentions: int = 1
    first_mentioned_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_mentioned_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)

    def update_mention(self):
        """Update mention count and timestamp."""
        self.mentions += 1
        self.last_mentioned_at = datetime.now(timezone.utc)


class SessionFact(BaseModel):
    """Key fact extracted from session."""

    fact: str
    confidence: float = 0.9
    extracted_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    sources: list[str] = Field(default_factory=list)  # Memory IDs that support this fact
    metadata: dict[str, Any] = Field(default_factory=dict)


class Session(BaseModel):
    """Session model for conversation tracking."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str
    title: Optional[str] = None
    summary: Optional[str] = None
    status: SessionStatus = SessionStatus.ACTIVE

    # Hierarchy
    parent_session_id: Optional[str] = None
    child_session_ids: list[str] = Field(default_factory=list)

    # Tracking
    message_count: int = 0
    memory_count: int = 0
    entities: dict[str, Entity] = Field(default_factory=dict)  # entity_name -> Entity
    facts: list[SessionFact] = Field(default_factory=list)

    # Metadata
    metadata: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)

    # Timestamps
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_activity_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    ended_at: Optional[datetime] = None

    # Statistics
    avg_importance: float = 0.0
    total_tokens: int = 0
    total_characters: int = 0

    def update_activity(self):
        """Update last activity timestamp."""
        self.last_activity_at = datetime.now(timezone.utc)
        self.message_count += 1

    def add_entity(self, name: str, entity_type: str, metadata: Optional[dict] = None):
        """Add or update an entity."""
        if name in self.entities:
            self.entities[name].update_mention()
        else:
            self.entities[name] = Entity(name=name, type=entity_type, metadata=metadata or {})

    def add_fact(self, fact: str, confidence: float = 0.9, sources: Optional[list[str]] = None):
        """Add a fact to the session."""
        self.facts.append(
            SessionFact(
                fact=fact,
                confidence=confidence,
                sources=sources or [],
            )
        )

    def add_child_session(self, child_id: str):
        """Add a child session ID."""
        if child_id not in self.child_session_ids:
            self.child_session_ids.append(child_id)

    def complete(self):
        """Mark session as completed."""
        self.status = SessionStatus.COMPLETED
        self.ended_at = datetime.now(timezone.utc)

    def archive(self):
        """Mark session as archived."""
        self.status = SessionStatus.ARCHIVED
        if not self.ended_at:
            self.ended_at = datetime.now(timezone.utc)

    def duration_seconds(self) -> float:
        """Get session duration in seconds."""
        end_time = self.ended_at or datetime.now(timezone.utc)
        return (end_time - self.started_at).total_seconds()

    def is_active(self) -> bool:
        """Check if session is active."""
        return self.status == SessionStatus.ACTIVE

    def get_top_entities(self, limit: int = 5) -> list[Entity]:
        """Get most mentioned entities."""
        sorted_entities = sorted(self.entities.values(), key=lambda e: e.mentions, reverse=True)
        return sorted_entities[:limit]

    def get_high_confidence_facts(self, min_confidence: float = 0.8) -> list[SessionFact]:
        """Get facts above confidence threshold."""
        return [f for f in self.facts if f.confidence >= min_confidence]


class SessionSearchResult(BaseModel):
    """Session search result with score."""

    session: Session
    score: float
    breakdown: dict[str, float] = Field(default_factory=dict)
