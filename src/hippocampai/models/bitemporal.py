"""Bi-temporal fact tracking models.

Bi-temporal modeling tracks two independent time dimensions:
- event_time: When the fact occurred or was stated in the real world
- valid_time: The interval during which the fact is/was true [valid_from, valid_to)
- system_time: When HippocampAI recorded/observed the fact (append-only ledger)

This enables queries like:
- "As of system_time T, what did we believe?"
- "What was valid during [valid_start, valid_end]?"
- "Latest known valid fact for entity/property"
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class FactStatus(str, Enum):
    """Status of a bi-temporal fact."""

    ACTIVE = "active"  # Currently valid
    SUPERSEDED = "superseded"  # Replaced by newer fact
    RETRACTED = "retracted"  # Explicitly invalidated
    EXPIRED = "expired"  # valid_to has passed


class BiTemporalFact(BaseModel):
    """A fact with bi-temporal tracking.

    Attributes:
        id: Unique identifier for this fact version
        fact_id: Logical fact identifier (same across revisions)
        text: The fact content
        user_id: Owner of the fact
        entity_id: Optional entity this fact is about
        property_name: Optional property name (e.g., "employer", "location")

        event_time: When the fact occurred/was stated (real-world time)
        valid_from: Start of validity interval (inclusive)
        valid_to: End of validity interval (exclusive), None = still valid
        system_time: When this record was created (immutable, append-only)

        status: Current status of this fact version
        superseded_by: ID of the fact that supersedes this one
        supersedes: ID of the fact this one supersedes
        confidence: Confidence score (0.0-1.0)
        source: Source of the fact (conversation, import, etc.)
        metadata: Additional metadata
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    fact_id: str = Field(default_factory=lambda: str(uuid4()))
    text: str
    user_id: str
    entity_id: Optional[str] = None
    property_name: Optional[str] = None

    # Bi-temporal timestamps
    event_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    valid_from: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    valid_to: Optional[datetime] = None  # None = currently valid (no end)
    system_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Status and lineage
    status: FactStatus = FactStatus.ACTIVE
    superseded_by: Optional[str] = None
    supersedes: Optional[str] = None

    # Quality metrics
    confidence: float = Field(default=0.9, ge=0.0, le=1.0)
    source: str = "conversation"
    metadata: dict[str, Any] = Field(default_factory=dict)

    def is_valid_at(self, point_in_time: datetime) -> bool:
        """Check if fact was valid at a specific point in time."""
        if point_in_time < self.valid_from:
            return False
        if self.valid_to is not None and point_in_time >= self.valid_to:
            return False
        return True

    def is_currently_valid(self) -> bool:
        """Check if fact is currently valid."""
        return self.is_valid_at(datetime.now(timezone.utc))

    def overlaps_interval(self, start: datetime, end: datetime) -> bool:
        """Check if fact validity overlaps with given interval."""
        # Fact validity: [valid_from, valid_to)
        # Query interval: [start, end)
        fact_end = self.valid_to or datetime.max.replace(tzinfo=timezone.utc)
        return self.valid_from < end and fact_end > start


class BiTemporalQuery(BaseModel):
    """Query parameters for bi-temporal fact retrieval.

    Supports three query modes:
    1. Current: Get currently valid facts (default)
    2. As-of system time: What did we believe at system_time T?
    3. Valid-time range: What was valid during [valid_start, valid_end]?
    """

    user_id: str
    entity_id: Optional[str] = None
    property_name: Optional[str] = None
    text_query: Optional[str] = None  # Semantic search query

    # System time filter (as-of query)
    as_of_system_time: Optional[datetime] = None

    # Valid time filters
    valid_at: Optional[datetime] = None  # Point-in-time validity
    valid_from: Optional[datetime] = None  # Range start
    valid_to: Optional[datetime] = None  # Range end

    # Status filter
    include_superseded: bool = False
    include_retracted: bool = False

    # Pagination
    limit: int = 100
    offset: int = 0


class FactRevision(BaseModel):
    """Represents a revision/correction to an existing fact.

    Used when updating a fact without deleting history.
    """

    original_fact_id: str
    new_text: str
    new_valid_from: Optional[datetime] = None
    new_valid_to: Optional[datetime] = None
    reason: str = "correction"
    confidence: float = Field(default=0.9, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class BiTemporalQueryResult(BaseModel):
    """Result of a bi-temporal query."""

    facts: list[BiTemporalFact]
    total_count: int
    query: BiTemporalQuery
    as_of_system_time: Optional[datetime] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
