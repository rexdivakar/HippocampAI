"""Memory lifecycle tracking and audit logging for HippocampAI."""

import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ==============================================================================
# Enums
# ==============================================================================


class MemoryEventType(str, Enum):
    """Types of memory lifecycle events."""

    CREATED = "created"
    UPDATED = "updated"
    DELETED = "deleted"
    RETRIEVED = "retrieved"
    SEARCHED = "searched"
    CONSOLIDATED = "consolidated"
    DEDUPLICATED = "deduplicated"
    HEALTH_CHECK = "health_check"
    CONFLICT_DETECTED = "conflict_detected"
    CONFLICT_RESOLVED = "conflict_resolved"
    ACCESS_PATTERN = "access_pattern"
    STALENESS_DETECTED = "staleness_detected"
    FRESHNESS_UPDATED = "freshness_updated"


class MemoryEventSeverity(str, Enum):
    """Severity levels for memory events."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


# ==============================================================================
# Models
# ==============================================================================


class MemoryEvent(BaseModel):
    """Represents a single memory lifecycle event."""

    event_id: str = Field(..., description="Unique event identifier")
    memory_id: str = Field(..., description="ID of the memory")
    user_id: str = Field(..., description="User ID who owns the memory")
    event_type: MemoryEventType = Field(..., description="Type of event")
    severity: MemoryEventSeverity = Field(
        default=MemoryEventSeverity.INFO, description="Event severity"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the event occurred",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional event metadata"
    )
    duration_ms: Optional[float] = Field(
        None, description="Operation duration in milliseconds"
    )
    success: bool = Field(default=True, description="Whether operation succeeded")
    error_message: Optional[str] = Field(None, description="Error message if failed")

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class MemoryAccessPattern(BaseModel):
    """Tracks access patterns for a memory."""

    memory_id: str
    user_id: str
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    first_accessed: Optional[datetime] = None
    search_hits: int = 0
    direct_retrievals: int = 0
    access_frequency: float = 0.0  # accesses per day
    access_sources: dict[str, int] = Field(
        default_factory=dict
    )  # endpoint: count


class MemoryHealthSnapshot(BaseModel):
    """Snapshot of memory health at a point in time."""

    memory_id: str
    user_id: str
    timestamp: datetime
    health_score: float
    staleness_score: float
    freshness_score: float
    access_frequency: float
    duplicate_likelihood: float
    issues: list[str] = Field(default_factory=list)


# ==============================================================================
# Memory Tracker
# ==============================================================================


class MemoryTracker:
    """Tracks memory lifecycle events and provides observability."""

    def __init__(self, storage_backend: Optional[str] = None):
        """
        Initialize memory tracker.

        Args:
            storage_backend: Optional storage backend ('redis', 'file', None)
                           If None, events are only logged
        """
        self.storage_backend = storage_backend
        self._events: list[MemoryEvent] = []  # In-memory buffer
        self._access_patterns: dict[str, MemoryAccessPattern] = {}
        self._health_snapshots: dict[str, list[MemoryHealthSnapshot]] = {}
        logger.info(f"MemoryTracker initialized with backend: {storage_backend}")

    def track_event(
        self,
        memory_id: str,
        user_id: str,
        event_type: MemoryEventType,
        severity: MemoryEventSeverity = MemoryEventSeverity.INFO,
        metadata: Optional[dict[str, Any]] = None,
        duration_ms: Optional[float] = None,
        success: bool = True,
        error_message: Optional[str] = None,
    ) -> MemoryEvent:
        """
        Track a memory lifecycle event.

        Args:
            memory_id: ID of the memory
            user_id: User ID who owns the memory
            event_type: Type of event
            severity: Event severity
            metadata: Additional event metadata
            duration_ms: Operation duration in milliseconds
            success: Whether operation succeeded
            error_message: Error message if failed

        Returns:
            MemoryEvent: The created event
        """
        import uuid

        event = MemoryEvent(
            event_id=str(uuid.uuid4()),
            memory_id=memory_id,
            user_id=user_id,
            event_type=event_type,
            severity=severity,
            timestamp=datetime.now(timezone.utc),
            metadata=metadata or {},
            duration_ms=duration_ms,
            success=success,
            error_message=error_message,
        )

        # Store event
        self._events.append(event)

        # Update access patterns
        self._update_access_pattern(memory_id, user_id, event_type)

        # Log event
        log_msg = (
            f"Memory {memory_id} [{event_type.value}] "
            f"user={user_id} success={success}"
        )
        if duration_ms:
            log_msg += f" duration={duration_ms:.2f}ms"

        if severity == MemoryEventSeverity.ERROR:
            logger.error(log_msg)
        elif severity == MemoryEventSeverity.WARNING:
            logger.warning(log_msg)
        else:
            logger.info(log_msg)

        # Store to backend if configured
        if self.storage_backend:
            self._store_event(event)

        return event

    def _update_access_pattern(
        self, memory_id: str, user_id: str, event_type: MemoryEventType
    ):
        """Update access pattern for a memory."""
        key = f"{user_id}:{memory_id}"
        now = datetime.now(timezone.utc)

        if key not in self._access_patterns:
            self._access_patterns[key] = MemoryAccessPattern(
                memory_id=memory_id,
                user_id=user_id,
                first_accessed=now,
            )

        pattern = self._access_patterns[key]
        pattern.access_count += 1
        pattern.last_accessed = now

        if event_type == MemoryEventType.RETRIEVED:
            pattern.direct_retrievals += 1
        elif event_type == MemoryEventType.SEARCHED:
            pattern.search_hits += 1

        # Calculate access frequency (accesses per day)
        if pattern.first_accessed:
            days_elapsed = (now - pattern.first_accessed).total_seconds() / 86400
            if days_elapsed > 0:
                pattern.access_frequency = pattern.access_count / days_elapsed

    def _store_event(self, event: MemoryEvent):
        """Store event to backend (placeholder for future implementation)."""
        # TODO: Implement Redis/file storage
        pass

    def get_memory_events(
        self,
        memory_id: Optional[str] = None,
        user_id: Optional[str] = None,
        event_type: Optional[MemoryEventType] = None,
        limit: int = 100,
    ) -> list[MemoryEvent]:
        """
        Get memory events with optional filtering.

        Args:
            memory_id: Filter by memory ID
            user_id: Filter by user ID
            event_type: Filter by event type
            limit: Maximum number of events to return

        Returns:
            List of MemoryEvent objects
        """
        filtered = self._events

        if memory_id:
            filtered = [e for e in filtered if e.memory_id == memory_id]

        if user_id:
            filtered = [e for e in filtered if e.user_id == user_id]

        if event_type:
            filtered = [e for e in filtered if e.event_type == event_type]

        # Sort by timestamp (newest first)
        filtered.sort(key=lambda e: e.timestamp, reverse=True)

        return filtered[:limit]

    def get_access_pattern(
        self, memory_id: str, user_id: str
    ) -> Optional[MemoryAccessPattern]:
        """Get access pattern for a specific memory."""
        key = f"{user_id}:{memory_id}"
        return self._access_patterns.get(key)

    def get_all_access_patterns(
        self, user_id: Optional[str] = None
    ) -> list[MemoryAccessPattern]:
        """Get all access patterns, optionally filtered by user."""
        patterns = list(self._access_patterns.values())

        if user_id:
            patterns = [p for p in patterns if p.user_id == user_id]

        return patterns

    def record_health_snapshot(
        self,
        memory_id: str,
        user_id: str,
        health_score: float,
        staleness_score: float,
        freshness_score: float,
        access_frequency: float,
        duplicate_likelihood: float,
        issues: Optional[list[str]] = None,
    ):
        """Record a health snapshot for a memory."""
        snapshot = MemoryHealthSnapshot(
            memory_id=memory_id,
            user_id=user_id,
            timestamp=datetime.now(timezone.utc),
            health_score=health_score,
            staleness_score=staleness_score,
            freshness_score=freshness_score,
            access_frequency=access_frequency,
            duplicate_likelihood=duplicate_likelihood,
            issues=issues or [],
        )

        key = f"{user_id}:{memory_id}"
        if key not in self._health_snapshots:
            self._health_snapshots[key] = []

        self._health_snapshots[key].append(snapshot)

    def get_health_history(
        self, memory_id: str, user_id: str, limit: int = 100
    ) -> list[MemoryHealthSnapshot]:
        """Get health history for a memory."""
        key = f"{user_id}:{memory_id}"
        snapshots = self._health_snapshots.get(key, [])

        # Sort by timestamp (newest first)
        snapshots.sort(key=lambda s: s.timestamp, reverse=True)

        return snapshots[:limit]

    def get_memory_stats(self, user_id: str) -> dict[str, Any]:
        """
        Get comprehensive memory statistics for a user.

        Args:
            user_id: User ID

        Returns:
            Dictionary with memory statistics
        """
        events = [e for e in self._events if e.user_id == user_id]
        patterns = [p for p in self._access_patterns.values() if p.user_id == user_id]

        # Count events by type
        event_counts = {}
        for event_type in MemoryEventType:
            count = len([e for e in events if e.event_type == event_type])
            if count > 0:
                event_counts[event_type.value] = count

        # Calculate success rate
        total_ops = len(events)
        successful_ops = len([e for e in events if e.success])
        success_rate = successful_ops / total_ops if total_ops > 0 else 0.0

        # Most accessed memories
        most_accessed = sorted(patterns, key=lambda p: p.access_count, reverse=True)[
            :10
        ]

        # Calculate average operation duration
        durations = [e.duration_ms for e in events if e.duration_ms is not None]
        avg_duration = sum(durations) / len(durations) if durations else 0.0

        return {
            "user_id": user_id,
            "total_events": total_ops,
            "success_rate": success_rate,
            "event_counts": event_counts,
            "total_memories_tracked": len(patterns),
            "total_accesses": sum(p.access_count for p in patterns),
            "avg_operation_duration_ms": avg_duration,
            "most_accessed_memories": [
                {
                    "memory_id": p.memory_id,
                    "access_count": p.access_count,
                    "last_accessed": p.last_accessed.isoformat()
                    if p.last_accessed
                    else None,
                }
                for p in most_accessed
            ],
        }

    def clear_old_events(self, days: int = 30):
        """
        Clear events older than specified days.

        Args:
            days: Number of days to keep
        """
        cutoff = datetime.now(timezone.utc).timestamp() - (days * 86400)
        self._events = [
            e for e in self._events if e.timestamp.timestamp() > cutoff
        ]
        logger.info(f"Cleared events older than {days} days")


# ==============================================================================
# Global Tracker Instance
# ==============================================================================

# Global instance for easy access
_global_tracker: Optional[MemoryTracker] = None


def get_tracker() -> MemoryTracker:
    """Get the global memory tracker instance."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = MemoryTracker()
    return _global_tracker


def initialize_tracker(storage_backend: Optional[str] = None):
    """Initialize the global memory tracker."""
    global _global_tracker
    _global_tracker = MemoryTracker(storage_backend=storage_backend)
    return _global_tracker
