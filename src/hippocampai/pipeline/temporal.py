"""Temporal reasoning and time-based analysis for memories.

This module provides:
- Time-based queries (last week, last month, date ranges)
- Chronological narrative construction
- Event sequence understanding
- Time-decay functions per memory type
- Future memory scheduling
- Timeline analysis and visualization
"""

import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from hippocampai.models.memory import Memory, MemoryType

logger = logging.getLogger(__name__)


class TimeRange(str, Enum):
    """Predefined time ranges for queries."""

    LAST_HOUR = "last_hour"
    LAST_DAY = "last_day"
    LAST_WEEK = "last_week"
    LAST_MONTH = "last_month"
    LAST_YEAR = "last_year"
    TODAY = "today"
    YESTERDAY = "yesterday"
    THIS_WEEK = "this_week"
    THIS_MONTH = "this_month"
    THIS_YEAR = "this_year"


class ScheduledMemory(BaseModel):
    """Represents a memory scheduled for future creation or reminder."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str
    text: str
    type: MemoryType = MemoryType.FACT
    scheduled_for: datetime
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    triggered: bool = False
    triggered_at: Optional[datetime] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)

    # Temporal metadata
    recurrence: Optional[str] = None  # "daily", "weekly", "monthly"
    reminder_offset: Optional[int] = None  # Minutes before scheduled_for


class TemporalEvent(BaseModel):
    """Represents an event extracted from memories with temporal context."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    memory_id: str
    text: str
    timestamp: datetime
    event_type: str  # "action", "state_change", "milestone", etc.
    participants: list[str] = Field(default_factory=list)
    location: Optional[str] = None
    duration: Optional[int] = None  # in seconds
    metadata: dict[str, Any] = Field(default_factory=dict)


class Timeline(BaseModel):
    """Represents a chronological timeline of events."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str
    title: str
    events: list[TemporalEvent] = Field(default_factory=list)
    start_time: datetime
    end_time: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)

    def add_event(self, event: TemporalEvent):
        """Add event to timeline in chronological order."""
        self.events.append(event)
        self.events.sort(key=lambda e: e.timestamp)

        # Update time bounds
        if event.timestamp < self.start_time:
            self.start_time = event.timestamp
        if event.timestamp > self.end_time:
            self.end_time = event.timestamp

    def get_duration(self) -> timedelta:
        """Get total duration of timeline."""
        return self.end_time - self.start_time


class TemporalAnalyzer:
    """Handles temporal reasoning and time-based analysis."""

    def __init__(self, llm=None):
        """Initialize temporal analyzer.

        Args:
            llm: Optional language model for advanced temporal analysis
        """
        self.llm = llm
        self.scheduled_memories: dict[str, ScheduledMemory] = {}

    def parse_time_range(self, time_range: TimeRange) -> tuple[datetime, datetime]:
        """Parse predefined time range to start and end datetimes.

        Args:
            time_range: Predefined time range

        Returns:
            Tuple of (start_time, end_time)
        """
        now = datetime.now(timezone.utc)

        if time_range == TimeRange.LAST_HOUR:
            start = now - timedelta(hours=1)
            end = now
        elif time_range == TimeRange.LAST_DAY:
            start = now - timedelta(days=1)
            end = now
        elif time_range == TimeRange.LAST_WEEK:
            start = now - timedelta(weeks=1)
            end = now
        elif time_range == TimeRange.LAST_MONTH:
            start = now - timedelta(days=30)
            end = now
        elif time_range == TimeRange.LAST_YEAR:
            start = now - timedelta(days=365)
            end = now
        elif time_range == TimeRange.TODAY:
            start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            end = now
        elif time_range == TimeRange.YESTERDAY:
            yesterday = now - timedelta(days=1)
            start = yesterday.replace(hour=0, minute=0, second=0, microsecond=0)
            end = start + timedelta(days=1)
        elif time_range == TimeRange.THIS_WEEK:
            # Start of week (Monday)
            days_since_monday = now.weekday()
            start = (now - timedelta(days=days_since_monday)).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            end = now
        elif time_range == TimeRange.THIS_MONTH:
            start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            end = now
        elif time_range == TimeRange.THIS_YEAR:
            start = now.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
            end = now
        else:
            # Default to last day
            start = now - timedelta(days=1)
            end = now

        return start, end

    def filter_by_time_range(
        self,
        memories: list[Memory],
        time_range: Optional[TimeRange] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> list[Memory]:
        """Filter memories by time range.

        Args:
            memories: List of memories to filter
            time_range: Predefined time range
            start_time: Custom start time
            end_time: Custom end time

        Returns:
            Filtered memories within time range
        """
        if time_range:
            start_time, end_time = self.parse_time_range(time_range)
        elif not (start_time and end_time):
            # No time filter specified, return all
            return memories

        filtered = []
        for memory in memories:
            mem_time = memory.created_at
            if start_time <= mem_time <= end_time:
                filtered.append(memory)

        logger.info(
            f"Filtered {len(memories)} memories to {len(filtered)} "
            f"between {start_time} and {end_time}"
        )
        return filtered

    def build_chronological_narrative(
        self, memories: list[Memory], title: Optional[str] = None
    ) -> str:
        """Build a chronological narrative from memories.

        Args:
            memories: List of memories to construct narrative from
            title: Optional title for the narrative

        Returns:
            Formatted narrative string
        """
        if not memories:
            return "No memories to construct narrative."

        # Sort by timestamp
        sorted_memories = sorted(memories, key=lambda m: m.created_at)

        # Build narrative
        narrative_parts = []
        if title:
            narrative_parts.append(f"# {title}\n")

        current_date = None
        for memory in sorted_memories:
            mem_date = memory.created_at.date()

            # Add date header if new day
            if mem_date != current_date:
                current_date = mem_date
                narrative_parts.append(f"\n## {mem_date.strftime('%B %d, %Y')}\n")

            # Format time and memory
            time_str = memory.created_at.strftime("%I:%M %p")
            narrative_parts.append(f"**{time_str}**: {memory.text}")

            # Add metadata if available
            if memory.tags:
                narrative_parts.append(f" _[{', '.join(memory.tags)}]_")
            narrative_parts.append("\n")

        return "".join(narrative_parts)

    def extract_temporal_events(self, memories: list[Memory]) -> list[TemporalEvent]:
        """Extract temporal events from memories.

        Args:
            memories: List of memories to extract events from

        Returns:
            List of temporal events
        """
        events = []

        for memory in memories:
            # Determine event type from memory type
            event_type = self._determine_event_type(memory)

            # Extract event information
            event = TemporalEvent(
                memory_id=memory.id,
                text=memory.text,
                timestamp=memory.created_at,
                event_type=event_type,
                metadata={
                    "memory_type": memory.type.value,
                    "importance": memory.importance,
                    "tags": memory.tags,
                },
            )

            # Extract participants, locations if using LLM
            if self.llm:
                enriched = self._enrich_event_with_llm(event, memory.text)
                if enriched:
                    event = enriched

            events.append(event)

        logger.info(f"Extracted {len(events)} temporal events from {len(memories)} memories")
        return events

    def _determine_event_type(self, memory: Memory) -> str:
        """Determine event type from memory."""
        if memory.type == MemoryType.EVENT:
            return "event"
        if memory.type == MemoryType.GOAL:
            return "milestone"
        if memory.type == MemoryType.HABIT:
            return "recurring"
        if memory.type == MemoryType.PREFERENCE:
            return "state_change"
        return "action"

    def _enrich_event_with_llm(self, event: TemporalEvent, text: str) -> Optional[TemporalEvent]:
        """Use LLM to extract additional event details."""
        prompt = f"""Extract event details from this text:
"{text}"

Return JSON with:
- participants: list of people/entities involved
- location: where it happened (if mentioned)
- duration: estimated duration in minutes (if applicable)

JSON:"""

        if self.llm is None:
            return None
        try:
            response = self.llm.generate(prompt, max_tokens=100)

            # Parse JSON response
            import json

            data = json.loads(response)

            if data.get("participants"):
                event.participants = data["participants"]
            if data.get("location"):
                event.location = data["location"]
            if data.get("duration"):
                event.duration = data["duration"] * 60  # Convert to seconds

            return event
        except Exception as e:
            logger.warning(f"LLM event enrichment failed: {e}")
            return None

    def create_timeline(
        self,
        memories: list[Memory],
        user_id: str,
        title: str = "Memory Timeline",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> Timeline:
        """Create a timeline from memories.

        Args:
            memories: List of memories
            user_id: User ID
            title: Timeline title
            start_time: Optional start time (defaults to earliest memory)
            end_time: Optional end time (defaults to latest memory)

        Returns:
            Timeline object
        """
        if not memories:
            now = datetime.now(timezone.utc)
            return Timeline(
                user_id=user_id,
                title=title,
                start_time=start_time or now,
                end_time=end_time or now,
            )

        # Extract events
        events = self.extract_temporal_events(memories)

        # Determine timeline bounds
        if not start_time:
            start_time = min(e.timestamp for e in events)
        if not end_time:
            end_time = max(e.timestamp for e in events)

        timeline = Timeline(
            user_id=user_id,
            title=title,
            events=events,
            start_time=start_time,
            end_time=end_time,
        )

        logger.info(
            f"Created timeline '{title}' with {len(events)} events "
            f"spanning {timeline.get_duration()}"
        )
        return timeline

    def analyze_event_sequences(
        self, memories: list[Memory], max_gap_hours: int = 24
    ) -> list[list[Memory]]:
        """Identify sequences of related events.

        Args:
            memories: List of memories
            max_gap_hours: Maximum time gap to consider events related

        Returns:
            List of event sequences (each sequence is a list of memories)
        """
        if not memories:
            return []

        # Sort by time
        sorted_mems = sorted(memories, key=lambda m: m.created_at)

        sequences = []
        current_sequence = [sorted_mems[0]]

        for i in range(1, len(sorted_mems)):
            prev_mem = sorted_mems[i - 1]
            curr_mem = sorted_mems[i]

            # Calculate time gap
            gap = (curr_mem.created_at - prev_mem.created_at).total_seconds() / 3600

            if gap <= max_gap_hours:
                # Same sequence
                current_sequence.append(curr_mem)
            else:
                # New sequence
                if len(current_sequence) > 1:
                    sequences.append(current_sequence)
                current_sequence = [curr_mem]

        # Add last sequence
        if len(current_sequence) > 1:
            sequences.append(current_sequence)

        logger.info(f"Found {len(sequences)} event sequences from {len(memories)} memories")
        return sequences

    def calculate_time_decay(self, memory: Memory, half_life_days: int = 30) -> float:
        """Calculate time decay factor for a memory.

        Args:
            memory: Memory to calculate decay for
            half_life_days: Half-life in days

        Returns:
            Decay factor (0.0 to 1.0)
        """
        now = datetime.now(timezone.utc)
        age_days = (now - memory.created_at).total_seconds() / 86400

        # Exponential decay: (0.5)^(age / half_life)
        decay_factor = 0.5 ** (age_days / half_life_days)

        return decay_factor

    def schedule_memory(
        self,
        text: str,
        user_id: str,
        scheduled_for: datetime,
        type: MemoryType = MemoryType.FACT,
        tags: Optional[list[str]] = None,
        recurrence: Optional[str] = None,
        reminder_offset: Optional[int] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> ScheduledMemory:
        """Schedule a memory for future creation.

        Args:
            text: Memory text
            user_id: User ID
            scheduled_for: When to create the memory
            type: Memory type
            tags: Optional tags
            recurrence: Optional recurrence pattern
            reminder_offset: Minutes before scheduled_for to trigger reminder
            metadata: Optional metadata

        Returns:
            ScheduledMemory object
        """
        scheduled = ScheduledMemory(
            user_id=user_id,
            text=text,
            type=type,
            scheduled_for=scheduled_for,
            tags=tags or [],
            recurrence=recurrence,
            reminder_offset=reminder_offset,
            metadata=metadata or {},
        )

        self.scheduled_memories[scheduled.id] = scheduled

        logger.info(f"Scheduled memory for {user_id} at {scheduled_for} (recurrence: {recurrence})")
        return scheduled

    def get_due_scheduled_memories(self) -> list[ScheduledMemory]:
        """Get scheduled memories that are due.

        Returns:
            List of due scheduled memories
        """
        now = datetime.now(timezone.utc)
        due = []

        for scheduled in self.scheduled_memories.values():
            if not scheduled.triggered and scheduled.scheduled_for <= now:
                due.append(scheduled)

        return due

    def trigger_scheduled_memory(self, scheduled_id: str) -> bool:
        """Mark a scheduled memory as triggered.

        Args:
            scheduled_id: Scheduled memory ID

        Returns:
            True if triggered successfully
        """
        if scheduled_id in self.scheduled_memories:
            scheduled = self.scheduled_memories[scheduled_id]
            scheduled.triggered = True
            scheduled.triggered_at = datetime.now(timezone.utc)

            # Handle recurrence
            if scheduled.recurrence:
                self._create_recurrence(scheduled)

            logger.info(f"Triggered scheduled memory {scheduled_id}")
            return True

        return False

    def _create_recurrence(self, original: ScheduledMemory):
        """Create next occurrence for recurring scheduled memory."""
        if original.recurrence == "daily":
            next_time = original.scheduled_for + timedelta(days=1)
        elif original.recurrence == "weekly":
            next_time = original.scheduled_for + timedelta(weeks=1)
        elif original.recurrence == "monthly":
            next_time = original.scheduled_for + timedelta(days=30)
        else:
            return

        # Create new scheduled memory
        self.schedule_memory(
            text=original.text,
            user_id=original.user_id,
            scheduled_for=next_time,
            type=original.type,
            tags=original.tags,
            recurrence=original.recurrence,
            reminder_offset=original.reminder_offset,
            metadata=original.metadata,
        )

    def get_temporal_summary(self, memories: list[Memory], user_id: str) -> dict[str, Any]:
        """Generate temporal summary statistics.

        Args:
            memories: List of memories
            user_id: User ID

        Returns:
            Dictionary with temporal statistics
        """
        if not memories:
            return {
                "total_memories": 0,
                "time_span_days": 0,
                "memories_per_day": 0,
                "peak_activity_hour": None,
                "peak_activity_day": None,
            }

        sorted_mems = sorted(memories, key=lambda m: m.created_at)
        first_mem = sorted_mems[0]
        last_mem = sorted_mems[-1]

        time_span = (last_mem.created_at - first_mem.created_at).total_seconds() / 86400

        # Count by hour
        hour_counts = defaultdict(int)
        day_counts = defaultdict(int)

        for mem in memories:
            hour = mem.created_at.hour
            day = mem.created_at.strftime("%A")
            hour_counts[hour] += 1
            day_counts[day] += 1

        peak_hour = max(hour_counts.items(), key=lambda x: x[1])[0] if hour_counts else None
        peak_day = max(day_counts.items(), key=lambda x: x[1])[0] if day_counts else None

        return {
            "total_memories": len(memories),
            "time_span_days": time_span,
            "memories_per_day": len(memories) / max(time_span, 1),
            "first_memory": first_mem.created_at.isoformat(),
            "last_memory": last_mem.created_at.isoformat(),
            "peak_activity_hour": f"{peak_hour}:00" if peak_hour is not None else None,
            "peak_activity_day": peak_day,
            "hourly_distribution": dict(hour_counts),
            "daily_distribution": dict(day_counts),
        }
