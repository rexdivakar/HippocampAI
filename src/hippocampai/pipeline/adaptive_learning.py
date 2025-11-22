"""Adaptive memory learning - access patterns, proactive refresh, and usage-based compression."""

import logging
from collections import defaultdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class AccessPattern(str, Enum):
    """Types of access patterns detected."""

    FREQUENT = "frequent"  # Accessed often
    PERIODIC = "periodic"  # Regular intervals
    BURST = "burst"  # Sudden spikes
    DECLINING = "declining"  # Decreasing access
    SPORADIC = "sporadic"  # Random, irregular
    CONTEXTUAL = "contextual"  # Accessed with specific contexts
    SEQUENTIAL = "sequential"  # Accessed in sequence
    CO_OCCURRING = "co_occurring"  # Accessed together with others


class CompressionLevel(str, Enum):
    """Compression levels for memories."""

    NONE = "none"  # No compression - full detail
    LIGHT = "light"  # Minor summarization
    MODERATE = "moderate"  # Standard compression
    AGGRESSIVE = "aggressive"  # Heavy compression
    ARCHIVED = "archived"  # Minimal representation


class RefreshPriority(str, Enum):
    """Priority levels for memory refresh."""

    CRITICAL = "critical"  # Refresh immediately
    HIGH = "high"  # Refresh within hours
    MEDIUM = "medium"  # Refresh within days
    LOW = "low"  # Refresh when convenient
    NONE = "none"  # No refresh needed


class AccessEvent(BaseModel):
    """Record of a memory access event."""

    memory_id: str = Field(description="Memory identifier")
    timestamp: datetime = Field(description="When accessed")
    access_type: str = Field(description="Type of access (read, update, etc.)")
    context: Optional[str] = Field(default=None, description="Access context/query")
    co_accessed_ids: list[str] = Field(
        default_factory=list, description="Other memories accessed in same request"
    )
    relevance_score: Optional[float] = Field(
        default=None, description="Relevance score if from search"
    )
    session_id: Optional[str] = Field(default=None, description="Session identifier")
    user_id: Optional[str] = Field(default=None, description="User identifier")


class AccessPatternAnalysis(BaseModel):
    """Analysis of memory access patterns."""

    memory_id: str = Field(description="Memory identifier")
    total_accesses: int = Field(description="Total number of accesses")
    first_access: datetime = Field(description="First access timestamp")
    last_access: datetime = Field(description="Most recent access")
    access_frequency: float = Field(
        description="Accesses per day"
    )  # accesses / days_since_first
    pattern_type: AccessPattern = Field(description="Detected pattern type")
    pattern_confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence in pattern detection"
    )
    avg_time_between_accesses: float = Field(
        description="Average hours between accesses"
    )
    access_contexts: list[str] = Field(
        default_factory=list, description="Common access contexts"
    )
    co_occurring_memories: dict[str, int] = Field(
        default_factory=dict, description="Memories accessed together (id: count)"
    )
    peak_hours: list[int] = Field(
        default_factory=list, description="Hours of day with most access (0-23)"
    )
    trend: str = Field(
        description="Trend direction (increasing, stable, decreasing)"
    )
    last_analyzed: datetime = Field(description="When analysis was performed")


class RefreshRecommendation(BaseModel):
    """Recommendation to refresh a memory."""

    memory_id: str = Field(description="Memory to refresh")
    priority: RefreshPriority = Field(description="Refresh priority")
    reason: str = Field(description="Why refresh is needed")
    staleness_score: float = Field(
        ge=0.0, le=1.0, description="How stale the memory is"
    )
    importance_score: float = Field(
        ge=0.0, le=10.0, description="Memory importance"
    )
    access_frequency: float = Field(description="Recent access frequency")
    suggested_sources: list[str] = Field(
        default_factory=list, description="Suggested sources for refresh"
    )
    estimated_effort: str = Field(
        description="Estimated effort (low, medium, high)"
    )
    last_updated: datetime = Field(description="When memory was last updated")


class CompressionRecommendation(BaseModel):
    """Recommendation for memory compression."""

    memory_id: str = Field(description="Memory to compress")
    current_level: CompressionLevel = Field(description="Current compression")
    recommended_level: CompressionLevel = Field(description="Recommended compression")
    reason: str = Field(description="Why compression is recommended")
    access_score: float = Field(
        ge=0.0, le=1.0, description="Access frequency score (0=never, 1=constant)"
    )
    importance_score: float = Field(
        ge=0.0, le=10.0, description="Memory importance"
    )
    age_days: int = Field(description="Age in days")
    estimated_space_savings: int = Field(
        description="Estimated bytes saved"
    )
    reversible: bool = Field(
        default=True, description="Can decompression restore full detail"
    )


class AdaptiveLearningEngine:
    """Engine for learning memory access patterns and adaptive optimization."""

    def __init__(
        self,
        access_history_limit: int = 10000,
        pattern_analysis_window_days: int = 30,
        refresh_check_interval_hours: int = 24,
        compression_analysis_interval_hours: int = 168,  # Weekly
    ):
        """
        Initialize adaptive learning engine.

        Args:
            access_history_limit: Maximum access events to keep in memory
            pattern_analysis_window_days: Days of history to analyze for patterns
            refresh_check_interval_hours: Hours between refresh checks
            compression_analysis_interval_hours: Hours between compression analysis
        """
        self.access_history: list[AccessEvent] = []
        self.access_history_limit = access_history_limit
        self.pattern_analysis_window_days = pattern_analysis_window_days
        self.refresh_check_interval_hours = refresh_check_interval_hours
        self.compression_analysis_interval_hours = compression_analysis_interval_hours

        # Pattern cache
        self.pattern_cache: dict[str, AccessPatternAnalysis] = {}
        self.last_pattern_analysis: dict[str, datetime] = {}

        # Statistics
        self.memory_access_counts: dict[str, int] = defaultdict(int)
        self.context_access_counts: dict[str, int] = defaultdict(int)
        self.co_occurrence_matrix: dict[tuple[str, str], int] = defaultdict(int)

    def record_access(self, event: AccessEvent) -> None:
        """
        Record a memory access event.

        Args:
            event: Access event to record
        """
        # Add to history
        self.access_history.append(event)

        # Maintain limit
        if len(self.access_history) > self.access_history_limit:
            self.access_history = self.access_history[-self.access_history_limit :]

        # Update statistics
        self.memory_access_counts[event.memory_id] += 1

        if event.context:
            self.context_access_counts[event.context] += 1

        # Track co-occurrences
        for co_id in event.co_accessed_ids:
            sorted_ids = sorted([event.memory_id, co_id])
            pair: tuple[str, str] = (sorted_ids[0], sorted_ids[1])
            self.co_occurrence_matrix[pair] += 1

        logger.debug(
            f"Recorded access for memory {event.memory_id} "
            f"(total: {self.memory_access_counts[event.memory_id]})"
        )

    def analyze_access_pattern(
        self, memory_id: str, force_refresh: bool = False
    ) -> Optional[AccessPatternAnalysis]:
        """
        Analyze access pattern for a memory.

        Args:
            memory_id: Memory to analyze
            force_refresh: Force new analysis even if cached

        Returns:
            Access pattern analysis or None if insufficient data
        """
        now = datetime.now(timezone.utc)

        # Check cache
        if not force_refresh and memory_id in self.pattern_cache:
            cached = self.pattern_cache[memory_id]
            # Use cache if analyzed recently (within 1 hour)
            if (now - cached.last_analyzed).total_seconds() < 3600:
                return cached

        # Get access events for this memory
        events = [e for e in self.access_history if e.memory_id == memory_id]

        if not events:
            return None

        # Sort by timestamp
        events.sort(key=lambda e: e.timestamp)

        # Calculate metrics
        total_accesses = len(events)
        first_access = events[0].timestamp
        last_access = events[-1].timestamp

        # Ensure timestamps are timezone-aware
        if first_access.tzinfo is None:
            first_access = first_access.replace(tzinfo=timezone.utc)
        if last_access.tzinfo is None:
            last_access = last_access.replace(tzinfo=timezone.utc)

        days_since_first = max(1, (now - first_access).total_seconds() / 86400)
        access_frequency = total_accesses / days_since_first

        # Calculate average time between accesses
        if total_accesses > 1:
            time_diffs = []
            for i in range(1, len(events)):
                prev_time = events[i - 1].timestamp
                curr_time = events[i].timestamp
                if prev_time.tzinfo is None:
                    prev_time = prev_time.replace(tzinfo=timezone.utc)
                if curr_time.tzinfo is None:
                    curr_time = curr_time.replace(tzinfo=timezone.utc)
                diff_hours = (curr_time - prev_time).total_seconds() / 3600
                time_diffs.append(diff_hours)
            avg_time_between = sum(time_diffs) / len(time_diffs)
        else:
            avg_time_between = 0.0

        # Extract contexts
        access_contexts = list(
            set([e.context for e in events if e.context])
        )[:10]  # Top 10

        # Co-occurring memories
        co_occurring: dict[str, int] = defaultdict(int)
        for event in events:
            for co_id in event.co_accessed_ids:
                co_occurring[co_id] += 1

        # Top 10 co-occurring
        co_occurring_sorted = dict(
            sorted(co_occurring.items(), key=lambda x: x[1], reverse=True)[:10]
        )

        # Peak hours
        hours = [e.timestamp.hour for e in events]
        hour_counts: defaultdict[int, int] = defaultdict(int)
        for h in hours:
            hour_counts[h] += 1
        peak_hours_with_counts = sorted(hour_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        peak_hours: list[int] = [h[0] for h in peak_hours_with_counts]

        # Detect pattern type
        pattern_type, pattern_confidence = self._detect_pattern_type(
            events, avg_time_between, access_frequency
        )

        # Detect trend
        trend = self._detect_trend(events)

        analysis = AccessPatternAnalysis(
            memory_id=memory_id,
            total_accesses=total_accesses,
            first_access=first_access,
            last_access=last_access,
            access_frequency=access_frequency,
            pattern_type=pattern_type,
            pattern_confidence=pattern_confidence,
            avg_time_between_accesses=avg_time_between,
            access_contexts=access_contexts,
            co_occurring_memories=co_occurring_sorted,
            peak_hours=peak_hours,
            trend=trend,
            last_analyzed=now,
        )

        # Cache result
        self.pattern_cache[memory_id] = analysis
        self.last_pattern_analysis[memory_id] = now

        logger.info(
            f"Analyzed access pattern for {memory_id}: "
            f"{pattern_type.value} ({pattern_confidence:.2f} confidence), "
            f"{access_frequency:.2f} accesses/day, trend: {trend}"
        )

        return analysis

    def _detect_pattern_type(
        self, events: list[AccessEvent], avg_time_between: float, access_frequency: float
    ) -> tuple[AccessPattern, float]:
        """
        Detect the access pattern type.

        Args:
            events: Access events
            avg_time_between: Average hours between accesses
            access_frequency: Accesses per day

        Returns:
            Tuple of (pattern_type, confidence)
        """
        if len(events) < 3:
            return (AccessPattern.SPORADIC, 0.5)

        # Frequent: High access frequency
        if access_frequency > 5.0:  # More than 5 times per day
            return (AccessPattern.FREQUENT, 0.9)

        # Periodic: Regular intervals
        if avg_time_between > 0:
            time_diffs = []
            for i in range(1, len(events)):
                prev_time = events[i - 1].timestamp
                curr_time = events[i].timestamp
                if prev_time.tzinfo is None:
                    prev_time = prev_time.replace(tzinfo=timezone.utc)
                if curr_time.tzinfo is None:
                    curr_time = curr_time.replace(tzinfo=timezone.utc)
                diff_hours = (curr_time - prev_time).total_seconds() / 3600
                time_diffs.append(diff_hours)

            # Calculate coefficient of variation
            if time_diffs:
                mean_diff = sum(time_diffs) / len(time_diffs)
                variance = sum((x - mean_diff) ** 2 for x in time_diffs) / len(
                    time_diffs
                )
                std_dev = variance**0.5
                cv = std_dev / mean_diff if mean_diff > 0 else float("inf")

                # Low CV indicates regularity
                if cv < 0.3:
                    return (AccessPattern.PERIODIC, 0.8)

        # Burst: Sudden spikes
        recent_events = [
            e
            for e in events
            if (datetime.now(timezone.utc) - e.timestamp).total_seconds() < 86400
        ]
        if len(recent_events) > len(events) * 0.5:  # More than 50% in last day
            return (AccessPattern.BURST, 0.7)

        # Declining: Decreasing access over time
        mid_point = len(events) // 2
        first_half = events[:mid_point]
        second_half = events[mid_point:]

        if first_half and second_half:
            first_half_rate = len(first_half) / max(
                1, (first_half[-1].timestamp - first_half[0].timestamp).total_seconds()
            )
            second_half_rate = len(second_half) / max(
                1,
                (second_half[-1].timestamp - second_half[0].timestamp).total_seconds(),
            )

            if first_half_rate > second_half_rate * 2:  # Declined by more than 50%
                return (AccessPattern.DECLINING, 0.7)

        # Contextual: Accessed with specific contexts
        contexts = [e.context for e in events if e.context]
        unique_contexts = set(contexts)
        if len(unique_contexts) <= 3 and len(contexts) > 5:  # Few distinct contexts
            return (AccessPattern.CONTEXTUAL, 0.6)

        # Default: Sporadic
        return (AccessPattern.SPORADIC, 0.5)

    def _detect_trend(self, events: list[AccessEvent]) -> str:
        """
        Detect access trend.

        Args:
            events: Access events

        Returns:
            Trend direction (increasing, stable, decreasing)
        """
        if len(events) < 6:
            return "stable"

        # Split into thirds
        third = len(events) // 3
        first_third = events[:third]
        middle_third = events[third : 2 * third]
        last_third = events[2 * third :]

        # Count accesses per day in each period
        def access_rate(event_list: list[AccessEvent]) -> float:
            if not event_list:
                return 0.0
            start = event_list[0].timestamp
            end = event_list[-1].timestamp
            if start.tzinfo is None:
                start = start.replace(tzinfo=timezone.utc)
            if end.tzinfo is None:
                end = end.replace(tzinfo=timezone.utc)
            days = max(1, (end - start).total_seconds() / 86400)
            return len(event_list) / days

        first_rate = access_rate(first_third)
        middle_rate = access_rate(middle_third)
        last_rate = access_rate(last_third)

        # Determine trend
        if last_rate > middle_rate * 1.2 and middle_rate > first_rate * 1.2:
            return "increasing"
        elif last_rate < middle_rate * 0.8 and middle_rate < first_rate * 0.8:
            return "decreasing"
        else:
            return "stable"

    def recommend_refresh(
        self,
        memory_id: str,
        current_importance: float,
        last_updated: datetime,
        access_pattern: Optional[AccessPatternAnalysis] = None,
    ) -> Optional[RefreshRecommendation]:
        """
        Recommend if memory should be refreshed.

        Args:
            memory_id: Memory identifier
            current_importance: Current importance score
            last_updated: When memory was last updated
            access_pattern: Optional pre-computed access pattern

        Returns:
            Refresh recommendation or None if no refresh needed
        """
        now = datetime.now(timezone.utc)

        # Ensure last_updated is timezone-aware
        if last_updated.tzinfo is None:
            last_updated = last_updated.replace(tzinfo=timezone.utc)

        # Calculate staleness
        days_since_update = (now - last_updated).total_seconds() / 86400
        staleness_score = min(1.0, days_since_update / 90.0)  # 90 days = fully stale

        # Get or analyze access pattern
        if access_pattern is None:
            access_pattern = self.analyze_access_pattern(memory_id)

        # Default values if no access pattern
        access_frequency = 0.0

        if access_pattern:
            access_frequency = access_pattern.access_frequency

        # Determine priority
        priority = RefreshPriority.NONE
        reason = ""
        effort = "low"

        # Critical: Important + frequently accessed + very stale
        if (
            current_importance >= 8.0
            and access_frequency > 2.0
            and staleness_score > 0.7
        ):
            priority = RefreshPriority.CRITICAL
            reason = "High importance memory with frequent access is very stale"
            effort = "high"

        # High: Important + moderately stale OR frequent + stale
        elif (
            current_importance >= 6.0 and staleness_score > 0.5
        ) or (access_frequency > 1.0 and staleness_score > 0.6):
            priority = RefreshPriority.HIGH
            reason = "Important or frequently accessed memory is becoming stale"
            effort = "medium"

        # Medium: Moderately important + accessed + somewhat stale
        elif (
            current_importance >= 4.0
            and access_frequency > 0.1
            and staleness_score > 0.4
        ):
            priority = RefreshPriority.MEDIUM
            reason = "Moderately important memory with some access is aging"
            effort = "medium"

        # Low: Any access + old
        elif access_frequency > 0.0 and staleness_score > 0.8:
            priority = RefreshPriority.LOW
            reason = "Occasionally accessed memory is quite old"
            effort = "low"

        if priority == RefreshPriority.NONE:
            return None

        # Suggest sources based on context
        suggested_sources = []
        if access_pattern and access_pattern.access_contexts:
            suggested_sources = access_pattern.access_contexts[:3]

        recommendation = RefreshRecommendation(
            memory_id=memory_id,
            priority=priority,
            reason=reason,
            staleness_score=staleness_score,
            importance_score=current_importance,
            access_frequency=access_frequency,
            suggested_sources=suggested_sources,
            estimated_effort=effort,
            last_updated=last_updated,
        )

        logger.info(
            f"Refresh recommendation for {memory_id}: {priority.value} - {reason}"
        )

        return recommendation

    def recommend_compression(
        self,
        memory_id: str,
        current_importance: float,
        age_days: int,
        text_length: int,
        current_compression: CompressionLevel = CompressionLevel.NONE,
        access_pattern: Optional[AccessPatternAnalysis] = None,
    ) -> Optional[CompressionRecommendation]:
        """
        Recommend compression level for a memory.

        Args:
            memory_id: Memory identifier
            current_importance: Importance score
            age_days: Age in days
            text_length: Length of memory text
            current_compression: Current compression level
            access_pattern: Optional pre-computed access pattern

        Returns:
            Compression recommendation or None if no change needed
        """
        # Get or analyze access pattern
        if access_pattern is None:
            access_pattern = self.analyze_access_pattern(memory_id)

        # Calculate access score (0 = never accessed, 1 = constantly accessed)
        access_score = 0.0
        if access_pattern:
            # Normalize frequency (5+ accesses/day = score of 1.0)
            access_score = min(1.0, access_pattern.access_frequency / 5.0)

        # Determine recommended compression
        recommended = current_compression
        reason = ""

        # Never compress if frequently accessed and important
        if access_score > 0.5 and current_importance >= 7.0:
            recommended = CompressionLevel.NONE
            reason = "Frequently accessed important memory - keep full detail"

        # Light compression: Old but occasionally accessed
        elif access_score > 0.2 and age_days > 90:
            recommended = CompressionLevel.LIGHT
            reason = "Aging memory with some access - light summarization"

        # Moderate compression: Old, low access, moderate importance
        elif access_score < 0.2 and age_days > 180 and current_importance >= 4.0:
            recommended = CompressionLevel.MODERATE
            reason = "Old memory with low access - standard compression"

        # Aggressive compression: Very old, rarely accessed, low importance
        elif access_score < 0.1 and age_days > 365 and current_importance < 4.0:
            recommended = CompressionLevel.AGGRESSIVE
            reason = "Very old, rarely accessed, low importance - heavy compression"

        # Archived: Ancient, never accessed, very low importance
        elif access_score == 0.0 and age_days > 730 and current_importance < 2.0:
            recommended = CompressionLevel.ARCHIVED
            reason = "Ancient, never accessed - minimal representation"

        # No change needed
        if recommended == current_compression:
            return None

        # Estimate space savings
        compression_ratios = {
            CompressionLevel.NONE: 1.0,
            CompressionLevel.LIGHT: 0.8,
            CompressionLevel.MODERATE: 0.5,
            CompressionLevel.AGGRESSIVE: 0.3,
            CompressionLevel.ARCHIVED: 0.1,
        }

        current_ratio = compression_ratios[current_compression]
        recommended_ratio = compression_ratios[recommended]
        estimated_savings = int(text_length * (current_ratio - recommended_ratio))

        # Compression is reversible unless archived
        reversible = recommended != CompressionLevel.ARCHIVED

        recommendation = CompressionRecommendation(
            memory_id=memory_id,
            current_level=current_compression,
            recommended_level=recommended,
            reason=reason,
            access_score=access_score,
            importance_score=current_importance,
            age_days=age_days,
            estimated_space_savings=max(0, estimated_savings),
            reversible=reversible,
        )

        logger.info(
            f"Compression recommendation for {memory_id}: "
            f"{current_compression.value} -> {recommended.value} - {reason}"
        )

        return recommendation

    def get_access_statistics(self) -> dict[str, Any]:
        """
        Get overall access statistics.

        Returns:
            Dictionary with access statistics
        """
        total_events = len(self.access_history)
        unique_memories = len(set(e.memory_id for e in self.access_history))

        # Top accessed memories
        top_accessed = sorted(
            self.memory_access_counts.items(), key=lambda x: x[1], reverse=True
        )[:10]

        # Top contexts
        top_contexts = sorted(
            self.context_access_counts.items(), key=lambda x: x[1], reverse=True
        )[:10]

        # Top co-occurrences
        top_co_occurrences = sorted(
            self.co_occurrence_matrix.items(), key=lambda x: x[1], reverse=True
        )[:10]

        return {
            "total_access_events": total_events,
            "unique_memories_accessed": unique_memories,
            "top_accessed_memories": [
                {"memory_id": k, "access_count": v} for k, v in top_accessed
            ],
            "top_access_contexts": [
                {"context": k, "count": v} for k, v in top_contexts
            ],
            "top_co_occurrences": [
                {"memory_pair": list(k), "co_occurrence_count": v}
                for k, v in top_co_occurrences
            ],
            "patterns_analyzed": len(self.pattern_cache),
        }
