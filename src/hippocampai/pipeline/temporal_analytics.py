"""Advanced temporal analytics for memory intelligence.

This module provides:
- Peak activity time analysis
- Temporal pattern detection
- Trend analysis over time
- Temporal clustering
- Periodicity detection
- Time-based predictions
"""

import logging
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

from hippocampai.models.memory import Memory, MemoryType

logger = logging.getLogger(__name__)


class TimeOfDay(str, Enum):
    """Time of day periods."""

    EARLY_MORNING = "early_morning"  # 5am-9am
    LATE_MORNING = "late_morning"  # 9am-12pm
    AFTERNOON = "afternoon"  # 12pm-5pm
    EVENING = "evening"  # 5pm-9pm
    NIGHT = "night"  # 9pm-1am
    LATE_NIGHT = "late_night"  # 1am-5am


class DayOfWeek(str, Enum):
    """Days of the week."""

    MONDAY = "monday"
    TUESDAY = "tuesday"
    WEDNESDAY = "wednesday"
    THURSDAY = "thursday"
    FRIDAY = "friday"
    SATURDAY = "saturday"
    SUNDAY = "sunday"


class TrendDirection(str, Enum):
    """Trend directions."""

    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"
    VOLATILE = "volatile"


class PeakActivityAnalysis(BaseModel):
    """Analysis of peak activity times."""

    peak_hour: int = Field(..., description="Hour of day with most activity (0-23)")
    peak_day: DayOfWeek = Field(..., description="Day of week with most activity")
    peak_time_period: TimeOfDay = Field(..., description="Time period with most activity")
    hourly_distribution: dict[int, int] = Field(default_factory=dict)
    daily_distribution: dict[str, int] = Field(default_factory=dict)
    time_period_distribution: dict[str, int] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TemporalPattern(BaseModel):
    """Detected temporal pattern."""

    pattern_type: str = Field(..., description="Type of pattern (daily, weekly, monthly, custom)")
    description: str = Field(..., description="Human-readable pattern description")
    frequency: float = Field(..., description="Pattern frequency (times per period)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Pattern confidence")
    occurrences: list[datetime] = Field(default_factory=list)
    next_predicted: Optional[datetime] = Field(None, description="Next predicted occurrence")
    regularity_score: float = Field(..., ge=0.0, le=1.0, description="How regular the pattern is")


class TrendAnalysis(BaseModel):
    """Trend analysis over time."""

    metric: str = Field(..., description="What metric is being analyzed")
    time_window_days: int = Field(..., description="Analysis time window")
    direction: TrendDirection = Field(..., description="Trend direction")
    strength: float = Field(..., ge=0.0, le=1.0, description="Trend strength")
    change_rate: float = Field(..., description="Rate of change (units per day)")
    current_value: float = Field(..., description="Current metric value")
    historical_values: list[tuple[datetime, float]] = Field(default_factory=list)
    forecast: Optional[float] = Field(None, description="Forecasted value for next period")


class PeriodicityAnalysis(BaseModel):
    """Analysis of periodic patterns."""

    period_hours: float = Field(..., description="Detected period in hours")
    period_label: str = Field(..., description="Human-readable period (hourly, daily, weekly, etc.)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")
    amplitude: float = Field(..., description="Pattern amplitude")
    phase_offset: float = Field(..., description="Phase offset from reference")
    examples: list[datetime] = Field(default_factory=list)


class TemporalCluster(BaseModel):
    """Cluster of memories in time."""

    cluster_id: str
    start_time: datetime
    end_time: datetime
    duration_hours: float
    memories: list[Memory] = Field(default_factory=list)
    density: float = Field(..., ge=0.0, description="Temporal density of memories")
    dominant_type: Optional[MemoryType] = Field(None)
    tags: list[str] = Field(default_factory=list)


class TemporalAnalytics:
    """Advanced temporal analytics for memories."""

    def __init__(self):
        """Initialize temporal analytics."""
        pass

    def analyze_peak_activity(
        self, memories: list[Memory], timezone_offset: int = 0
    ) -> PeakActivityAnalysis:
        """Analyze peak activity times.

        Args:
            memories: List of memories to analyze
            timezone_offset: Timezone offset in hours from UTC

        Returns:
            Peak activity analysis
        """
        if not memories:
            return PeakActivityAnalysis(
                peak_hour=12,
                peak_day=DayOfWeek.MONDAY,
                peak_time_period=TimeOfDay.AFTERNOON,
            )

        # Extract timestamps with timezone adjustment
        hourly_counts: dict[int, int] = defaultdict(int)
        daily_counts: dict[str, int] = defaultdict(int)
        period_counts: dict[str, int] = defaultdict(int)

        for memory in memories:
            # Adjust for timezone
            local_time = memory.created_at + timedelta(hours=timezone_offset)

            # Hour of day (0-23)
            hour = local_time.hour
            hourly_counts[hour] += 1

            # Day of week
            day = list(DayOfWeek)[local_time.weekday()]
            daily_counts[day.value] += 1

            # Time period
            period = self._get_time_period(hour)
            period_counts[period.value] += 1

        # Find peaks
        peak_hour = max(hourly_counts.items(), key=lambda x: x[1])[0]
        peak_day_str = max(daily_counts.items(), key=lambda x: x[1])[0]
        peak_day = DayOfWeek(peak_day_str)
        peak_period_str = max(period_counts.items(), key=lambda x: x[1])[0]
        peak_period = TimeOfDay(peak_period_str)

        return PeakActivityAnalysis(
            peak_hour=peak_hour,
            peak_day=peak_day,
            peak_time_period=peak_period,
            hourly_distribution=dict(hourly_counts),
            daily_distribution=dict(daily_counts),
            time_period_distribution=dict(period_counts),
            metadata={
                "total_memories": len(memories),
                "peak_hour_count": hourly_counts[peak_hour],
                "peak_day_count": daily_counts[peak_day_str],
            },
        )

    def _get_time_period(self, hour: int) -> TimeOfDay:
        """Convert hour to time period."""
        if 5 <= hour < 9:
            return TimeOfDay.EARLY_MORNING
        elif 9 <= hour < 12:
            return TimeOfDay.LATE_MORNING
        elif 12 <= hour < 17:
            return TimeOfDay.AFTERNOON
        elif 17 <= hour < 21:
            return TimeOfDay.EVENING
        elif 21 <= hour < 24 or hour == 0:
            return TimeOfDay.NIGHT
        else:
            return TimeOfDay.LATE_NIGHT

    def detect_temporal_patterns(
        self, memories: list[Memory], min_occurrences: int = 3
    ) -> list[TemporalPattern]:
        """Detect recurring temporal patterns.

        Args:
            memories: List of memories to analyze
            min_occurrences: Minimum occurrences to consider a pattern

        Returns:
            List of detected temporal patterns
        """
        if len(memories) < min_occurrences:
            return []

        patterns = []

        # Sort memories by time
        sorted_memories = sorted(memories, key=lambda m: m.created_at)

        # Check for daily patterns (similar time each day)
        daily_pattern = self._detect_daily_pattern(sorted_memories, min_occurrences)
        if daily_pattern:
            patterns.append(daily_pattern)

        # Check for weekly patterns (same day of week)
        weekly_pattern = self._detect_weekly_pattern(sorted_memories, min_occurrences)
        if weekly_pattern:
            patterns.append(weekly_pattern)

        # Check for custom interval patterns
        interval_patterns = self._detect_interval_patterns(sorted_memories, min_occurrences)
        patterns.extend(interval_patterns)

        return patterns

    def _detect_daily_pattern(
        self, memories: list[Memory], min_occurrences: int
    ) -> Optional[TemporalPattern]:
        """Detect daily recurring pattern."""
        # Group by hour of day
        hour_occurrences: dict[int, list[datetime]] = defaultdict(list)

        for memory in memories:
            hour = memory.created_at.hour
            hour_occurrences[hour].append(memory.created_at)

        # Find hour with most occurrences
        for hour, timestamps in hour_occurrences.items():
            if len(timestamps) >= min_occurrences:
                # Check if occurrences are on different days
                unique_days = len(set(ts.date() for ts in timestamps))

                if unique_days >= min_occurrences:
                    # Compute regularity (how consistently it occurs)
                    intervals = []
                    sorted_ts = sorted(timestamps)
                    for i in range(1, len(sorted_ts)):
                        interval = (sorted_ts[i] - sorted_ts[i - 1]).total_seconds() / 3600
                        intervals.append(interval)

                    # Regularity based on variance of intervals
                    if intervals:
                        mean_interval = sum(intervals) / len(intervals)
                        variance = sum((x - mean_interval) ** 2 for x in intervals) / len(intervals)
                        regularity = max(0.0, 1.0 - (variance / (24 ** 2)))  # Normalize by day
                    else:
                        regularity = 1.0

                    # Predict next occurrence
                    last_occurrence = sorted_ts[-1]
                    next_predicted = last_occurrence + timedelta(days=1)

                    return TemporalPattern(
                        pattern_type="daily",
                        description=f"Activity around {hour:02d}:00 daily",
                        frequency=len(timestamps) / unique_days,
                        confidence=min(1.0, len(timestamps) / (min_occurrences * 3)),
                        occurrences=timestamps,
                        next_predicted=next_predicted,
                        regularity_score=regularity,
                    )

        return None

    def _detect_weekly_pattern(
        self, memories: list[Memory], min_occurrences: int
    ) -> Optional[TemporalPattern]:
        """Detect weekly recurring pattern."""
        # Group by day of week
        dow_occurrences: dict[int, list[datetime]] = defaultdict(list)

        for memory in memories:
            dow = memory.created_at.weekday()  # 0=Monday, 6=Sunday
            dow_occurrences[dow].append(memory.created_at)

        # Find day with most occurrences
        for dow, timestamps in dow_occurrences.items():
            if len(timestamps) >= min_occurrences:
                # Check if occurrences span multiple weeks
                sorted_ts = sorted(timestamps)
                weeks_span = (sorted_ts[-1] - sorted_ts[0]).days / 7

                if weeks_span >= 2:  # At least 2 weeks
                    day_name = list(DayOfWeek)[dow].value

                    # Compute regularity
                    intervals = []
                    for i in range(1, len(sorted_ts)):
                        interval = (sorted_ts[i] - sorted_ts[i - 1]).total_seconds() / (3600 * 24)
                        intervals.append(interval)

                    if intervals:
                        mean_interval = sum(intervals) / len(intervals)
                        variance = sum((x - mean_interval) ** 2 for x in intervals) / len(intervals)
                        regularity = max(0.0, 1.0 - (variance / (7 ** 2)))
                    else:
                        regularity = 1.0

                    # Predict next occurrence
                    last_occurrence = sorted_ts[-1]
                    days_until_next = (7 - last_occurrence.weekday() + dow) % 7
                    if days_until_next == 0:
                        days_until_next = 7
                    next_predicted = last_occurrence + timedelta(days=days_until_next)

                    return TemporalPattern(
                        pattern_type="weekly",
                        description=f"Activity on {day_name}s",
                        frequency=len(timestamps) / weeks_span,
                        confidence=min(1.0, len(timestamps) / (min_occurrences * 2)),
                        occurrences=timestamps,
                        next_predicted=next_predicted,
                        regularity_score=regularity,
                    )

        return None

    def _detect_interval_patterns(
        self, memories: list[Memory], min_occurrences: int
    ) -> list[TemporalPattern]:
        """Detect custom interval patterns."""
        if len(memories) < min_occurrences:
            return []

        patterns = []

        # Compute intervals between consecutive memories
        timestamps = [m.created_at for m in memories]
        intervals = []
        for i in range(1, len(timestamps)):
            interval_hours = (timestamps[i] - timestamps[i - 1]).total_seconds() / 3600
            intervals.append(interval_hours)

        if not intervals:
            return []

        # Find common intervals (cluster intervals)
        interval_groups: dict[float, list[int]] = defaultdict(list)

        for i, interval in enumerate(intervals):
            # Round to nearest hour for grouping
            rounded = round(interval)

            # Group intervals within ±20% tolerance
            found_group = False
            for group_interval in list(interval_groups.keys()):
                if abs(rounded - group_interval) / max(rounded, group_interval) < 0.2:
                    interval_groups[group_interval].append(i)
                    found_group = True
                    break

            if not found_group:
                interval_groups[rounded].append(i)

        # Create patterns for significant interval groups
        for interval_hours, indices in interval_groups.items():
            if len(indices) >= min_occurrences - 1:  # -1 because intervals = n-1
                # Get timestamps for this pattern
                pattern_timestamps = [timestamps[0]]
                for idx in indices:
                    pattern_timestamps.append(timestamps[idx + 1])

                # Describe interval
                if interval_hours < 24:
                    description = f"Activity every {interval_hours:.1f} hours"
                elif interval_hours < 168:  # Less than a week
                    days = interval_hours / 24
                    description = f"Activity every {days:.1f} days"
                else:
                    weeks = interval_hours / 168
                    description = f"Activity every {weeks:.1f} weeks"

                regularity = 1.0 - (len(set(intervals)) / len(intervals))

                # Predict next
                last_time = pattern_timestamps[-1]
                next_predicted = last_time + timedelta(hours=interval_hours)

                patterns.append(
                    TemporalPattern(
                        pattern_type="interval",
                        description=description,
                        frequency=1.0 / interval_hours if interval_hours > 0 else 0.0,
                        confidence=min(1.0, len(indices) / (min_occurrences * 2)),
                        occurrences=pattern_timestamps,
                        next_predicted=next_predicted,
                        regularity_score=regularity,
                    )
                )

        return patterns

    def analyze_trends(
        self, memories: list[Memory], time_window_days: int = 30, metric: str = "activity"
    ) -> TrendAnalysis:
        """Analyze trends over time.

        Args:
            memories: List of memories to analyze
            time_window_days: Time window for analysis
            metric: Metric to analyze (activity, importance, types)

        Returns:
            Trend analysis
        """
        if not memories:
            return TrendAnalysis(
                metric=metric,
                time_window_days=time_window_days,
                direction=TrendDirection.STABLE,
                strength=0.0,
                change_rate=0.0,
                current_value=0.0,
            )

        # Filter to time window
        cutoff = datetime.now(timezone.utc) - timedelta(days=time_window_days)
        recent_memories = [m for m in memories if m.created_at >= cutoff]

        if metric == "activity":
            return self._analyze_activity_trend(recent_memories, time_window_days)
        elif metric == "importance":
            return self._analyze_importance_trend(recent_memories, time_window_days)
        elif metric == "types":
            return self._analyze_type_diversity_trend(recent_memories, time_window_days)
        else:
            # Default to activity
            return self._analyze_activity_trend(recent_memories, time_window_days)

    def _analyze_activity_trend(
        self, memories: list[Memory], time_window_days: int
    ) -> TrendAnalysis:
        """Analyze activity trend (memories per day)."""
        if not memories:
            return TrendAnalysis(
                metric="activity",
                time_window_days=time_window_days,
                direction=TrendDirection.STABLE,
                strength=0.0,
                change_rate=0.0,
                current_value=0.0,
            )

        # Group by day
        daily_counts: dict[str, int] = defaultdict(int)
        for memory in memories:
            date_str = memory.created_at.date().isoformat()
            daily_counts[date_str] += 1

        # Convert to time series
        sorted_dates = sorted(daily_counts.keys())
        values = [daily_counts[date] for date in sorted_dates]
        timestamps = [
            datetime.fromisoformat(date + "T00:00:00").replace(tzinfo=timezone.utc)
            for date in sorted_dates
        ]

        historical_values = list(zip(timestamps, [float(v) for v in values]))

        # Compute trend using simple linear regression
        if len(values) < 2:
            return TrendAnalysis(
                metric="activity",
                time_window_days=time_window_days,
                direction=TrendDirection.STABLE,
                strength=0.0,
                change_rate=0.0,
                current_value=float(values[0]) if values else 0.0,
                historical_values=historical_values,
            )

        # Simple linear regression
        n = len(values)
        x_vals = list(range(n))
        mean_x = sum(x_vals) / n
        mean_y = sum(values) / n

        numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_vals, values))
        denominator = sum((x - mean_x) ** 2 for x in x_vals)

        slope = numerator / denominator if denominator != 0 else 0.0

        # Determine direction and strength
        if abs(slope) < 0.1:
            direction = TrendDirection.STABLE
        elif slope > 0:
            direction = TrendDirection.INCREASING
        else:
            direction = TrendDirection.DECREASING

        # Strength based on R-squared
        y_pred = [slope * x + (mean_y - slope * mean_x) for x in x_vals]
        ss_res = sum((y - y_p) ** 2 for y, y_p in zip(values, y_pred))
        ss_tot = sum((y - mean_y) ** 2 for y in values)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
        strength = max(0.0, min(1.0, r_squared))

        # Forecast next period
        forecast = y_pred[-1] + slope if values else 0.0

        return TrendAnalysis(
            metric="activity",
            time_window_days=time_window_days,
            direction=direction,
            strength=strength,
            change_rate=slope,
            current_value=float(values[-1]) if values else 0.0,
            historical_values=historical_values,
            forecast=forecast,
        )

    def _analyze_importance_trend(
        self, memories: list[Memory], time_window_days: int
    ) -> TrendAnalysis:
        """Analyze importance trend over time."""
        # Group by week and compute average importance
        weekly_importance: dict[str, list[float]] = defaultdict(list)

        for memory in memories:
            # Get week number
            week = memory.created_at.isocalendar()[1]
            year = memory.created_at.year
            week_key = f"{year}-W{week:02d}"
            weekly_importance[week_key].append(memory.importance)

        # Compute averages
        sorted_weeks = sorted(weekly_importance.keys())
        values = [sum(weekly_importance[week]) / len(weekly_importance[week]) for week in sorted_weeks]

        if not values:
            return TrendAnalysis(
                metric="importance",
                time_window_days=time_window_days,
                direction=TrendDirection.STABLE,
                strength=0.0,
                change_rate=0.0,
                current_value=0.0,
            )

        # Simple trend detection
        if len(values) >= 2:
            recent_avg = sum(values[-3:]) / len(values[-3:])
            earlier_avg = sum(values[:3]) / min(3, len(values))
            change = recent_avg - earlier_avg

            if abs(change) < 0.5:
                direction = TrendDirection.STABLE
            elif change > 0:
                direction = TrendDirection.INCREASING
            else:
                direction = TrendDirection.DECREASING

            strength = min(1.0, abs(change) / 10.0)  # Normalize by max importance
        else:
            direction = TrendDirection.STABLE
            strength = 0.0
            change = 0.0

        return TrendAnalysis(
            metric="importance",
            time_window_days=time_window_days,
            direction=direction,
            strength=strength,
            change_rate=change / time_window_days,
            current_value=values[-1] if values else 0.0,
        )

    def _analyze_type_diversity_trend(
        self, memories: list[Memory], time_window_days: int
    ) -> TrendAnalysis:
        """Analyze memory type diversity trend."""
        # Group by week and count unique types
        weekly_types: dict[str, set[MemoryType]] = defaultdict(set)

        for memory in memories:
            week = memory.created_at.isocalendar()[1]
            year = memory.created_at.year
            week_key = f"{year}-W{week:02d}"
            weekly_types[week_key].add(memory.type)

        # Count unique types per week
        sorted_weeks = sorted(weekly_types.keys())
        values = [len(weekly_types[week]) for week in sorted_weeks]

        if not values:
            return TrendAnalysis(
                metric="type_diversity",
                time_window_days=time_window_days,
                direction=TrendDirection.STABLE,
                strength=0.0,
                change_rate=0.0,
                current_value=0.0,
            )

        # Trend detection
        if len(values) >= 2:
            recent_avg = sum(values[-3:]) / len(values[-3:])
            earlier_avg = sum(values[:3]) / min(3, len(values))
            change = recent_avg - earlier_avg

            if abs(change) < 0.5:
                direction = TrendDirection.STABLE
            elif change > 0:
                direction = TrendDirection.INCREASING
            else:
                direction = TrendDirection.DECREASING

            strength = min(1.0, abs(change) / len(MemoryType))
        else:
            direction = TrendDirection.STABLE
            strength = 0.0

        return TrendAnalysis(
            metric="type_diversity",
            time_window_days=time_window_days,
            direction=direction,
            strength=strength,
            change_rate=0.0,
            current_value=float(values[-1]) if values else 0.0,
        )

    def cluster_by_time(
        self, memories: list[Memory], max_gap_hours: float = 24.0
    ) -> list[TemporalCluster]:
        """Cluster memories by temporal proximity.

        Args:
            memories: List of memories to cluster
            max_gap_hours: Maximum gap between memories in same cluster

        Returns:
            List of temporal clusters
        """
        if not memories:
            return []

        # Sort by time
        sorted_memories = sorted(memories, key=lambda m: m.created_at)

        clusters = []
        current_cluster = [sorted_memories[0]]

        for i in range(1, len(sorted_memories)):
            time_gap = (
                sorted_memories[i].created_at - current_cluster[-1].created_at
            ).total_seconds() / 3600

            if time_gap <= max_gap_hours:
                # Add to current cluster
                current_cluster.append(sorted_memories[i])
            else:
                # Finalize current cluster
                if current_cluster:
                    cluster = self._create_temporal_cluster(current_cluster, len(clusters))
                    clusters.append(cluster)

                # Start new cluster
                current_cluster = [sorted_memories[i]]

        # Add final cluster
        if current_cluster:
            cluster = self._create_temporal_cluster(current_cluster, len(clusters))
            clusters.append(cluster)

        return clusters

    def _create_temporal_cluster(
        self, memories: list[Memory], cluster_idx: int
    ) -> TemporalCluster:
        """Create a temporal cluster from memories."""
        start_time = memories[0].created_at
        end_time = memories[-1].created_at
        duration_hours = (end_time - start_time).total_seconds() / 3600

        # Compute density (memories per hour)
        density = len(memories) / max(duration_hours, 1.0)

        # Find dominant type
        type_counts = Counter(m.type for m in memories)
        dominant_type = type_counts.most_common(1)[0][0] if type_counts else None

        # Collect all tags
        all_tags = set()
        for m in memories:
            all_tags.update(m.tags)

        return TemporalCluster(
            cluster_id=f"temporal_cluster_{cluster_idx}",
            start_time=start_time,
            end_time=end_time,
            duration_hours=duration_hours,
            memories=memories,
            density=density,
            dominant_type=dominant_type,
            tags=list(all_tags),
        )
