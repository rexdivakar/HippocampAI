"""Enhanced temporal features: forecasting, time-decay, freshness scoring."""

import logging
import math
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field

from hippocampai.models.memory import Memory

logger = logging.getLogger(__name__)


class MemoryFreshnessScore(BaseModel):
    """Freshness score for a memory."""

    memory_id: str
    freshness_score: float = Field(ge=0.0, le=1.0, description="How fresh/up-to-date (0-1)")
    age_days: float
    last_accessed_days: Optional[float] = None
    access_frequency: float = 0.0
    temporal_relevance: float = 1.0
    factors: dict[str, float] = Field(default_factory=dict)


class TimeDecayFunction(BaseModel):
    """Customizable time decay function."""

    name: str
    decay_type: str = Field(
        description="exponential, linear, logarithmic, or step"
    )
    half_life_days: Optional[float] = None  # For exponential
    decay_rate: Optional[float] = None  # For linear
    steps: Optional[list[tuple[int, float]]] = None  # For step functions
    min_score: float = 0.0  # Floor value


class TemporalContextWindow(BaseModel):
    """Auto-adjusting temporal context window."""

    start_date: datetime
    end_date: datetime
    window_size_days: int
    context_type: str = Field(description="recent, relevant, or seasonal")
    confidence: float = Field(ge=0.0, le=1.0)


class MemoryForecast(BaseModel):
    """Forecast of future memory patterns."""

    forecast_type: str = Field(description="usage, importance, or topic")
    time_period: str
    predictions: list[dict[str, Any]] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    basis: str = Field(description="What historical data this is based on")


class PatternPrediction(BaseModel):
    """Predicted future pattern."""

    pattern_type: str
    predicted_date: datetime
    confidence: float
    description: str
    historical_basis: list[datetime] = Field(default_factory=list)


class EnhancedTemporalAnalyzer:
    """Enhanced temporal analysis with forecasting and decay functions."""

    def __init__(
        self,
        default_half_life_days: int = 90,
        freshness_window_days: int = 30,
    ):
        """Initialize enhanced temporal analyzer.

        Args:
            default_half_life_days: Default half-life for exponential decay
            freshness_window_days: Days within which memories are considered fresh
        """
        self.default_half_life = default_half_life_days
        self.freshness_window = freshness_window_days
        self.decay_functions: dict[str, TimeDecayFunction] = {}

        # Register default decay functions
        self._register_default_decay_functions()

    def calculate_freshness_score(
        self,
        memory: Memory,
        reference_date: Optional[datetime] = None,
    ) -> MemoryFreshnessScore:
        """Calculate comprehensive freshness score for a memory.

        Args:
            memory: Memory to score
            reference_date: Reference date (defaults to now)

        Returns:
            MemoryFreshnessScore with breakdown
        """
        if reference_date is None:
            reference_date = datetime.now(timezone.utc)

        # Age factor
        age_days = (reference_date - memory.created_at).days
        age_factor = self._calculate_age_freshness(age_days)

        # Access factor
        last_accessed_days = None
        access_factor = 0.5  # Default
        if memory.updated_at:
            last_accessed_days = (reference_date - memory.updated_at).days
            access_factor = self._calculate_access_freshness(
                memory.access_count,
                last_accessed_days
            )

        # Access frequency
        if age_days > 0:
            access_frequency = memory.access_count / age_days
        else:
            access_frequency = 0.0

        # Temporal relevance (check for time-sensitive content)
        temporal_relevance = self._calculate_temporal_relevance(memory.text, reference_date)

        # Combined freshness score (weighted)
        freshness_score = (
            age_factor * 0.35 +
            access_factor * 0.35 +
            temporal_relevance * 0.30
        )

        return MemoryFreshnessScore(
            memory_id=memory.id,
            freshness_score=freshness_score,
            age_days=age_days,
            last_accessed_days=last_accessed_days,
            access_frequency=access_frequency,
            temporal_relevance=temporal_relevance,
            factors={
                "age": age_factor,
                "access": access_factor,
                "temporal_relevance": temporal_relevance,
            },
        )

    def apply_time_decay(
        self,
        memory: Memory,
        decay_function: Optional[TimeDecayFunction] = None,
        reference_date: Optional[datetime] = None,
    ) -> float:
        """Apply time decay function to memory importance.

        Args:
            memory: Memory to decay
            decay_function: Custom decay function (uses default if None)
            reference_date: Reference date

        Returns:
            Decayed importance score
        """
        if reference_date is None:
            reference_date = datetime.now(timezone.utc)

        age_days = (reference_date - memory.created_at).days

        if decay_function is None:
            decay_function = self.decay_functions["default_exponential"]

        decay_multiplier = self._compute_decay_multiplier(age_days, decay_function)

        # Apply decay to importance
        decayed_importance = memory.importance * decay_multiplier

        # Apply floor
        return max(decay_function.min_score, decayed_importance)

    def get_adaptive_context_window(
        self,
        query: str,
        memories: list[Memory],
        context_type: str = "relevant",
    ) -> TemporalContextWindow:
        """Auto-adjust temporal context window based on query and data.

        Args:
            query: Query text
            memories: Available memories
            context_type: Type of context (recent, relevant, seasonal)

        Returns:
            TemporalContextWindow with adaptive date range
        """
        now = datetime.now(timezone.utc)

        if context_type == "recent":
            # Recent memories (last N days based on activity)
            window_size = self._calculate_recent_window_size(memories)
            start_date = now - timedelta(days=window_size)
            confidence = 0.9

        elif context_type == "seasonal":
            # Same time period in previous years
            window_size = 30  # +/- 30 days around current date
            start_date = now - timedelta(days=window_size)
            confidence = 0.7

        else:  # relevant
            # Analyze query for temporal hints
            window_size, confidence = self._infer_window_from_query(query, memories)
            start_date = now - timedelta(days=window_size)

        return TemporalContextWindow(
            start_date=start_date,
            end_date=now,
            window_size_days=window_size,
            context_type=context_type,
            confidence=confidence,
        )

    def forecast_memory_patterns(
        self,
        memories: list[Memory],
        forecast_days: int = 30,
    ) -> list[MemoryForecast]:
        """Predict future memory patterns based on historical data.

        Args:
            memories: Historical memories
            forecast_days: Days to forecast ahead

        Returns:
            List of forecasts
        """
        if not memories:
            return []

        forecasts = []

        # Usage forecast
        usage_forecast = self._forecast_usage_patterns(memories, forecast_days)
        if usage_forecast:
            forecasts.append(usage_forecast)

        # Topic forecast
        topic_forecast = self._forecast_topic_trends(memories, forecast_days)
        if topic_forecast:
            forecasts.append(topic_forecast)

        # Importance forecast
        importance_forecast = self._forecast_importance_trends(memories, forecast_days)
        if importance_forecast:
            forecasts.append(importance_forecast)

        return forecasts

    def predict_future_patterns(
        self,
        memories: list[Memory],
        pattern_type: str = "recurring",
    ) -> list[PatternPrediction]:
        """Predict when patterns might recur.

        Args:
            memories: Historical memories
            pattern_type: Type of pattern (recurring, seasonal)

        Returns:
            List of pattern predictions
        """
        predictions = []

        if pattern_type == "recurring":
            # Find recurring temporal patterns
            predictions.extend(self._detect_recurring_patterns(memories))
        elif pattern_type == "seasonal":
            # Find seasonal patterns
            predictions.extend(self._detect_seasonal_patterns(memories))

        return predictions

    def register_decay_function(self, func: TimeDecayFunction) -> None:
        """Register a custom decay function."""
        self.decay_functions[func.name] = func
        logger.info(f"Registered decay function: {func.name}")

    # Private helper methods

    def _register_default_decay_functions(self) -> None:
        """Register built-in decay functions."""
        # Exponential decay (default)
        self.decay_functions["default_exponential"] = TimeDecayFunction(
            name="default_exponential",
            decay_type="exponential",
            half_life_days=self.default_half_life,
            min_score=0.1,
        )

        # Linear decay
        self.decay_functions["linear"] = TimeDecayFunction(
            name="linear",
            decay_type="linear",
            decay_rate=0.01,  # 1% per day
            min_score=0.0,
        )

        # Logarithmic decay (slower decay)
        self.decay_functions["logarithmic"] = TimeDecayFunction(
            name="logarithmic",
            decay_type="logarithmic",
            half_life_days=180,
            min_score=0.2,
        )

        # Step decay (sharp drops at intervals)
        self.decay_functions["step"] = TimeDecayFunction(
            name="step",
            decay_type="step",
            steps=[
                (0, 1.0),    # 0-7 days: 100%
                (7, 0.9),    # 7-30 days: 90%
                (30, 0.7),   # 30-90 days: 70%
                (90, 0.5),   # 90-180 days: 50%
                (180, 0.3),  # 180-365 days: 30%
                (365, 0.1),  # 365+ days: 10%
            ],
            min_score=0.1,
        )

    def _compute_decay_multiplier(
        self,
        age_days: float,
        decay_func: TimeDecayFunction,
    ) -> float:
        """Compute decay multiplier based on age and function."""
        if decay_func.decay_type == "exponential":
            half_life = decay_func.half_life_days or self.default_half_life
            return math.exp(-0.693 * age_days / half_life)

        elif decay_func.decay_type == "linear":
            rate = decay_func.decay_rate or 0.01
            return max(0.0, 1.0 - (rate * age_days))

        elif decay_func.decay_type == "logarithmic":
            if age_days == 0:
                return 1.0
            half_life = decay_func.half_life_days or 180
            return 1.0 / (1.0 + math.log(age_days + 1) / math.log(half_life))

        elif decay_func.decay_type == "step":
            if not decay_func.steps:
                return 1.0

            # Find appropriate step
            for threshold, multiplier in reversed(decay_func.steps):
                if age_days >= threshold:
                    return multiplier

            return 1.0

        return 1.0  # No decay

    def _calculate_age_freshness(self, age_days: float) -> float:
        """Calculate freshness score based on age."""
        if age_days <= self.freshness_window:
            return 1.0

        # Exponential decay after freshness window
        decay_rate = 0.01  # 1% per day
        return math.exp(-decay_rate * (age_days - self.freshness_window))

    def _calculate_access_freshness(
        self,
        access_count: int,
        days_since_access: float,
    ) -> float:
        """Calculate freshness based on access patterns."""
        # Recency component
        if days_since_access <= 7:
            recency = 1.0
        elif days_since_access <= 30:
            recency = 0.7
        elif days_since_access <= 90:
            recency = 0.4
        else:
            recency = 0.1

        # Frequency component
        if access_count == 0:
            frequency = 0.0
        elif access_count < 5:
            frequency = 0.3
        elif access_count < 20:
            frequency = 0.6
        else:
            frequency = 1.0

        return (recency * 0.6 + frequency * 0.4)

    def _calculate_temporal_relevance(
        self,
        text: str,
        reference_date: datetime,
    ) -> float:
        """Calculate temporal relevance from text content."""
        import re

        text_lower = text.lower()
        current_year = reference_date.year

        # Check for time-sensitive keywords
        urgent_keywords = ['now', 'today', 'currently', 'latest', 'new']
        time_neutral = ['always', 'forever', 'never', 'generally', 'usually']

        if any(kw in text_lower for kw in urgent_keywords):
            # Time-sensitive content - check how old it is
            return 0.5  # Medium relevance

        if any(kw in text_lower for kw in time_neutral):
            # Time-neutral content
            return 1.0

        # Check for year mentions
        year_pattern = r'\b(20\d{2})\b'
        years = [int(y) for y in re.findall(year_pattern, text)]

        if years:
            most_recent_year = max(years)
            years_old = current_year - most_recent_year
            if years_old > 3:
                return 0.3  # Outdated
            elif years_old > 1:
                return 0.6  # Somewhat dated

        return 0.8  # Default relevance

    def _calculate_recent_window_size(self, memories: list[Memory]) -> int:
        """Calculate adaptive window size for recent context."""
        if not memories:
            return 30

        # Analyze memory frequency
        now = datetime.now(timezone.utc)
        recent_count = sum(
            1 for m in memories
            if (now - m.created_at).days <= 7
        )

        # More recent activity = shorter window
        if recent_count > 20:
            return 7
        elif recent_count > 10:
            return 14
        elif recent_count > 5:
            return 30
        else:
            return 90

    def _infer_window_from_query(
        self,
        query: str,
        memories: list[Memory],
    ) -> tuple[int, float]:
        """Infer temporal window from query text."""
        query_lower = query.lower()

        # Explicit time references
        if any(word in query_lower for word in ['today', 'now', 'current']):
            return (7, 0.9)
        if any(word in query_lower for word in ['recent', 'lately', 'this week']):
            return (14, 0.85)
        if any(word in query_lower for word in ['this month', 'past month']):
            return (30, 0.85)
        if any(word in query_lower for word in ['this year', 'past year']):
            return (365, 0.8)

        # Historical references
        if any(word in query_lower for word in ['ever', 'always', 'all time']):
            return (3650, 0.9)  # 10 years

        # Default: analyze memory distribution
        avg_window = self._calculate_recent_window_size(memories)
        return (avg_window, 0.6)

    def _forecast_usage_patterns(
        self,
        memories: list[Memory],
        forecast_days: int,
    ) -> Optional[MemoryForecast]:
        """Forecast memory usage patterns."""
        if not memories:
            return None

        # Analyze historical access patterns
        now = datetime.now(timezone.utc)
        daily_access: dict[int, int] = defaultdict(int)

        for mem in memories:
            if mem.access_count > 0:
                age_days = (now - mem.created_at).days
                if age_days >= 0:
                    daily_access[age_days] += mem.access_count

        if not daily_access:
            return None

        # Simple trend analysis
        recent_days = 30
        recent_accesses = sum(
            count for day, count in daily_access.items()
            if day <= recent_days
        )
        avg_daily = recent_accesses / recent_days if recent_days > 0 else 0

        # Forecast
        predictions = []
        for i in range(1, forecast_days + 1):
            predicted_accesses = avg_daily  # Simple constant forecast
            predictions.append({
                "day": i,
                "predicted_accesses": round(predicted_accesses, 2),
            })

        return MemoryForecast(
            forecast_type="usage",
            time_period=f"next {forecast_days} days",
            predictions=predictions,
            confidence=0.6,
            basis=f"Based on {recent_days} days of historical access data",
        )

    def _forecast_topic_trends(
        self,
        memories: list[Memory],
        forecast_days: int,
    ) -> Optional[MemoryForecast]:
        """Forecast topic trends."""
        # Analyze tag frequencies over time
        now = datetime.now(timezone.utc)
        tag_timeline: dict[str, list[datetime]] = defaultdict(list)

        for mem in memories:
            for tag in mem.tags:
                tag_timeline[tag].append(mem.created_at)

        if not tag_timeline:
            return None

        # Find trending topics (increasing frequency)
        trending = []
        for tag, dates in tag_timeline.items():
            if len(dates) < 3:
                continue

            sorted_dates = sorted(dates)
            recent_count = sum(1 for d in sorted_dates if (now - d).days <= 30)
            older_count = len(dates) - recent_count

            if recent_count > older_count:
                trending.append({
                    "topic": tag,
                    "trend": "increasing",
                    "recent_frequency": recent_count,
                })

        return MemoryForecast(
            forecast_type="topic",
            time_period=f"next {forecast_days} days",
            predictions=trending[:5],  # Top 5
            confidence=0.5,
            basis="Based on historical tag frequency trends",
        )

    def _forecast_importance_trends(
        self,
        memories: list[Memory],
        forecast_days: int,
    ) -> Optional[MemoryForecast]:
        """Forecast importance score trends."""
        if not memories:
            return None

        # Analyze average importance over time
        now = datetime.now(timezone.utc)
        time_buckets: dict[int, list[float]] = defaultdict(list)

        for mem in memories:
            days_old = (now - mem.created_at).days
            bucket = days_old // 30  # Monthly buckets
            time_buckets[bucket].append(mem.importance)

        if len(time_buckets) < 2:
            return None

        # Calculate trend
        bucket_avgs = {
            bucket: sum(scores) / len(scores)
            for bucket, scores in time_buckets.items()
        }

        recent_avg = bucket_avgs.get(0, 5.0)

        return MemoryForecast(
            forecast_type="importance",
            time_period=f"next {forecast_days} days",
            predictions=[{
                "predicted_avg_importance": round(recent_avg, 2),
                "trend": "stable",
            }],
            confidence=0.5,
            basis="Based on recent importance score patterns",
        )

    def _detect_recurring_patterns(
        self,
        memories: list[Memory],
    ) -> list[PatternPrediction]:
        """Detect recurring temporal patterns."""
        predictions = []

        # Group by tags to find recurring topics
        tag_timeline: dict[str, list[datetime]] = defaultdict(list)
        for mem in memories:
            for tag in mem.tags:
                tag_timeline[tag].append(mem.created_at)

        # Find patterns with regular intervals
        for tag, dates in tag_timeline.items():
            if len(dates) < 3:
                continue

            sorted_dates = sorted(dates)
            intervals = [
                (sorted_dates[i + 1] - sorted_dates[i]).days
                for i in range(len(sorted_dates) - 1)
            ]

            if not intervals:
                continue

            # Check for regularity
            avg_interval = sum(intervals) / len(intervals)
            variance = sum((i - avg_interval) ** 2 for i in intervals) / len(intervals)

            if variance < avg_interval * 0.3:  # Low variance = regular pattern
                last_date = sorted_dates[-1]
                predicted_date = last_date + timedelta(days=avg_interval)

                predictions.append(
                    PatternPrediction(
                        pattern_type="recurring",
                        predicted_date=predicted_date,
                        confidence=0.7,
                        description=f"Topic '{tag}' recurs approximately every {int(avg_interval)} days",
                        historical_basis=sorted_dates,
                    )
                )

        return predictions

    def _detect_seasonal_patterns(
        self,
        memories: list[Memory],
    ) -> list[PatternPrediction]:
        """Detect seasonal patterns."""
        predictions = []

        # Group by month/day to find annual patterns
        monthly_activity: dict[int, int] = defaultdict(int)
        for mem in memories:
            month = mem.created_at.month
            monthly_activity[month] += 1

        if not monthly_activity:
            return []

        # Find peak months
        avg_activity = sum(monthly_activity.values()) / len(monthly_activity)
        peak_months = [
            month for month, count in monthly_activity.items()
            if count > avg_activity * 1.5
        ]

        now = datetime.now(timezone.utc)
        for month in peak_months:
            # Predict next occurrence
            if month > now.month:
                predicted_year = now.year
            else:
                predicted_year = now.year + 1

            predicted_date = datetime(predicted_year, month, 15, tzinfo=timezone.utc)

            predictions.append(
                PatternPrediction(
                    pattern_type="seasonal",
                    predicted_date=predicted_date,
                    confidence=0.6,
                    description=f"Increased activity in month {month}",
                    historical_basis=[],
                )
            )

        return predictions
