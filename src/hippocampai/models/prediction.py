"""Models for predictive analytics and recommendations."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from hippocampai.models.memory import MemoryType


class PredictionType(str, Enum):
    """Types of predictions."""

    NEXT_OCCURRENCE = "next_occurrence"  # When user will remember similar thing
    TREND_FORECAST = "trend_forecast"  # Future trend in memory activity
    ANOMALY_ALERT = "anomaly_alert"  # Unusual behavior detected
    PATTERN_CONTINUATION = "pattern_continuation"  # Continuation of detected pattern
    RECOMMENDATION = "recommendation"  # Proactive suggestion


class MemoryPrediction(BaseModel):
    """Prediction about future memory behavior."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str
    prediction_type: PredictionType
    memory_type: Optional[MemoryType] = None
    predicted_datetime: datetime
    confidence: float = Field(ge=0.0, le=1.0, description="Prediction confidence")
    factors: list[str] = Field(
        default_factory=list, description="Factors that led to this prediction"
    )
    recommendation: str
    action: Optional[str] = None  # Suggested action
    related_memory_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None
    was_accurate: Optional[bool] = None  # Feedback on prediction accuracy


class AnomalyType(str, Enum):
    """Types of anomalies."""

    UNUSUAL_ACTIVITY = "unusual_activity"  # Activity spike or drop
    MISSING_PATTERN = "missing_pattern"  # Expected pattern didn't occur
    UNEXPECTED_CONTENT = "unexpected_content"  # Memory content is unusual
    BEHAVIOR_CHANGE = "behavior_change"  # User behavior shifted
    DATA_QUALITY = "data_quality"  # Data quality issue


class AnomalySeverity(str, Enum):
    """Severity levels for anomalies."""

    LOW = "low"  # Minor deviation
    MEDIUM = "medium"  # Notable deviation
    HIGH = "high"  # Significant deviation
    CRITICAL = "critical"  # Urgent attention needed


class AnomalyDetection(BaseModel):
    """Detected anomaly in memory patterns."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str
    detected_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    anomaly_type: AnomalyType
    severity: AnomalySeverity
    title: str
    description: str
    expected_behavior: str
    actual_behavior: str
    suggestions: list[str] = Field(default_factory=list)
    affected_memory_ids: list[str] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)
    is_resolved: bool = False
    resolved_at: Optional[datetime] = None


class RecommendationType(str, Enum):
    """Types of recommendations."""

    REMEMBER_THIS = "remember_this"  # Suggest creating memory
    REVIEW_MEMORY = "review_memory"  # Suggest reviewing existing memory
    CREATE_HABIT = "create_habit"  # Suggest creating habit memory
    UPDATE_PREFERENCE = "update_preference"  # Suggest updating preference
    CONSOLIDATE_MEMORIES = "consolidate_memories"  # Suggest merging memories
    ARCHIVE_MEMORY = "archive_memory"  # Suggest archiving old memory
    SET_GOAL = "set_goal"  # Suggest creating goal
    CREATE_SESSION = "create_session"  # Suggest starting session


class Recommendation(BaseModel):
    """Proactive recommendation for user."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str
    recommendation_type: RecommendationType
    priority: int = Field(ge=1, le=10, description="Priority (1=low, 10=urgent)")
    title: str
    reason: str
    action: str  # What user should do
    related_memory_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None
    is_dismissed: bool = False
    is_completed: bool = False
    completed_at: Optional[datetime] = None

    def dismiss(self) -> None:
        """Dismiss this recommendation."""
        self.is_dismissed = True

    def complete(self) -> None:
        """Mark recommendation as completed."""
        self.is_completed = True
        self.completed_at = datetime.now(timezone.utc)


class ForecastMetric(str, Enum):
    """Metrics that can be forecasted."""

    ACTIVITY_LEVEL = "activity_level"  # Number of memories per period
    IMPORTANCE_AVERAGE = "importance_average"  # Average importance
    TYPE_DISTRIBUTION = "type_distribution"  # Distribution of memory types
    ENGAGEMENT = "engagement"  # User engagement level
    MEMORY_QUALITY = "memory_quality"  # Overall quality score


class ForecastHorizon(str, Enum):
    """Time horizons for forecasts."""

    NEXT_DAY = "next_day"
    NEXT_WEEK = "next_week"
    NEXT_MONTH = "next_month"
    NEXT_QUARTER = "next_quarter"


class MemoryForecast(BaseModel):
    """Forecast of future memory metrics."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str
    metric: ForecastMetric
    horizon: ForecastHorizon
    forecast_date: datetime
    predicted_value: float
    confidence_interval: tuple[float, float]  # (lower, upper)
    confidence: float = Field(ge=0.0, le=1.0)
    method: str  # Forecasting method used
    historical_data: list[tuple[datetime, float]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class PredictiveInsight(BaseModel):
    """High-level predictive insight."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str
    insight_type: Literal[
        "pattern_emerging",
        "behavior_shift",
        "opportunity",
        "risk",
        "achievement",
        "milestone",
    ]
    title: str
    description: str
    evidence: list[str] = Field(
        default_factory=list, description="Supporting evidence for this insight"
    )
    predictions: list[str] = Field(default_factory=list, description="Related prediction IDs")
    recommendations: list[str] = Field(
        default_factory=list, description="Related recommendation IDs"
    )
    confidence: float = Field(ge=0.0, le=1.0)
    impact: Literal["low", "medium", "high"] = "medium"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    is_acknowledged: bool = False
