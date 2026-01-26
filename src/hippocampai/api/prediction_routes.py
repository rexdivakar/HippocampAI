"""REST API routes for predictive analytics."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from hippocampai.client import MemoryClient
from hippocampai.models.prediction import ForecastHorizon, ForecastMetric
from hippocampai.pipeline.predictive_analytics import PredictiveAnalyticsEngine
from hippocampai.pipeline.temporal_analytics import TemporalAnalytics

router = APIRouter(prefix="/v1/predictions", tags=["predictions"])

# Initialize engines
temporal_analytics = TemporalAnalytics()
predictive_engine = PredictiveAnalyticsEngine(temporal_analytics)


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================


class PredictionRequest(BaseModel):
    user_id: str
    min_occurrences: int = 3


class AnomalyRequest(BaseModel):
    user_id: str
    lookback_days: int = 30


class RecommendationRequest(BaseModel):
    user_id: str
    max_recommendations: int = 10


class ForecastRequest(BaseModel):
    user_id: str
    metric: str  # "activity_level", "importance_average", "engagement"
    horizon: str  # "next_day", "next_week", "next_month", "next_quarter"


# ============================================================================
# PATTERN DETECTION
# ============================================================================


@router.post("/patterns")
async def detect_patterns(request: PredictionRequest):
    """Detect temporal patterns in memories."""
    try:
        # Get client and memories
        client = MemoryClient()
        memories = client.get_memories(user_id=request.user_id)

        # Detect patterns
        patterns = temporal_analytics.detect_temporal_patterns(
            memories=memories, min_occurrences=request.min_occurrences
        )

        return {
            "patterns": [
                {
                    "pattern_type": pattern.pattern_type,
                    "description": pattern.description,
                    "frequency": pattern.frequency,
                    "confidence": pattern.confidence,
                    "regularity_score": pattern.regularity_score,
                    "occurrences_count": len(pattern.occurrences),
                    "next_predicted": pattern.next_predicted.isoformat()
                    if pattern.next_predicted
                    else None,
                }
                for pattern in patterns
            ],
            "count": len(patterns),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/patterns/predict")
async def predict_next_occurrence(request: PredictionRequest):
    """Predict next occurrence for all detected patterns."""
    try:
        client = MemoryClient()
        memories = client.get_memories(user_id=request.user_id)

        patterns = temporal_analytics.detect_temporal_patterns(memories, request.min_occurrences)

        predictions = []
        for pattern in patterns:
            prediction = predictive_engine.predict_next_occurrence(
                user_id=request.user_id, pattern=pattern
            )
            predictions.append(
                {
                    "prediction_type": prediction.prediction_type.value,
                    "predicted_datetime": prediction.predicted_datetime.isoformat(),
                    "confidence": prediction.confidence,
                    "factors": prediction.factors,
                    "recommendation": prediction.recommendation,
                }
            )

        return {"predictions": predictions, "count": len(predictions)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# ANOMALY DETECTION
# ============================================================================


@router.post("/anomalies")
async def detect_anomalies(request: AnomalyRequest):
    """Detect anomalies in memory patterns."""
    try:
        client = MemoryClient()
        memories = client.get_memories(user_id=request.user_id)

        anomalies = predictive_engine.detect_anomalies(
            user_id=request.user_id, memories=memories, lookback_days=request.lookback_days
        )

        return {
            "anomalies": [
                {
                    "id": anomaly.id,
                    "title": anomaly.title,
                    "description": anomaly.description,
                    "anomaly_type": anomaly.anomaly_type.value,
                    "severity": anomaly.severity.value,
                    "expected_behavior": anomaly.expected_behavior,
                    "actual_behavior": anomaly.actual_behavior,
                    "suggestions": anomaly.suggestions,
                    "detected_at": anomaly.detected_at.isoformat(),
                }
                for anomaly in anomalies
            ],
            "count": len(anomalies),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# RECOMMENDATIONS
# ============================================================================


@router.post("/recommendations")
async def generate_recommendations(request: RecommendationRequest):
    """Generate proactive recommendations."""
    try:
        client = MemoryClient()
        memories = client.get_memories(user_id=request.user_id)

        recommendations = predictive_engine.generate_recommendations(
            user_id=request.user_id,
            memories=memories,
            max_recommendations=request.max_recommendations,
        )

        return {
            "recommendations": [
                {
                    "id": rec.id,
                    "type": rec.recommendation_type.value,
                    "priority": rec.priority,
                    "title": rec.title,
                    "reason": rec.reason,
                    "action": rec.action,
                    "confidence": rec.confidence,
                    "created_at": rec.created_at.isoformat(),
                }
                for rec in recommendations
            ],
            "count": len(recommendations),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# FORECASTING
# ============================================================================


@router.post("/forecast")
async def forecast_metric(request: ForecastRequest):
    """Forecast a future memory metric."""
    try:
        client = MemoryClient()
        memories = client.get_memories(user_id=request.user_id)

        # Convert string enums
        metric = ForecastMetric(request.metric)
        horizon = ForecastHorizon(request.horizon)

        forecast = predictive_engine.forecast_metric(
            user_id=request.user_id, memories=memories, metric=metric, horizon=horizon
        )

        return {
            "metric": forecast.metric.value,
            "horizon": forecast.horizon.value,
            "forecast_date": forecast.forecast_date.isoformat(),
            "predicted_value": forecast.predicted_value,
            "confidence_interval": {
                "lower": forecast.confidence_interval[0],
                "upper": forecast.confidence_interval[1],
            },
            "confidence": forecast.confidence,
            "method": forecast.method,
            "historical_data": [
                {"date": dt.isoformat(), "value": val} for dt, val in forecast.historical_data
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# INSIGHTS
# ============================================================================


@router.post("/insights")
async def generate_insights(request: PredictionRequest):
    """Generate predictive insights."""
    try:
        client = MemoryClient()
        memories = client.get_memories(user_id=request.user_id)

        insights = predictive_engine.generate_predictive_insights(
            user_id=request.user_id, memories=memories
        )

        return {
            "insights": [
                {
                    "id": insight.id,
                    "type": insight.insight_type,
                    "title": insight.title,
                    "description": insight.description,
                    "evidence": insight.evidence,
                    "confidence": insight.confidence,
                    "impact": insight.impact,
                    "created_at": insight.created_at.isoformat(),
                }
                for insight in insights
            ],
            "count": len(insights),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# PEAK ACTIVITY ANALYSIS
# ============================================================================


@router.post("/activity/peaks")
async def analyze_peak_activity(user_id: str, timezone_offset: int = 0):
    """Analyze peak activity times."""
    try:
        client = MemoryClient()
        memories = client.get_memories(user_id=user_id)

        analysis = temporal_analytics.analyze_peak_activity(
            memories=memories, timezone_offset=timezone_offset
        )

        return {
            "peak_hour": analysis.peak_hour,
            "peak_day": analysis.peak_day.value,
            "peak_time_period": analysis.peak_time_period.value,
            "hourly_distribution": analysis.hourly_distribution,
            "daily_distribution": analysis.daily_distribution,
            "time_period_distribution": analysis.time_period_distribution,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# TREND ANALYSIS
# ============================================================================


@router.post("/trends")
async def analyze_trends(user_id: str, time_window_days: int = 30, metric: str = "activity"):
    """Analyze trends over time."""
    try:
        client = MemoryClient()
        memories = client.get_memories(user_id=user_id)

        trend = temporal_analytics.analyze_trends(
            memories=memories, time_window_days=time_window_days, metric=metric
        )

        return {
            "metric": trend.metric,
            "time_window_days": trend.time_window_days,
            "direction": trend.direction.value,
            "strength": trend.strength,
            "change_rate": trend.change_rate,
            "current_value": trend.current_value,
            "forecast": trend.forecast,
            "historical_values": [
                {"date": dt.isoformat(), "value": val} for dt, val in trend.historical_values
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
