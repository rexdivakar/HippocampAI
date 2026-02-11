"""Predictive analytics engine for memory intelligence.

This module provides:
- Pattern-based predictions
- Anomaly detection
- Proactive recommendations
- Forecasting future memory behavior
"""

import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np

from hippocampai.models.memory import Memory, MemoryType
from hippocampai.models.prediction import (
    AnomalyDetection,
    AnomalySeverity,
    AnomalyType,
    ForecastHorizon,
    ForecastMetric,
    MemoryForecast,
    MemoryPrediction,
    PredictionType,
    PredictiveInsight,
    Recommendation,
    RecommendationType,
)
from hippocampai.pipeline.temporal_analytics import TemporalAnalytics, TemporalPattern

logger = logging.getLogger(__name__)


class PredictiveAnalyticsEngine:
    """
    Predictive analytics engine for proactive memory intelligence.

    Features:
    - Predict when patterns will continue
    - Detect anomalies in memory behavior
    - Generate proactive recommendations
    - Forecast future memory metrics
    """

    def __init__(self, temporal_analytics: Optional[TemporalAnalytics] = None):
        """Initialize predictive analytics engine."""
        self.temporal_analytics = temporal_analytics or TemporalAnalytics()

    def predict_next_occurrence(
        self, user_id: str, pattern: TemporalPattern, context: Optional[dict] = None
    ) -> MemoryPrediction:
        """
        Predict when a pattern will occur next.

        Args:
            user_id: User ID
            pattern: Detected temporal pattern
            context: Optional context

        Returns:
            Memory prediction
        """
        prediction = MemoryPrediction(
            user_id=user_id,
            prediction_type=PredictionType.NEXT_OCCURRENCE,
            predicted_datetime=pattern.next_predicted or datetime.now(timezone.utc),
            confidence=pattern.confidence * pattern.regularity_score,
            factors=[
                f"Pattern type: {pattern.pattern_type}",
                f"Frequency: {pattern.frequency:.2f}",
                f"Regularity: {pattern.regularity_score:.2f}",
                f"Observed {len(pattern.occurrences)} times",
            ],
            recommendation=f"Based on {pattern.description}, expect similar activity around {pattern.next_predicted.strftime('%Y-%m-%d %H:%M') if pattern.next_predicted else 'N/A'}",
        )

        return prediction

    def detect_anomalies(
        self, user_id: str, memories: list[Memory], lookback_days: int = 30
    ) -> list[AnomalyDetection]:
        """
        Detect anomalies in memory patterns.

        Args:
            user_id: User ID
            memories: List of memories to analyze
            lookback_days: Days to look back for baseline

        Returns:
            List of detected anomalies
        """
        anomalies = []

        if not memories:
            return anomalies

        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(days=lookback_days)
        recent_memories = [m for m in memories if m.created_at >= cutoff]

        # 1. Activity anomaly detection
        activity_anomaly = self._detect_activity_anomaly(user_id, memories, recent_memories, now)
        if activity_anomaly:
            anomalies.append(activity_anomaly)

        # 2. Content anomaly detection
        content_anomaly = self._detect_content_anomaly(user_id, memories, recent_memories)
        if content_anomaly:
            anomalies.append(content_anomaly)

        # 3. Missing pattern detection
        missing_pattern = self._detect_missing_pattern(user_id, memories, recent_memories)
        if missing_pattern:
            anomalies.append(missing_pattern)

        # 4. Behavior change detection
        behavior_change = self._detect_behavior_change(user_id, memories)
        if behavior_change:
            anomalies.append(behavior_change)

        logger.info(f"Detected {len(anomalies)} anomalies for user {user_id}")
        return anomalies

    def generate_recommendations(
        self, user_id: str, memories: list[Memory], max_recommendations: int = 10
    ) -> list[Recommendation]:
        """
        Generate proactive recommendations.

        Args:
            user_id: User ID
            memories: List of memories
            max_recommendations: Maximum recommendations to return

        Returns:
            List of recommendations
        """
        recommendations = []

        if not memories:
            # First-time user recommendation
            recommendations.append(
                Recommendation(
                    user_id=user_id,
                    recommendation_type=RecommendationType.REMEMBER_THIS,
                    priority=8,
                    title="Start building your memory",
                    reason="You haven't stored any memories yet",
                    action="Create your first memory to begin tracking important information",
                    confidence=1.0,
                )
            )
            return recommendations

        # 1. Review stale memories
        stale_recs = self._recommend_review_stale(user_id, memories)
        recommendations.extend(stale_recs)

        # 2. Create habits from patterns
        habit_recs = self._recommend_create_habits(user_id, memories)
        recommendations.extend(habit_recs)

        # 3. Consolidate similar memories
        consolidate_recs = self._recommend_consolidation(user_id, memories)
        recommendations.extend(consolidate_recs)

        # 4. Update preferences based on behavior
        preference_recs = self._recommend_preference_updates(user_id, memories)
        recommendations.extend(preference_recs)

        # 5. Set goals based on trends
        goal_recs = self._recommend_goals(user_id, memories)
        recommendations.extend(goal_recs)

        # Sort by priority and limit
        recommendations.sort(key=lambda r: r.priority, reverse=True)
        return recommendations[:max_recommendations]

    def forecast_metric(
        self,
        user_id: str,
        memories: list[Memory],
        metric: ForecastMetric,
        horizon: ForecastHorizon,
    ) -> MemoryForecast:
        """
        Forecast a future memory metric.

        Args:
            user_id: User ID
            memories: List of memories
            metric: Metric to forecast
            horizon: Time horizon

        Returns:
            Memory forecast
        """
        # Determine forecast date based on horizon
        now = datetime.now(timezone.utc)
        if horizon == ForecastHorizon.NEXT_DAY:
            forecast_date = now + timedelta(days=1)
            lookback_days = 30
        elif horizon == ForecastHorizon.NEXT_WEEK:
            forecast_date = now + timedelta(days=7)
            lookback_days = 60
        elif horizon == ForecastHorizon.NEXT_MONTH:
            forecast_date = now + timedelta(days=30)
            lookback_days = 90
        else:  # NEXT_QUARTER
            forecast_date = now + timedelta(days=90)
            lookback_days = 180

        # Get historical data
        cutoff = now - timedelta(days=lookback_days)
        recent_memories = [m for m in memories if m.created_at >= cutoff]

        if metric == ForecastMetric.ACTIVITY_LEVEL:
            return self._forecast_activity(user_id, recent_memories, forecast_date, horizon)
        elif metric == ForecastMetric.IMPORTANCE_AVERAGE:
            return self._forecast_importance(user_id, recent_memories, forecast_date, horizon)
        elif metric == ForecastMetric.ENGAGEMENT:
            return self._forecast_engagement(user_id, recent_memories, forecast_date, horizon)
        else:
            # Default forecast
            return MemoryForecast(
                user_id=user_id,
                metric=metric,
                horizon=horizon,
                forecast_date=forecast_date,
                predicted_value=0.0,
                confidence_interval=(0.0, 0.0),
                confidence=0.5,
                method="baseline",
            )

    def generate_predictive_insights(
        self, user_id: str, memories: list[Memory]
    ) -> list[PredictiveInsight]:
        """
        Generate high-level predictive insights.

        Args:
            user_id: User ID
            memories: List of memories

        Returns:
            List of predictive insights
        """
        insights = []

        if not memories or len(memories) < 10:
            return insights

        # 1. Detect emerging patterns
        patterns = self.temporal_analytics.detect_temporal_patterns(memories, min_occurrences=3)
        for pattern in patterns:
            if pattern.confidence > 0.7:
                insights.append(
                    PredictiveInsight(
                        user_id=user_id,
                        insight_type="pattern_emerging",
                        title=f"Emerging pattern: {pattern.description}",
                        description=f"You consistently {pattern.description} with {pattern.confidence:.0%} confidence",
                        evidence=[
                            f"Detected {len(pattern.occurrences)} occurrences",
                            f"Regularity score: {pattern.regularity_score:.2f}",
                        ],
                        confidence=pattern.confidence,
                        impact="medium",
                    )
                )

        # 2. Detect behavior shifts
        trends = self.temporal_analytics.analyze_trends(
            memories, time_window_days=30, metric="activity"
        )
        if trends.strength > 0.6:
            insights.append(
                PredictiveInsight(
                    user_id=user_id,
                    insight_type="behavior_shift",
                    title=f"Memory activity is {trends.direction.value}",
                    description=f"Your memory activity has been {trends.direction.value} over the past 30 days",
                    evidence=[
                        f"Current level: {trends.current_value:.1f} memories/day",
                        f"Trend strength: {trends.strength:.0%}",
                        f"Change rate: {trends.change_rate:.2f} memories/day",
                    ],
                    confidence=trends.strength,
                    impact="high" if abs(trends.change_rate) > 0.5 else "medium",
                )
            )

        # 3. Identify opportunities
        memory_by_type = defaultdict(int)
        for memory in memories[-30:]:  # Last 30 memories
            memory_by_type[memory.type] += 1

        # Check for underutilized memory types
        all_types = set(MemoryType)
        used_types = set(memory_by_type.keys())
        unused_types = all_types - used_types

        if unused_types:
            insights.append(
                PredictiveInsight(
                    user_id=user_id,
                    insight_type="opportunity",
                    title="Expand your memory types",
                    description=f"You're not using {len(unused_types)} memory types",
                    evidence=[f"Unused types: {', '.join(t.value for t in unused_types)}"],
                    confidence=0.9,
                    impact="low",
                )
            )

        return insights

    # === HELPER METHODS ===

    def _detect_activity_anomaly(
        self, user_id: str, all_memories: list[Memory], recent_memories: list[Memory], now: datetime
    ) -> Optional[AnomalyDetection]:
        """Detect anomalous activity levels."""
        if len(all_memories) < 30:
            return None

        # Calculate baseline (exclude recent memories)
        baseline_memories = [m for m in all_memories if m not in recent_memories]
        if not baseline_memories:
            return None

        # Calculate daily activity
        baseline_days = (
            baseline_memories[-1].created_at - baseline_memories[0].created_at
        ).days or 1
        baseline_rate = len(baseline_memories) / baseline_days

        recent_days = 7
        recent_rate = (
            len([m for m in recent_memories if (now - m.created_at).days <= recent_days])
            / recent_days
        )

        # Check if significantly different (>2 standard deviations)
        if abs(recent_rate - baseline_rate) / max(baseline_rate, 1) > 0.5:
            return AnomalyDetection(
                user_id=user_id,
                anomaly_type=AnomalyType.UNUSUAL_ACTIVITY,
                severity=AnomalySeverity.MEDIUM
                if recent_rate < baseline_rate
                else AnomalySeverity.LOW,
                title="Unusual memory activity detected",
                description="Your memory creation rate has changed significantly",
                expected_behavior=f"Typical: {baseline_rate:.1f} memories/day",
                actual_behavior=f"Recent: {recent_rate:.1f} memories/day",
                suggestions=[
                    "Review your recent memory creation patterns",
                    "Consider if this change is intentional",
                ],
            )

        return None

    def _detect_content_anomaly(
        self, user_id: str, all_memories: list[Memory], recent_memories: list[Memory]
    ) -> Optional[AnomalyDetection]:
        """Detect anomalous content patterns."""
        if len(recent_memories) < 5:
            return None

        # Check importance scores
        recent_importance = [m.importance for m in recent_memories[-10:]]
        avg_recent = np.mean(recent_importance)

        baseline_importance = [m.importance for m in all_memories if m not in recent_memories]
        if baseline_importance:
            avg_baseline = np.mean(baseline_importance)

            if abs(avg_recent - avg_baseline) > 3.0:  # Significant shift in importance
                return AnomalyDetection(
                    user_id=user_id,
                    anomaly_type=AnomalyType.UNEXPECTED_CONTENT,
                    severity=AnomalySeverity.LOW,
                    title="Change in memory importance detected",
                    description="Recent memories have different importance levels than usual",
                    expected_behavior=f"Typical importance: {avg_baseline:.1f}/10",
                    actual_behavior=f"Recent importance: {avg_recent:.1f}/10",
                    suggestions=["Review importance scoring", "Ensure consistency in assessment"],
                )

        return None

    def _detect_missing_pattern(
        self, user_id: str, all_memories: list[Memory], recent_memories: list[Memory]
    ) -> Optional[AnomalyDetection]:
        """Detect if expected pattern didn't occur."""
        patterns = self.temporal_analytics.detect_temporal_patterns(all_memories, min_occurrences=3)

        now = datetime.now(timezone.utc)
        for pattern in patterns:
            if pattern.next_predicted and pattern.next_predicted < now:
                # Pattern was expected but didn't happen
                days_late = (now - pattern.next_predicted).days
                if days_late > 3:  # More than 3 days late
                    return AnomalyDetection(
                        user_id=user_id,
                        anomaly_type=AnomalyType.MISSING_PATTERN,
                        severity=AnomalySeverity.MEDIUM,
                        title="Expected pattern did not occur",
                        description=f"{pattern.description} was expected but didn't happen",
                        expected_behavior=f"Expected around {pattern.next_predicted.strftime('%Y-%m-%d')}",
                        actual_behavior=f"No occurrence detected ({days_late} days late)",
                        suggestions=[
                            "Check if this pattern is still relevant",
                            "Consider updating your memory habits",
                        ],
                    )

        return None

    def _detect_behavior_change(
        self, user_id: str, all_memories: list[Memory]
    ) -> Optional[AnomalyDetection]:
        """Detect significant behavior changes."""
        if len(all_memories) < 50:
            return None

        # Compare type distribution: first half vs second half
        mid_point = len(all_memories) // 2
        first_half = all_memories[:mid_point]
        second_half = all_memories[mid_point:]

        first_types = defaultdict(int)
        second_types = defaultdict(int)

        for m in first_half:
            first_types[m.type] += 1
        for m in second_half:
            second_types[m.type] += 1

        # Normalize
        first_dist = {k: v / len(first_half) for k, v in first_types.items()}
        second_dist = {k: v / len(second_half) for k, v in second_types.items()}

        # Calculate divergence
        all_types = set(first_dist.keys()) | set(second_dist.keys())
        divergence = sum(abs(first_dist.get(t, 0) - second_dist.get(t, 0)) for t in all_types)

        if divergence > 0.5:  # Significant change
            return AnomalyDetection(
                user_id=user_id,
                anomaly_type=AnomalyType.BEHAVIOR_CHANGE,
                severity=AnomalySeverity.MEDIUM,
                title="Your memory patterns have shifted",
                description="The types of memories you create have changed significantly",
                expected_behavior=f"Earlier distribution: {dict(first_dist)}",
                actual_behavior=f"Current distribution: {dict(second_dist)}",
                suggestions=["Review if this change aligns with your goals"],
            )

        return None

    def _recommend_review_stale(self, user_id: str, memories: list[Memory]) -> list[Recommendation]:
        """Recommend reviewing stale memories."""
        recommendations = []
        now = datetime.now(timezone.utc)

        stale_threshold = 60  # days
        stale_memories = [
            m
            for m in memories
            if (now - m.created_at).days > stale_threshold
            and (now - (m.updated_at or m.created_at)).days > 30
        ]

        if len(stale_memories) > 5:
            recommendations.append(
                Recommendation(
                    user_id=user_id,
                    recommendation_type=RecommendationType.REVIEW_MEMORY,
                    priority=6,
                    title=f"Review {len(stale_memories)} stale memories",
                    reason="You have memories that haven't been accessed in a while",
                    action="Review and update or archive old memories",
                    related_memory_ids=[m.id for m in stale_memories[:10]],
                    confidence=0.8,
                )
            )

        return recommendations

    def _recommend_create_habits(
        self, user_id: str, memories: list[Memory]
    ) -> list[Recommendation]:
        """Recommend creating habits from patterns."""
        recommendations = []

        patterns = self.temporal_analytics.detect_temporal_patterns(memories, min_occurrences=4)

        for pattern in patterns[:2]:  # Top 2 patterns
            if pattern.regularity_score > 0.7 and pattern.confidence > 0.7:
                recommendations.append(
                    Recommendation(
                        user_id=user_id,
                        recommendation_type=RecommendationType.CREATE_HABIT,
                        priority=7,
                        title=f"Turn pattern into a habit: {pattern.description}",
                        reason=f"You consistently do this (regularity: {pattern.regularity_score:.0%})",
                        action="Create a habit memory to track this behavior",
                        confidence=pattern.confidence,
                    )
                )

        return recommendations

    def _recommend_consolidation(
        self, user_id: str, memories: list[Memory]
    ) -> list[Recommendation]:
        """Recommend consolidating similar memories."""
        # Simple check for memories with same tags
        tag_groups = defaultdict(list)
        for memory in memories:
            for tag in memory.tags:
                tag_groups[tag].append(memory)

        recommendations = []
        for tag, mems in tag_groups.items():
            if len(mems) > 5:
                recommendations.append(
                    Recommendation(
                        user_id=user_id,
                        recommendation_type=RecommendationType.CONSOLIDATE_MEMORIES,
                        priority=5,
                        title=f"Consolidate {len(mems)} memories about '{tag}'",
                        reason=f"You have many memories with tag '{tag}'",
                        action="Consider merging similar memories to reduce redundancy",
                        related_memory_ids=[m.id for m in mems],
                        confidence=0.7,
                    )
                )
                break  # Only suggest one consolidation at a time

        return recommendations

    def _recommend_preference_updates(
        self, user_id: str, memories: list[Memory]
    ) -> list[Recommendation]:
        """Recommend updating preferences based on behavior."""
        # Check for repeated facts that could be preferences
        preference_memories = [m for m in memories if m.type == MemoryType.PREFERENCE]
        fact_memories = [m for m in memories if m.type == MemoryType.FACT]

        # If many facts about same topic, suggest making it a preference
        if len(fact_memories) > len(preference_memories) * 3:
            return [
                Recommendation(
                    user_id=user_id,
                    recommendation_type=RecommendationType.UPDATE_PREFERENCE,
                    priority=6,
                    title="Convert repeated facts to preferences",
                    reason="You have many factual memories that might be preferences",
                    action="Review your facts and identify preferences",
                    confidence=0.6,
                )
            ]

        return []

    def _recommend_goals(self, user_id: str, memories: list[Memory]) -> list[Recommendation]:
        """Recommend setting goals based on trends."""
        goal_memories = [m for m in memories if m.type == MemoryType.GOAL]

        if len(goal_memories) == 0 and len(memories) > 20:
            return [
                Recommendation(
                    user_id=user_id,
                    recommendation_type=RecommendationType.SET_GOAL,
                    priority=7,
                    title="Set your first goal",
                    reason="You have memories but no goals defined",
                    action="Create a goal memory to track aspirations",
                    confidence=0.8,
                )
            ]

        return []

    def _forecast_activity(
        self,
        user_id: str,
        memories: list[Memory],
        forecast_date: datetime,
        horizon: ForecastHorizon,
    ) -> MemoryForecast:
        """Forecast activity level."""
        if len(memories) < 7:
            return MemoryForecast(
                user_id=user_id,
                metric=ForecastMetric.ACTIVITY_LEVEL,
                horizon=horizon,
                forecast_date=forecast_date,
                predicted_value=0.0,
                confidence_interval=(0.0, 0.0),
                confidence=0.3,
                method="insufficient_data",
            )

        # Simple moving average forecast
        daily_counts = defaultdict(int)
        for memory in memories:
            date = memory.created_at.date()
            daily_counts[date] += 1

        if not daily_counts:
            return MemoryForecast(
                user_id=user_id,
                metric=ForecastMetric.ACTIVITY_LEVEL,
                horizon=horizon,
                forecast_date=forecast_date,
                predicted_value=0.0,
                confidence_interval=(0.0, 0.0),
                confidence=0.3,
                method="no_data",
            )

        recent_counts = list(daily_counts.values())[-7:]
        avg_activity = np.mean(recent_counts)
        std_activity = np.std(recent_counts) if len(recent_counts) > 1 else avg_activity * 0.2

        # Confidence interval
        lower = max(0, avg_activity - 1.96 * std_activity)
        upper = avg_activity + 1.96 * std_activity

        # Historical data for context
        historical_data = [
            (datetime.combine(date, datetime.min.time()).replace(tzinfo=timezone.utc), float(count))
            for date, count in sorted(daily_counts.items())[-30:]
        ]

        return MemoryForecast(
            user_id=user_id,
            metric=ForecastMetric.ACTIVITY_LEVEL,
            horizon=horizon,
            forecast_date=forecast_date,
            predicted_value=float(avg_activity),
            confidence_interval=(float(lower), float(upper)),
            confidence=0.7,
            method="moving_average",
            historical_data=historical_data,
        )

    def _forecast_importance(
        self,
        user_id: str,
        memories: list[Memory],
        forecast_date: datetime,
        horizon: ForecastHorizon,
    ) -> MemoryForecast:
        """Forecast average importance."""
        if not memories:
            return MemoryForecast(
                user_id=user_id,
                metric=ForecastMetric.IMPORTANCE_AVERAGE,
                horizon=horizon,
                forecast_date=forecast_date,
                predicted_value=5.0,
                confidence_interval=(4.0, 6.0),
                confidence=0.5,
                method="baseline",
            )

        recent_importance = [m.importance for m in memories[-30:]]
        avg_importance = np.mean(recent_importance)
        std_importance = np.std(recent_importance)

        return MemoryForecast(
            user_id=user_id,
            metric=ForecastMetric.IMPORTANCE_AVERAGE,
            horizon=horizon,
            forecast_date=forecast_date,
            predicted_value=float(avg_importance),
            confidence_interval=(float(avg_importance - std_importance), float(avg_importance + std_importance)),
            confidence=0.7,
            method="historical_average",
        )

    def _forecast_engagement(
        self,
        user_id: str,
        memories: list[Memory],
        forecast_date: datetime,
        horizon: ForecastHorizon,
    ) -> MemoryForecast:
        """Forecast user engagement level."""
        if not memories:
            return MemoryForecast(
                user_id=user_id,
                metric=ForecastMetric.ENGAGEMENT,
                horizon=horizon,
                forecast_date=forecast_date,
                predicted_value=0.0,
                confidence_interval=(0.0, 0.0),
                confidence=0.5,
                method="baseline",
            )

        # Engagement = average access count
        recent_engagement = [m.access_count for m in memories[-30:]]
        avg_engagement = np.mean(recent_engagement)
        std_engagement = np.std(recent_engagement)

        return MemoryForecast(
            user_id=user_id,
            metric=ForecastMetric.ENGAGEMENT,
            horizon=horizon,
            forecast_date=forecast_date,
            predicted_value=float(avg_engagement),
            confidence_interval=(
                float(max(0, avg_engagement - std_engagement)),
                float(avg_engagement + std_engagement),
            ),
            confidence=0.6,
            method="access_pattern_analysis",
        )
