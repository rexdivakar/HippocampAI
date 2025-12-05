"""Demo: Predictive analytics and proactive recommendations."""

from datetime import datetime, timedelta

from hippocampai.client import MemoryClient
from hippocampai.models.prediction import (
    ForecastHorizon,
    ForecastMetric,
)
from hippocampai.pipeline.predictive_analytics import PredictiveAnalyticsEngine
from hippocampai.pipeline.temporal_analytics import TemporalAnalytics


def main():
    """Demonstrate predictive analytics capabilities."""

    # Initialize client
    client = MemoryClient(user_id="user_predictive_demo")

    # Create temporal and predictive engines
    temporal_analytics = TemporalAnalytics()
    predictive_engine = PredictiveAnalyticsEngine(temporal_analytics)

    print("=" * 80)
    print("PREDICTIVE ANALYTICS & RECOMMENDATIONS DEMO")
    print("=" * 80)

    # 1. Create sample memories with patterns
    print("\n" + "=" * 80)
    print("1. Creating Sample Memories with Patterns")
    print("=" * 80)

    # Simulate daily workout pattern
    for i in range(15):
        date = datetime.now() - timedelta(days=i)
        client.remember(
            f"Morning workout completed on day {15-i}",
            type="habit",
            importance=7.0,
            tags=["fitness", "morning"],
            metadata={"created_at": date.isoformat()}
        )
    print("✓ Created 15 workout memories (daily pattern)")

    # Simulate weekly meeting pattern
    for i in range(4):
        date = datetime.now() - timedelta(weeks=i)
        client.remember(
            f"Team standup meeting - week {4-i}",
            type="event",
            importance=6.0,
            tags=["work", "meeting"],
            metadata={"created_at": date.isoformat()}
        )
    print("✓ Created 4 meeting memories (weekly pattern)")

    # Add some random memories
    for i in range(10):
        client.remember(
            f"Random note about topic {i}",
            type="fact",
            importance=5.0 + (i % 5),
            tags=[f"topic_{i % 3}"]
        )
    print("✓ Created 10 random memories")

    # Get all memories
    memories = client.get_memories()
    print(f"\n✓ Total memories: {len(memories)}")

    # 2. Detect patterns
    print("\n" + "=" * 80)
    print("2. Detecting Temporal Patterns")
    print("=" * 80)

    patterns = temporal_analytics.detect_temporal_patterns(memories, min_occurrences=3)
    print(f"✓ Detected {len(patterns)} pattern(s):")

    for pattern in patterns:
        print(f"\n  Pattern: {pattern.description}")
        print(f"  Type: {pattern.pattern_type}")
        print(f"  Frequency: {pattern.frequency:.2f}")
        print(f"  Confidence: {pattern.confidence:.0%}")
        print(f"  Regularity: {pattern.regularity_score:.0%}")
        if pattern.next_predicted:
            print(f"  Next predicted: {pattern.next_predicted.strftime('%Y-%m-%d %H:%M')}")

    # 3. Generate predictions
    print("\n" + "=" * 80)
    print("3. Generating Predictions")
    print("=" * 80)

    for pattern in patterns[:2]:  # Top 2 patterns
        prediction = predictive_engine.predict_next_occurrence(
            user_id=client.user_id,
            pattern=pattern
        )

        print(f"\n✓ Prediction: {prediction.prediction_type.value}")
        print(f"  When: {prediction.predicted_datetime.strftime('%Y-%m-%d %H:%M')}")
        print(f"  Confidence: {prediction.confidence:.0%}")
        print(f"  Recommendation: {prediction.recommendation}")
        print("  Factors:")
        for factor in prediction.factors:
            print(f"    - {factor}")

    # 4. Detect anomalies
    print("\n" + "=" * 80)
    print("4. Detecting Anomalies")
    print("=" * 80)

    # Add anomalous behavior (sudden spike in activity)
    print("  Simulating anomalous behavior...")
    for i in range(20):  # Sudden spike
        client.remember(
            f"Sudden burst of activity - item {i}",
            type="fact",
            importance=8.0,
            tags=["anomaly_test"]
        )

    memories = client.get_memories()
    anomalies = predictive_engine.detect_anomalies(
        user_id=client.user_id,
        memories=memories,
        lookback_days=30
    )

    print(f"✓ Detected {len(anomalies)} anomal(ies):")
    for anomaly in anomalies:
        print(f"\n  {anomaly.title}")
        print(f"  Type: {anomaly.anomaly_type.value}")
        print(f"  Severity: {anomaly.severity.value}")
        print(f"  Description: {anomaly.description}")
        print(f"  Expected: {anomaly.expected_behavior}")
        print(f"  Actual: {anomaly.actual_behavior}")
        print("  Suggestions:")
        for suggestion in anomaly.suggestions:
            print(f"    - {suggestion}")

    # 5. Generate recommendations
    print("\n" + "=" * 80)
    print("5. Generating Proactive Recommendations")
    print("=" * 80)

    recommendations = predictive_engine.generate_recommendations(
        user_id=client.user_id,
        memories=memories,
        max_recommendations=5
    )

    print(f"✓ Generated {len(recommendations)} recommendation(s):")
    for i, rec in enumerate(recommendations, 1):
        print(f"\n  {i}. {rec.title}")
        print(f"     Type: {rec.recommendation_type.value}")
        print(f"     Priority: {rec.priority}/10")
        print(f"     Reason: {rec.reason}")
        print(f"     Action: {rec.action}")
        print(f"     Confidence: {rec.confidence:.0%}")

    # 6. Forecast metrics
    print("\n" + "=" * 80)
    print("6. Forecasting Future Metrics")
    print("=" * 80)

    # Forecast activity for next week
    activity_forecast = predictive_engine.forecast_metric(
        user_id=client.user_id,
        memories=memories,
        metric=ForecastMetric.ACTIVITY_LEVEL,
        horizon=ForecastHorizon.NEXT_WEEK
    )

    print("✓ Activity Forecast (Next Week):")
    print(f"  Predicted value: {activity_forecast.predicted_value:.1f} memories/day")
    print(f"  Confidence interval: {activity_forecast.confidence_interval[0]:.1f} - {activity_forecast.confidence_interval[1]:.1f}")
    print(f"  Confidence: {activity_forecast.confidence:.0%}")
    print(f"  Method: {activity_forecast.method}")

    # Forecast importance
    importance_forecast = predictive_engine.forecast_metric(
        user_id=client.user_id,
        memories=memories,
        metric=ForecastMetric.IMPORTANCE_AVERAGE,
        horizon=ForecastHorizon.NEXT_MONTH
    )

    print("\n✓ Importance Forecast (Next Month):")
    print(f"  Predicted average: {importance_forecast.predicted_value:.1f}/10")
    print(f"  Confidence interval: {importance_forecast.confidence_interval[0]:.1f} - {importance_forecast.confidence_interval[1]:.1f}")

    # 7. Generate predictive insights
    print("\n" + "=" * 80)
    print("7. Generating Predictive Insights")
    print("=" * 80)

    insights = predictive_engine.generate_predictive_insights(
        user_id=client.user_id,
        memories=memories
    )

    print(f"✓ Generated {len(insights)} insight(s):")
    for insight in insights:
        print(f"\n  📊 {insight.title}")
        print(f"     Type: {insight.insight_type}")
        print(f"     Description: {insight.description}")
        print(f"     Impact: {insight.impact}")
        print(f"     Confidence: {insight.confidence:.0%}")
        print("     Evidence:")
        for evidence in insight.evidence:
            print(f"       - {evidence}")

    # 8. Trend analysis with forecasting
    print("\n" + "=" * 80)
    print("8. Trend Analysis with Forecasting")
    print("=" * 80)

    # Get trend
    trend = temporal_analytics.analyze_trends(
        memories=memories,
        time_window_days=30,
        metric="activity"
    )

    print("✓ Activity Trend (Last 30 Days):")
    print(f"  Direction: {trend.direction.value}")
    print(f"  Strength: {trend.strength:.0%}")
    print(f"  Current value: {trend.current_value:.1f} memories/day")
    print(f"  Change rate: {trend.change_rate:+.2f} memories/day")
    if trend.forecast:
        print(f"  Forecast (next period): {trend.forecast:.1f} memories/day")

    # 9. Summary
    print("\n" + "=" * 80)
    print("9. Summary")
    print("=" * 80)

    print("✓ Predictive analytics demo completed!")
    print(f"  - Analyzed {len(memories)} memories")
    print(f"  - Detected {len(patterns)} temporal pattern(s)")
    print(f"  - Found {len(anomalies)} anomal(ies)")
    print(f"  - Generated {len(recommendations)} recommendation(s)")
    print(f"  - Created {len(insights)} predictive insight(s)")
    print("  - Forecasted 2 metric(s)")

    print("\n💡 Key Features Demonstrated:")
    print("  1. Pattern detection and prediction")
    print("  2. Anomaly detection with severity levels")
    print("  3. Proactive recommendations")
    print("  4. Metric forecasting (activity, importance, engagement)")
    print("  5. Predictive insights generation")
    print("  6. Trend analysis with forecasts")


if __name__ == "__main__":
    main()
