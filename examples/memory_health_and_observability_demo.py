"""
Demo: Memory Health, Quality Monitoring, and Observability Features

This example demonstrates:
1. Memory health scoring and monitoring
2. Stale memory detection
3. Duplicate detection and clustering
4. Coverage analysis
5. Temporal features (freshness, decay, forecasting)
6. Retrieval explainability
7. Performance profiling and observability
"""

import time

from hippocampai import MemoryClient
from hippocampai.models.memory import MemoryType


def print_section(title: str):
    """Print a section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def demo_health_monitoring(client: MemoryClient, user_id: str):
    """Demonstrate memory health monitoring."""
    print_section("1. MEMORY HEALTH MONITORING")

    # Get comprehensive health score
    print("\n📊 Calculating health score...")
    health = client.get_memory_health_score(
        user_id=user_id,
        include_stale_detection=True,
        include_duplicate_detection=True,
    )

    # Overall score
    score = health["overall_score"]
    if score >= 80:
        emoji = "✅"
        status = "Excellent"
    elif score >= 60:
        emoji = "⚠️"
        status = "Good"
    else:
        emoji = "❌"
        status = "Needs Attention"

    print(f"\n{emoji} Overall Health Score: {score:.1f}/100 ({status})")

    # Component breakdown
    print("\n📈 Component Scores:")
    components = [
        ("Quality", health["quality_score"]),
        ("Freshness", health["freshness_score"]),
        ("Diversity", health["diversity_score"]),
        ("Coverage", health["coverage_score"]),
        ("Duplication", health["duplication_score"]),
        ("Staleness", health["staleness_score"]),
    ]

    for name, value in components:
        bar = "█" * int(value / 5)
        print(f"  {name:12s}: {bar:20s} {value:.1f}/100")

    # Statistics
    print("\n📊 Statistics:")
    print(f"  Total memories:    {health['total_memories']}")
    print(f"  Stale memories:    {health['stale_count']}")
    print(f"  Duplicate clusters: {health['duplicate_clusters']}")
    print(f"  Avg importance:    {health['avg_importance']:.2f}")
    print(f"  Avg confidence:    {health['avg_confidence']:.2f}")
    print(f"  Avg age:           {health['avg_age_days']:.1f} days")

    # Recommendations
    if health["recommendations"]:
        print("\n💡 Recommendations:")
        for rec in health["recommendations"]:
            print(f"  • {rec}")


def demo_stale_detection(client: MemoryClient, user_id: str):
    """Demonstrate stale memory detection."""
    print_section("2. STALE MEMORY DETECTION")

    stale_memories = client.detect_stale_memories(user_id)

    if not stale_memories:
        print("\n✅ No stale memories detected!")
        return

    print(f"\n🔍 Found {len(stale_memories)} potentially stale memories:")

    # Group by reason
    reasons = {}
    for mem in stale_memories:
        reason = mem["staleness_reason"].split(";")[0]
        reasons[reason] = reasons.get(reason, 0) + 1

    print("\n📊 Staleness Reasons:")
    for reason, count in sorted(reasons.items(), key=lambda x: x[1], reverse=True):
        print(f"  • {reason}: {count} memories")

    # Show top stale memories
    print("\n🔝 Top 5 Stale Memories:")
    for i, mem in enumerate(stale_memories[:5], 1):
        print(f"\n  {i}. Age: {mem['age_days']:.0f} days | Confidence: {mem['confidence']:.2f}")
        print(f"     Text: {mem['text'][:100]}...")
        print(f"     Reason: {mem['staleness_reason']}")


def demo_duplicate_detection(client: MemoryClient, user_id: str):
    """Demonstrate duplicate detection."""
    print_section("3. DUPLICATE DETECTION")

    # Detect clusters
    clusters = client.detect_duplicate_clusters(user_id, min_cluster_size=2)

    if not clusters:
        print("\n✅ No duplicate clusters found!")
    else:
        print(f"\n🔄 Found {len(clusters)} duplicate clusters:")

        total_duplicates = sum(c["cluster_size"] - 1 for c in clusters)
        print(f"   💾 Could potentially remove {total_duplicates} duplicates")

        for i, cluster in enumerate(clusters[:3], 1):
            print(f"\n  Cluster {i}:")
            print(f"    Size: {cluster['cluster_size']} memories")
            print(f"    Avg similarity: {cluster['avg_similarity']:.2f}")
            print(f"    Representative: {cluster['representative_text'][:100]}...")

    # Near-duplicates
    print("\n🔎 Checking for near-duplicates...")
    warnings = client.detect_near_duplicates(user_id, suggest_merge=True)

    if warnings:
        print(f"\n⚠️  Found {len(warnings)} near-duplicate pairs:")

        for i, warning in enumerate(warnings[:3], 1):
            print(f"\n  Pair {i} (Similarity: {warning['similarity_score']:.2f}):")
            print(f"    Memory 1: {warning['text_1'][:70]}...")
            print(f"    Memory 2: {warning['text_2'][:70]}...")
            if warning["merge_suggestion"]:
                print(f"    💡 Suggestion: {warning['merge_suggestion']}")
    else:
        print("✅ No near-duplicates found!")


def demo_coverage_analysis(client: MemoryClient, user_id: str):
    """Demonstrate coverage analysis."""
    print_section("4. COVERAGE ANALYSIS")

    coverage = client.analyze_memory_coverage(user_id)

    # Topic distribution
    print("\n📚 Topic Distribution:")
    topic_dist = coverage["topic_distribution"]
    if topic_dist:
        for topic, count in sorted(topic_dist.items(), key=lambda x: x[1], reverse=True)[:10]:
            bar = "█" * count
            print(f"  {topic:20s}: {bar} ({count})")
    else:
        print("  No tagged topics found")

    # Type distribution
    print("\n📊 Memory Type Distribution:")
    for mem_type, count in coverage["type_distribution"].items():
        bar = "█" * (count // 5 or 1)
        print(f"  {mem_type:15s}: {bar} ({count})")

    # Well/poorly covered
    if coverage["well_covered_topics"]:
        print(f"\n✅ Well-covered topics: {', '.join(coverage['well_covered_topics'][:5])}")
    if coverage["poorly_covered_topics"]:
        print(f"⚠️  Poorly-covered topics: {', '.join(coverage['poorly_covered_topics'][:5])}")

    # Gaps
    if coverage["coverage_gaps"]:
        print("\n❌ Coverage gaps:")
        for gap in coverage["coverage_gaps"]:
            print(f"  • {gap}")

    # Recommendations
    if coverage["recommendations"]:
        print("\n💡 Recommendations:")
        for rec in coverage["recommendations"]:
            print(f"  • {rec}")


def demo_temporal_features(client: MemoryClient, user_id: str):
    """Demonstrate enhanced temporal features."""
    print_section("5. ENHANCED TEMPORAL FEATURES")

    memories = client.get_memories(user_id, limit=5)

    if not memories:
        print("\n⚠️  No memories found")
        return

    memory = memories[0]

    # Freshness scoring
    print("\n🕐 Freshness Analysis:")
    freshness = client.calculate_memory_freshness(memory)
    print(f"  Memory: {memory.text[:60]}...")
    print(f"  Freshness Score: {freshness['freshness_score']:.2f}")
    print(f"  Age: {freshness['age_days']:.1f} days")
    print(f"  Access frequency: {freshness['access_frequency']:.3f}")
    print(f"  Temporal relevance: {freshness['temporal_relevance']:.2f}")

    # Time decay
    print("\n⏳ Time Decay Functions:")
    decay_types = ["exponential", "linear", "logarithmic", "step"]
    print(f"  Original importance: {memory.importance:.2f}")

    for decay_type in decay_types:
        decayed = client.apply_time_decay(memory, decay_type=decay_type)
        print(f"  {decay_type:12s}: {memory.importance:.2f} → {decayed:.2f}")

    # Adaptive time windows
    print("\n📅 Adaptive Temporal Windows:")
    queries = [
        ("recent updates", "recent"),
        ("historical data", "relevant"),
        ("seasonal patterns", "seasonal"),
    ]

    for query, context_type in queries:
        window = client.get_adaptive_time_window(query, user_id, context_type)
        print(f"\n  Query: '{query}' ({context_type})")
        print(f"    Window size: {window['window_size_days']} days")
        print(f"    Confidence: {window['confidence']:.2f}")

    # Forecasting
    print("\n🔮 Memory Pattern Forecasting:")
    forecasts = client.forecast_memory_patterns(user_id, forecast_days=30)

    if forecasts:
        for forecast in forecasts:
            print(f"\n  {forecast['forecast_type'].upper()} Forecast:")
            print(f"    Period: {forecast['time_period']}")
            print(f"    Confidence: {forecast['confidence']:.2f}")
            print(f"    Basis: {forecast['basis']}")
    else:
        print("  Not enough data for forecasting")

    # Pattern predictions
    print("\n🔁 Recurring Pattern Predictions:")
    predictions = client.predict_future_patterns(user_id, "recurring")

    if predictions:
        for pred in predictions[:3]:
            print(f"\n  {pred['description']}")
            print(f"    Next: {pred['predicted_date'].strftime('%Y-%m-%d')}")
            print(f"    Confidence: {pred['confidence']:.2f}")
    else:
        print("  No recurring patterns detected")


def demo_retrieval_explainability(client: MemoryClient, user_id: str):
    """Demonstrate retrieval explainability."""
    print_section("6. RETRIEVAL EXPLAINABILITY")

    query = "python programming tips"
    print(f"\n🔍 Query: '{query}'")

    # Search
    results = client.search_memories(user_id, query, top_k=5)

    if not results:
        print("\n⚠️  No results found")
        return

    # Explain retrieval
    print("\n📝 Retrieval Explanations:")
    explanations = client.explain_retrieval(query, results)

    for exp in explanations[:3]:
        print(f"\n  Rank #{exp['rank']} - Score: {exp['final_score']:.3f}")

        # Score breakdown
        print("  Components:")
        for component, score in exp["score_breakdown"].items():
            bar = "█" * int(score * 10)
            print(f"    {component:12s}: {bar:10s} {score:.3f}")

        # Explanation
        print(f"  💬 {exp['explanation']}")
        print(f"  🔑 Key factors: {', '.join(exp['contributing_factors'])}")

    # Visualize scores
    print("\n📊 Score Distribution:")
    viz = client.visualize_similarity_scores(query, results, top_k=5)

    for bucket, count in viz["score_distribution"].items():
        if count > 0:
            bar = "█" * count
            print(f"  {bucket}: {bar} ({count})")

    print("\n  Statistics:")
    print(f"    Average: {viz['avg_score']:.3f}")
    print(f"    Max:     {viz['max_score']:.3f}")
    print(f"    Min:     {viz['min_score']:.3f}")


def demo_performance_profiling(client: MemoryClient, user_id: str):
    """Demonstrate performance profiling."""
    print_section("7. PERFORMANCE PROFILING")

    # Run some queries
    print("\n🏃 Running test queries...")
    queries = [
        "python tips",
        "machine learning",
        "database design",
        "api patterns",
    ]

    for query in queries:
        start = time.time()
        results = client.search_memories(user_id, query, top_k=5)
        elapsed_ms = (time.time() - start) * 1000
        print(f"  '{query}': {elapsed_ms:.2f}ms ({len(results)} results)")

    # Performance snapshot
    print("\n📊 Performance Snapshot:")
    snapshot = client.get_performance_snapshot()

    print(f"  Total queries:  {snapshot['total_queries']}")
    print(f"  Avg time:       {snapshot['avg_query_time_ms']:.2f}ms")
    print(f"  Slow queries:   {snapshot['slow_queries']}")
    print(f"  Performance:    {snapshot['performance_score']:.1f}/100")

    # Performance report
    print("\n📈 Performance Report:")
    report = client.get_performance_report()

    if report["stage_averages_ms"]:
        print("\n  Stage Timings:")
        for stage, time_ms in report["stage_averages_ms"].items():
            print(f"    {stage:15s}: {time_ms:.2f}ms")

    if report["common_bottlenecks"]:
        print("\n  ⚠️  Common Bottlenecks:")
        for bottleneck, count in list(report["common_bottlenecks"].items())[:3]:
            print(f"    • {bottleneck}: {count}x")

    if report["recommendations"]:
        print("\n  💡 Optimization Recommendations:")
        for rec in report["recommendations"][:3]:
            print(f"    • {rec}")

    # Slow queries
    slow = client.identify_slow_queries(threshold_ms=100)
    if slow:
        print(f"\n⏱️  Found {len(slow)} slow queries:")
        for q in slow[:3]:
            print(f"\n    Query: {q['query'][:50]}...")
            print(f"    Time: {q['total_time_ms']:.2f}ms")
            if q["bottlenecks"]:
                print(f"    Bottlenecks: {', '.join(q['bottlenecks'][:2])}")


def demo_access_patterns(client: MemoryClient, user_id: str):
    """Demonstrate access pattern analysis."""
    print_section("8. ACCESS PATTERN ANALYSIS")

    heatmap = client.generate_access_heatmap(user_id, time_period_days=30)

    print("\n📊 Access Statistics (Last 30 days):")
    print(f"  Total accesses: {heatmap['total_accesses']}")

    # By hour
    if heatmap["access_by_hour"]:
        print("\n🕐 Access by Hour:")
        for hour in sorted(heatmap["access_by_hour"].keys())[:8]:
            count = heatmap["access_by_hour"][hour]
            bar = "█" * min(count, 20)
            print(f"  {hour:02d}:00 - {bar} ({count})")

        if heatmap["peak_hours"]:
            print(f"\n  Peak hours: {', '.join(map(lambda h: f'{h:02d}:00', heatmap['peak_hours']))}")

    # By day
    if heatmap["access_by_day"]:
        print("\n📅 Access by Day:")
        for day, count in sorted(heatmap["access_by_day"].items(), key=lambda x: x[1], reverse=True):
            bar = "█" * min(count // 2, 20)
            print(f"  {day:9s}: {bar} ({count})")

    # By type
    if heatmap["access_by_type"]:
        print("\n📋 Access by Memory Type:")
        for mem_type, count in sorted(
            heatmap["access_by_type"].items(), key=lambda x: x[1], reverse=True
        ):
            bar = "█" * min(count // 5, 20)
            print(f"  {mem_type:12s}: {bar} ({count})")

    # Hot/cold memories
    if heatmap["hot_memories"]:
        print("\n🔥 Hot Memories (Top 5):")
        for mem_id, count in heatmap["hot_memories"][:5]:
            print(f"  {mem_id[:40]}... : {count} accesses")

    if heatmap["cold_memories"]:
        print(f"\n❄️  Cold Memories: {len(heatmap['cold_memories'])} never accessed")


def main():
    """Run the complete demo."""
    print("\n" + "=" * 70)
    print("  HIPPOCAMPAI: MEMORY HEALTH & OBSERVABILITY DEMO")
    print("=" * 70)

    # Initialize client
    print("\n⚙️  Initializing HippocampAI client...")
    client = MemoryClient.from_preset("local")
    user_id = "demo_user"

    # Create some sample memories if needed
    print("📝 Setting up demo memories...")

    sample_memories = [
        ("Python is great for data science", MemoryType.FACT, ["python", "data-science"]),
        ("User prefers dark mode", MemoryType.PREFERENCE, ["ui", "preferences"]),
        ("Complete ML project by next month", MemoryType.GOAL, ["ml", "project"]),
        ("Morning coffee at 8am daily", MemoryType.HABIT, ["routine", "morning"]),
        ("Attended Python conference", MemoryType.EVENT, ["python", "conference"]),
        ("Machine learning requires good data", MemoryType.FACT, ["ml", "data"]),
        ("Python has great libraries", MemoryType.FACT, ["python", "libraries"]),
    ]

    for text, mem_type, tags in sample_memories:
        try:
            client.add_memory(user_id=user_id, text=text, type=mem_type, tags=tags)
        except Exception:
            pass  # Memory might already exist

    time.sleep(1)  # Let indexing complete

    # Run demos
    try:
        demo_health_monitoring(client, user_id)
        demo_stale_detection(client, user_id)
        demo_duplicate_detection(client, user_id)
        demo_coverage_analysis(client, user_id)
        demo_temporal_features(client, user_id)
        demo_retrieval_explainability(client, user_id)
        demo_performance_profiling(client, user_id)
        demo_access_patterns(client, user_id)

    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback

        traceback.print_exc()

    # Summary
    print_section("DEMO COMPLETE")
    print("\n✅ All features demonstrated successfully!")
    print("\n💡 Next Steps:")
    print("  • Review the documentation: docs/MEMORY_QUALITY_AND_OBSERVABILITY.md")
    print("  • Integrate health monitoring into your workflows")
    print("  • Use explainability for debugging")
    print("  • Monitor performance regularly")
    print("  • Experiment with temporal features")


if __name__ == "__main__":
    main()
