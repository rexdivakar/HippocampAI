# Memory Quality, Health Monitoring & Observability Guide

This guide covers HippocampAI's advanced features for monitoring memory health, quality scoring, temporal enhancements, and debugging/observability capabilities.

## Table of Contents

1. [Memory Health & Quality Monitoring](#memory-health--quality-monitoring)
2. [Enhanced Temporal Features](#enhanced-temporal-features)
3. [Debugging & Observability](#debugging--observability)
4. [Complete Examples](#complete-examples)

---

## Memory Health & Quality Monitoring

### Overview

HippocampAI provides comprehensive health monitoring to assess the quality and state of your memory store.

### Health Score Components

The overall health score (0-100) is calculated from these components:

- **Quality Score** (25% weight): Based on importance, confidence, and metadata completeness
- **Diversity Score** (15% weight): Measures variety across memory types and content
- **Freshness Score** (15% weight): Evaluates how up-to-date memories are
- **Coverage Score** (15% weight): Analyzes topic and type coverage
- **Duplication Score** (15% weight): Penalizes duplicate clusters
- **Staleness Score** (15% weight): Inverse of stale memory ratio

### Get Overall Health Score

```python
from hippocampai import MemoryClient

client = MemoryClient.from_preset("local")

# Get comprehensive health score
health = client.get_memory_health_score(
    user_id="user123",
    include_stale_detection=True,
    include_duplicate_detection=True
)

print(f"Overall Health: {health['overall_score']:.1f}/100")
print(f"Quality: {health['quality_score']:.1f}")
print(f"Freshness: {health['freshness_score']:.1f}")
print(f"Diversity: {health['diversity_score']:.1f}")
print(f"Coverage: {health['coverage_score']:.1f}")

print(f"\nStatistics:")
print(f"Total memories: {health['total_memories']}")
print(f"Stale memories: {health['stale_count']}")
print(f"Duplicate clusters: {health['duplicate_clusters']}")
print(f"Avg importance: {health['avg_importance']:.2f}")
print(f"Avg age: {health['avg_age_days']:.1f} days")

print(f"\nRecommendations:")
for rec in health['recommendations']:
    print(f"  • {rec}")
```

### Detect Stale Memories

Identify potentially outdated or obsolete memories:

```python
# Detect stale memories
stale = client.detect_stale_memories("user123")

print(f"Found {len(stale)} stale memories:\n")

for mem in stale[:5]:
    print(f"ID: {mem['memory_id']}")
    print(f"Age: {mem['age_days']} days")
    print(f"Text: {mem['text']}")
    print(f"Reason: {mem['staleness_reason']}")
    print(f"Confidence: {mem['confidence']:.2f}")
    print()
```

**Staleness Detection Factors:**
- Age (>180 days by default)
- Never accessed
- Low importance + aging
- Outdated temporal information (old year mentions, "currently", "now")

### Detect Duplicate Clusters

Find clusters of similar/duplicate memories:

```python
# Detect duplicate clusters
clusters = client.detect_duplicate_clusters(
    user_id="user123",
    min_cluster_size=2
)

print(f"Found {len(clusters)} duplicate clusters:\n")

for cluster in clusters:
    print(f"Cluster ID: {cluster['cluster_id']}")
    print(f"Size: {cluster['cluster_size']} memories")
    print(f"Avg similarity: {cluster['avg_similarity']:.2f}")
    print(f"Representative: {cluster['representative_text'][:100]}...")
    print(f"Memory IDs: {cluster['memory_ids']}")
    print()
```

### Detect Near-Duplicates with Merge Suggestions

Find near-duplicate pairs and get merge suggestions:

```python
# Detect near-duplicates
warnings = client.detect_near_duplicates(
    user_id="user123",
    suggest_merge=True
)

print(f"Found {len(warnings)} near-duplicate pairs:\n")

for warning in warnings[:3]:
    print(f"Similarity: {warning['similarity_score']:.2f}")
    print(f"Memory 1: {warning['text_1'][:80]}...")
    print(f"Memory 2: {warning['text_2'][:80]}...")
    print(f"Suggestion: {warning['merge_suggestion']}")
    print(f"Confidence: {warning['confidence']:.2f}")
    print()
```

### Analyze Memory Coverage

Understand which topics are well/poorly covered:

```python
# Analyze coverage
coverage = client.analyze_memory_coverage("user123")

print("Topic Distribution:")
for topic, count in coverage['topic_distribution'].items():
    print(f"  {topic}: {count}")

print(f"\nWell-covered topics: {coverage['well_covered_topics']}")
print(f"Poorly-covered topics: {coverage['poorly_covered_topics']}")

print("\nType Distribution:")
for mem_type, count in coverage['type_distribution'].items():
    print(f"  {mem_type}: {count}")

print(f"\nCoverage gaps: {coverage['coverage_gaps']}")

print("\nRecommendations:")
for rec in coverage['recommendations']:
    print(f"  • {rec}")
```

---

## Enhanced Temporal Features

### Memory Freshness Scoring

Calculate comprehensive freshness scores:

```python
# Get a memory
memories = client.get_memories("user123", limit=1)
memory = memories[0]

# Calculate freshness
freshness = client.calculate_memory_freshness(memory)

print(f"Freshness Score: {freshness['freshness_score']:.2f}")
print(f"Age: {freshness['age_days']} days")
print(f"Last accessed: {freshness['last_accessed_days']} days ago")
print(f"Access frequency: {freshness['access_frequency']:.3f}")
print(f"Temporal relevance: {freshness['temporal_relevance']:.2f}")

print("\nFactor breakdown:")
for factor, value in freshness['factors'].items():
    print(f"  {factor}: {value:.2f}")
```

### Time Decay Functions

Apply customizable decay to memory importance:

```python
# Apply exponential decay (default)
decayed = client.apply_time_decay(
    memory,
    decay_type="exponential"
)
print(f"Exponential decay: {memory.importance} → {decayed:.2f}")

# Apply linear decay
decayed = client.apply_time_decay(memory, decay_type="linear")
print(f"Linear decay: {memory.importance} → {decayed:.2f}")

# Apply logarithmic decay (slower)
decayed = client.apply_time_decay(memory, decay_type="logarithmic")
print(f"Logarithmic decay: {memory.importance} → {decayed:.2f}")

# Apply step decay (sharp drops at intervals)
decayed = client.apply_time_decay(memory, decay_type="step")
print(f"Step decay: {memory.importance} → {decayed:.2f}")
```

**Decay Types:**
- **Exponential**: Standard exponential decay with configurable half-life
- **Linear**: Constant decay rate per day
- **Logarithmic**: Slower decay for long-term memories
- **Step**: Sharp drops at specific age thresholds

### Adaptive Temporal Context Windows

Auto-adjust time windows based on query and data:

```python
# Get adaptive window for recent context
window = client.get_adaptive_time_window(
    query="recent updates",
    user_id="user123",
    context_type="recent"
)

print(f"Window size: {window['window_size_days']} days")
print(f"From: {window['start_date']}")
print(f"To: {window['end_date']}")
print(f"Type: {window['context_type']}")
print(f"Confidence: {window['confidence']:.2f}")

# Try different context types
window = client.get_adaptive_time_window(
    query="seasonal patterns",
    user_id="user123",
    context_type="seasonal"
)

window = client.get_adaptive_time_window(
    query="all information about python",
    user_id="user123",
    context_type="relevant"
)
```

**Context Types:**
- **recent**: Last N days based on activity level
- **seasonal**: Same time period in previous years
- **relevant**: Inferred from query keywords

### Memory Pattern Forecasting

Predict future patterns based on historical data:

```python
# Forecast memory patterns
forecasts = client.forecast_memory_patterns(
    user_id="user123",
    forecast_days=30
)

for forecast in forecasts:
    print(f"\nForecast Type: {forecast['forecast_type']}")
    print(f"Period: {forecast['time_period']}")
    print(f"Confidence: {forecast['confidence']:.2f}")
    print(f"Basis: {forecast['basis']}")

    if forecast['predictions']:
        print("Predictions:")
        for pred in forecast['predictions'][:3]:
            print(f"  {pred}")
```

**Forecast Types:**
- **Usage**: Predict access patterns and frequency
- **Topic**: Identify trending topics
- **Importance**: Forecast importance score trends

### Predict Recurring Patterns

Detect and predict when patterns will recur:

```python
# Predict recurring patterns
predictions = client.predict_future_patterns(
    user_id="user123",
    pattern_type="recurring"
)

print("Recurring Patterns:")
for pred in predictions:
    print(f"\nPattern: {pred['description']}")
    print(f"Next occurrence: {pred['predicted_date']}")
    print(f"Confidence: {pred['confidence']:.2f}")
    print(f"Historical basis: {len(pred['historical_basis'])} occurrences")

# Predict seasonal patterns
predictions = client.predict_future_patterns(
    user_id="user123",
    pattern_type="seasonal"
)

print("\nSeasonal Patterns:")
for pred in predictions:
    print(f"Pattern: {pred['description']}")
    print(f"Next occurrence: {pred['predicted_date']}")
```

---

## Debugging & Observability

### Retrieval Explainability

Understand why memories were retrieved:

```python
# Search and explain
results = client.search_memories("user123", "python tips", top_k=5)

explanations = client.explain_retrieval("python tips", results)

for exp in explanations:
    print(f"\nRank #{exp['rank']}")
    print(f"Memory: {exp['memory_id']}")
    print(f"Final Score: {exp['final_score']:.3f}")

    print("Score breakdown:")
    for component, score in exp['score_breakdown'].items():
        print(f"  {component}: {score:.3f}")

    print(f"Explanation: {exp['explanation']}")

    print(f"Contributing factors: {', '.join(exp['contributing_factors'])}")
```

### Similarity Score Visualization

Get data for visualizing similarity distributions:

```python
# Get visualization data
viz = client.visualize_similarity_scores(
    query="python programming",
    results=results,
    top_k=10
)

print(f"Query: {viz['query']}")
print(f"Avg score: {viz['avg_score']:.3f}")
print(f"Max score: {viz['max_score']:.3f}")
print(f"Min score: {viz['min_score']:.3f}")

print("\nScore distribution:")
for bucket, count in viz['score_distribution'].items():
    print(f"  {bucket}: {'█' * count} ({count})")

print("\nTop results:")
for result in viz['results'][:5]:
    print(f"  Rank {result['rank']}: {result['score']:.3f} - {result['text_preview'][:60]}...")
```

### Memory Access Heatmaps

Visualize access patterns over time:

```python
# Generate access heatmap
heatmap = client.generate_access_heatmap(
    user_id="user123",
    time_period_days=30
)

print(f"Total accesses: {heatmap['total_accesses']}")
print(f"Peak hours: {heatmap['peak_hours']}")

print("\nAccess by hour:")
for hour in sorted(heatmap['access_by_hour'].keys()):
    count = heatmap['access_by_hour'][hour]
    print(f"  {hour:02d}:00 - {'█' * count} ({count})")

print("\nAccess by day:")
for day, count in heatmap['access_by_day'].items():
    print(f"  {day}: {count}")

print("\nAccess by type:")
for mem_type, count in heatmap['access_by_type'].items():
    print(f"  {mem_type}: {count}")

print(f"\nHot memories (top 10):")
for mem_id, count in heatmap['hot_memories'][:10]:
    print(f"  {mem_id}: {count} accesses")

print(f"\nCold memories (never accessed): {len(heatmap['cold_memories'])}")
```

### Performance Profiling

Monitor and optimize query performance:

```python
# Get current performance snapshot
snapshot = client.get_performance_snapshot()

print(f"Total queries: {snapshot['total_queries']}")
print(f"Avg query time: {snapshot['avg_query_time_ms']:.2f}ms")
print(f"Slow queries: {snapshot['slow_queries']}")
print(f"Performance score: {snapshot['performance_score']:.1f}/100")
print(f"Avg results returned: {snapshot['avg_results_returned']:.1f}")

# Get detailed performance report
report = client.get_performance_report()

print("\nStage Averages:")
for stage, time_ms in report['stage_averages_ms'].items():
    print(f"  {stage}: {time_ms:.2f}ms")

print("\nCommon Bottlenecks:")
for bottleneck, count in list(report['common_bottlenecks'].items())[:5]:
    print(f"  {bottleneck}: {count} occurrences")

print("\nRecommendations:")
for rec in report['recommendations']:
    print(f"  • {rec}")

# Identify slow queries
slow_queries = client.identify_slow_queries(threshold_ms=500)

print(f"\nFound {len(slow_queries)} slow queries:")
for query in slow_queries[:3]:
    print(f"\nQuery: {query['query']}")
    print(f"Time: {query['total_time_ms']:.2f}ms")
    print(f"Bottlenecks: {query['bottlenecks']}")
    print(f"Recommendations: {query['recommendations']}")
```

---

## Complete Examples

### Example 1: Memory Health Audit

```python
from hippocampai import MemoryClient

client = MemoryClient.from_preset("local")
user_id = "user123"

print("=" * 60)
print("MEMORY HEALTH AUDIT")
print("=" * 60)

# 1. Overall health
health = client.get_memory_health_score(user_id)
print(f"\n📊 Overall Health Score: {health['overall_score']:.1f}/100")

if health['overall_score'] > 80:
    print("✅ Excellent health!")
elif health['overall_score'] > 60:
    print("⚠️  Good, but could be improved")
else:
    print("❌ Needs attention")

# 2. Component scores
print(f"\nComponent Scores:")
print(f"  Quality:     {health['quality_score']:.1f}/100")
print(f"  Freshness:   {health['freshness_score']:.1f}/100")
print(f"  Diversity:   {health['diversity_score']:.1f}/100")
print(f"  Coverage:    {health['coverage_score']:.1f}/100")
print(f"  Duplication: {health['duplication_score']:.1f}/100")
print(f"  Staleness:   {health['staleness_score']:.1f}/100")

# 3. Stale memories
stale = client.detect_stale_memories(user_id)
if stale:
    print(f"\n🔍 Found {len(stale)} stale memories")
    print(f"Top staleness reasons:")
    reasons = {}
    for mem in stale:
        reason = mem['staleness_reason'].split(';')[0]
        reasons[reason] = reasons.get(reason, 0) + 1
    for reason, count in sorted(reasons.items(), key=lambda x: x[1], reverse=True)[:3]:
        print(f"  • {reason}: {count}")

# 4. Duplicates
clusters = client.detect_duplicate_clusters(user_id)
if clusters:
    print(f"\n🔄 Found {len(clusters)} duplicate clusters")
    total_dupes = sum(c['cluster_size'] - 1 for c in clusters)
    print(f"   Could remove {total_dupes} duplicates")

# 5. Coverage
coverage = client.analyze_memory_coverage(user_id)
print(f"\n📈 Coverage Analysis:")
print(f"  Topic diversity: {len(coverage['topic_distribution'])} topics")
print(f"  Type coverage: {len(coverage['type_distribution'])}/6 types")

# 6. Recommendations
print(f"\n💡 Recommendations:")
for rec in health['recommendations']:
    print(f"  • {rec}")
```

### Example 2: Query Performance Monitoring

```python
from hippocampai import MemoryClient
import time

client = MemoryClient.from_preset("local")
user_id = "user123"

print("=" * 60)
print("QUERY PERFORMANCE MONITORING")
print("=" * 60)

# Run several queries
queries = [
    "python best practices",
    "machine learning algorithms",
    "database optimization",
    "api design patterns",
]

for query in queries:
    start = time.time()
    results = client.search_memories(user_id, query, top_k=10)
    elapsed_ms = (time.time() - start) * 1000
    print(f"\n'{query}': {elapsed_ms:.2f}ms ({len(results)} results)")

# Get performance snapshot
print("\n" + "=" * 60)
snapshot = client.get_performance_snapshot()

print(f"\n📊 Performance Summary:")
print(f"  Total queries: {snapshot['total_queries']}")
print(f"  Avg time: {snapshot['avg_query_time_ms']:.2f}ms")
print(f"  Slow queries: {snapshot['slow_queries']}")
print(f"  Performance score: {snapshot['performance_score']:.1f}/100")

# Identify bottlenecks
report = client.get_performance_report()

print(f"\n⚠️  Common Bottlenecks:")
for bottleneck, count in list(report['common_bottlenecks'].items())[:3]:
    print(f"  • {bottleneck}: {count}x")

print(f"\n💡 Optimization Recommendations:")
for rec in report['recommendations']:
    print(f"  • {rec}")
```

### Example 3: Retrieval Explainability

```python
from hippocampai import MemoryClient

client = MemoryClient.from_preset("local")
user_id = "user123"

query = "python error handling"
print(f"Query: '{query}'")
print("=" * 60)

# Search with explanation
results = client.search_memories(user_id, query, top_k=5)
explanations = client.explain_retrieval(query, results)

for exp in explanations:
    mem = results[exp['rank'] - 1].memory

    print(f"\n🔍 Result #{exp['rank']}")
    print(f"Memory: {mem.text[:100]}...")
    print(f"Final Score: {exp['final_score']:.3f}")

    # Visual score breakdown
    print("\nScore Components:")
    breakdown = exp['score_breakdown']
    for component, score in breakdown.items():
        bar = '█' * int(score * 20)
        print(f"  {component:12s}: {bar:20s} {score:.3f}")

    # Explanation
    print(f"\n💬 Why retrieved: {exp['explanation']}")
    print(f"   Key factors: {', '.join(exp['contributing_factors'])}")

# Visualize score distribution
viz = client.visualize_similarity_scores(query, results)

print("\n" + "=" * 60)
print("SCORE DISTRIBUTION")
print("=" * 60)
for bucket, count in viz['score_distribution'].items():
    bar = '█' * count
    print(f"{bucket}: {bar} ({count})")

print(f"\nStatistics:")
print(f"  Avg: {viz['avg_score']:.3f}")
print(f"  Max: {viz['max_score']:.3f}")
print(f"  Min: {viz['min_score']:.3f}")
```

---

## Best Practices

### Health Monitoring

1. **Regular Audits**: Run health checks weekly or monthly
2. **Act on Recommendations**: Follow the automated recommendations
3. **Clean Up**: Remove stale memories and merge duplicates periodically
4. **Monitor Trends**: Track health scores over time

### Temporal Features

1. **Choose Appropriate Decay**: Match decay function to use case
   - Exponential: General purpose
   - Linear: Predictable decay
   - Logarithmic: Long-term memories
   - Step: Clear time boundaries

2. **Leverage Forecasting**: Use predictions for proactive memory management
3. **Adaptive Windows**: Let the system adjust time ranges automatically

### Performance Optimization

1. **Monitor Regularly**: Check performance snapshots after changes
2. **Profile Slow Queries**: Identify and optimize bottlenecks
3. **Use Explainability**: Understand retrieval behavior
4. **Track Access Patterns**: Optimize for common usage patterns

---

## Configuration Options

### Health Monitor

```python
from hippocampai.pipeline.memory_health import MemoryHealthMonitor

monitor = MemoryHealthMonitor(
    qdrant_store=client.qdrant,
    embedder=client.embedder,
    stale_threshold_days=180,      # Age threshold for staleness
    near_duplicate_threshold=0.85,  # Similarity for near-duplicates
    exact_duplicate_threshold=0.95, # Similarity for exact duplicates
)
```

### Temporal Analyzer

```python
from hippocampai.pipeline.temporal_enhancement import EnhancedTemporalAnalyzer

temporal = EnhancedTemporalAnalyzer(
    default_half_life_days=90,    # Half-life for exponential decay
    freshness_window_days=30,     # Days considered "fresh"
)
```

### Observability Monitor

```python
from hippocampai.pipeline.memory_observability import MemoryObservabilityMonitor

observability = MemoryObservabilityMonitor(
    enable_profiling=True,              # Enable performance profiling
    slow_query_threshold_ms=1000.0,     # Threshold for slow queries
    track_access_patterns=True,         # Track access patterns
)
```

---

## API Reference

See the full API documentation for detailed parameter descriptions and return types:

- [Memory Health API](API_REFERENCE.md#memory-health)
- [Temporal Features API](API_REFERENCE.md#temporal-features)
- [Observability API](API_REFERENCE.md#observability)

---

## Next Steps

- Integrate health monitoring into your workflows
- Set up automated performance tracking
- Use explainability for debugging
- Experiment with different decay functions
- Leverage forecasting for proactive management

For more information, see:
- [User Guide](USER_GUIDE.md)
- [Advanced Features](FEATURES.md)
- [API Reference](API_REFERENCE.md)
