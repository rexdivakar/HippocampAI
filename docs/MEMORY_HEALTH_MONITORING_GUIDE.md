# Memory Quality & Health Monitoring Guide

## Overview

HippocampAI's Memory Health Monitoring system provides comprehensive quality assurance, duplicate detection, stale memory identification, and coverage analysis with built-in metrics and tracing.

## Key Features

### 1. Memory Health Scoring
- **Overall Health Score** (0-100): Comprehensive quality assessment
- **Component Scores**: Freshness, diversity, consistency, coverage
- **Status Levels**: Excellent, Good, Fair, Poor, Critical
- **Actionable Recommendations**: Automated suggestions for improvement

### 2. Advanced Duplicate Detection
- **Exact Duplicates**: Identical or near-identical text (>98% similarity)
- **Soft Duplicates**: Semantically similar content (>85% similarity)
- **Paraphrase Detection**: Same meaning, different wording
- **Cluster Analysis**: Groups duplicates with merge suggestions
- **Representative Selection**: Identifies best memory in each cluster

### 3. Stale Memory Detection
- **Outdated Memories**: Very old with no recent access
- **Low Confidence**: Memories with decayed confidence
- **No Activity**: Never accessed memories
- **Temporal Context**: Time-sensitive expired information
- **Automatic Recommendations**: Archive, delete, or review suggestions

### 4. Topic Coverage Analysis
- **Coverage Levels**: Comprehensive, Adequate, Sparse, Minimal, Missing
- **Quality Scoring**: Per-topic quality assessment
- **Gap Identification**: Missing or under-represented topics
- **Representative Sampling**: Top memories per topic

### 5. Metrics & Tracing
- **Operation Metrics**: Timing, success rates, error tracking
- **Store Statistics**: Memory counts, health scores, duplicates
- **Quality Metrics**: Freshness, diversity, consistency tracking
- **Distributed Tracing**: Operation traces with spans
- **Export Formats**: Prometheus, JSON

## Quick Start

### Basic Health Check

```python
from hippocampai.monitoring import MemoryHealthMonitor
from hippocampai.embed.embedder import Embedder

# Initialize monitor
embedder = Embedder(model_name="BAAI/bge-small-en-v1.5")
monitor = MemoryHealthMonitor(embedder=embedder)

# Get all memories for a user
memories = await client.get_memories(user_id="user123", limit=1000)

# Calculate health score
health_score = monitor.calculate_health_score(memories, detailed=True)

print(f"Health Score: {health_score.overall_score:.1f}/100")
print(f"Status: {health_score.status}")
print(f"Total Memories: {health_score.total_memories}")
print(f"Stale Memories: {health_score.stale_memories}")
print(f"Duplicate Clusters: {health_score.duplicate_clusters}")

# View recommendations
for rec in health_score.recommendations:
    print(f"  - {rec}")
```

### Generate Quality Report

```python
# Generate comprehensive report
report = monitor.generate_quality_report(
    memories=memories,
    user_id="user123",
    include_topics=True
)

# Health overview
print(f"Overall Health: {report.health_score.overall_score:.1f}/100")
print(f"Freshness: {report.health_score.freshness_score:.1f}")
print(f"Diversity: {report.health_score.diversity_score:.1f}")
print(f"Consistency: {report.health_score.consistency_score:.1f}")
print(f"Coverage: {report.health_score.coverage_score:.1f}")

# Duplicate clusters
print(f"\nDuplicate Clusters: {len(report.duplicate_clusters)}")
for cluster in report.duplicate_clusters[:5]:
    print(f"  - {cluster.cluster_type}: {len(cluster.memories)} memories")
    print(f"    Suggestion: {cluster.merge_suggestion}")

# Stale memories
print(f"\nStale Memories: {len(report.stale_memories)}")
for stale in report.stale_memories[:5]:
    print(f"  - {stale.memory.text[:50]}...")
    print(f"    Reason: {stale.reason}, Score: {stale.staleness_score:.2f}")
    print(f"    Recommendation: {stale.recommendation}")

# Topic coverage
print(f"\nTopic Coverage:")
for topic in report.topic_coverage[:10]:
    print(f"  - {topic.topic}: {topic.coverage_level} ({topic.memory_count} memories)")
    print(f"    Quality: {topic.quality_score:.1f}/100")
```

## Duplicate Detection

### Detect and Cluster Duplicates

```python
# Detect exact duplicates
exact_clusters = monitor.detect_duplicate_clusters(
    memories=memories,
    cluster_type="exact",  # >98% similarity
    min_cluster_size=2
)

# Detect soft duplicates (semantic similarity)
soft_clusters = monitor.detect_duplicate_clusters(
    memories=memories,
    cluster_type="soft",  # >85% similarity
    min_cluster_size=2
)

# Detect all types
all_clusters = monitor.detect_duplicate_clusters(
    memories=memories,
    cluster_type="all",  # Exact, soft, and paraphrases
    min_cluster_size=2
)

# Process clusters
for cluster in exact_clusters:
    print(f"Cluster ID: {cluster.cluster_id}")
    print(f"Type: {cluster.cluster_type}")
    print(f"Memories: {len(cluster.memories)}")
    print(f"Confidence: {cluster.confidence:.2f}")
    print(f"Representative: {cluster.representative_memory_id}")
    print(f"Merge Suggestion: {cluster.merge_suggestion}")

    # Get representative memory
    rep = next(m for m in cluster.memories if m.id == cluster.representative_memory_id)
    print(f"Best memory: {rep.text}")

    # Delete duplicates (keep representative)
    for memory in cluster.memories:
        if memory.id != cluster.representative_memory_id:
            await client.delete_memory(memory.id)
```

### Configuration

```python
# Custom thresholds
monitor = MemoryHealthMonitor(
    embedder=embedder,
    exact_duplicate_threshold=0.98,  # Default: 0.98
    soft_duplicate_threshold=0.85,   # Default: 0.85
)
```

## Stale Memory Detection

### Identify and Handle Stale Memories

```python
# Detect stale memories
stale_memories = monitor.detect_stale_memories(
    memories=memories,
    threshold_days=90  # Custom threshold (default: 90)
)

# Process by reason
by_reason = defaultdict(list)
for stale in stale_memories:
    by_reason[stale.reason].append(stale)

print(f"Outdated: {len(by_reason[StaleReason.OUTDATED])}")
print(f"Low Confidence: {len(by_reason[StaleReason.LOW_CONFIDENCE])}")
print(f"No Activity: {len(by_reason[StaleReason.NO_ACTIVITY])}")

# Auto-cleanup based on recommendations
for stale in stale_memories:
    if stale.should_delete:
        print(f"Deleting: {stale.memory.text[:50]}...")
        await client.delete_memory(stale.memory.id)
    elif stale.should_archive:
        print(f"Archiving: {stale.memory.text[:50]}...")
        # Move to archive or mark as archived
        await client.update_memory(
            memory_id=stale.memory.id,
            metadata={**stale.memory.metadata, "archived": True}
        )
```

### Custom Thresholds

```python
monitor = MemoryHealthMonitor(
    embedder=embedder,
    stale_days_threshold=90,          # Days until stale
    no_access_days_threshold=60,      # Days without access
    min_confidence_threshold=0.3,     # Minimum confidence
)
```

## Topic Coverage Analysis

### Analyze Coverage Across Topics

```python
# Auto-detect topics
coverage = monitor.analyze_topic_coverage(memories)

# Check specific topics
coverage = monitor.analyze_topic_coverage(
    memories,
    topics=["work", "hobbies", "health", "relationships"]
)

# Process coverage results
for topic_coverage in coverage:
    print(f"\nTopic: {topic_coverage.topic}")
    print(f"Coverage: {topic_coverage.coverage_level} ({topic_coverage.memory_count} memories)")
    print(f"Quality: {topic_coverage.quality_score:.1f}/100")

    # Representative memories
    print("Examples:")
    for example in topic_coverage.representative_memories:
        print(f"  - {example}")

    # Identify gaps
    if topic_coverage.gaps:
        print("Gaps:")
        for gap in topic_coverage.gaps:
            print(f"  - {gap}")
```

## Metrics & Tracing

### Enable Metrics Collection

```python
from hippocampai.monitoring import (
    get_metrics_collector,
    record_memory_operation,
    record_memory_store_stats,
    OperationType
)

# Get global collector
collector = get_metrics_collector()

# Record operations
record_memory_operation(
    OperationType.CREATE,
    success=True,
    duration_ms=25.5,
    metadata={"user_id": "user123"}
)

# Record store statistics
record_memory_store_stats(
    total_memories=100,
    healthy_memories=85,
    stale_memories=15,
    duplicate_clusters=3,
    health_score=82.5
)

# Get metrics summary
summary = collector.get_metrics_summary()
print(f"Total operations: {summary['total_traces']}")
print(f"Success rate: {summary['successful_operations'] / summary['total_traces'] * 100:.1f}%")
```

### Distributed Tracing

```python
# Trace operations
with collector.trace_operation(OperationType.CREATE, metadata={"user_id": "user123"}) as trace:
    # Span for embedding
    with collector.span(trace, "embed_text"):
        vector = embedder.encode_single(text)

    # Span for storage
    with collector.span(trace, "store_vector"):
        await qdrant.upsert(collection, id, vector, payload)

# View traces
recent_traces = collector.get_recent_traces(limit=10)
for trace in recent_traces:
    print(f"Operation: {trace.operation}")
    print(f"Duration: {trace.duration_ms:.2f}ms")
    print(f"Success: {trace.success}")
    print(f"Spans: {len(trace.spans)}")
```

### Decorator-Based Tracing

```python
@collector.time_function(OperationType.CREATE)
async def create_memory(text: str, user_id: str):
    # Automatically traced
    memory = Memory(text=text, user_id=user_id, ...)
    await store_memory(memory)
    return memory
```

### Export Metrics

```python
# Prometheus format
prometheus_metrics = collector.export_metrics(format="prometheus")
print(prometheus_metrics)

# JSON format
json_metrics = collector.export_metrics(format="json")
print(json_metrics)
```

## Configuration

### Environment Variables

```bash
# Stale detection
STALE_DAYS_THRESHOLD=90
NO_ACCESS_DAYS_THRESHOLD=60
MIN_CONFIDENCE_THRESHOLD=0.3

# Duplicate detection
EXACT_DUPLICATE_THRESHOLD=0.98
SOFT_DUPLICATE_THRESHOLD=0.85

# Metrics
ENABLE_METRICS=true
ENABLE_TRACING=true
```

### Programmatic Configuration

```python
# Health monitor
monitor = MemoryHealthMonitor(
    embedder=embedder,
    exact_duplicate_threshold=0.98,
    soft_duplicate_threshold=0.85,
    stale_days_threshold=90,
    no_access_days_threshold=60,
    min_confidence_threshold=0.3,
)

# Metrics collector
from hippocampai.monitoring import configure_metrics
configure_metrics(enable_tracing=True)
```

## Integration Examples

### Scheduled Health Checks

```python
import asyncio
from datetime import timedelta

async def daily_health_check():
    """Run daily health check for all users."""
    while True:
        # Get all users
        users = await get_all_users()

        for user in users:
            # Get memories
            memories = await client.get_memories(user_id=user.id, limit=1000)

            # Generate report
            report = monitor.generate_quality_report(memories, user_id=user.id)

            # Send alert if critical
            if report.health_score.status == HealthStatus.CRITICAL:
                await send_alert(f"Critical health for user {user.id}")

            # Auto-cleanup stale memories
            if report.health_score.stale_memories > 20:
                for stale in report.stale_memories:
                    if stale.should_delete:
                        await client.delete_memory(stale.memory.id)

        # Wait 24 hours
        await asyncio.sleep(86400)

# Start background task
asyncio.create_task(daily_health_check())
```

### API Endpoint

```python
from fastapi import FastAPI, Depends

app = FastAPI()

@app.get("/api/health/{user_id}")
async def get_health_report(user_id: str):
    """Get memory health report for user."""
    # Get memories
    memories = await client.get_memories(user_id=user_id, limit=1000)

    # Generate report
    report = monitor.generate_quality_report(memories, user_id=user_id)

    return {
        "health_score": report.health_score.overall_score,
        "status": report.health_score.status,
        "total_memories": report.health_score.total_memories,
        "stale_memories": report.health_score.stale_memories,
        "duplicate_clusters": len(report.duplicate_clusters),
        "recommendations": report.health_score.recommendations,
        "metrics": report.health_score.metrics,
    }

@app.get("/api/metrics")
async def get_metrics():
    """Get system metrics."""
    collector = get_metrics_collector()
    return collector.get_metrics_summary()

@app.get("/api/metrics/prometheus")
async def get_prometheus_metrics():
    """Get Prometheus-formatted metrics."""
    collector = get_metrics_collector()
    return collector.export_metrics(format="prometheus")
```

## Best Practices

### 1. Regular Health Checks
- Run health checks daily or weekly
- Set up alerts for critical status
- Monitor trends over time

### 2. Proactive Duplicate Management
- Run duplicate detection weekly
- Automatically merge obvious duplicates
- Review soft duplicates manually

### 3. Stale Memory Cleanup
- Archive old unused memories
- Delete very stale low-value memories
- Update or verify important stale memories

### 4. Coverage Monitoring
- Track coverage for key topics
- Identify and fill gaps
- Maintain diversity across types

### 5. Metrics & Alerting
- Monitor operation latencies
- Track error rates
- Set up alerting for anomalies

## Troubleshooting

### Low Health Score

**Problem**: Overall health score is low (<60)

**Solutions**:
1. Clean up stale memories
2. Remove duplicate clusters
3. Add more diverse memories
4. Update low-confidence memories

### Too Many Duplicates

**Problem**: Many duplicate clusters detected

**Solutions**:
1. Run deduplication
2. Lower similarity threshold
3. Improve input validation
4. Merge similar memories

### High Staleness

**Problem**: Many stale memories

**Solutions**:
1. Archive old unused memories
2. Update important stale memories
3. Adjust staleness thresholds
4. Implement TTL policies

### Poor Coverage

**Problem**: Missing or sparse topic coverage

**Solutions**:
1. Prompt users for missing topics
2. Extract more from conversations
3. Add placeholder memories
4. Import from external sources

## API Reference

### MemoryHealthMonitor

```python
class MemoryHealthMonitor:
    def __init__(
        self,
        embedder: Embedder,
        exact_duplicate_threshold: float = 0.98,
        soft_duplicate_threshold: float = 0.85,
        stale_days_threshold: int = 90,
        no_access_days_threshold: int = 60,
        min_confidence_threshold: float = 0.3,
    )

    def calculate_health_score(
        self,
        memories: list[Memory],
        detailed: bool = True
    ) -> MemoryHealthScore

    def detect_duplicate_clusters(
        self,
        memories: list[Memory],
        cluster_type: str = "soft",
        min_cluster_size: int = 2,
    ) -> list[DuplicateCluster]

    def detect_stale_memories(
        self,
        memories: list[Memory],
        threshold_days: Optional[int] = None
    ) -> list[StaleMemory]

    def analyze_topic_coverage(
        self,
        memories: list[Memory],
        topics: Optional[list[str]] = None
    ) -> list[TopicCoverage]

    def generate_quality_report(
        self,
        memories: list[Memory],
        user_id: str,
        include_topics: bool = True
    ) -> MemoryQualityReport
```

### MetricsCollector

```python
class MetricsCollector:
    def __init__(self, enable_tracing: bool = True)

    def record_metric(
        self,
        name: str,
        value: float,
        metric_type: MetricType,
        tags: Optional[dict[str, str]] = None,
    )

    def trace_operation(
        self,
        operation: OperationType,
        trace_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    )

    def get_metrics_summary(self) -> dict[str, Any]

    def export_metrics(self, format: str = "prometheus") -> str
```

## Summary

HippocampAI's Memory Health Monitoring provides:

- ✅ **Comprehensive Health Scoring** - Overall quality assessment
- ✅ **Advanced Duplicate Detection** - Exact, soft, and semantic clustering
- ✅ **Stale Memory Identification** - Automatic detection and recommendations
- ✅ **Topic Coverage Analysis** - Gap identification and quality scoring
- ✅ **Metrics & Tracing** - Performance monitoring and debugging
- ✅ **Production Ready** - Tested with 55 test cases
- ✅ **Easy Integration** - Simple API, sensible defaults

Maintain a healthy memory store with automated monitoring and actionable insights!
