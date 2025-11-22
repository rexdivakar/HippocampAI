# Memory Health Monitoring - Quick Start

## What Is It?

Memory Health Monitoring automatically tracks memory quality, detects duplicates, identifies stale memories, and provides actionable insights with metrics and tracing.

## Installation

No additional installation needed! Health monitoring is built into HippocampAI.

## 5-Minute Quick Start

### 1. Basic Health Check

```python
from hippocampai.unified_client import UnifiedHippocampAI
from hippocampai.monitoring import MemoryHealthMonitor
from hippocampai.embed.embedder import Embedder

# Initialize
client = UnifiedHippocampAI()
embedder = Embedder(model_name="BAAI/bge-small-en-v1.5")
monitor = MemoryHealthMonitor(embedder=embedder)

# Get memories
memories = await client.get_memories(user_id="user123", limit=1000)

# Check health
health = monitor.calculate_health_score(memories)

print(f"Health Score: {health.overall_score:.1f}/100 ({health.status})")
print(f"Recommendations:")
for rec in health.recommendations:
    print(f"  - {rec}")
```

Output:
```
Health Score: 82.5/100 (good)
Recommendations:
  - ✅ Memory store is healthy with minor improvements possible
  - 🔄 3 duplicate clusters detected. Run deduplication.
```

### 2. Detect Duplicates

```python
# Find duplicate clusters
clusters = monitor.detect_duplicate_clusters(memories, cluster_type="soft")

print(f"Found {len(clusters)} duplicate clusters")

for cluster in clusters:
    print(f"\nCluster: {len(cluster.memories)} memories")
    print(f"Suggestion: {cluster.merge_suggestion}")

    # Auto-merge: keep best, delete rest
    for memory in cluster.memories:
        if memory.id != cluster.representative_memory_id:
            await client.delete_memory(memory.id)
```

### 3. Clean Up Stale Memories

```python
# Find stale memories
stale = monitor.detect_stale_memories(memories)

print(f"Found {len(stale)} stale memories")

for memory in stale:
    if memory.should_delete:
        print(f"Deleting: {memory.memory.text[:50]}...")
        await client.delete_memory(memory.memory.id)
    elif memory.should_archive:
        print(f"Archiving: {memory.memory.text[:50]}...")
        # Archive logic here
```

### 4. Analyze Coverage

```python
# Check topic coverage
coverage = monitor.analyze_topic_coverage(memories)

for topic in coverage[:10]:
    print(f"{topic.topic}: {topic.coverage_level} ({topic.memory_count} memories)")
    print(f"  Quality: {topic.quality_score:.1f}/100")

    if topic.gaps:
        print(f"  Gaps: {', '.join(topic.gaps)}")
```

### 5. Enable Metrics

```python
from hippocampai.monitoring import get_metrics_collector, OperationType

collector = get_metrics_collector()

# Trace operations automatically
with collector.trace_operation(OperationType.CREATE):
    await client.add("I love coffee", user_id="user123")

# View metrics
summary = collector.get_metrics_summary()
print(f"Operations: {summary['total_traces']}")
print(f"Success rate: {summary['successful_operations'] / summary['total_traces'] * 100:.1f}%")
```

## Complete Example

```python
from hippocampai.unified_client import UnifiedHippocampAI
from hippocampai.monitoring import MemoryHealthMonitor, get_metrics_collector
from hippocampai.embed.embedder import Embedder

async def monitor_memory_health(user_id: str):
    """Complete health monitoring workflow."""

    # Initialize
    client = UnifiedHippocampAI()
    embedder = Embedder(model_name="BAAI/bge-small-en-v1.5")
    monitor = MemoryHealthMonitor(embedder=embedder)

    # Get memories
    memories = await client.get_memories(user_id=user_id, limit=1000)

    print(f"Analyzing {len(memories)} memories for user {user_id}...")

    # Generate comprehensive report
    report = monitor.generate_quality_report(memories, user_id=user_id)

    # Health overview
    print(f"\n=== HEALTH OVERVIEW ===")
    print(f"Score: {report.health_score.overall_score:.1f}/100 ({report.health_score.status})")
    print(f"Freshness: {report.health_score.freshness_score:.1f}")
    print(f"Diversity: {report.health_score.diversity_score:.1f}")
    print(f"Consistency: {report.health_score.consistency_score:.1f}")
    print(f"Coverage: {report.health_score.coverage_score:.1f}")

    # Duplicates
    print(f"\n=== DUPLICATES ===")
    print(f"Clusters found: {len(report.duplicate_clusters)}")
    for cluster in report.duplicate_clusters[:3]:
        print(f"  - {len(cluster.memories)} memories")
        print(f"    {cluster.merge_suggestion}")

    # Stale memories
    print(f"\n=== STALE MEMORIES ===")
    print(f"Stale found: {len(report.stale_memories)}")
    for stale in report.stale_memories[:3]:
        print(f"  - {stale.memory.text[:50]}...")
        print(f"    {stale.reason}: {stale.recommendation}")

    # Coverage
    print(f"\n=== TOPIC COVERAGE ===")
    for topic in report.topic_coverage[:5]:
        print(f"  - {topic.topic}: {topic.coverage_level} ({topic.memory_count} memories)")

    # Recommendations
    print(f"\n=== RECOMMENDATIONS ===")
    for rec in report.health_score.recommendations:
        print(f"  {rec}")

    # Auto-cleanup
    if report.health_score.status in ["poor", "critical"]:
        print(f"\n=== AUTO-CLEANUP ===")

        # Remove exact duplicates
        for cluster in report.duplicate_clusters:
            if cluster.cluster_type == "exact":
                for memory in cluster.memories:
                    if memory.id != cluster.representative_memory_id:
                        await client.delete_memory(memory.id)
                        print(f"Deleted duplicate: {memory.text[:40]}...")

        # Remove very stale memories
        for stale in report.stale_memories:
            if stale.should_delete:
                await client.delete_memory(stale.memory.id)
                print(f"Deleted stale: {stale.memory.text[:40]}...")

        print("Cleanup complete!")

    return report

# Run
if __name__ == "__main__":
    import asyncio
    report = asyncio.run(monitor_memory_health("user123"))
```

## Key Features

### Health Score Components

| Component | Weight | Description |
|-----------|--------|-------------|
| **Freshness** | 30% | How recent and accessed |
| **Diversity** | 25% | Variety of types and topics |
| **Consistency** | 25% | Quality and completeness |
| **Coverage** | 20% | Breadth and depth |

### Duplicate Types

| Type | Threshold | Use Case |
|------|-----------|----------|
| **Exact** | 98% | Identical text |
| **Soft** | 85% | Semantic similarity |
| **Paraphrase** | 75% | Same meaning |
| **Variant** | <75% | Related but different |

### Stale Reasons

| Reason | Trigger | Action |
|--------|---------|--------|
| **Outdated** | 90+ days old, rarely accessed | Archive or delete |
| **Low Confidence** | Confidence < 0.3 | Review or delete |
| **No Activity** | 60+ days, never accessed | Archive |
| **Temporal Context** | Expired time-sensitive info | Delete |

### Coverage Levels

| Level | Memory Count | Quality |
|-------|--------------|---------|
| **Comprehensive** | 10+ | Excellent |
| **Adequate** | 5-9 | Good |
| **Sparse** | 2-4 | Fair |
| **Minimal** | 1 | Poor |
| **Missing** | 0 | Critical |

## Configuration

```python
# Custom thresholds
monitor = MemoryHealthMonitor(
    embedder=embedder,
    exact_duplicate_threshold=0.98,   # Exact similarity
    soft_duplicate_threshold=0.85,    # Soft similarity
    stale_days_threshold=90,          # Days until stale
    no_access_days_threshold=60,      # Days without access
    min_confidence_threshold=0.3,     # Minimum confidence
)
```

## Metrics & Tracing

```python
from hippocampai.monitoring import (
    get_metrics_collector,
    record_memory_store_stats,
    OperationType
)

collector = get_metrics_collector()

# Record store stats
record_memory_store_stats(
    total_memories=len(memories),
    healthy_memories=report.health_score.healthy_memories,
    stale_memories=report.health_score.stale_memories,
    duplicate_clusters=len(report.duplicate_clusters),
    health_score=report.health_score.overall_score
)

# Export metrics
prometheus_metrics = collector.export_metrics(format="prometheus")
json_metrics = collector.export_metrics(format="json")
```

## Integration Examples

### FastAPI Endpoint

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/health/{user_id}")
async def health_check(user_id: str):
    memories = await client.get_memories(user_id=user_id, limit=1000)
    health = monitor.calculate_health_score(memories)

    return {
        "score": health.overall_score,
        "status": health.status,
        "recommendations": health.recommendations,
    }
```

### Scheduled Task

```python
from celery import Celery
from celery.schedules import crontab

app = Celery()

@app.task
def daily_health_check():
    """Run health checks daily."""
    users = get_all_users()

    for user in users:
        memories = await client.get_memories(user_id=user.id, limit=1000)
        health = monitor.calculate_health_score(memories)

        if health.status == "critical":
            send_alert(f"Critical health for {user.id}")

@app.on_after_configure.connect
def setup_periodic_tasks(sender, **kwargs):
    # Run daily at 2 AM
    sender.add_periodic_task(
        crontab(hour=2, minute=0),
        daily_health_check.s(),
    )
```

## Troubleshooting

### Low Health Score
```python
# Identify issues
if health.stale_memories > 20:
    print("Too many stale memories - run cleanup")

if health.duplicate_clusters > 5:
    print("Too many duplicates - run deduplication")

if health.diversity_score < 50:
    print("Low diversity - add more varied memories")
```

### High Duplicate Count
```python
# Auto-merge duplicates
for cluster in clusters:
    if cluster.confidence > 0.95:
        # Keep best, delete rest
        for mem in cluster.memories:
            if mem.id != cluster.representative_memory_id:
                await client.delete_memory(mem.id)
```

## Next Steps

- **Read full guide:** See `docs/MEMORY_HEALTH_MONITORING_GUIDE.md`
- **Set up monitoring:** Add to your application
- **Configure alerts:** Monitor health trends
- **Automate cleanup:** Schedule regular maintenance

## Summary

Memory Health Monitoring provides:

- ✅ **Health Scoring** - Comprehensive quality assessment
- ✅ **Duplicate Detection** - Automatic clustering and merging
- ✅ **Stale Identification** - Smart cleanup recommendations
- ✅ **Coverage Analysis** - Gap identification
- ✅ **Metrics & Tracing** - Performance monitoring
- ✅ **Production Ready** - 55 tests passing

Monitor your memory store health effortlessly! 🎉
