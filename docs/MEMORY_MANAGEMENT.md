# Memory Management & Health Monitoring Guide

**Complete guide to memory quality, health monitoring, tracking, and conflict resolution**

Last Updated: 2025-11-21

---

## Table of Contents

1. [Overview](#overview)
2. [Memory Health Monitoring](#memory-health-monitoring)
3. [Duplicate Detection & Resolution](#duplicate-detection--resolution)
4. [Stale Memory Management](#stale-memory-management)
5. [Topic Coverage Analysis](#topic-coverage-analysis)
6. [Memory Tracking & Events](#memory-tracking--events)
7. [Conflict Detection & Resolution](#conflict-detection--resolution)
8. [Metrics & Observability](#metrics--observability)
9. [Best Practices](#best-practices)
10. [API Reference](#api-reference)

---

## Overview

HippocampAI's Memory Management system provides comprehensive tools for maintaining memory quality, detecting issues, and optimizing memory stores. This includes:

- **Health Scoring**: Comprehensive quality assessment with component scores
- **Duplicate Detection**: Advanced clustering with semantic similarity
- **Staleness Tracking**: Temporal monitoring and freshness scoring
- **Coverage Analysis**: Topic representation and gap identification
- **Event Tracking**: Complete lifecycle and access pattern monitoring
- **Conflict Resolution**: Automated detection and resolution strategies
- **Metrics & Tracing**: Performance monitoring and observability

---

## Memory Health Monitoring

### Health Score Components

The overall health score (0-100) combines four key components:

| Component | Weight | Description |
|-----------|--------|-------------|
| **Freshness** | 30% | How recent and accessed |
| **Diversity** | 25% | Variety of types and topics |
| **Consistency** | 25% | Quality and completeness |
| **Coverage** | 20% | Breadth and depth |

### Health Status Levels

| Status | Score Range | Description |
|--------|-------------|-------------|
| **Excellent** | 90-100 | Optimal memory store health |
| **Good** | 75-89 | Minor improvements possible |
| **Fair** | 60-74 | Some issues need attention |
| **Poor** | 40-59 | Significant problems exist |
| **Critical** | 0-39 | Urgent action required |

### Quick Start

```python
from hippocampai.monitoring import MemoryHealthMonitor
from hippocampai.embed.embedder import Embedder
from hippocampai import MemoryClient

# Initialize
client = MemoryClient()
embedder = Embedder(model_name="BAAI/bge-small-en-v1.5")
monitor = MemoryHealthMonitor(embedder=embedder)

# Get memories
memories = await client.get_memories(user_id="user123", limit=1000)

# Calculate health score
health = monitor.calculate_health_score(memories)

print(f"Health Score: {health.overall_score:.1f}/100 ({health.status})")
print(f"Freshness: {health.freshness_score:.1f}")
print(f"Diversity: {health.diversity_score:.1f}")
print(f"Consistency: {health.consistency_score:.1f}")
print(f"Coverage: {health.coverage_score:.1f}")

# View recommendations
for rec in health.recommendations:
    print(f"  - {rec}")
```

### Component Score Calculation

#### Freshness Score (30%)

Measures how recently memories were created and accessed:

```python
# Calculation
age_score = exp(-age_days / 30)  # Exponential decay, 30-day half-life
access_score = exp(-days_since_access / 60)  # 60-day half-life
freshness = (age_score * 0.7 + access_score * 0.3) * 100
```

**Factors:**
- Memory age (70%): Days since creation
- Access recency (30%): Days since last access

#### Diversity Score (25%)

Evaluates variety in memory types, topics, and tags:

```python
# Calculation
type_diversity = unique_types / total_possible_types
topic_diversity = unique_topics / max(10, total_memories * 0.1)
tag_diversity = unique_tags / max(20, total_memories * 0.2)
diversity = (type_diversity * 0.4 + topic_diversity * 0.3 + tag_diversity * 0.3) * 100
```

**Factors:**
- Type variety (40%): Different memory types used
- Topic variety (30%): Breadth of topics covered
- Tag variety (30%): Richness of tagging

#### Consistency Score (25%)

Assesses memory completeness and quality:

```python
# Calculation
completeness = avg(text_length_score, tag_count_score, metadata_richness)
confidence = avg(confidence_scores)
quality = avg(quality_scores)
consistency = (completeness * 0.4 + confidence * 0.3 + quality * 0.3) * 100
```

**Factors:**
- Completeness (40%): Text length, tags, metadata
- Confidence (30%): Average confidence scores
- Quality (30%): Overall memory quality

#### Coverage Score (20%)

Measures topic representation and depth:

```python
# Calculation
topic_count_score = min(100, unique_topics / target_topics * 100)
avg_per_topic = total_memories / unique_topics
depth_score = min(100, avg_per_topic / 5 * 100)  # Target: 5+ per topic
coverage = (topic_count_score * 0.6 + depth_score * 0.4)
```

**Factors:**
- Topic breadth (60%): Number of unique topics
- Topic depth (40%): Memories per topic

### Generate Comprehensive Report

```python
# Generate full quality report
report = monitor.generate_quality_report(
    memories=memories,
    user_id="user123",
    include_topics=True
)

# Health overview
print(f"\n=== HEALTH OVERVIEW ===")
print(f"Score: {report.health_score.overall_score:.1f}/100 ({report.health_score.status})")
print(f"Total Memories: {report.health_score.total_memories}")
print(f"Healthy: {report.health_score.healthy_memories}")
print(f"Stale: {report.health_score.stale_memories}")
print(f"Duplicates: {len(report.duplicate_clusters)}")

# Duplicates
print(f"\n=== DUPLICATES ===")
for cluster in report.duplicate_clusters[:5]:
    print(f"  - {cluster.cluster_type}: {len(cluster.memories)} memories")
    print(f"    Similarity: {cluster.avg_similarity:.2f}")
    print(f"    Suggestion: {cluster.merge_suggestion}")

# Stale memories
print(f"\n=== STALE MEMORIES ===")
for stale in report.stale_memories[:5]:
    print(f"  - {stale.memory.text[:50]}...")
    print(f"    Age: {stale.age_days} days")
    print(f"    Recommendation: {stale.recommendation}")

# Topic coverage
print(f"\n=== TOPIC COVERAGE ===")
for topic in report.topic_coverage[:10]:
    print(f"  - {topic.topic}: {topic.coverage_level}")
    print(f"    Memories: {topic.memory_count}, Quality: {topic.quality_score:.1f}")
```

---

## Duplicate Detection & Resolution

### Duplicate Types

| Type | Threshold | Description | Action |
|------|-----------|-------------|--------|
| **Exact** | >98% | Identical or near-identical text | Auto-merge |
| **Soft** | 85-98% | Semantically very similar | Review merge |
| **Paraphrase** | 75-85% | Same meaning, different wording | Manual review |
| **Variant** | <75% | Related but distinct | Keep separate |

### Detect Duplicate Clusters

```python
# Find duplicates
clusters = monitor.detect_duplicate_clusters(
    memories,
    cluster_type="soft",  # "exact", "soft", "paraphrase"
    min_cluster_size=2
)

print(f"Found {len(clusters)} duplicate clusters")

for cluster in clusters:
    print(f"\nCluster: {len(cluster.memories)} memories")
    print(f"Type: {cluster.cluster_type}")
    print(f"Avg Similarity: {cluster.avg_similarity:.2f}")
    print(f"Representative: {cluster.representative_memory_id}")
    print(f"Suggestion: {cluster.merge_suggestion}")

    # Show cluster members
    for memory in cluster.memories:
        print(f"  - [{memory.id}] {memory.text[:50]}...")
```

### Auto-Merge Duplicates

```python
# Merge exact duplicates automatically
for cluster in clusters:
    if cluster.cluster_type == "exact" and cluster.confidence > 0.95:
        # Keep the representative (usually most recent or highest quality)
        representative = cluster.representative_memory_id

        # Delete duplicates
        for memory in cluster.memories:
            if memory.id != representative:
                await client.delete_memory(memory.id)
                print(f"Deleted duplicate: {memory.id}")

        print(f"Kept representative: {representative}")
```

### Manual Review Workflow

```python
# Review soft duplicates
for cluster in clusters:
    if cluster.cluster_type in ["soft", "paraphrase"]:
        print(f"\n=== Review Required ===")
        print(f"Cluster: {len(cluster.memories)} memories")

        # Show all members
        for i, memory in enumerate(cluster.memories, 1):
            print(f"\n{i}. [{memory.id}] Created: {memory.created_at}")
            print(f"   {memory.text}")
            print(f"   Importance: {memory.importance}, Tags: {memory.tags}")

        # User decision
        action = input("\nAction? (merge/keep-all/skip): ")

        if action == "merge":
            # Implement merge logic
            pass
        elif action == "keep-all":
            continue
```

### Advanced Clustering Options

```python
# Custom configuration
monitor = MemoryHealthMonitor(
    embedder=embedder,
    exact_duplicate_threshold=0.98,
    soft_duplicate_threshold=0.85,
    paraphrase_threshold=0.75,
    use_semantic_clustering=True
)

# Detect with custom parameters
clusters = monitor.detect_duplicate_clusters(
    memories,
    cluster_type="soft",
    min_cluster_size=2,
    max_cluster_size=10,
    similarity_threshold=0.90  # Override default
)
```

---

## Stale Memory Management

### Stale Reasons & Triggers

| Reason | Trigger | Recommendation |
|--------|---------|----------------|
| **Outdated** | 90+ days old, rarely accessed | Archive or delete |
| **Low Confidence** | Confidence < 0.3 | Review or delete |
| **No Activity** | 60+ days, never accessed | Archive |
| **Temporal Context** | Expired time-sensitive info | Delete |

### Detect Stale Memories

```python
# Find stale memories
stale_memories = monitor.detect_stale_memories(
    memories,
    stale_days_threshold=90,
    no_access_days_threshold=60,
    min_confidence_threshold=0.3
)

print(f"Found {len(stale_memories)} stale memories")

for stale in stale_memories:
    print(f"\n{stale.memory.text[:50]}...")
    print(f"  Age: {stale.age_days} days")
    print(f"  Last accessed: {stale.days_since_access} days ago")
    print(f"  Reason: {stale.reason}")
    print(f"  Staleness score: {stale.staleness_score:.2f}")
    print(f"  Recommendation: {stale.recommendation}")
    print(f"  Actions: Delete={stale.should_delete}, Archive={stale.should_archive}")
```

### Auto-Cleanup Stale Memories

```python
# Automatic cleanup based on rules
for stale in stale_memories:
    if stale.should_delete:
        # Delete very stale or low-confidence memories
        await client.delete_memory(stale.memory.id)
        print(f"Deleted: {stale.memory.text[:40]}...")

    elif stale.should_archive:
        # Archive moderately stale memories
        await client.update_memory(
            stale.memory.id,
            metadata={"archived": True, "archived_at": datetime.now().isoformat()}
        )
        print(f"Archived: {stale.memory.text[:40]}...")

    else:
        # Mark for manual review
        print(f"Review: {stale.memory.text[:40]}...")
```

### Freshness Scoring

```python
# Calculate freshness for individual memory
def calculate_freshness(memory):
    age_days = (datetime.now() - memory.created_at).days
    days_since_access = (datetime.now() - memory.last_accessed).days

    # Exponential decay with 30-day half-life
    age_score = math.exp(-age_days / 30) * 100
    access_score = math.exp(-days_since_access / 60) * 100

    # Weighted combination
    freshness = age_score * 0.7 + access_score * 0.3

    return freshness

# Update memories with freshness scores
for memory in memories:
    freshness = calculate_freshness(memory)
    print(f"{memory.id}: Freshness = {freshness:.1f}")
```

### Staleness Prevention

```python
# Implement staleness prevention strategies

# 1. Update access timestamps regularly
await client.touch_memory(memory_id)  # Updates last_accessed

# 2. Periodic freshness boosts for important memories
if memory.importance >= 8.0:
    await client.update_memory(
        memory_id,
        metadata={"refreshed_at": datetime.now().isoformat()}
    )

# 3. User engagement reminders
if days_since_access > 30:
    send_reminder(user_id, memory)
```

---

## Topic Coverage Analysis

### Coverage Levels

| Level | Memory Count | Quality | Status |
|-------|--------------|---------|--------|
| **Comprehensive** | 10+ | Excellent | ✅ Well covered |
| **Adequate** | 5-9 | Good | ✅ Sufficient |
| **Sparse** | 2-4 | Fair | ⚠️ Limited |
| **Minimal** | 1 | Poor | ❌ Very limited |
| **Missing** | 0 | N/A | ❌ Gap identified |

### Analyze Topic Coverage

```python
# Analyze coverage
coverage = monitor.analyze_topic_coverage(
    memories,
    topics=None  # Auto-extract from tags, or provide custom list
)

print(f"=== TOPIC COVERAGE ANALYSIS ===")
print(f"Total Topics: {len(coverage)}\n")

for topic in coverage:
    print(f"{topic.topic}: {topic.coverage_level}")
    print(f"  Memories: {topic.memory_count}")
    print(f"  Quality: {topic.quality_score:.1f}/100")
    print(f"  Avg Recency: {topic.avg_recency_days:.0f} days")
    print(f"  Top Keywords: {', '.join(topic.keywords[:5])}")

    if topic.gaps:
        print(f"  Gaps: {', '.join(topic.gaps)}")

    # Show representative memories
    for mem in topic.representative_memories[:2]:
        print(f"    - {mem.text[:50]}...")
    print()
```

### Identify Coverage Gaps

```python
# Find under-represented topics
gaps = [topic for topic in coverage if topic.coverage_level in ["sparse", "minimal", "missing"]]

print(f"=== COVERAGE GAPS ===")
print(f"Found {len(gaps)} topics needing more coverage\n")

for topic in gaps:
    print(f"{topic.topic}:")
    print(f"  Current: {topic.memory_count} memories ({topic.coverage_level})")
    print(f"  Target: 5+ memories for adequate coverage")
    print(f"  Suggested action: Add more {topic.topic}-related memories")

    if topic.gaps:
        print(f"  Missing aspects: {', '.join(topic.gaps)}")
```

### Coverage Improvement Suggestions

```python
# Generate improvement plan
def generate_coverage_plan(coverage):
    plan = []

    for topic in coverage:
        if topic.coverage_level == "missing":
            plan.append({
                "topic": topic.topic,
                "action": "add",
                "target_count": 5,
                "priority": "high"
            })
        elif topic.coverage_level == "minimal":
            plan.append({
                "topic": topic.topic,
                "action": "expand",
                "target_count": 5,
                "priority": "medium"
            })
        elif topic.quality_score < 50:
            plan.append({
                "topic": topic.topic,
                "action": "improve_quality",
                "target_quality": 70,
                "priority": "medium"
            })

    return plan

plan = generate_coverage_plan(coverage)

for item in plan:
    print(f"{item['priority'].upper()}: {item['action']} for {item['topic']}")
```

---

## Memory Tracking & Events

### Event Types

| Event Type | Description | Severity |
|------------|-------------|----------|
| `created` | Memory was created | INFO |
| `updated` | Memory was modified | INFO |
| `deleted` | Memory was removed | INFO |
| `retrieved` | Memory was directly fetched | DEBUG |
| `searched` | Memory appeared in search | DEBUG |
| `consolidated` | Memory was consolidated | INFO |
| `deduplicated` | Duplicate detected | WARNING |
| `health_check` | Health check performed | DEBUG |
| `conflict_detected` | Conflict found | WARNING |
| `conflict_resolved` | Conflict resolved | INFO |
| `staleness_detected` | Memory marked stale | WARNING |
| `freshness_updated` | Freshness boosted | INFO |

### Track Memory Events

```python
from hippocampai.monitoring import MemoryTracker

# Initialize tracker
tracker = MemoryTracker(
    storage_backend="redis",  # "memory", "redis", "file"
    max_events=10000
)

# Events are automatically tracked during operations
memory = await client.add_memory(
    user_id="user123",
    text="Important information",
    memory_type="fact"
)
# Tracker logs: event_type="created"

# Retrieve events for a memory
events = tracker.get_memory_events(memory_id=memory.id)

for event in events:
    print(f"{event.timestamp}: {event.event_type}")
    print(f"  Metadata: {event.metadata}")
```

### Access Pattern Analysis

```python
# Get access patterns for a memory
pattern = tracker.get_access_pattern(memory_id=memory.id)

print(f"=== ACCESS PATTERN ===")
print(f"Total accesses: {pattern.access_count}")
print(f"Last accessed: {pattern.last_accessed}")
print(f"Access frequency: {pattern.frequency}")
print(f"Hot/Cold: {pattern.temperature}")
print(f"Access sources: {pattern.sources}")

# Get all patterns for a user
user_patterns = tracker.get_user_access_patterns(user_id="user123")

# Find hot and cold memories
hot_memories = [p for p in user_patterns if p.temperature == "hot"]
cold_memories = [p for p in user_patterns if p.temperature == "cold"]

print(f"\nHot memories: {len(hot_memories)}")
print(f"Cold memories: {len(cold_memories)}")
```

### Health History Tracking

```python
# Record health snapshots
tracker.record_health_snapshot(
    memory_id=memory.id,
    health_score=85.5,
    issues=["stale"],
    metadata={"component_scores": {...}}
)

# Get health history
history = tracker.get_health_history(
    memory_id=memory.id,
    limit=10
)

print("=== HEALTH HISTORY ===")
for snapshot in history:
    print(f"{snapshot.timestamp}: Score = {snapshot.health_score}")
    print(f"  Issues: {snapshot.issues}")
```

### Query Events

```python
# Query by event type
created_events = tracker.query_events(
    event_type="created",
    user_id="user123",
    start_time=datetime.now() - timedelta(days=7),
    limit=100
)

# Query by severity
warnings = tracker.query_events(
    severity="WARNING",
    user_id="user123",
    limit=50
)

# Audit trail for compliance
audit_trail = tracker.get_audit_trail(
    memory_id=memory.id,
    include_metadata=True
)

for event in audit_trail:
    print(f"{event.timestamp}: {event.event_type} by {event.user_id}")
    print(f"  Changes: {event.metadata.get('changes', {})}")
```

---

## Conflict Detection & Resolution

### Conflict Types

| Type | Description | Detection |
|------|-------------|-----------|
| **Temporal** | Different time periods | Date comparison |
| **Semantic** | Contradicting content | Embedding similarity |
| **Factual** | Incompatible facts | NLI model |
| **Update** | Newer overwrites older | Timestamp comparison |

### Detect Conflicts

```python
from hippocampai.monitoring import ConflictDetector

# Initialize detector
detector = ConflictDetector(embedder=embedder)

# Detect conflicts in memory set
conflicts = detector.detect_conflicts(
    memories,
    conflict_types=["temporal", "semantic", "factual"]
)

print(f"Found {len(conflicts)} conflicts")

for conflict in conflicts:
    print(f"\n=== CONFLICT ===")
    print(f"Type: {conflict.conflict_type}")
    print(f"Severity: {conflict.severity}")
    print(f"Memories involved: {len(conflict.memory_ids)}")

    # Show conflicting memories
    for memory_id in conflict.memory_ids:
        memory = next(m for m in memories if m.id == memory_id)
        print(f"  - [{memory.created_at}] {memory.text[:50]}...")

    print(f"Suggested resolution: {conflict.resolution_strategy}")
```

### Resolve Conflicts

```python
# Automatic conflict resolution
for conflict in conflicts:
    if conflict.severity == "high":
        # Auto-resolve high-severity conflicts
        if conflict.resolution_strategy == "keep_latest":
            # Keep most recent, delete older
            sorted_memories = sorted(
                [m for m in memories if m.id in conflict.memory_ids],
                key=lambda m: m.created_at,
                reverse=True
            )

            keep = sorted_memories[0]
            delete = sorted_memories[1:]

            for memory in delete:
                await client.delete_memory(memory.id)

            print(f"Resolved: Kept {keep.id}, deleted {len(delete)} older versions")

        elif conflict.resolution_strategy == "merge":
            # Merge conflicting information
            merged_text = detector.merge_conflicting_memories(
                [m for m in memories if m.id in conflict.memory_ids]
            )

            # Create new merged memory
            await client.add_memory(
                user_id=memories[0].user_id,
                text=merged_text,
                memory_type="consolidated"
            )

            # Delete originals
            for memory_id in conflict.memory_ids:
                await client.delete_memory(memory_id)
```

### Resolution Strategies

| Strategy | When to Use | Action |
|----------|-------------|--------|
| **keep_latest** | Temporal conflicts | Keep most recent |
| **keep_highest_confidence** | Quality conflicts | Keep highest confidence |
| **merge** | Complementary info | Combine memories |
| **manual_review** | Complex conflicts | Flag for human review |
| **versioning** | Historical tracking | Keep all versions |

---

## Metrics & Observability

### Metrics Collection

```python
from hippocampai.monitoring import (
    get_metrics_collector,
    record_memory_store_stats,
    OperationType
)

# Get global collector
collector = get_metrics_collector()

# Trace operations
with collector.trace_operation(OperationType.CREATE):
    await client.add_memory(user_id="user123", text="Test")

# Record custom metrics
record_memory_store_stats(
    total_memories=len(memories),
    healthy_memories=health.healthy_memories,
    stale_memories=health.stale_memories,
    duplicate_clusters=len(clusters),
    health_score=health.overall_score
)
```

### View Metrics

```python
# Get metrics summary
summary = collector.get_metrics_summary()

print(f"=== METRICS SUMMARY ===")
print(f"Total operations: {summary['total_traces']}")
print(f"Successful: {summary['successful_operations']}")
print(f"Failed: {summary['failed_operations']}")
print(f"Success rate: {summary['successful_operations'] / summary['total_traces'] * 100:.1f}%")
print(f"Avg duration: {summary['avg_duration_ms']:.2f}ms")

# Get operation breakdown
for op_type, count in summary['operations_by_type'].items():
    print(f"  {op_type}: {count}")
```

### Export Metrics

```python
# Export to Prometheus format
prometheus_metrics = collector.export_metrics(format="prometheus")
print(prometheus_metrics)

# Export to JSON
json_metrics = collector.export_metrics(format="json")

# Export to file
collector.export_to_file("/path/to/metrics.txt", format="prometheus")
```

### Distributed Tracing

```python
# Get trace details
traces = collector.get_traces(
    operation_type=OperationType.RECALL,
    limit=10
)

for trace in traces:
    print(f"\n=== TRACE {trace.trace_id} ===")
    print(f"Operation: {trace.operation_type}")
    print(f"Duration: {trace.duration_ms}ms")
    print(f"Success: {trace.success}")

    # Show spans
    for span in trace.spans:
        print(f"  {span.name}: {span.duration_ms}ms")
```

---

## Best Practices

### 1. Regular Health Checks

```python
# Schedule daily health checks
async def daily_health_check():
    users = await get_all_users()

    for user in users:
        memories = await client.get_memories(user_id=user.id, limit=1000)
        health = monitor.calculate_health_score(memories)

        if health.status in ["poor", "critical"]:
            # Send alert
            send_alert(f"User {user.id} health: {health.status}")

        # Record metrics
        record_memory_store_stats(
            total_memories=health.total_memories,
            health_score=health.overall_score,
            user_id=user.id
        )
```

### 2. Proactive Duplicate Prevention

```python
# Check for duplicates before adding
async def add_memory_with_duplicate_check(text, user_id):
    # Get existing memories
    existing = await client.get_memories(user_id=user_id)

    # Check similarity
    new_embedding = embedder.embed([text])[0]

    for memory in existing:
        similarity = cosine_similarity(new_embedding, memory.embedding)

        if similarity > 0.95:
            print(f"Duplicate detected! Similar to: {memory.text[:50]}...")
            return None

    # Add if no duplicate
    return await client.add_memory(user_id=user_id, text=text)
```

### 3. Automated Cleanup Workflows

```python
# Weekly cleanup routine
async def weekly_cleanup():
    # 1. Remove exact duplicates
    clusters = monitor.detect_duplicate_clusters(memories, cluster_type="exact")
    for cluster in clusters:
        # Auto-merge
        await auto_merge_cluster(cluster)

    # 2. Archive stale memories
    stale = monitor.detect_stale_memories(memories)
    for memory in stale:
        if memory.should_archive:
            await archive_memory(memory.memory.id)

    # 3. Delete very old low-importance memories
    for memory in memories:
        age_days = (datetime.now() - memory.created_at).days
        if age_days > 365 and memory.importance < 3.0:
            await client.delete_memory(memory.id)
```

### 4. Quality Gates

```python
# Enforce minimum quality standards
async def add_memory_with_quality_gate(text, user_id, **kwargs):
    # Check text length
    if len(text) < 10:
        raise ValueError("Text too short (min 10 chars)")

    # Check confidence
    confidence = kwargs.get('confidence', 0.5)
    if confidence < 0.3:
        raise ValueError("Confidence too low (min 0.3)")

    # Add memory
    return await client.add_memory(user_id=user_id, text=text, **kwargs)
```

### 5. Monitoring Integration

```python
# Integrate with FastAPI
from fastapi import FastAPI

app = FastAPI()

@app.get("/health/{user_id}")
async def get_user_health(user_id: str):
    memories = await client.get_memories(user_id=user_id, limit=1000)
    health = monitor.calculate_health_score(memories)

    return {
        "score": health.overall_score,
        "status": health.status,
        "components": {
            "freshness": health.freshness_score,
            "diversity": health.diversity_score,
            "consistency": health.consistency_score,
            "coverage": health.coverage_score
        },
        "recommendations": health.recommendations
    }

@app.post("/cleanup/{user_id}")
async def run_cleanup(user_id: str):
    memories = await client.get_memories(user_id=user_id, limit=1000)

    # Run cleanup
    report = await run_full_cleanup(memories, user_id)

    return {
        "duplicates_removed": report['duplicates_removed'],
        "stale_archived": report['stale_archived'],
        "conflicts_resolved": report['conflicts_resolved']
    }
```

---

## API Reference

### MemoryHealthMonitor

```python
from hippocampai.monitoring import MemoryHealthMonitor

monitor = MemoryHealthMonitor(
    embedder=embedder,
    exact_duplicate_threshold=0.98,
    soft_duplicate_threshold=0.85,
    stale_days_threshold=90,
    no_access_days_threshold=60,
    min_confidence_threshold=0.3
)

# Methods
health = monitor.calculate_health_score(memories, detailed=True)
report = monitor.generate_quality_report(memories, user_id)
clusters = monitor.detect_duplicate_clusters(memories, cluster_type="soft")
stale = monitor.detect_stale_memories(memories)
coverage = monitor.analyze_topic_coverage(memories, topics=None)
```

### MemoryTracker

```python
from hippocampai.monitoring import MemoryTracker

tracker = MemoryTracker(
    storage_backend="redis",
    max_events=10000
)

# Methods
events = tracker.get_memory_events(memory_id)
pattern = tracker.get_access_pattern(memory_id)
patterns = tracker.get_user_access_patterns(user_id)
history = tracker.get_health_history(memory_id, limit=10)
audit = tracker.get_audit_trail(memory_id)
```

### MetricsCollector

```python
from hippocampai.monitoring import get_metrics_collector, OperationType

collector = get_metrics_collector()

# Methods
with collector.trace_operation(OperationType.CREATE):
    # ... operation ...
    pass

summary = collector.get_metrics_summary()
metrics = collector.export_metrics(format="prometheus")
traces = collector.get_traces(operation_type, limit=10)
```

---

## Related Documentation

- [GETTING_STARTED.md](GETTING_STARTED.md) - Setup and basic usage
- [API_REFERENCE.md](API_REFERENCE.md) - Complete API documentation
- [MONITORING.md](MONITORING.md) - System-wide monitoring
- [USER_GUIDE.md](USER_GUIDE.md) - Production deployment guide

---

**Built with ❤️ by the HippocampAI community**
