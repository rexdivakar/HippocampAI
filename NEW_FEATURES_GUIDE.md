# 🚀 New Features Guide - HippocampAI

This guide covers all the newly implemented advanced features in HippocampAI.

## 📋 Table of Contents

1. [Memory Conflict Resolution](#1-memory-conflict-resolution)
2. [Memory Health Monitoring](#2-memory-health-monitoring)
3. [Auto-Summarization](#3-auto-summarization)
4. [Auto-Consolidation](#4-auto-consolidation)
5. [Advanced Compression](#5-advanced-compression)
6. [Importance Decay](#6-importance-decay)
7. [Metrics & Tracing](#7-metrics--tracing)
8. [Qdrant Monitoring Storage](#8-qdrant-monitoring-storage)
9. [Testing the Features](#9-testing-the-features)
10. [SaaS Deployment Considerations](#10-saas-deployment-considerations)

---

## 1. Memory Conflict Resolution

Automatically detects and resolves contradictory memories.

### Features:
- ✅ Contradiction detection (semantic + pattern-based)
- ✅ 6 resolution strategies (temporal, confidence, importance, user review, auto merge, keep both)
- ✅ Configurable thresholds
- ✅ LLM-powered smart merging

### Usage:

```python
from hippocampai.pipeline import MemoryConflictResolver
from hippocampai.adapters import GroqLLM

llm = GroqLLM(api_key="your-key")
resolver = MemoryConflictResolver(
    embedder=embedder,
    llm=llm,
    default_strategy="temporal"  # Latest wins
)

# Detect conflicts
conflicts = resolver.detect_conflicts(new_memory, existing_memories)

# Resolve conflict
resolution = resolver.resolve_conflict(conflicts[0])
print(f"Action: {resolution.action}")
print(f"Result: {resolution.result_memory.text}")
```

### Configuration:

```python
# In config.py or environment variables
ENABLE_CONFLICT_RESOLUTION=true
CONFLICT_RESOLUTION_STRATEGY=temporal  # temporal, confidence, importance, user_review, auto_merge, keep_both
CONFLICT_SIMILARITY_THRESHOLD=0.75
CONFLICT_CONTRADICTION_THRESHOLD=0.85
AUTO_RESOLVE_CONFLICTS=true
```

### Testing:
```bash
python -m pytest tests/test_conflict_resolution.py -v
```

---

## 2. Memory Health Monitoring

Comprehensive quality assessment of your memory store.

### Features:
- ✅ 4-component health scoring (Freshness, Diversity, Consistency, Coverage)
- ✅ Duplicate detection with 4 cluster types (Exact, Soft, Paraphrase, Variant)
- ✅ Stale memory detection (5 staleness reasons)
- ✅ Topic coverage analysis
- ✅ Actionable recommendations

### Usage:

```python
from hippocampai.monitoring import MemoryHealthMonitor

monitor = MemoryHealthMonitor(embedder=embedder)

# Generate comprehensive health report
memories = memory_service.get_memories(user_id="user123", limit=1000)
report = monitor.generate_quality_report(memories, user_id="user123")

print(f"Health Score: {report.health_score.overall_score}/100")
print(f"Status: {report.health_score.status}")
print(f"Recommendations: {report.health_score.recommendations}")

# Detect duplicates
duplicates = monitor.detect_duplicate_clusters(memories, cluster_type="soft")
print(f"Found {len(duplicates)} duplicate clusters")

# Detect stale memories
stale = monitor.detect_stale_memories(memories, threshold_days=90)
print(f"Found {len(stale)} stale memories")

# Analyze topic coverage
coverage = monitor.analyze_topic_coverage(memories)
for topic in coverage:
    print(f"{topic.topic}: {topic.memory_count} memories ({topic.coverage_level})")
```

### Testing:
```bash
python -m pytest tests/test_memory_health.py -v
```

---

## 3. Auto-Summarization

Automatically summarize and compress older memories.

### Features:
- ✅ Time-based summarization (older memories first)
- ✅ Smart grouping by topic/type
- ✅ LLM-powered summaries
- ✅ Configurable batch sizes and time windows
- ✅ Archive original memories

### Usage:

```python
from hippocampai.pipeline import AutoSummarization

summarizer = AutoSummarization(
    llm=llm,
    embedder=embedder,
    memory_service=memory_service,
    time_window_days=30,  # Summarize memories older than 30 days
    batch_size=10  # Group 10 memories per summary
)

# Run summarization
result = summarizer.summarize_memories(user_id="user123")

print(f"Processed: {result['memories_processed']}")
print(f"Summaries created: {result['summaries_created']}")
print(f"Space saved: {result['space_saved']:.1%}")
```

### Configuration:

```python
AUTO_SUMMARIZATION_ENABLED=true
AUTO_SUMMARIZATION_TIME_WINDOW_DAYS=30
AUTO_SUMMARIZATION_BATCH_SIZE=10
AUTO_SUMMARIZATION_MIN_MEMORIES=50
```

### Testing:
```bash
python -m pytest tests/test_auto_summarization.py -v
```

---

## 4. Auto-Consolidation

Merge related memories to reduce redundancy.

### Features:
- ✅ Semantic similarity clustering
- ✅ Smart merging (preserves all information)
- ✅ Configurable similarity threshold
- ✅ Archive original memories

### Usage:

```python
from hippocampai.pipeline import AutoConsolidation

consolidator = AutoConsolidation(
    llm=llm,
    embedder=embedder,
    memory_service=memory_service,
    similarity_threshold=0.85
)

# Run consolidation
result = consolidator.consolidate_memories(user_id="user123")

print(f"Clusters found: {result['clusters_found']}")
print(f"Consolidated: {result['consolidated_memories']}")
print(f"Reduction: {result['reduction_percentage']:.1%}")
```

### Testing:
```bash
python -m pytest tests/test_auto_consolidation.py -v
```

---

## 5. Advanced Compression

Intelligent memory compression with quality preservation.

### Features:
- ✅ 3 compression levels (light, medium, aggressive)
- ✅ Quality-aware compression
- ✅ Smart deduplication
- ✅ Importance preservation

### Usage:

```python
from hippocampai.pipeline import AdvancedCompression, CompressionQuality

compressor = AdvancedCompression(
    llm=llm,
    embedder=embedder,
    memory_service=memory_service
)

# Compress with quality control
result = compressor.compress_memories(
    user_id="user123",
    target_reduction=0.30,  # 30% reduction
    min_quality=CompressionQuality.MEDIUM
)

print(f"Compressed: {result['compressed_memories']}")
print(f"Compression ratio: {result['compression_ratio']:.2f}x")
print(f"Space saved: {result['space_saved']:.1%}")
```

### Testing:
```bash
python -m pytest tests/test_advanced_compression.py -v
```

---

## 6. Importance Decay

Gradually decay importance scores over time.

### Features:
- ✅ Exponential decay
- ✅ Configurable half-life
- ✅ Access-based boosting
- ✅ Type-based decay rates

### Usage:

```python
from hippocampai.pipeline import ImportanceDecay

decay = ImportanceDecay(
    memory_service=memory_service,
    half_life_days=90  # Importance halves every 90 days
)

# Apply decay
result = decay.apply_decay(user_id="user123")

print(f"Memories updated: {result['memories_updated']}")
print(f"Average decay: {result['average_decay_factor']:.3f}")
```

### Testing:
```bash
python -m pytest tests/test_importance_decay.py -v
```

---

## 7. Metrics & Tracing

Comprehensive observability for all operations.

### Features:
- ✅ Distributed tracing with spans
- ✅ 4 metric types (Counter, Gauge, Histogram, Timer)
- ✅ Tag-based filtering
- ✅ Multi-dimensional querying
- ✅ Prometheus & JSON export

### Usage:

```python
from hippocampai.monitoring import MetricsCollector, OperationType

collector = MetricsCollector(enable_tracing=True)

# Trace an operation
with collector.trace_operation(
    OperationType.CREATE,
    tags={"environment": "production", "version": "1.0"},
    user_id="user123",
    memory_type="fact"
) as trace:
    # Your operation
    memory = create_memory(...)

# Query traces
traces = collector.query_traces(
    tags={"environment": "production"},
    user_id="user123",
    min_duration_ms=10.0
)

# Get statistics
stats = collector.get_trace_statistics(
    user_id="user123",
    tags={"environment": "production"}
)

print(f"Operations: {stats['count']}")
print(f"Success rate: {stats['success_rate']:.1f}%")
print(f"Avg duration: {stats['duration_stats']['mean_ms']:.1f}ms")
```

### Testing:
```bash
python -m pytest tests/test_metrics.py -v
```

---

## 8. Qdrant Monitoring Storage

Persistent storage for monitoring data in Qdrant.

### Features:
- ✅ Store health reports in Qdrant
- ✅ Store traces with full metadata
- ✅ Advanced filtering (tags, user_id, time ranges, duration)
- ✅ Historical trending
- ✅ Cleanup old data

### Usage:

```python
from hippocampai.monitoring import MonitoringStorage

storage = MonitoringStorage(qdrant_store=qdrant_store)

# Store health report
report_id = storage.store_health_report(
    report,
    tags={"environment": "production", "version": "1.0"}
)

# Store trace
trace_id = storage.store_trace(
    trace,
    additional_tags={"region": "us-west"}
)

# Query health reports
reports = storage.query_health_reports(
    user_id="user123",
    min_health_score=70.0,
    tags={"environment": "production"},
    start_time=datetime.now() - timedelta(days=7)
)

# Query traces
traces = storage.query_traces(
    operation="create",
    user_id="user123",
    tags={"environment": "production"},
    min_duration_ms=50.0
)

# Get health history (trending)
history = storage.get_health_history(user_id="user123", days=30)

# Get trace statistics
stats = storage.get_trace_statistics(
    user_id="user123",
    tags={"environment": "production"},
    days=7
)

# Cleanup old data
cleanup_result = storage.cleanup_old_data(
    health_retention_days=90,
    trace_retention_days=30
)
```

### Testing:
```bash
python -m pytest tests/test_monitoring_tags_storage.py -v
```

---

## 9. Testing the Features

### Basic Chat (Original):
```bash
python chat.py
```

### Advanced Chat (All Features):
```bash
python chat_advanced.py
```

### Available Commands in Advanced Chat:

**Basic:**
- `/help` - Show commands
- `/stats` - Memory statistics
- `/memories` - Recent memories
- `/patterns` - Behavioral patterns
- `/search <query>` - Search memories

**Advanced:**
- `/health` - Generate comprehensive health report
- `/summarize` - Run auto-summarization
- `/consolidate` - Run memory consolidation
- `/compress` - Run advanced compression
- `/metrics` - Show metrics and tracing stats
- `/traces` - Show recent operation traces

### Run All Tests:
```bash
# All new feature tests
python -m pytest tests/test_conflict_resolution.py tests/test_memory_health.py tests/test_auto_summarization.py tests/test_auto_consolidation.py tests/test_advanced_compression.py tests/test_importance_decay.py tests/test_metrics.py tests/test_monitoring_tags_storage.py -v

# Total: 110+ tests across 8 test files
```

---

## 10. SaaS Deployment Considerations

### Making Features Automatic

The features are **library components** that need to be **triggered**. For a SaaS deployment, you have several options:

#### A. Background Task Scheduler (Recommended for SaaS)

Use **Celery** or similar task scheduler:

```python
from celery import Celery
from celery.schedules import crontab

celery = Celery('hippocampai_tasks')

# Configure periodic tasks
celery.conf.beat_schedule = {
    'auto-summarize-daily': {
        'task': 'tasks.auto_summarize_all_users',
        'schedule': crontab(hour=2, minute=0),  # 2 AM daily
    },
    'consolidate-weekly': {
        'task': 'tasks.consolidate_all_users',
        'schedule': crontab(day_of_week=0, hour=3, minute=0),  # Sunday 3 AM
    },
    'apply-decay-daily': {
        'task': 'tasks.apply_importance_decay',
        'schedule': crontab(hour=1, minute=0),  # 1 AM daily
    },
    'health-check-daily': {
        'task': 'tasks.generate_health_reports',
        'schedule': crontab(hour=4, minute=0),  # 4 AM daily
    }
}

@celery.task
def auto_summarize_all_users():
    """Summarize memories for all users."""
    users = get_all_active_users()
    for user_id in users:
        summarizer.summarize_memories(user_id=user_id)

@celery.task
def consolidate_all_users():
    """Consolidate memories for all users."""
    users = get_all_active_users()
    for user_id in users:
        consolidator.consolidate_memories(user_id=user_id)

@celery.task
def apply_importance_decay():
    """Apply decay to all users."""
    users = get_all_active_users()
    for user_id in users:
        decay.apply_decay(user_id=user_id)

@celery.task
def generate_health_reports():
    """Generate health reports for all users."""
    users = get_all_active_users()
    for user_id in users:
        memories = memory_service.get_memories(user_id=user_id, limit=1000)
        report = monitor.generate_quality_report(memories, user_id=user_id)
        storage.store_health_report(report, tags={"automated": "true"})
```

#### B. Threshold-Based Triggers

Trigger operations based on memory count or usage:

```python
async def create_memory(text: str, user_id: str):
    """Create memory with automatic optimization."""

    # Create the memory
    memory = await memory_service.create_memory(text, user_id)

    # Check memory count
    stats = await memory_service.get_memory_statistics(user_id)
    memory_count = stats['total_memories']

    # Auto-summarize if too many memories
    if memory_count > 1000:
        # Queue summarization task
        auto_summarize.delay(user_id)

    # Auto-consolidate if many duplicates
    if memory_count > 500 and memory_count % 100 == 0:
        # Check for duplicates periodically
        consolidate_memories.delay(user_id)

    # Check health if many memories
    if memory_count % 250 == 0:
        # Generate health report periodically
        generate_health_report.delay(user_id)

    return memory
```

#### C. API Endpoints for Manual Triggers

Expose endpoints for users/admins to trigger operations:

```python
from fastapi import FastAPI

app = FastAPI()

@app.post("/api/v1/users/{user_id}/memories/summarize")
async def trigger_summarization(user_id: str):
    """Manually trigger summarization."""
    result = summarizer.summarize_memories(user_id=user_id)
    return result

@app.post("/api/v1/users/{user_id}/memories/consolidate")
async def trigger_consolidation(user_id: str):
    """Manually trigger consolidation."""
    result = consolidator.consolidate_memories(user_id=user_id)
    return result

@app.get("/api/v1/users/{user_id}/health")
async def get_health_report(user_id: str):
    """Get memory health report."""
    memories = await memory_service.get_memories(user_id=user_id, limit=1000)
    report = monitor.generate_quality_report(memories, user_id=user_id)
    return report

@app.get("/api/v1/users/{user_id}/metrics")
async def get_metrics(user_id: str, days: int = 7):
    """Get usage metrics."""
    stats = storage.get_trace_statistics(
        user_id=user_id,
        days=days
    )
    return stats
```

### Docker Compose Example

```yaml
version: '3.8'

services:
  qdrant:
    image: qdrant/qdrant
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage

  redis:
    image: redis:7
    ports:
      - "6379:6379"

  celery_worker:
    build: .
    command: celery -A tasks worker --loglevel=info
    depends_on:
      - redis
      - qdrant
    environment:
      - CELERY_BROKER_URL=redis://redis:6379/0

  celery_beat:
    build: .
    command: celery -A tasks beat --loglevel=info
    depends_on:
      - redis
      - celery_worker
    environment:
      - CELERY_BROKER_URL=redis://redis:6379/0

  api:
    build: .
    command: uvicorn main:app --host 0.0.0.0 --port 8000
    ports:
      - "8000:8000"
    depends_on:
      - qdrant
      - redis
    environment:
      - GROQ_API_KEY=${GROQ_API_KEY}

volumes:
  qdrant_data:
```

---

## Summary

✅ **All features are implemented and tested**
✅ **110+ tests passing across 8 test suites**
✅ **Comprehensive documentation**
✅ **Example implementation in `chat_advanced.py`**
✅ **Ready for SaaS deployment with task schedulers**

### Key Points for SaaS:

1. **Features work but need triggers** - They're library components, not automatic daemons
2. **Use Celery or similar** - Schedule periodic tasks (daily summarization, weekly consolidation, etc.)
3. **Threshold-based triggers** - Auto-run when memory count exceeds thresholds
4. **Expose API endpoints** - Let users/admins trigger operations manually
5. **Monitor with Qdrant storage** - Store all health reports and traces for trending

### Next Steps:

1. Test features with `python chat_advanced.py`
2. Set up Celery for periodic tasks if deploying to SaaS
3. Configure thresholds in environment variables
4. Set up monitoring dashboards (Grafana + Prometheus)
