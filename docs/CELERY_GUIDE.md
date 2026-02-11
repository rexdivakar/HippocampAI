# Celery Background Task Processing Guide

**Complete guide to using Celery for distributed background task processing in HippocampAI**

Last Updated: 2026-02-11

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Architecture](#architecture)
4. [Configuration & Optimization](#configuration--optimization)
5. [Task Management](#task-management)
6. [Worker Control](#worker-control)
7. [Queue Management](#queue-management)
8. [Monitoring with Flower](#monitoring-with-flower)
9. [Performance Tuning](#performance-tuning)
10. [Troubleshooting](#troubleshooting)
11. [Best Practices](#best-practices)

---

## Overview

HippocampAI integrates **Celery** for distributed task processing and **Flower** for real-time monitoring. This enables:

- **Asynchronous background tasks** - Long-running operations don't block API responses
- **Scheduled jobs** - Automatic deduplication, consolidation, and cleanup
- **Distributed processing** - Scale workers independently from the API
- **Task monitoring** - Beautiful Flower UI to track all tasks in real-time
- **Retry logic** - Automatic retries for failed tasks
- **Task queues** - Separate queues for different priority levels

### Key Components

| Component | Purpose | Port |
|-----------|---------|------|
| **Celery Workers** | Execute background tasks | N/A |
| **Celery Beat** | Scheduler for periodic tasks | N/A |
| **Redis** | Message broker & result backend | 6379 |
| **Flower** | Monitoring UI | 5555 |

---

## Quick Start

### 1. Start All Services

```bash
# Start everything with docker-compose
docker-compose up -d

# This starts:
# - hippocampai-api (FastAPI) on port 8000
# - hippocampai-celery-worker (background tasks)
# - hippocampai-celery-beat (scheduler)
# - hippocampai-flower (monitoring UI) on port 5555
# - hippocampai-redis (broker/cache) on port 6379
# - hippocampai-qdrant (vector DB) on port 6333
```

### 2. Access Flower UI

Open your browser to http://localhost:5555

**Features:**
- Active tasks per worker
- Task success/failure rates
- Queue lengths
- Worker resource usage
- Task history & logs

### 3. Submit Background Tasks

**Via Python Client:**
```python
from hippocampai import MemoryClient

# Initialize client in remote mode
client = MemoryClient(mode="remote", api_url="http://localhost:8000")

# Submit background task
task_id = client.submit_deduplication_task(user_id="user123")
print(f"Task submitted: {task_id}")

# Check task status
status = client.get_task_status(task_id)
print(f"Status: {status['state']}")
```

**Via REST API:**
```bash
# Submit deduplication task
curl -X POST http://localhost:8000/api/v1/tasks/deduplication \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "similarity_threshold": 0.85
  }'

# Response
{
  "task_id": "abc-123-def",
  "status": "PENDING",
  "queue": "memory_ops"
}

# Check task status
curl http://localhost:8000/api/v1/tasks/abc-123-def/status
```

### 4. Monitor Tasks

```bash
# View task logs
docker logs hippocampai-celery-worker -f

# View beat scheduler logs
docker logs hippocampai-celery-beat -f

# Check worker status
docker exec hippocampai-celery-worker celery -A src.hippocampai.celery_app inspect active
```

---

## Architecture

### Component Breakdown

```
┌─────────────────────────────────────────────────────────────┐
│                    USER / APPLICATION                        │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │ FastAPI      │    │ Python SDK   │    │ Direct Task  │  │
│  │ Async Routes │    │ client.py    │    │ Submission   │  │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘  │
└─────────┼────────────────────┼────────────────────┼──────────┘
          │                    │                    │
          └────────────────────┼────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────┐
│              REDIS (Message Broker & Result Backend)         │
│  • DB 1: Task Queue (broker)                                 │
│  • DB 2: Task Results (backend)                              │
└────────────────────┬─────────────────────────────────────────┘
                     │
          ┌──────────┴──────────┬──────────────┬─────────────┐
          ▼                     ▼              ▼             ▼
┌─────────────────┐  ┌─────────────────┐  ┌────────┐  ┌──────────┐
│ CELERY WORKER 1 │  │ CELERY WORKER 2 │  │ BEAT   │  │ FLOWER   │
│  (4 threads)    │  │  (4 threads)    │  │Schedule│  │ Monitor  │
│                 │  │                 │  │        │  │  :5555   │
│ Queues:         │  │ Queues:         │  │Triggers│  │          │
│  • default      │  │  • default      │  │periodic│  │          │
│  • memory_ops   │  │  • memory_ops   │  │tasks   │  │          │
│  • background   │  │  • background   │  │        │  │          │
│  • scheduled    │  │  • scheduled    │  │        │  │          │
└─────────┬───────┘  └─────────┬───────┘  └────────┘  └──────────┘
          │                    │
          └────────────────────┼───────────────────────────────────┐
                               ▼                                    │
         ┌──────────────────────────────────────────┐              │
         │   HIPPOCAMPAI SERVICES                   │              │
         │   • MemoryManagementService              │              │
         │   • QdrantStore (Vector DB)              │              │
         │   • Embedder (BGE-Small)                 │              │
         │   • Reranker (Cross-Encoder)             │              │
         └──────────────────────────────────────────┘              │
                               │                                    │
         ┌─────────────────────┴────────────────┐                  │
         ▼                                      ▼                   │
┌──────────────────┐                   ┌──────────────────┐        │
│ QDRANT           │                   │ REDIS (Cache)    │        │
│ Vector Storage   │                   │ DB 0             │        │
└──────────────────┘                   └──────────────────┘        │
                                                                    │
         ┌──────────────────────────────────────────────────────────┘
         ▼
┌──────────────────────────────────────────────────────────────────┐
│                       MONITORING                                  │
│  Flower UI shows:                                                 │
│  • Active tasks per worker                                        │
│  • Task success/failure rates                                     │
│  • Queue lengths                                                  │
│  • Worker resource usage                                          │
│  • Task history & logs                                            │
└──────────────────────────────────────────────────────────────────┘
```

### Queue Types

| Queue | Purpose | Routing Key | Priority |
|-------|---------|-------------|----------|
| **default** | General tasks | `default` | Normal |
| **memory_ops** | Memory CRUD operations | `memory.#` | High |
| **background** | Background maintenance | `background.#` | Low |
| **scheduled** | Periodic tasks (Beat) | `scheduled.#` | Low |

---

## Configuration & Optimization

### Environment Variables

```bash
# Broker & Backend
CELERY_BROKER_URL=redis://localhost:6379/1
CELERY_RESULT_BACKEND=redis://localhost:6379/2

# Worker Configuration
CELERY_WORKER_CONCURRENCY=4              # Number of worker processes
CELERY_WORKER_PREFETCH_MULTIPLIER=4      # Tasks to prefetch per worker
CELERY_WORKER_MAX_TASKS_PER_CHILD=1000   # Max tasks before worker restart
CELERY_TASK_ACKS_LATE=true               # Acknowledge task after completion

# Task Execution
CELERY_TASK_SOFT_TIME_LIMIT=300          # Soft timeout (5 minutes)
CELERY_TASK_TIME_LIMIT=600               # Hard timeout (10 minutes)
CELERY_TASK_REJECT_ON_WORKER_LOST=true   # Re-queue on worker crash

# Result Backend
CELERY_RESULT_EXPIRES=3600               # Results expire after 1 hour
CELERY_RESULT_PERSISTENT=true            # Persist results to disk
CELERY_RESULT_COMPRESSION=gzip           # Compress results

# Scheduled Tasks (Beat)
AUTO_DEDUP_ENABLED=true                  # Enable auto-deduplication
AUTO_CONSOLIDATION_ENABLED=false         # Enable auto-consolidation
DEDUP_INTERVAL_HOURS=24                  # Deduplication interval
CONSOLIDATION_INTERVAL_HOURS=168         # Consolidation interval (1 week)
EXPIRATION_INTERVAL_HOURS=1              # Memory expiration check interval

# Monitoring
FLOWER_PORT=5555
FLOWER_USER=admin
FLOWER_PASSWORD=admin
```

### Optimal Configuration Presets

#### Development Environment
```bash
CELERY_WORKER_CONCURRENCY=2
CELERY_WORKER_PREFETCH_MULTIPLIER=2
CELERY_WORKER_MAX_TASKS_PER_CHILD=100
CELERY_TASK_ACKS_LATE=false
```

#### Production (Medium Load)
```bash
CELERY_WORKER_CONCURRENCY=4
CELERY_WORKER_PREFETCH_MULTIPLIER=4
CELERY_WORKER_MAX_TASKS_PER_CHILD=1000
CELERY_TASK_ACKS_LATE=true
```

#### Production (High Load)
```bash
CELERY_WORKER_CONCURRENCY=8
CELERY_WORKER_PREFETCH_MULTIPLIER=2
CELERY_WORKER_MAX_TASKS_PER_CHILD=500
CELERY_TASK_ACKS_LATE=true
```

#### CPU-Intensive Workloads
```bash
CELERY_WORKER_CONCURRENCY=$(nproc)  # Match CPU cores
CELERY_WORKER_PREFETCH_MULTIPLIER=1
CELERY_WORKER_MAX_TASKS_PER_CHILD=200
```

---

## Task Management

### Available Background Tasks

| Task | Description | Queue | Typical Duration |
|------|-------------|-------|------------------|
| `deduplication_task` | Find and merge duplicates | memory_ops | 10-60s |
| `consolidation_task` | Consolidate related memories | memory_ops | 30-120s |
| `health_check_task` | Generate health report | background | 5-30s |
| `auto_summarization_task` | Create summaries | background | 20-90s |
| `compression_task` | Advanced compression | background | 15-60s |
| `expire_memories_task` | Clean up expired memories | scheduled | 5-20s |
| `migrate_embeddings_task` | Re-encode all memories for a new embedding model | memory_ops | 10-60min |
| `consolidate_procedural_rules` | Merge redundant procedural rules per user | background | 10-60s |

### Submit Tasks Programmatically

```python
from hippocampai.celery_app import (
    deduplication_task,
    consolidation_task,
    health_check_task,
    auto_summarization_task,
)
from hippocampai.tasks import migrate_embeddings_task

# Submit deduplication task
task = deduplication_task.delay(
    user_id="user123",
    similarity_threshold=0.85
)
print(f"Task ID: {task.id}")

# Wait for result (blocking)
result = task.get(timeout=300)
print(f"Result: {result}")

# Check status (non-blocking)
if task.ready():
    print(f"Result: {task.result}")
elif task.failed():
    print(f"Error: {task.traceback}")
else:
    print(f"Status: {task.state}")
```

### New v0.5.0 Tasks

#### migrate_embeddings_task

Re-encodes all stored memories when changing the embedding model. Dispatched by `POST /v1/admin/embeddings/migrate`.

- **Queue:** `memory_ops`
- **soft_time_limit:** 3600 seconds (1 hour)
- **Accepts:** `migration_id` (str) — the migration record ID
- **Progress:** Updates `migrated_count` and `failed_count` on the migration record

```python
from hippocampai.tasks import migrate_embeddings_task

# Dispatched automatically by the migration API endpoint
task = migrate_embeddings_task.delay(migration_id="mig_abc")
print(f"Migration task: {task.id}")
```

#### consolidate_procedural_rules

Merges redundant procedural rules for a user using LLM-based semantic comparison. Keeps the total rule count under `PROCEDURAL_RULE_MAX_COUNT`.

- **Queue:** `background`
- **Accepts:** `user_id` (str)

```python
# Can be triggered via POST /v1/procedural/consolidate?user_id=alice
# or programmatically
```

---

### Submit Tasks via REST API

```bash
# Deduplication
curl -X POST http://localhost:8000/api/v1/tasks/deduplication \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "similarity_threshold": 0.85,
    "async": true
  }'

# Consolidation
curl -X POST http://localhost:8000/api/v1/tasks/consolidation \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "similarity_threshold": 0.80
  }'

# Health Check
curl -X POST http://localhost:8000/api/v1/tasks/health-check \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user123"}'

# Auto-Summarization
curl -X POST http://localhost:8000/api/v1/tasks/summarization \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "min_age_days": 30
  }'
```

### Query Task Status

```bash
# Get task status
curl http://localhost:8000/api/v1/tasks/{task_id}/status

# Response
{
  "task_id": "abc-123-def",
  "state": "SUCCESS",
  "result": {
    "duplicates_found": 12,
    "duplicates_merged": 10,
    "space_saved_mb": 2.5
  },
  "traceback": null
}
```

### Cancel Running Task

```bash
# Cancel/revoke task
curl -X DELETE http://localhost:8000/api/v1/tasks/{task_id}

# Or via Python
from hippocampai.celery_app import celery_app
celery_app.control.revoke(task_id, terminate=True)
```

---

## Worker Control

### Start/Stop Workers

```bash
# Start worker (single)
celery -A src.hippocampai.celery_app worker \
  --loglevel=info \
  --concurrency=4 \
  --max-tasks-per-child=1000

# Start worker with specific queues
celery -A src.hippocampai.celery_app worker \
  --queues=memory_ops,default \
  --loglevel=info

# Start beat scheduler
celery -A src.hippocampai.celery_app beat --loglevel=info

# Start worker + beat (development)
celery -A src.hippocampai.celery_app worker --beat --loglevel=info
```

### Worker Inspection

```bash
# View active tasks
celery -A src.hippocampai.celery_app inspect active

# View scheduled tasks
celery -A src.hippocampai.celery_app inspect scheduled

# View registered tasks
celery -A src.hippocampai.celery_app inspect registered

# View worker stats
celery -A src.hippocampai.celery_app inspect stats

# Ping workers
celery -A src.hippocampai.celery_app inspect ping
```

### Worker Control Commands

```bash
# Graceful shutdown (wait for tasks to finish)
celery -A src.hippocampai.celery_app control shutdown

# Pool restart
celery -A src.hippocampai.celery_app control pool_restart

# Autoscale workers
celery -A src.hippocampai.celery_app control autoscale 10 3  # max=10, min=3

# Add consumer (queue)
celery -A src.hippocampai.celery_app control add_consumer memory_ops

# Cancel consumer (queue)
celery -A src.hippocampai.celery_app control cancel_consumer background
```

---

## Queue Management

### Queue Priority

**Configure in celeryconfig.py:**
```python
from kombu import Queue, Exchange

task_queues = (
    Queue('memory_ops', Exchange('memory_ops'), routing_key='memory.#', priority=9),
    Queue('default', Exchange('default'), routing_key='default', priority=5),
    Queue('background', Exchange('background'), routing_key='background.#', priority=3),
    Queue('scheduled', Exchange('scheduled'), routing_key='scheduled.#', priority=1),
)

task_default_queue = 'default'
task_default_exchange = 'default'
task_default_routing_key = 'default'
```

### Route Tasks to Specific Queues

```python
# Define routing in celeryconfig.py
task_routes = {
    'src.hippocampai.celery_tasks.deduplication_task': {
        'queue': 'memory_ops',
        'routing_key': 'memory.dedup'
    },
    'src.hippocampai.celery_tasks.consolidation_task': {
        'queue': 'memory_ops',
        'routing_key': 'memory.consolidate'
    },
    'src.hippocampai.celery_tasks.health_check_task': {
        'queue': 'background',
        'routing_key': 'background.health'
    },
}
```

### Monitor Queue Length

```bash
# Via Redis CLI
redis-cli -n 1
LLEN celery  # Default queue
LLEN memory_ops
LLEN background
LLEN scheduled

# Via Python
from redis import Redis
redis = Redis(host='localhost', port=6379, db=1)
queue_lengths = {
    'default': redis.llen('celery'),
    'memory_ops': redis.llen('memory_ops'),
    'background': redis.llen('background')
}
print(queue_lengths)
```

---

## Monitoring with Flower

### Access Flower Dashboard

Open http://localhost:5555

**Default Credentials:**
- Username: `admin`
- Password: `admin` (change in production!)

### Flower Features

**Task Monitoring:**
- Real-time task execution
- Success/failure rates
- Task duration histograms
- Task details and arguments

**Worker Monitoring:**
- Active workers
- Worker resource usage (CPU, memory)
- Concurrency and pool info
- Worker uptime

**Queue Monitoring:**
- Queue lengths
- Task routing
- Message rates

**Task Management:**
- Revoke tasks
- Retry failed tasks
- View task results
- Inspect task arguments

### Flower API

```bash
# Get all tasks
curl http://localhost:5555/api/tasks

# Get task info
curl http://localhost:5555/api/task/info/{task_id}

# Get worker info
curl http://localhost:5555/api/workers

# Revoke task
curl -X POST http://localhost:5555/api/task/revoke/{task_id}
```

### Configure Flower

**Environment Variables:**
```bash
FLOWER_PORT=5555
FLOWER_USER=admin
FLOWER_PASSWORD=admin
FLOWER_URL_PREFIX=/flower
```

**Command Line:**
```bash
celery -A src.hippocampai.celery_app flower \
  --port=5555 \
  --basic_auth=admin:admin \
  --url_prefix=/flower
```

---

## Performance Tuning

### Concurrency Strategies

#### Prefork Pool (Default)
```bash
celery -A src.hippocampai.celery_app worker \
  --pool=prefork \
  --concurrency=4
```
- Best for: CPU-intensive tasks
- Pros: Process isolation, good for blocking operations
- Cons: Higher memory usage

#### Gevent Pool
```bash
celery -A src.hippocampai.celery_app worker \
  --pool=gevent \
  --concurrency=100
```
- Best for: I/O-intensive tasks, many concurrent operations
- Pros: Very lightweight, high concurrency
- Cons: Requires gevent-friendly libraries

#### Solo Pool
```bash
celery -A src.hippocampai.celery_app worker \
  --pool=solo
```
- Best for: Debugging, development
- Pros: Simple, no multiprocessing overhead
- Cons: No concurrency

### Optimize Task Execution

```python
# Use compression for large results
@celery_app.task(compression='gzip')
def large_result_task():
    return large_data

# Set custom time limits per task
@celery_app.task(soft_time_limit=60, time_limit=120)
def time_sensitive_task():
    pass

# Ignore results if not needed
@celery_app.task(ignore_result=True)
def fire_and_forget_task():
    pass

# Set task priority
task.apply_async(args=[...], priority=9)  # 0-9, higher = more priority
```

### Connection Pool Tuning

```python
# In celeryconfig.py
broker_pool_limit = 10  # Max connections to broker
broker_connection_retry_on_startup = True
broker_connection_max_retries = 10

redis_backend_health_check_interval = 30
redis_max_connections = 50
```

### Memory Management

```python
# Restart workers periodically
CELERY_WORKER_MAX_TASKS_PER_CHILD = 1000

# Limit task prefetch
CELERY_WORKER_PREFETCH_MULTIPLIER = 4

# Enable task result expiration
CELERY_RESULT_EXPIRES = 3600  # 1 hour
```

---

## Troubleshooting

### Common Issues

#### Workers Not Starting

```bash
# Check logs
docker logs hippocampai-celery-worker

# Verify Redis connection
redis-cli -h localhost -p 6379 ping

# Check for port conflicts
netstat -tuln | grep 6379
```

#### Tasks Not Executing

```bash
# Check if workers are consuming from queue
celery -A src.hippocampai.celery_app inspect active

# Check queue length
redis-cli -n 1 LLEN celery

# Verify task is registered
celery -A src.hippocampai.celery_app inspect registered
```

#### Task Timeout Issues

```python
# Increase time limits in celeryconfig.py
task_soft_time_limit = 600  # 10 minutes
task_time_limit = 1200  # 20 minutes

# Or per-task
@celery_app.task(soft_time_limit=900, time_limit=1800)
def long_running_task():
    pass
```

#### Memory Leaks

```bash
# Enable max tasks per child
CELERY_WORKER_MAX_TASKS_PER_CHILD=500

# Monitor memory usage
docker stats hippocampai-celery-worker

# Restart workers regularly
celery -A src.hippocampai.celery_app control pool_restart
```

#### Result Backend Issues

```bash
# Check Redis result backend
redis-cli -n 2
KEYS celery-task-meta-*

# Clear old results
redis-cli -n 2 FLUSHDB

# Verify result expiration is set
CELERY_RESULT_EXPIRES=3600
```

---

## Best Practices

### 1. Task Design

```python
# Keep tasks idempotent (can run multiple times safely)
@celery_app.task(bind=True, max_retries=3)
def idempotent_task(self, user_id):
    try:
        # Check if already processed
        if is_already_processed(user_id):
            return {"status": "already_processed"}

        # Do work
        result = process_user(user_id)

        # Mark as processed
        mark_as_processed(user_id)

        return result
    except Exception as exc:
        # Retry with exponential backoff
        raise self.retry(exc=exc, countdown=2 ** self.request.retries)
```

### 2. Error Handling

```python
# Proper exception handling and retries
@celery_app.task(bind=True, autoretry_for=(Exception,), retry_kwargs={'max_retries': 3}, retry_backoff=True)
def robust_task(self, data):
    try:
        return process_data(data)
    except CriticalError:
        # Don't retry critical errors
        logger.error("Critical error, not retrying")
        raise
    except RetryableError as exc:
        # Custom retry logic
        logger.warning(f"Retryable error: {exc}")
        raise self.retry(exc=exc, countdown=60)
```

### 3. Monitoring & Logging

```python
# Add comprehensive logging
import logging
logger = logging.getLogger(__name__)

@celery_app.task(bind=True)
def monitored_task(self):
    logger.info(f"Task started: {self.request.id}")

    try:
        result = do_work()
        logger.info(f"Task completed: {self.request.id}")
        return result
    except Exception as e:
        logger.error(f"Task failed: {self.request.id}, Error: {e}")
        raise
```

### 4. Resource Management

```python
# Clean up resources properly
@celery_app.task
def resource_intensive_task():
    connection = None
    try:
        connection = create_connection()
        return process_with_connection(connection)
    finally:
        if connection:
            connection.close()
```

### 5. Testing

```python
# Test tasks synchronously during development
from celery.contrib.testing import worker_start

# Unit test
def test_deduplication_task():
    result = deduplication_task.apply(args=["user123", 0.85]).get()
    assert result['duplicates_found'] > 0

# Integration test
def test_task_with_worker():
    with worker_start(celery_app):
        result = deduplication_task.delay("user123", 0.85).get(timeout=10)
        assert result['success'] == True
```

---

## Related Documentation

- [SAAS_GUIDE.md](SAAS_GUIDE.md) - SaaS deployment and automation
- [MONITORING.md](MONITORING.md) - System-wide monitoring
- [API_REFERENCE.md](API_REFERENCE.md) - Complete API documentation
- [USER_GUIDE.md](USER_GUIDE.md) - Production deployment guide

### External Resources

- [Celery Documentation](https://docs.celeryproject.org/)
- [Flower Documentation](https://flower.readthedocs.io/)
- [Redis Documentation](https://redis.io/documentation)

---

**Built with ❤️ by the HippocampAI community**
