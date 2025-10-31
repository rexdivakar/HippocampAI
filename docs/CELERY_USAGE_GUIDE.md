# Celery & Flower Integration Guide

## Overview

HippocampAI now includes **Celery** for distributed task processing and **Flower** for real-time monitoring. This enables:

- **Asynchronous background tasks** - Long-running operations don't block API responses
- **Scheduled jobs** - Automatic deduplication, consolidation, and cleanup
- **Distributed processing** - Scale workers independently from the API
- **Task monitoring** - Beautiful Flower UI to track all tasks in real-time
- **Retry logic** - Automatic retries for failed tasks
- **Task queues** - Separate queues for different priority levels

---

## Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                    USER / APPLICATION                           │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    │
│  │ FastAPI      │    │ Python SDK   │    │ Direct Task  │    │
│  │ Async Routes │    │ client.py    │    │ Submission   │    │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘    │
└─────────┼────────────────────┼────────────────────┼────────────┘
          │                    │                    │
          └────────────────────┼────────────────────┘
                               ↓
┌──────────────────────────────────────────────────────────────────┐
│              REDIS (Message Broker & Result Backend)             │
│  • DB 1: Task Queue (broker)                                     │
│  • DB 2: Task Results (backend)                                  │
└────────────────────┬─────────────────────────────────────────────┘
                     │
          ┌──────────┴──────────┬──────────────┬─────────────┐
          ↓                     ↓              ↓             ↓
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
                               ↓                                    │
         ┌──────────────────────────────────────────┐              │
         │   HIPPOCAMPAI SERVICES                   │              │
         │   • MemoryManagementService              │              │
         │   • QdrantStore (Vector DB)              │              │
         │   • Embedder (BGE-Small)                 │              │
         │   • Reranker (Cross-Encoder)             │              │
         └──────────────────────────────────────────┘              │
                               │                                    │
         ┌─────────────────────┴────────────────┐                  │
         ↓                                      ↓                   │
┌──────────────────┐                   ┌──────────────────┐        │
│ QDRANT           │                   │ REDIS (Cache)    │        │
│ Vector Storage   │                   │ DB 0             │        │
└──────────────────┘                   └──────────────────┘        │
                                                                    │
         ┌──────────────────────────────────────────────────────────┘
         ↓
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
# - hippocampai-qdrant (vector DB) on port 6333
# - hippocampai-redis (broker/cache) on port 6379
```

### 2. Access Flower UI

Open your browser:

```
http://localhost:5555
```

**Default credentials:**

- Username: `admin`
- Password: `admin` (change in `.env` file via `FLOWER_PASSWORD`)

---

## Task Queues

HippocampAI uses 4 separate queues for different task types:

| Queue | Purpose | Priority | Examples |
|-------|---------|----------|----------|
| **memory_ops** | Memory CRUD operations | High | create_memory, batch_create, recall |
| **scheduled** | Periodic maintenance tasks | Medium | deduplication, consolidation, decay |
| **background** | Health checks & monitoring | Low | health_check_task |
| **default** | General tasks | Normal | Everything else |

---

## Using Celery Tasks

### Method 1: Via FastAPI Async Endpoints

Submit tasks asynchronously and track their status:

```bash
# Create a memory asynchronously
curl -X POST http://localhost:8000/api/v1/tasks/memory/create \
  -H "Content-Type: application/json" \
  -d '{
    "text": "I prefer dark chocolate over milk chocolate",
    "user_id": "alice",
    "memory_type": "preference",
    "importance": 8.0,
    "tags": ["food", "preferences"]
  }'

# Response:
# {
#   "task_id": "a3f5b2c1-4d8e-9f0a-1b2c-3d4e5f6g7h8i",
#   "status": "submitted",
#   "message": "Memory creation task submitted..."
# }

# Check task status
curl http://localhost:8000/api/v1/tasks/status/a3f5b2c1-4d8e-9f0a-1b2c-3d4e5f6g7h8i

# Response when completed:
# {
#   "task_id": "a3f5b2c1-4d8e-9f0a-1b2c-3d4e5f6g7h8i",
#   "status": "SUCCESS",
#   "result": {
#     "id": "mem_123",
#     "text": "I prefer dark chocolate...",
#     "user_id": "alice",
#     "importance": 8.0
#   }
# }
```

### Method 2: Direct Task Submission (Python)

```python
from hippocampai.tasks import create_memory_task

# Submit task to Celery
task = create_memory_task.delay(
    text="Paris is the capital of France",
    user_id="bob",
    memory_type="fact",
    importance=7.0,
    tags=["geography", "france"]
)

# Get task ID
print(f"Task ID: {task.id}")

# Check status
print(f"Status: {task.status}")

# Wait for result (blocking)
result = task.get(timeout=30)
print(f"Created memory: {result}")
```

### Method 3: Batch Operations

```python
from hippocampai.tasks import batch_create_memories_task

memories = [
    {"text": "Memory 1", "user_id": "alice", "type": "fact"},
    {"text": "Memory 2", "user_id": "alice", "type": "preference"},
    {"text": "Memory 3", "user_id": "alice", "type": "goal"},
]

# Submit batch task (much faster than individual tasks)
task = batch_create_memories_task.delay(
    memories=memories,
    check_duplicates=True
)

# This uses QdrantStore.bulk_upsert() which is 3-5x faster
result = task.get(timeout=60)
print(f"Created {len(result)} memories")
```

---

## Scheduled Tasks (Celery Beat)

These tasks run automatically on a schedule:

### 1. **Auto-Deduplication** (Every 24 hours)

```bash
# Runs: deduplicate_all_memories
# When: Every 24 hours (configurable via DEDUP_INTERVAL_HOURS)
# What: Scans for and removes duplicate memories across all users
```

### 2. **Memory Consolidation** (Weekly)

```bash
# Runs: consolidate_all_memories
# When: Every 168 hours / 7 days (configurable)
# What: Merges related memories to reduce redundancy
# Enable: Set AUTO_CONSOLIDATION_ENABLED=true
```

### 3. **Expired Memory Cleanup** (Hourly)

```bash
# Runs: cleanup_expired_memories
# When: Every 1 hour (configurable)
# What: Deletes memories that have passed their TTL
```

### 4. **Importance Decay** (Daily at 2am)

```bash
# Runs: decay_memory_importance
# When: Daily at 2:00 AM UTC
# What: Applies time-based decay to memory importance scores
```

### 5. **Collection Snapshots** (Hourly)

```bash
# Runs: create_collection_snapshots
# When: Every hour
# What: Creates Qdrant collection snapshots for backup
```

### 6. **Health Check** (Every 5 minutes)

```bash
# Runs: health_check_task
# When: Every 5 minutes
# What: Verifies Qdrant, Redis, and Embedder are healthy
```

### Manually Trigger Scheduled Tasks

```bash
# Via API
curl -X POST http://localhost:8000/api/v1/tasks/scheduled/auto-deduplicate-memories/run

# Via Python
from hippocampai.tasks import deduplicate_all_memories

task = deduplicate_all_memories.delay()
print(f"Task ID: {task.id}")
```

---

## Monitoring with Flower

### Dashboard Features

**1. Workers Tab**

- Shows all active workers
- CPU & memory usage per worker
- Number of completed/failed tasks
- Worker uptime

**2. Tasks Tab**

- Real-time task execution
- Task success/failure rates
- Average execution time
- Task arguments and results

**3. Monitor Tab**

- Live task stream
- See tasks as they're executed
- Color-coded by status (success/failure/retry)

**4. Broker Tab**

- Queue lengths
- Messages per queue
- Broker health status

### Flower API

```bash
# Get worker stats (JSON)
curl http://localhost:5555/api/workers

# Get task info
curl http://localhost:5555/api/task/info/<task-id>

# List active tasks
curl http://localhost:5555/api/tasks
```

---

## Configuration

### Environment Variables (.env)

```bash
# Celery Broker & Backend
CELERY_BROKER_URL=redis://redis:6379/1
CELERY_RESULT_BACKEND=redis://redis:6379/2

# Worker Settings
CELERY_WORKER_CONCURRENCY=4              # Threads per worker
CELERY_WORKER_PREFETCH_MULTIPLIER=4      # Tasks to prefetch
CELERY_TASK_ACKS_LATE=true               # Ack after completion
CELERY_WORKER_MAX_TASKS_PER_CHILD=1000   # Restart worker after N tasks

# Flower UI
FLOWER_PORT=5555
FLOWER_USER=admin
FLOWER_PASSWORD=changeme_in_production

# Scheduled Task Controls
AUTO_DEDUP_ENABLED=true
AUTO_CONSOLIDATION_ENABLED=false
DEDUP_INTERVAL_HOURS=24
CONSOLIDATION_INTERVAL_HOURS=168
EXPIRATION_INTERVAL_HOURS=1
```

---

## Scaling Workers

### Add More Workers

```bash
# Start additional worker containers
docker-compose up -d --scale celery-worker=3

# This creates:
# - hippocampai-celery-worker-1
# - hippocampai-celery-worker-2
# - hippocampai-celery-worker-3
```

### Queue-Specific Workers

Run workers that only process specific queues:

```bash
# Worker for only memory operations
celery -A hippocampai.celery_app worker \
  --loglevel=info \
  --queues=memory_ops \
  --concurrency=8

# Worker for only scheduled tasks
celery -A hippocampai.celery_app worker \
  --loglevel=info \
  --queues=scheduled \
  --concurrency=2
```

---

## Production Best Practices

### 1. **Resource Limits**

```yaml
# docker-compose.yml
celery-worker:
  deploy:
    resources:
      limits:
        cpus: '2.0'
        memory: 4G
      reservations:
        cpus: '1.0'
        memory: 2G
```

### 2. **Worker Autoscaling**

```bash
celery -A hippocampai.celery_app worker \
  --autoscale=10,3  # Max 10, min 3 worker processes
```

### 3. **Task Time Limits**

Already configured in `celery_app.py`:

```python
task_soft_time_limit=300   # 5 minutes (raises exception)
task_time_limit=600        # 10 minutes (kills process)
```

### 4. **Result Expiration**

```python
result_expires=3600  # Results kept for 1 hour
```

### 5. **Task Retries**

Tasks auto-retry on failure:

```python
@celery_app.task(bind=True, max_retries=3)
def my_task(self):
    try:
        # ... task logic
    except Exception as exc:
        raise self.retry(exc=exc, countdown=60)  # Retry after 60s
```

---

## Troubleshooting

### Worker Not Processing Tasks

```bash
# Check if workers are running
docker-compose ps | grep celery

# View worker logs
docker logs hippocampai-celery-worker -f

# Inspect active workers
celery -A hippocampai.celery_app inspect active

# Purge all tasks (CAREFUL!)
celery -A hippocampai.celery_app purge
```

### Tasks Stuck in PENDING

```bash
# Check broker connection
docker exec hippocampai-redis redis-cli ping

# Check queue lengths
docker exec hippocampai-redis redis-cli -n 1 LLEN celery

# Restart workers
docker-compose restart celery-worker
```

### Flower Not Accessible

```bash
# Check Flower logs
docker logs hippocampai-flower -f

# Access Flower directly on container
docker exec hippocampai-flower curl localhost:5555/healthcheck

# Rebuild and restart
docker-compose up -d --build flower
```

---

## API Endpoints Summary

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/tasks/memory/create` | POST | Submit async memory creation |
| `/api/v1/tasks/memory/batch-create` | POST | Submit batch creation |
| `/api/v1/tasks/memory/recall` | POST | Submit async recall |
| `/api/v1/tasks/memory/update` | POST | Submit async update |
| `/api/v1/tasks/memory/delete` | POST | Submit async deletion |
| `/api/v1/tasks/status/{task_id}` | GET | Check task status |
| `/api/v1/tasks/cancel/{task_id}` | POST | Cancel running task |
| `/api/v1/tasks/inspect/stats` | GET | Get worker statistics |
| `/api/v1/tasks/inspect/queues` | GET | Get queue information |
| `/api/v1/tasks/scheduled` | GET | List scheduled tasks |
| `/api/v1/tasks/scheduled/{name}/run` | POST | Trigger scheduled task |

---

## Next Steps

1. **Monitor Flower Dashboard** - Watch your tasks in real-time
2. **Configure Schedules** - Adjust task intervals in `.env`
3. **Scale Workers** - Add more workers for higher throughput
4. **Create Custom Tasks** - Add your own tasks to `tasks.py`
5. **Set Up Alerts** - Integrate Flower with Prometheus/Grafana

---

## Support

- **Flower Documentation**: <https://flower.readthedocs.io/>
- **Celery Documentation**: <https://docs.celeryproject.org/>
- **HippocampAI Issues**: <https://github.com/rexdivakar/HippocampAI/issues>
