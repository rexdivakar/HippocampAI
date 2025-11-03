# Celery Optimization, Control & Tracing Guide

**Version**: V0.2.5
**Last Updated**: 2025-11-03

Complete guide to optimizing Celery background tasks, controlling worker behavior, and implementing comprehensive tracing and monitoring.

---

## 📋 Table of Contents

1. [Celery Architecture Overview](#celery-architecture-overview)
2. [Configuration & Optimization](#configuration--optimization)
3. [Worker Control & Management](#worker-control--management)
4. [Queue Management](#queue-management)
5. [Task Routing & Prioritization](#task-routing--prioritization)
6. [Performance Tuning](#performance-tuning)
7. [Monitoring & Tracing](#monitoring--tracing)
8. [Flower Dashboard](#flower-dashboard)
9. [Troubleshooting](#troubleshooting)
10. [Best Practices](#best-practices)

---

## 🏗️ Celery Architecture Overview

### Component Breakdown

```
┌─────────────────────────────────────────────────────────────┐
│                     HippocampAI Application                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   FastAPI   │  │   Client    │  │   Tasks     │         │
│  │   Server    │  │  Requests   │  │   Module    │         │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘         │
│         │                 │                 │                 │
└─────────┼─────────────────┼─────────────────┼────────────────┘
          │                 │                 │
          ▼                 ▼                 ▼
┌─────────────────────────────────────────────────────────────┐
│                      Message Broker (Redis)                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   Queue:    │  │   Queue:    │  │   Queue:    │         │
│  │   default   │  │ memory_ops  │  │  scheduled  │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────┘
          │                 │                 │
          ▼                 ▼                 ▼
┌─────────────────────────────────────────────────────────────┐
│                      Celery Workers                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │  Worker 1   │  │  Worker 2   │  │  Worker 3   │         │
│  │  (4 threads)│  │  (4 threads)│  │  (4 threads)│         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────┘
          │                 │                 │
          ▼                 ▼                 ▼
┌─────────────────────────────────────────────────────────────┐
│                    Result Backend (Redis)                    │
│              Task Results & State Storage                    │
└─────────────────────────────────────────────────────────────┘
```

### Queue Types

| Queue | Purpose | Routing Key | Priority |
|-------|---------|-------------|----------|
| **default** | General tasks | `default` | Normal |
| **memory_ops** | Memory CRUD operations | `memory.#` | High |
| **background** | Background maintenance | `background.#` | Low |
| **scheduled** | Periodic tasks (Beat) | `scheduled.#` | Low |

---

## ⚙️ Configuration & Optimization

### Environment Variables

Complete list of Celery configuration options:

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

## 🎮 Worker Control & Management

### Starting Workers

```bash
# Basic worker start
celery -A hippocampai.celery_app worker --loglevel=info

# With concurrency control
celery -A hippocampai.celery_app worker --concurrency=4 --loglevel=info

# With specific queues
celery -A hippocampai.celery_app worker \
  --queues=memory_ops,background \
  --loglevel=info

# With autoscaling
celery -A hippocampai.celery_app worker \
  --autoscale=10,3 \
  --loglevel=info
  # Max 10 processes, min 3 processes

# With pool type
celery -A hippocampai.celery_app worker \
  --pool=prefork \  # prefork (default), eventlet, gevent, solo
  --concurrency=4 \
  --loglevel=info
```

### Worker Control Commands

```bash
# Inspect active tasks
celery -A hippocampai.celery_app inspect active

# Inspect scheduled tasks
celery -A hippocampai.celery_app inspect scheduled

# Inspect registered tasks
celery -A hippocampai.celery_app inspect registered

# Worker statistics
celery -A hippocampai.celery_app inspect stats

# Check active queues
celery -A hippocampai.celery_app inspect active_queues

# Worker pool status
celery -A hippocampai.celery_app inspect pool

# Worker reserved tasks
celery -A hippocampai.celery_app inspect reserved
```

### Worker Management

```bash
# Graceful shutdown (finish current tasks)
celery -A hippocampai.celery_app control shutdown

# Restart workers
celery -A hippocampai.celery_app control pool_restart

# Enable/disable event monitoring
celery -A hippocampai.celery_app control enable_events
celery -A hippocampai.celery_app control disable_events

# Cancel consuming from queue
celery -A hippocampai.celery_app control cancel_consumer memory_ops

# Add consumer to queue
celery -A hippocampai.celery_app control add_consumer memory_ops

# Set concurrency
celery -A hippocampai.celery_app control pool_grow 2  # Add 2 workers
celery -A hippocampai.celery_app control pool_shrink 1 # Remove 1 worker

# Set rate limit
celery -A hippocampai.celery_app control rate_limit hippocampai.tasks.create_memory_task 10/m
```

### Monitoring Workers

```bash
# Events monitoring (real-time)
celery -A hippocampai.celery_app events

# Events snapshot
celery -A hippocampai.celery_app events --dump

# Task execution report
celery -A hippocampai.celery_app report
```

---

## 📊 Queue Management

### Queue Configuration

Queues are defined in `src/hippocampai/celery_app.py`:

```python
celery_app.conf.task_queues = (
    Queue("default", Exchange("default"), routing_key="default"),
    Queue("memory_ops", Exchange("memory_ops"), routing_key="memory.#"),
    Queue("background", Exchange("background"), routing_key="background.#"),
    Queue("scheduled", Exchange("scheduled"), routing_key="scheduled.#"),
)
```

### Checking Queue Length

```python
from hippocampai.celery_app import celery_app

# Get queue lengths
inspector = celery_app.control.inspect()
active = inspector.active()
scheduled = inspector.scheduled()
reserved = inspector.reserved()

print(f"Active tasks: {len(active) if active else 0}")
print(f"Scheduled tasks: {len(scheduled) if scheduled else 0}")
print(f"Reserved tasks: {len(reserved) if reserved else 0}")
```

Using Redis CLI:
```bash
# Connect to Redis
redis-cli

# Check queue lengths
LLEN celery

# Peek at queue items
LRANGE celery 0 10

# Check all Celery keys
KEYS celery*
```

### Purging Queues

```bash
# Purge all tasks from all queues
celery -A hippocampai.celery_app purge

# Purge specific queue
celery -A hippocampai.celery_app purge -Q memory_ops

# Force purge (no confirmation)
celery -A hippocampai.celery_app purge -f
```

---

## 🎯 Task Routing & Prioritization

### Task Routes

Configured in `src/hippocampai/celery_app.py`:

```python
celery_app.conf.task_routes = {
    # Memory operations → memory_ops queue
    "hippocampai.tasks.create_memory_task": {"queue": "memory_ops"},
    "hippocampai.tasks.batch_create_memories_task": {"queue": "memory_ops"},
    "hippocampai.tasks.recall_memories_task": {"queue": "memory_ops"},
    "hippocampai.tasks.update_memory_task": {"queue": "memory_ops"},
    "hippocampai.tasks.delete_memory_task": {"queue": "memory_ops"},

    # Scheduled maintenance → scheduled queue
    "hippocampai.tasks.deduplicate_all_memories": {"queue": "scheduled"},
    "hippocampai.tasks.consolidate_all_memories": {"queue": "scheduled"},
    "hippocampai.tasks.cleanup_expired_memories": {"queue": "scheduled"},
    "hippocampai.tasks.decay_memory_importance": {"queue": "scheduled"},

    # Background tasks → background queue
    "hippocampai.tasks.health_check_task": {"queue": "background"},
}
```

### Custom Task Routing

```python
from hippocampai.celery_app import celery_app

# Send task to specific queue
task = celery_app.send_task(
    "hippocampai.tasks.create_memory_task",
    args=["I love Python", "alice"],
    queue="memory_ops",
    routing_key="memory.create"
)

# Set task priority (0-9, higher = more important)
task = celery_app.send_task(
    "hippocampai.tasks.create_memory_task",
    args=["Urgent memory", "alice"],
    queue="memory_ops",
    priority=9
)

# Set task expiration
from datetime import timedelta
task = celery_app.send_task(
    "hippocampai.tasks.create_memory_task",
    args=["Temporary task", "alice"],
    expires=timedelta(minutes=5)
)
```

### Priority Queue Pattern

```python
# Define priority queues
from kombu import Queue, Exchange

high_priority_queue = Queue(
    "high_priority",
    Exchange("priority"),
    routing_key="high",
    queue_arguments={"x-max-priority": 10}
)

normal_priority_queue = Queue(
    "normal_priority",
    Exchange("priority"),
    routing_key="normal",
    queue_arguments={"x-max-priority": 5}
)

# Submit with priority
task = celery_app.send_task(
    "hippocampai.tasks.create_memory_task",
    args=["High priority memory", "alice"],
    queue="high_priority",
    priority=9
)
```

---

## 🚀 Performance Tuning

### Concurrency Models

#### 1. Prefork (Default - Best for CPU-bound)
```bash
# Multi-process workers
celery -A hippocampai.celery_app worker \
  --pool=prefork \
  --concurrency=4 \
  --loglevel=info
```

**Pros:**
- True parallel execution
- Isolated processes
- Best for CPU-intensive tasks

**Cons:**
- Higher memory usage
- Process startup overhead

#### 2. Eventlet (Best for I/O-bound)
```bash
# Install eventlet
pip install eventlet

# Start with eventlet
celery -A hippocampai.celery_app worker \
  --pool=eventlet \
  --concurrency=100 \
  --loglevel=info
```

**Pros:**
- Handle thousands of concurrent tasks
- Low memory overhead
- Great for API calls, DB queries

**Cons:**
- Not for CPU-intensive tasks
- Greenlet limitations

#### 3. Gevent (Alternative for I/O-bound)
```bash
# Install gevent
pip install gevent

# Start with gevent
celery -A hippocampai.celery_app worker \
  --pool=gevent \
  --concurrency=100 \
  --loglevel=info
```

### Worker Autoscaling

```bash
# Dynamic worker scaling
celery -A hippocampai.celery_app worker \
  --autoscale=10,3 \
  --loglevel=info

# Scale based on queue length
celery -A hippocampai.celery_app worker \
  --autoscale=10,2 \
  --max-tasks-per-child=100 \
  --loglevel=info
```

Autoscaling configuration in code:
```python
celery_app.conf.update(
    worker_autoscaler='celery.worker.autoscale:Autoscaler',
    worker_max_tasks_per_child=1000,
    worker_max_memory_per_child=200000,  # 200MB
)
```

### Task Execution Optimization

#### Retry Configuration
```python
from hippocampai.celery_app import celery_app

@celery_app.task(
    bind=True,
    max_retries=3,
    default_retry_delay=60,  # 60 seconds
    autoretry_for=(Exception,),
    retry_backoff=True,       # Exponential backoff
    retry_backoff_max=600,    # Max 10 minutes
    retry_jitter=True         # Add random jitter
)
def my_task(self, *args, **kwargs):
    try:
        # Task logic
        pass
    except Exception as exc:
        # Manual retry with custom countdown
        raise self.retry(exc=exc, countdown=120)
```

#### Task Result Configuration
```python
celery_app.conf.update(
    # Result expiration
    result_expires=3600,  # 1 hour

    # Result compression
    result_compression='gzip',

    # Result serialization
    result_serializer='json',

    # Extended task result
    result_extended=True,  # Include args, kwargs in result

    # Ignore results for certain tasks
    task_ignore_result=False,
)

# Per-task result configuration
@celery_app.task(ignore_result=True)  # Don't store result
def fire_and_forget_task():
    pass
```

### Memory Management

```python
celery_app.conf.update(
    # Restart worker after N tasks (prevent memory leaks)
    worker_max_tasks_per_child=1000,

    # Restart worker when memory exceeds limit
    worker_max_memory_per_child=200000,  # 200MB in KB

    # Prefetch settings
    worker_prefetch_multiplier=4,

    # Task acknowledgment
    task_acks_late=True,  # Ack after completion
    task_reject_on_worker_lost=True,
)
```

### Redis Optimization

```bash
# Redis configuration for Celery
redis-server \
  --maxmemory 2gb \
  --maxmemory-policy allkeys-lru \
  --appendonly yes \
  --save 900 1 \
  --save 300 10 \
  --tcp-backlog 511 \
  --timeout 300
```

Redis connection pooling:
```python
celery_app.conf.update(
    broker_transport_options={
        'visibility_timeout': 3600,  # 1 hour
        'max_connections': 100,
        'socket_connect_timeout': 5,
        'socket_keepalive': True,
        'retry_on_timeout': True,
    },

    redis_max_connections=100,
    redis_socket_connect_timeout=5,
    redis_socket_keepalive=True,
)
```

---

## 📈 Monitoring & Tracing

### Built-in Celery Monitoring

#### Events Monitoring
```python
from celery import Celery
from celery.events import EventReceiver

app = Celery('hippocampai')

def monitor_events():
    state = app.events.State()

    with app.connection() as connection:
        recv = EventReceiver(connection, handlers={
            'task-sent': lambda event: print(f"Task sent: {event['uuid']}"),
            'task-received': lambda event: print(f"Task received: {event['uuid']}"),
            'task-started': lambda event: print(f"Task started: {event['uuid']}"),
            'task-succeeded': lambda event: print(f"Task succeeded: {event['uuid']}"),
            'task-failed': lambda event: print(f"Task failed: {event['uuid']}: {event['exception']}"),
            'task-retried': lambda event: print(f"Task retried: {event['uuid']}"),
        })
        recv.capture(limit=None, timeout=None, wakeup=True)

# Run in background thread
import threading
monitor_thread = threading.Thread(target=monitor_events, daemon=True)
monitor_thread.start()
```

### Task Execution Tracing

#### Custom Task Tracing
```python
from celery.signals import (
    task_prerun, task_postrun, task_success,
    task_failure, task_retry, task_revoked
)
import time
import logging

logger = logging.getLogger(__name__)

# Track task execution time
task_start_times = {}

@task_prerun.connect
def task_prerun_handler(sender=None, task_id=None, task=None, args=None, kwargs=None, **extra):
    """Called before task execution."""
    task_start_times[task_id] = time.time()
    logger.info(f"Task {task.name}[{task_id}] starting with args={args}, kwargs={kwargs}")

@task_postrun.connect
def task_postrun_handler(sender=None, task_id=None, task=None, args=None, kwargs=None,
                         retval=None, **extra):
    """Called after task execution."""
    duration = time.time() - task_start_times.pop(task_id, time.time())
    logger.info(f"Task {task.name}[{task_id}] completed in {duration:.2f}s")

@task_success.connect
def task_success_handler(sender=None, result=None, **kwargs):
    """Called when task succeeds."""
    logger.info(f"Task {sender.name} succeeded with result: {result}")

@task_failure.connect
def task_failure_handler(sender=None, task_id=None, exception=None, args=None,
                        kwargs=None, traceback=None, **extra):
    """Called when task fails."""
    logger.error(f"Task {sender.name}[{task_id}] failed: {exception}")
    logger.error(f"Traceback: {traceback}")

@task_retry.connect
def task_retry_handler(sender=None, task_id=None, reason=None, **kwargs):
    """Called when task is retried."""
    logger.warning(f"Task {sender.name}[{task_id}] retrying: {reason}")
```

### Metrics Collection

#### Built-in Monitoring: Flower Dashboard

**Status**: ✅ Fully implemented and running

The primary monitoring tool for Celery tasks is **Flower**, which is included in docker-compose:

**Access**: `http://localhost:5555`
**Login**: `admin:admin` (configurable via `FLOWER_USER` and `FLOWER_PASSWORD`)

**Features:**
- Real-time task monitoring
- Worker status and performance metrics
- Task history and statistics
- Queue lengths and backlog
- Worker concurrency and pool info
- Task success/failure rates
- Task duration tracking
- Broker and result backend monitoring

**API Endpoints:**
```bash
# Get task statistics
curl http://admin:admin@localhost:5555/api/tasks

# Get worker information
curl http://admin:admin@localhost:5555/api/workers

# Get task details
curl http://admin:admin@localhost:5555/api/task/info/{task_id}
```

---

#### Prometheus Metrics (Optional Enhancement)

**Status**: ⚠️ Infrastructure configured, metrics export not yet implemented

While Prometheus and Grafana containers are configured in docker-compose, the metrics exporter is not yet integrated into the Python codebase.

**To implement Prometheus metrics** (if needed for your use case):

1. **Install prometheus_client:**
```bash
pip install prometheus-client
```

2. **Add metrics to your Celery signals:**
```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server
from celery.signals import task_prerun, task_postrun, task_failure

# Define metrics
task_sent_counter = Counter(
    'celery_task_sent_total',
    'Total tasks sent',
    ['task_name', 'queue']
)

task_succeeded_counter = Counter(
    'celery_task_succeeded_total',
    'Total successful tasks',
    ['task_name']
)

task_failed_counter = Counter(
    'celery_task_failed_total',
    'Total failed tasks',
    ['task_name', 'exception_type']
)

task_duration_histogram = Histogram(
    'celery_task_duration_seconds',
    'Task execution duration',
    ['task_name']
)

active_tasks_gauge = Gauge(
    'celery_active_tasks',
    'Currently executing tasks',
    ['task_name']
)

# Connect to signals
task_execution_times = {}

@task_prerun.connect
def record_task_start(sender=None, task_id=None, task=None, **kwargs):
    task_execution_times[task_id] = time.time()
    active_tasks_gauge.labels(task_name=task.name).inc()

@task_postrun.connect
def record_task_complete(sender=None, task_id=None, task=None, **kwargs):
    duration = time.time() - task_execution_times.pop(task_id, time.time())
    task_duration_histogram.labels(task_name=task.name).observe(duration)
    active_tasks_gauge.labels(task_name=task.name).dec()

@task_failure.connect
def record_task_failure(sender=None, task_id=None, exception=None, **kwargs):
    exception_type = type(exception).__name__
    task_failed_counter.labels(
        task_name=sender.name,
        exception_type=exception_type
    ).inc()

# Start Prometheus server
start_http_server(8001)
```

3. **Add `/metrics` endpoint to FastAPI** (in `src/hippocampai/api/async_app.py`):
```python
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from fastapi import Response

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
```

**Until then, use Flower for comprehensive Celery monitoring.**

### Task Result Tracking

```python
from hippocampai.celery_app import celery_app
from celery.result import AsyncResult

class TaskTracker:
    """Track task execution and results."""

    def __init__(self):
        self.tracked_tasks = {}

    def submit_and_track(self, task_name, *args, **kwargs):
        """Submit task and track its execution."""
        task = celery_app.send_task(task_name, args=args, kwargs=kwargs)

        self.tracked_tasks[task.id] = {
            'task_name': task_name,
            'submitted_at': time.time(),
            'args': args,
            'kwargs': kwargs,
            'result': task
        }

        return task.id

    def get_task_status(self, task_id):
        """Get comprehensive task status."""
        if task_id not in self.tracked_tasks:
            return None

        tracked = self.tracked_tasks[task_id]
        result = tracked['result']

        status = {
            'task_id': task_id,
            'task_name': tracked['task_name'],
            'status': result.status,
            'submitted_at': tracked['submitted_at'],
            'duration': time.time() - tracked['submitted_at'],
        }

        if result.ready():
            if result.successful():
                status['result'] = result.result
                status['success'] = True
            else:
                status['error'] = str(result.info)
                status['success'] = False

        return status

    def get_all_pending(self):
        """Get all pending tasks."""
        return [
            task_id
            for task_id, tracked in self.tracked_tasks.items()
            if not tracked['result'].ready()
        ]

    def cleanup_completed(self, older_than_seconds=3600):
        """Remove completed tasks older than threshold."""
        now = time.time()
        to_remove = [
            task_id
            for task_id, tracked in self.tracked_tasks.items()
            if tracked['result'].ready() and
               (now - tracked['submitted_at']) > older_than_seconds
        ]

        for task_id in to_remove:
            del self.tracked_tasks[task_id]

        return len(to_remove)

# Usage
tracker = TaskTracker()

# Submit and track
task_id = tracker.submit_and_track(
    "hippocampai.tasks.create_memory_task",
    "I love Python",
    "alice"
)

# Check status
status = tracker.get_task_status(task_id)
print(f"Task status: {status}")

# Get pending
pending = tracker.get_all_pending()
print(f"Pending tasks: {len(pending)}")

# Cleanup old tasks
removed = tracker.cleanup_completed(older_than_seconds=1800)
print(f"Cleaned up {removed} old tasks")
```

---

## 🌺 Flower Dashboard

### Accessing Flower

**URL**: `http://localhost:5555`
**Default Login**: `admin:admin` (configured via environment variables)

### Flower Features

#### 1. **Dashboard Overview**
- Active tasks count
- Processed tasks count
- Failed tasks count
- Worker status

#### 2. **Tasks View**
- Real-time task monitoring
- Task history
- Task details (args, kwargs, result)
- Task timeline
- Retry information

#### 3. **Workers View**
- Worker list and status
- Worker pool info
- Active tasks per worker
- Worker statistics
- Shutdown/restart workers

#### 4. **Broker View**
- Queue lengths
- Message rates
- Connections

#### 5. **API Access**

```python
import requests

# Get task info
response = requests.get("http://localhost:5555/api/task/info/TASK-ID")
task_info = response.json()

# Get task list
response = requests.get("http://localhost:5555/api/tasks")
tasks = response.json()

# Get worker info
response = requests.get("http://localhost:5555/api/workers")
workers = response.json()

# Revoke task
response = requests.post(
    "http://localhost:5555/api/task/revoke/TASK-ID",
    data={'terminate': True}
)
```

### Custom Flower Configuration

```python
# flower_config.py
broker_api = "redis://localhost:6379/1"
max_tasks = 10000
max_workers = 100
enable_events = True
persistent = True
db = "/app/flower.db"
port = 5555
```

Start Flower with custom config:
```bash
celery -A hippocampai.celery_app flower --conf=flower_config.py
```

---

## 🔧 Troubleshooting

### Common Issues

#### 1. Tasks Stuck in PENDING

**Symptoms:**
- Tasks never start executing
- Status remains PENDING

**Solutions:**
```bash
# Check if workers are running
celery -A hippocampai.celery_app inspect active

# Check if workers are consuming from correct queue
celery -A hippocampai.celery_app inspect active_queues

# Restart workers
docker-compose restart celery-worker

# Purge stuck tasks
celery -A hippocampai.celery_app purge
```

#### 2. High Memory Usage

**Solutions:**
```bash
# Reduce prefetch multiplier
export CELERY_WORKER_PREFETCH_MULTIPLIER=1

# Enable max tasks per child
export CELERY_WORKER_MAX_TASKS_PER_CHILD=100

# Use eventlet for I/O-bound tasks
celery -A hippocampai.celery_app worker --pool=eventlet --concurrency=100
```

#### 3. Slow Task Execution

**Diagnosis:**
```python
# Enable task execution logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check task duration in Flower
# Look for tasks taking longer than expected
```

**Solutions:**
- Increase worker concurrency
- Use autoscaling
- Optimize task code
- Check database/Redis connection pooling

#### 4. Tasks Failing with Timeout

**Solutions:**
```python
# Increase task time limits
celery_app.conf.update(
    task_soft_time_limit=600,  # 10 minutes
    task_time_limit=1200,      # 20 minutes
)

# Per-task timeout
@celery_app.task(time_limit=300, soft_time_limit=240)
def long_running_task():
    pass
```

#### 5. Connection Errors

**Check Redis:**
```bash
# Test Redis connection
redis-cli ping

# Check Redis connections
redis-cli CLIENT LIST | grep celery

# Check Redis memory
redis-cli INFO memory
```

**Check Celery connection:**
```python
from hippocampai.celery_app import celery_app

# Test connection
celery_app.connection().ensure_connection(max_retries=3)
```

---

## ✅ Best Practices

### 1. Task Design

```python
# ✅ Good: Idempotent task
@celery_app.task(bind=True)
def create_memory_idempotent(self, text, user_id, idempotency_key):
    # Check if already processed
    if redis.exists(f"processed:{idempotency_key}"):
        return redis.get(f"processed:{idempotency_key}")

    # Process
    result = process_memory(text, user_id)

    # Store result
    redis.setex(f"processed:{idempotency_key}", 3600, result)
    return result

# ❌ Bad: Not idempotent (duplicate processing possible)
@celery_app.task
def create_memory_not_idempotent(text, user_id):
    return process_memory(text, user_id)
```

### 2. Error Handling

```python
# ✅ Good: Comprehensive error handling
@celery_app.task(bind=True, max_retries=3)
def robust_task(self, data):
    try:
        result = process_data(data)
        return result
    except TemporaryError as exc:
        # Retry on temporary errors
        raise self.retry(exc=exc, countdown=60)
    except PermanentError as exc:
        # Log and fail on permanent errors
        logger.error(f"Permanent error: {exc}")
        raise
    except Exception as exc:
        # Catch-all with logging
        logger.exception("Unexpected error")
        raise self.retry(exc=exc, countdown=30, max_retries=1)
```

### 3. Task Monitoring

```python
# ✅ Good: Comprehensive monitoring
@celery_app.task(bind=True)
def monitored_task(self, data):
    # Update progress
    self.update_state(
        state='PROGRESS',
        meta={'current': 0, 'total': 100}
    )

    try:
        for i in range(100):
            # Process
            process_item(i)

            # Update progress
            if i % 10 == 0:
                self.update_state(
                    state='PROGRESS',
                    meta={'current': i, 'total': 100}
                )

        return {'status': 'complete', 'processed': 100}
    except Exception as exc:
        self.update_state(
            state='FAILURE',
            meta={'exc': str(exc)}
        )
        raise
```

### 4. Resource Management

```python
# ✅ Good: Proper resource cleanup
@celery_app.task
def task_with_resources():
    connection = None
    try:
        connection = get_database_connection()
        result = process_with_connection(connection)
        return result
    finally:
        if connection:
            connection.close()

# Use context managers
@celery_app.task
def task_with_context_manager():
    with get_database_connection() as conn:
        return process_with_connection(conn)
```

### 5. Batch Processing

```python
# ✅ Good: Batch similar operations
@celery_app.task
def batch_create_memories(memories_data):
    # Process in batches of 100
    batch_size = 100
    results = []

    for i in range(0, len(memories_data), batch_size):
        batch = memories_data[i:i+batch_size]
        batch_results = bulk_insert_memories(batch)
        results.extend(batch_results)

    return results

# ❌ Bad: One task per item
for memory_data in memories_data:
    create_memory_task.delay(memory_data)  # Creates many tasks
```

### 6. Task Chaining

```python
from celery import chain, group, chord

# Sequential execution
result = chain(
    task1.s(arg1),
    task2.s(),
    task3.s()
).apply_async()

# Parallel execution
result = group(
    task1.s(arg1),
    task2.s(arg2),
    task3.s(arg3)
).apply_async()

# Parallel then aggregate
result = chord(
    group(task1.s(i) for i in range(10)),
    aggregate_task.s()
).apply_async()
```

---

## 📊 Performance Metrics

### Key Performance Indicators

1. **Task Throughput**: Tasks completed per second
2. **Task Latency**: Time from submission to completion
3. **Queue Length**: Number of waiting tasks
4. **Worker Utilization**: % of workers actively processing
5. **Error Rate**: Failed tasks / Total tasks
6. **Retry Rate**: Retried tasks / Total tasks

### Monitoring Query Examples

```python
# Get metrics from Flower API
import requests

response = requests.get("http://localhost:5555/api/workers")
workers = response.json()

for worker_name, worker_info in workers.items():
    stats = worker_info.get('stats', {})
    print(f"Worker: {worker_name}")
    print(f"  Active tasks: {len(worker_info.get('active', []))}")
    print(f"  Total processed: {stats.get('total', 0)}")
    print(f"  Pool processes: {stats.get('pool', {}).get('max-concurrency', 0)}")
```

---

## 📝 Summary

### Quick Reference Card

| Task | Command |
|------|---------|
| Start worker | `celery -A hippocampai.celery_app worker --loglevel=info` |
| Start beat | `celery -A hippocampai.celery_app beat --loglevel=info` |
| Start flower | `celery -A hippocampai.celery_app flower` |
| Check active tasks | `celery -A hippocampai.celery_app inspect active` |
| Get worker stats | `celery -A hippocampai.celery_app inspect stats` |
| Purge queues | `celery -A hippocampai.celery_app purge` |
| Shutdown workers | `celery -A hippocampai.celery_app control shutdown` |

### Recommended Settings

| Environment | Concurrency | Prefetch | Max Tasks/Child |
|-------------|-------------|----------|-----------------|
| Development | 2 | 2 | 100 |
| Production (Low) | 4 | 4 | 1000 |
| Production (High) | 8 | 2 | 500 |
| I/O Heavy | 100 (eventlet) | 10 | 1000 |

---

**Last Updated**: 2025-11-03
**Version**: V0.2.5
**Status**: Production Ready
