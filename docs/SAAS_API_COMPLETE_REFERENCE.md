# HippocampAI SaaS API Complete Reference

**Version**: V0.2.5
**Last Updated**: 2025-11-03

Complete reference for all HippocampAI REST API endpoints, monitoring interfaces, and task queue management.

---

## 📋 Table of Contents

1. [FastAPI Endpoints](#fastapi-endpoints)
2. [Celery Task Queue APIs](#celery-task-queue-apis)
3. [Intelligence APIs](#intelligence-apis)
4. [Monitoring & Observability](#monitoring--observability)
5. [Connection Guide](#connection-guide)
6. [API Authentication](#api-authentication)
7. [Error Handling](#error-handling)
8. [Rate Limiting](#rate-limiting)

---

## 🌐 Service Overview

HippocampAI provides multiple API layers:

| Service | Port | Purpose | URL |
|---------|------|---------|-----|
| **FastAPI** | 8000 | Core memory operations | `http://localhost:8000` |
| **Celery Tasks** | 8000 | Async task management | `http://localhost:8000/api/v1/tasks` |
| **Intelligence** | 8000 | Advanced AI features | `http://localhost:8000/v1/intelligence` |
| **Flower** | 5555 | Celery monitoring UI | `http://localhost:5555` |
| **Prometheus** | 9090 | Metrics collection | `http://localhost:9090` |
| **Grafana** | 3000 | Dashboards & visualization | `http://localhost:3000` |
| **Qdrant** | 6333 | Vector database | `http://localhost:6333` |
| **Redis** | 6379 | Cache & message broker | `redis://localhost:6379` |

---

## 🚀 FastAPI Endpoints

### Base URL: `http://localhost:8000`

### Health & Status

#### `GET /healthz`

Health check endpoint.

**Response:**
```json
{
  "status": "ok"
}
```

**Example:**
```bash
curl http://localhost:8000/healthz
```

---

### Core Memory Operations

#### `POST /v1/memories:remember`

Store a new memory.

**Request Body:**
```json
{
  "text": "I prefer oat milk in my coffee",
  "user_id": "alice",
  "session_id": "session_123",  // optional
  "type": "preference",          // fact, preference, goal, habit, event
  "importance": 8.0,             // optional, 0-10
  "tags": ["food", "beverages"], // optional
  "ttl_days": 30                 // optional
}
```

**Response:**
```json
{
  "id": "mem_abc123",
  "text": "I prefer oat milk in my coffee",
  "user_id": "alice",
  "session_id": "session_123",
  "type": "preference",
  "importance": 8.0,
  "tags": ["food", "beverages"],
  "created_at": "2025-11-03T10:00:00Z",
  "updated_at": "2025-11-03T10:00:00Z",
  "expires_at": "2025-12-03T10:00:00Z",
  "extracted_facts": ["beverage_preference: oat milk"],
  "text_length": 32,
  "token_count": 8
}
```

**Example:**
```python
import requests

response = requests.post(
    "http://localhost:8000/v1/memories:remember",
    json={
        "text": "I prefer oat milk in my coffee",
        "user_id": "alice",
        "type": "preference",
        "importance": 8.0,
        "tags": ["food", "beverages"]
    }
)
memory = response.json()
print(f"Created memory: {memory['id']}")
```

```bash
curl -X POST http://localhost:8000/v1/memories:remember \
  -H "Content-Type: application/json" \
  -d '{
    "text": "I prefer oat milk in my coffee",
    "user_id": "alice",
    "type": "preference",
    "importance": 8.0,
    "tags": ["food", "beverages"]
  }'
```

---

#### `POST /v1/memories:recall`

Retrieve relevant memories.

**Request Body:**
```json
{
  "query": "coffee preferences",
  "user_id": "alice",
  "session_id": "session_123",  // optional
  "k": 5,                        // number of results
  "filters": {                   // optional
    "type": "preference",
    "tags": ["food"]
  }
}
```

**Response:**
```json
[
  {
    "memory": {
      "id": "mem_abc123",
      "text": "I prefer oat milk in my coffee",
      "user_id": "alice",
      "type": "preference",
      "importance": 8.0,
      "tags": ["food", "beverages"],
      "created_at": "2025-11-03T10:00:00Z"
    },
    "score": 0.92,
    "breakdown": {
      "sim": 0.85,
      "rerank": 0.90,
      "recency": 0.95,
      "importance": 0.80
    }
  }
]
```

**Example:**
```python
response = requests.post(
    "http://localhost:8000/v1/memories:recall",
    json={
        "query": "coffee preferences",
        "user_id": "alice",
        "k": 5
    }
)
results = response.json()
for result in results:
    print(f"Memory: {result['memory']['text']}")
    print(f"Score: {result['score']:.2f}")
```

---

#### `POST /v1/memories:extract`

Extract memories from conversation.

**Request Body:**
```json
{
  "conversation": "User: I really enjoy drinking green tea in the morning.\nAssistant: That's great! Green tea is healthy.\nUser: Yes, and I usually have it without sugar.",
  "user_id": "bob",
  "session_id": "session_456"
}
```

**Response:**
```json
[
  {
    "id": "mem_def456",
    "text": "User enjoys green tea in the morning",
    "type": "preference",
    "extracted_facts": ["beverage_preference: green tea", "time_preference: morning"]
  },
  {
    "id": "mem_def457",
    "text": "User prefers tea without sugar",
    "type": "preference"
  }
]
```

---

#### `PATCH /v1/memories:update`

Update an existing memory.

**Request Body:**
```json
{
  "memory_id": "mem_abc123",
  "text": "I strongly prefer oat milk in my coffee",  // optional
  "importance": 9.0,                              // optional
  "tags": ["food", "beverages", "health"],        // optional
  "metadata": {"updated_reason": "user_feedback"},// optional
  "expires_at": "2025-12-31T23:59:59Z"           // optional
}
```

**Response:**
```json
{
  "id": "mem_abc123",
  "text": "I strongly prefer oat milk in my coffee",
  "importance": 9.0,
  "tags": ["food", "beverages", "health"],
  "updated_at": "2025-11-03T11:00:00Z"
}
```

---

#### `DELETE /v1/memories:delete`

Delete a memory.

**Request Body:**
```json
{
  "memory_id": "mem_abc123",
  "user_id": "alice"  // optional, for authorization
}
```

**Response:**
```json
{
  "success": true,
  "memory_id": "mem_abc123"
}
```

---

#### `POST /v1/memories:get`

Get memories with advanced filtering.

**Request Body:**
```json
{
  "user_id": "alice",
  "filters": {
    "type": "preference",
    "tags": ["food"],
    "min_importance": 7.0,
    "created_after": "2025-11-01T00:00:00Z"
  },
  "limit": 100
}
```

**Response:**
```json
[
  {
    "id": "mem_abc123",
    "text": "I prefer oat milk in my coffee",
    "type": "preference",
    "importance": 9.0,
    "tags": ["food", "beverages", "health"]
  }
]
```

---

#### `POST /v1/memories:expire`

Clean up expired memories.

**Request Body:**
```json
{
  "user_id": "alice"  // optional, if omitted cleans all users
}
```

**Response:**
```json
{
  "success": true,
  "expired_count": 15
}
```

---

## 🔄 Celery Task Queue APIs

### Base URL: `http://localhost:8000/api/v1/tasks`

All Celery operations are asynchronous. Submit a task and receive a `task_id` to track progress.

### Task Submission

#### `POST /api/v1/tasks/memory/create`

Create memory asynchronously.

**Request Body:**
```json
{
  "text": "I love Python programming",
  "user_id": "alice",
  "memory_type": "fact",
  "importance": 7.5,
  "tags": ["programming", "languages"],
  "metadata": {"source": "conversation"}
}
```

**Response:**
```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "submitted",
  "message": "Memory creation task submitted. Track with task_id: 550e8400-e29b-41d4-a716-446655440000"
}
```

---

#### `POST /api/v1/tasks/memory/batch-create`

Batch create memories asynchronously.

**Request Body:**
```json
{
  "memories": [
    {
      "text": "I love Python",
      "user_id": "alice",
      "type": "fact"
    },
    {
      "text": "I prefer dark mode",
      "user_id": "alice",
      "type": "preference"
    }
  ],
  "check_duplicates": true
}
```

**Response:**
```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440001",
  "status": "submitted",
  "message": "Batch memory creation task submitted for 2 memories. Track with task_id: 550e8400-e29b-41d4-a716-446655440001"
}
```

---

#### `POST /api/v1/tasks/memory/recall`

Recall memories asynchronously.

**Request Body:**
```json
{
  "query": "programming languages",
  "user_id": "alice",
  "k": 10,
  "filters": {"type": "fact"}
}
```

---

#### `POST /api/v1/tasks/memory/update`

Update memory asynchronously.

**Request Body:**
```json
{
  "memory_id": "mem_abc123",
  "user_id": "alice",
  "updates": {
    "text": "I absolutely love Python programming",
    "importance": 9.0
  }
}
```

---

#### `POST /api/v1/tasks/memory/delete`

Delete memory asynchronously.

**Request Body:**
```json
{
  "memory_id": "mem_abc123",
  "user_id": "alice"
}
```

---

### Task Management

#### `GET /api/v1/tasks/status/{task_id}`

Check task status and result.

**Response:**
```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "SUCCESS",  // PENDING, STARTED, SUCCESS, FAILURE, RETRY
  "result": {
    "id": "mem_abc123",
    "text": "I love Python programming",
    "user_id": "alice",
    "type": "fact",
    "importance": 7.5
  },
  "error": null,
  "progress": null
}
```

**Example:**
```python
# Submit task
response = requests.post("http://localhost:8000/api/v1/tasks/memory/create", json={...})
task_id = response.json()["task_id"]

# Check status
import time
while True:
    status_response = requests.get(f"http://localhost:8000/api/v1/tasks/status/{task_id}")
    status_data = status_response.json()

    if status_data["status"] in ["SUCCESS", "FAILURE"]:
        break
    time.sleep(1)

if status_data["status"] == "SUCCESS":
    print(f"Memory created: {status_data['result']}")
```

---

#### `POST /api/v1/tasks/cancel/{task_id}`

Cancel a running task.

**Response:**
```json
{
  "message": "Task 550e8400-e29b-41d4-a716-446655440000 cancelled",
  "status": "cancelled"
}
```

---

### Worker Inspection

#### `GET /api/v1/tasks/inspect/stats`

Get Celery worker statistics.

**Response:**
```json
{
  "active_tasks": {
    "celery@worker1": [
      {
        "id": "550e8400-e29b-41d4-a716-446655440000",
        "name": "hippocampai.tasks.create_memory_task",
        "args": ["I love Python", "alice"],
        "time_start": 1730635200.0
      }
    ]
  },
  "scheduled_tasks": {
    "celery@worker1": []
  },
  "registered_tasks": {
    "celery@worker1": [
      "hippocampai.tasks.create_memory_task",
      "hippocampai.tasks.batch_create_memories_task",
      "hippocampai.tasks.recall_memories_task"
    ]
  },
  "stats": {
    "celery@worker1": {
      "total": 1523,
      "pool": {
        "max-concurrency": 4,
        "processes": [1234, 1235, 1236, 1237]
      }
    }
  }
}
```

---

#### `GET /api/v1/tasks/inspect/queues`

Get queue information.

**Response:**
```json
{
  "active_queues": {
    "celery@worker1": [
      {
        "name": "memory_ops",
        "exchange": {"name": "memory_ops", "type": "direct"},
        "routing_key": "memory.#",
        "durable": true
      },
      {
        "name": "background",
        "exchange": {"name": "background", "type": "direct"},
        "routing_key": "background.#"
      }
    ]
  }
}
```

---

### Scheduled Tasks

#### `GET /api/v1/tasks/scheduled`

List all scheduled periodic tasks.

**Response:**
```json
{
  "scheduled_tasks": {
    "auto-deduplicate-memories": {
      "task": "hippocampai.tasks.deduplicate_all_memories",
      "schedule": "<crontab: */24 (hours)>",
      "options": {"queue": "scheduled"}
    },
    "cleanup-expired-memories": {
      "task": "hippocampai.tasks.cleanup_expired_memories",
      "schedule": "<crontab: */1 (hours)>",
      "options": {"queue": "scheduled"}
    },
    "decay-memory-importance": {
      "task": "hippocampai.tasks.decay_memory_importance",
      "schedule": "<crontab: 2 0 (hour, minute)>",
      "options": {"queue": "scheduled"}
    }
  }
}
```

---

#### `POST /api/v1/tasks/scheduled/{task_name}/run`

Manually trigger a scheduled task.

**Example:**
```bash
curl -X POST http://localhost:8000/api/v1/tasks/scheduled/auto-deduplicate-memories/run
```

**Response:**
```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440002",
  "status": "submitted",
  "message": "Scheduled task 'auto-deduplicate-memories' triggered manually. Track with task_id: 550e8400-e29b-41d4-a716-446655440002"
}
```

---

## 🧠 Intelligence APIs

### Base URL: `http://localhost:8000/v1/intelligence`

Advanced AI-powered analysis and intelligence features.

### Fact Extraction

#### `POST /v1/intelligence/facts:extract`

Extract structured facts from text.

**Request Body:**
```json
{
  "text": "John works at Google in San Francisco. He studied Computer Science at MIT and knows Python, Java, and React.",
  "source": "profile",
  "user_id": "john",
  "with_quality": true
}
```

**Response:**
```json
{
  "facts": [
    {
      "fact": "John works at Google",
      "category": "employment",
      "confidence": 0.95,
      "quality_score": 0.92,
      "source": "profile"
    },
    {
      "fact": "Work location: San Francisco",
      "category": "location",
      "confidence": 0.90
    },
    {
      "fact": "Education: MIT Computer Science",
      "category": "education",
      "confidence": 0.88
    },
    {
      "fact": "Skills: Python, Java, React",
      "category": "skills",
      "confidence": 0.93
    }
  ],
  "count": 4,
  "metadata": {
    "source": "profile",
    "with_quality": true
  }
}
```

---

### Entity Recognition

#### `POST /v1/intelligence/entities:extract`

Extract and recognize named entities.

**Request Body:**
```json
{
  "text": "Elon Musk founded SpaceX in California. The company launched Falcon 9 in 2010.",
  "context": {"domain": "technology"}
}
```

**Response:**
```json
{
  "entities": [
    {
      "canonical_name": "Elon Musk",
      "entity_type": "person",
      "mentions": 1,
      "confidence": 0.98
    },
    {
      "canonical_name": "SpaceX",
      "entity_type": "organization",
      "mentions": 1,
      "confidence": 0.95
    },
    {
      "canonical_name": "California",
      "entity_type": "location",
      "mentions": 1
    },
    {
      "canonical_name": "Falcon 9",
      "entity_type": "product",
      "mentions": 1
    }
  ],
  "count": 4,
  "statistics": {
    "by_type": {
      "person": 1,
      "organization": 1,
      "location": 1,
      "product": 1
    },
    "total_mentions": 4
  }
}
```

---

#### `POST /v1/intelligence/entities:search`

Search for entities.

**Request Body:**
```json
{
  "query": "musk",
  "entity_type": "person",  // optional
  "min_mentions": 1
}
```

---

#### `GET /v1/intelligence/entities/{entity_id}`

Get entity profile.

**Response:**
```json
{
  "entity": {
    "id": "entity_person_elon_musk",
    "canonical_name": "Elon Musk",
    "entity_type": "person",
    "aliases": ["Elon", "Musk"],
    "mentions": 15,
    "first_seen": "2025-11-01T10:00:00Z",
    "last_seen": "2025-11-03T10:00:00Z"
  },
  "timeline": [
    {
      "memory_id": "mem_abc123",
      "text": "Elon Musk founded SpaceX",
      "timestamp": "2025-11-01T10:00:00Z"
    }
  ]
}
```

---

### Relationship Analysis

#### `POST /v1/intelligence/relationships:analyze`

Analyze entity relationships.

**Request Body:**
```json
{
  "text": "Steve Jobs co-founded Apple with Steve Wozniak. They worked together in California.",
  "entity_ids": null  // optional, analyze specific entities
}
```

**Response:**
```json
{
  "relationships": [
    {
      "source_entity": "Steve Jobs",
      "target_entity": "Apple",
      "relation_type": "founded",
      "strength": 0.95,
      "evidence": ["co-founded Apple"]
    },
    {
      "source_entity": "Steve Wozniak",
      "target_entity": "Apple",
      "relation_type": "founded",
      "strength": 0.95
    },
    {
      "source_entity": "Steve Jobs",
      "target_entity": "Steve Wozniak",
      "relation_type": "worked_with",
      "strength": 0.85
    }
  ],
  "network": {
    "entities": 3,
    "relationships": 3,
    "density": 0.67,
    "central_entities": ["Apple"]
  },
  "visualization_data": {
    "nodes": [...],
    "edges": [...]
  }
}
```

---

#### `GET /v1/intelligence/relationships/{entity_id}`

Get relationships for an entity.

**Query Parameters:**
- `relation_type`: Filter by relationship type (optional)
- `min_strength`: Minimum strength score (default: 0.0)

---

#### `GET /v1/intelligence/relationships:network`

Get complete relationship network.

**Response:**
```json
{
  "network": {
    "total_entities": 150,
    "total_relationships": 320,
    "network_density": 0.42,
    "clusters": 5,
    "central_entities": [
      {"entity_id": "entity_org_apple", "centrality": 0.95}
    ]
  },
  "visualization": {...},
  "statistics": {...}
}
```

---

### Semantic Clustering

#### `POST /v1/intelligence/clustering:analyze`

Cluster memories by semantic similarity.

**Request Body:**
```json
{
  "memories": [
    {"id": "mem_1", "text": "I love Python programming"},
    {"id": "mem_2", "text": "JavaScript is great for web dev"},
    {"id": "mem_3", "text": "I prefer oat milk in coffee"},
    {"id": "mem_4", "text": "Green tea in the morning is relaxing"}
  ],
  "max_clusters": 10,
  "hierarchical": false
}
```

**Response:**
```json
{
  "clusters": [
    {
      "topic": "programming_languages",
      "memories": [
        {"id": "mem_1", "text": "I love Python programming"},
        {"id": "mem_2", "text": "JavaScript is great for web dev"}
      ],
      "tags": ["programming", "languages"],
      "size": 2
    },
    {
      "topic": "beverage_preferences",
      "memories": [
        {"id": "mem_3", "text": "I prefer oat milk in coffee"},
        {"id": "mem_4", "text": "Green tea in the morning is relaxing"}
      ],
      "tags": ["beverages", "preferences"],
      "size": 2
    }
  ],
  "count": 2,
  "quality_metrics": {
    "per_cluster": {
      "cluster_0": {
        "cohesion": 0.85,
        "separation": 0.72
      },
      "cluster_1": {
        "cohesion": 0.88,
        "separation": 0.75
      }
    }
  }
}
```

---

#### `POST /v1/intelligence/clustering:optimize`

Determine optimal cluster count.

**Request Body:**
```json
{
  "memories": [...],
  "min_k": 2,
  "max_k": 15
}
```

**Response:**
```json
{
  "optimal_cluster_count": 5,
  "min_k": 2,
  "max_k": 15
}
```

---

### Temporal Analytics

#### `POST /v1/intelligence/temporal:analyze`

Perform temporal analysis.

**Request Body:**
```json
{
  "memories": [...],
  "analysis_type": "peak_activity",  // peak_activity, patterns, trends, clusters
  "time_window_days": 30,
  "timezone_offset": -8  // PST
}
```

**Response (peak_activity):**
```json
{
  "analysis": {
    "peak_hour": 14,  // 2 PM
    "peak_day": "Monday",
    "busiest_period": "afternoon",
    "hourly_distribution": {
      "0": 5, "1": 2, ..., "14": 45, ..., "23": 8
    },
    "daily_distribution": {
      "Monday": 120,
      "Tuesday": 95,
      ...
    }
  },
  "metadata": {
    "analysis_type": "peak_activity"
  }
}
```

**Response (patterns):**
```json
{
  "analysis": {
    "patterns": [
      {
        "pattern_type": "daily_routine",
        "description": "User typically creates memories at 9 AM and 6 PM",
        "frequency": "daily",
        "confidence": 0.85
      }
    ],
    "count": 3
  },
  "metadata": {
    "analysis_type": "patterns"
  }
}
```

---

#### `POST /v1/intelligence/temporal:peak-times`

Get detailed peak activity times.

---

### Health Check

#### `GET /v1/intelligence/health`

Check intelligence services status.

**Response:**
```json
{
  "status": "healthy",
  "services": "Advanced Intelligence APIs",
  "version": "0.2.5"
}
```

---

## 📊 Monitoring & Observability

### Flower - Celery Monitoring UI

**URL**: `http://localhost:5555`
**Default Auth**: `admin:admin`

**Features:**
- Real-time task monitoring
- Worker status and performance
- Task history and statistics
- Broker and result backend monitoring
- Task details and tracebacks

**Endpoints:**
- Dashboard: `http://localhost:5555`
- Tasks: `http://localhost:5555/tasks`
- Workers: `http://localhost:5555/workers`
- API: `http://localhost:5555/api/tasks`

---

### Prometheus - Metrics Collection

**URL**: `http://localhost:9090`
**Status**: ⚠️ Infrastructure configured, application metrics not yet implemented

**Current State:**
- ✅ Prometheus container runs in docker-compose
- ✅ Configuration file exists (monitoring/prometheus.yml)
- ❌ Application `/metrics` endpoint not yet implemented
- ❌ Custom HippocampAI metrics not exposed

**What Works:**
- System-level metrics (container stats, resource usage)
- Basic health checks
- Infrastructure monitoring

**Planned Metrics** (coming in future release):
- `hippocampai_memory_operations_total` - Total memory operations
- `hippocampai_memory_operation_duration_seconds` - Operation latency
- `hippocampai_cache_hits_total` - Cache hit count
- `hippocampai_cache_misses_total` - Cache miss count
- `celery_task_sent` - Tasks sent to queue
- `celery_task_succeeded` - Successful tasks
- `celery_task_failed` - Failed tasks

**Alternative:** Use the built-in Telemetry API (see below) for operation tracking.

---

### Grafana - Dashboards

**URL**: `http://localhost:3000`
**Default Login**: `admin:admin`
**Status**: ⚠️ Infrastructure configured, custom dashboards not yet created

**Current State:**
- ✅ Grafana container runs in docker-compose
- ✅ Prometheus datasource can be configured
- ❌ Pre-built HippocampAI dashboards not included
- ❌ Application-specific visualizations need manual setup

**What You Can Do:**
1. Monitor infrastructure metrics (CPU, memory, disk)
2. Create custom dashboards manually
3. Query system-level Prometheus metrics

**Manual Dashboard Creation:**
1. Navigate to `http://localhost:3000/dashboard/new`
2. Add panel → Select Prometheus datasource
3. Create visualizations based on available metrics
4. Save your custom dashboard

**Alternative:** Use Flower (port 5555) for Celery task monitoring

---

### Qdrant Dashboard

**URL**: `http://localhost:6333/dashboard`

**Features:**
- Collection statistics
- Vector count and size
- Search performance metrics
- Index optimization status

---

### HippocampAI Telemetry API (Built-in)

**Status**: ✅ Fully implemented and production-ready

HippocampAI includes a built-in telemetry system that tracks all operations without requiring external monitoring tools.

**Available via Python Client:**

```python
from hippocampai import MemoryClient

client = MemoryClient()

# Get comprehensive metrics
metrics = client.get_telemetry_metrics()
print(f"Average recall duration: {metrics['recall_duration']['avg_ms']:.2f}ms")
print(f"P95 latency: {metrics['recall_duration']['p95']:.2f}ms")

# Get recent operation traces
recent_ops = client.get_recent_operations(limit=10)
for op in recent_ops:
    print(f"{op.operation}: {op.duration_ms:.2f}ms (trace_id: {op.trace_id})")

# Export traces for external analysis
traces = client.export_telemetry(format="json")
# Can be sent to OpenTelemetry, Datadog, etc.
```

**Tracked Metrics:**
- Operation durations (remember, recall, extract, update, delete)
- Retrieval counts and patterns
- Memory sizes (characters and tokens)
- Percentile latencies (P50, P95, P99)
- Error rates and types
- Trace IDs for distributed tracing

**Trace Format:**
Each operation gets a unique trace_id and tracks:
- Start/end timestamps
- Duration in milliseconds
- User ID and session ID
- Operation metadata
- Status (success/error)
- Sub-events within the operation

**Export Formats:**
- JSON (OpenTelemetry compatible)
- Datadog APM format
- Custom structured logs

**See Also:**
- [Telemetry Guide](TELEMETRY.md) - Complete telemetry documentation
- [Library Reference](LIBRARY_COMPLETE_REFERENCE.md) - Telemetry methods

---

## 🔌 Connection Guide

### Python Client

```python
import requests
from typing import Any, Dict

class HippocampAIClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url

    def remember(self, text: str, user_id: str, **kwargs) -> Dict[str, Any]:
        """Store a memory."""
        response = requests.post(
            f"{self.base_url}/v1/memories:remember",
            json={"text": text, "user_id": user_id, **kwargs}
        )
        response.raise_for_status()
        return response.json()

    def recall(self, query: str, user_id: str, k: int = 5) -> list:
        """Retrieve memories."""
        response = requests.post(
            f"{self.base_url}/v1/memories:recall",
            json={"query": query, "user_id": user_id, "k": k}
        )
        response.raise_for_status()
        return response.json()

    def submit_task(self, task_type: str, **params) -> str:
        """Submit async task."""
        response = requests.post(
            f"{self.base_url}/api/v1/tasks/memory/{task_type}",
            json=params
        )
        response.raise_for_status()
        return response.json()["task_id"]

    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Check task status."""
        response = requests.get(
            f"{self.base_url}/api/v1/tasks/status/{task_id}"
        )
        response.raise_for_status()
        return response.json()

# Usage
client = HippocampAIClient()

# Synchronous operation
memory = client.remember("I love Python", "alice", type="fact")
print(f"Created: {memory['id']}")

# Async operation
task_id = client.submit_task("create", text="I love Python", user_id="alice")
status = client.get_task_status(task_id)
print(f"Task status: {status['status']}")
```

---

### JavaScript/Node.js Client

```javascript
class HippocampAIClient {
  constructor(baseUrl = 'http://localhost:8000') {
    this.baseUrl = baseUrl;
  }

  async remember(text, userId, options = {}) {
    const response = await fetch(`${this.baseUrl}/v1/memories:remember`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({text, user_id: userId, ...options})
    });
    return response.json();
  }

  async recall(query, userId, k = 5) {
    const response = await fetch(`${this.baseUrl}/v1/memories:recall`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({query, user_id: userId, k})
    });
    return response.json();
  }
}

// Usage
const client = new HippocampAIClient();

const memory = await client.remember('I love Python', 'alice', {
  type: 'fact',
  importance: 8.0
});
console.log(`Created: ${memory.id}`);

const results = await client.recall('programming', 'alice');
console.log(`Found ${results.length} memories`);
```

---

### cURL Examples

```bash
# Remember
curl -X POST http://localhost:8000/v1/memories:remember \
  -H "Content-Type: application/json" \
  -d '{"text": "I love Python", "user_id": "alice", "type": "fact"}'

# Recall
curl -X POST http://localhost:8000/v1/memories:recall \
  -H "Content-Type: application/json" \
  -d '{"query": "programming", "user_id": "alice", "k": 5}'

# Submit async task
curl -X POST http://localhost:8000/api/v1/tasks/memory/create \
  -H "Content-Type: application/json" \
  -d '{"text": "I love Python", "user_id": "alice"}'

# Check task status
curl http://localhost:8000/api/v1/tasks/status/550e8400-e29b-41d4-a716-446655440000

# Get worker stats
curl http://localhost:8000/api/v1/tasks/inspect/stats

# Extract facts
curl -X POST http://localhost:8000/v1/intelligence/facts:extract \
  -H "Content-Type: application/json" \
  -d '{"text": "John works at Google", "with_quality": true}'
```

---

## 🔐 API Authentication

### Current Status
**V0.2.5**: No authentication required (development/local deployment)

### Future: API Key Authentication

Coming in future versions:

```python
# With API key
headers = {
    "X-API-Key": "your-api-key-here",
    "Content-Type": "application/json"
}

response = requests.post(
    "http://localhost:8000/v1/memories:remember",
    headers=headers,
    json={...}
)
```

### Production Recommendations

For production deployments, add authentication layer:

**Option 1: Reverse Proxy (Nginx/Traefik)**
```nginx
location /v1/ {
    auth_basic "Restricted";
    auth_basic_user_file /etc/nginx/.htpasswd;
    proxy_pass http://hippocampai:8000;
}
```

**Option 2: API Gateway (Kong/AWS API Gateway)**

**Option 3: OAuth2/JWT** (implement custom middleware)

---

## ⚠️ Error Handling

### Standard Error Response

```json
{
  "detail": "Error description",
  "status_code": 500,
  "error_type": "InternalServerError"
}
```

### HTTP Status Codes

| Code | Meaning | Common Causes |
|------|---------|---------------|
| 200 | Success | Operation completed |
| 400 | Bad Request | Invalid parameters |
| 404 | Not Found | Memory/task not found |
| 422 | Validation Error | Invalid request body |
| 500 | Internal Server Error | Server error |
| 503 | Service Unavailable | Database/Redis down |

### Error Handling Example

```python
try:
    memory = client.remember("text", "user_id")
except requests.HTTPError as e:
    if e.response.status_code == 404:
        print("Memory not found")
    elif e.response.status_code == 500:
        print(f"Server error: {e.response.json()['detail']}")
    else:
        print(f"Error: {e}")
```

---

## 🚦 Rate Limiting

### Current Limits (Configurable)

- **Default**: No rate limiting
- **Recommended Production**:
  - 100 requests/minute per IP
  - 1000 requests/hour per API key
  - 10,000 requests/day per user

### Future: Rate Limit Headers

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1730635200
```

---

## 📚 Additional Resources

- **[API Reference](API_REFERENCE.md)** - Complete method documentation
- **[Configuration](CONFIGURATION.md)** - Environment variables and settings
- **[Celery Guide](CELERY_USAGE_GUIDE.md)** - Background task optimization
- **[Telemetry Guide](TELEMETRY.md)** - Monitoring and observability
- **[User Guide](USER_GUIDE.md)** - Production deployment

---

**Last Updated**: 2025-11-03
**API Version**: V0.2.5
**Status**: Production Ready
