# HippocampAI Memory Tracking & Monitoring Guide

## Overview

HippocampAI includes comprehensive **memory-level tracking** that allows users to monitor what's happening with individual memories on the backend. This system provides complete visibility into memory lifecycle events, access patterns, and health metrics.

**Status**: ✅ Fully Integrated

---

## Key Features

### 1. **Memory Lifecycle Tracking**
Track every event in a memory's lifecycle:
- Creation, updates, deletions
- Searches and retrievals
- Consolidations and deduplication
- Health checks and conflict detection
- Staleness detection and freshness updates

### 2. **Access Pattern Analysis**
Understand how memories are being used:
- Access frequency and recency
- Search hits vs direct retrievals
- Hot and cold memory identification
- Access sources tracking

### 3. **Health Monitoring**
Track memory health over time:
- Health score snapshots
- Staleness and freshness tracking
- Duplicate likelihood detection
- Issue identification

### 4. **Real-time Event Streaming**
- All events are tracked in real-time
- Events are logged with structured metadata
- Query events by memory ID, user ID, or event type
- Complete audit trail for compliance

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    Memory Operations                      │
│  (remember, recall, update, delete, consolidate, etc.)   │
└───────────────────────┬──────────────────────────────────┘
                        │
                        │ tracks events
                        ▼
┌──────────────────────────────────────────────────────────┐
│                    MemoryTracker                         │
│  - Event logging                                         │
│  - Access pattern tracking                               │
│  - Health snapshot recording                             │
└───────────────────────┬──────────────────────────────────┘
                        │
                        │ stores to
                        ▼
┌──────────────────────────────────────────────────────────┐
│               In-Memory / Redis / File                   │
│  - Event buffer (last 10k events)                       │
│  - Access patterns by memory                            │
│  - Health history snapshots                             │
└───────────────────────┬──────────────────────────────────┘
                        │
                        │ exposed via
                        ▼
┌──────────────────────────────────────────────────────────┐
│                 Monitoring API Endpoints                 │
│  /v1/monitoring/events                                   │
│  /v1/monitoring/stats                                    │
│  /v1/monitoring/access-pattern                           │
│  /v1/monitoring/access-patterns/{user_id}                │
│  /v1/monitoring/health-history                           │
└──────────────────────────────────────────────────────────┘
```

---

## Event Types

### Memory Lifecycle Events

| Event Type | Description | Severity |
|------------|-------------|----------|
| `created` | Memory was created | INFO |
| `updated` | Memory was modified | INFO |
| `deleted` | Memory was removed | INFO |
| `retrieved` | Memory was directly fetched | DEBUG |
| `searched` | Memory appeared in search results | DEBUG |
| `consolidated` | Memory was consolidated with others | INFO |
| `deduplicated` | Duplicate memory detected | WARNING |
| `health_check` | Health check performed | DEBUG |
| `conflict_detected` | Conflict with another memory | WARNING |
| `conflict_resolved` | Conflict was resolved | INFO |
| `access_pattern` | Access pattern analyzed | DEBUG |
| `staleness_detected` | Memory marked as stale | WARNING |
| `freshness_updated` | Memory freshness score updated | DEBUG |

---

## API Endpoints

### 1. Get Memory Events

**Endpoint**: `GET /v1/monitoring/events`

Get lifecycle events for memories with optional filtering.

**Query Parameters**:
- `memory_id` (optional): Filter by specific memory
- `user_id` (optional): Filter by user
- `event_type` (optional): Filter by event type (e.g., "created", "updated", "searched")
- `limit` (optional, default: 100): Maximum number of events to return

**Example Request**:
```bash
# Get all events for a specific memory
curl "http://localhost:8000/v1/monitoring/events?memory_id=abc123&limit=50"

# Get all creation events for a user
curl "http://localhost:8000/v1/monitoring/events?user_id=alice&event_type=created"

# Get recent events (all types)
curl "http://localhost:8000/v1/monitoring/events?limit=100"
```

**Example Response**:
```json
{
  "total": 25,
  "events": [
    {
      "event_id": "evt-789",
      "memory_id": "abc123",
      "user_id": "alice",
      "event_type": "created",
      "severity": "info",
      "timestamp": "2025-11-08T10:30:00Z",
      "metadata": {
        "type": "fact",
        "importance": 7.5,
        "tags": ["work", "project"],
        "auto_resolved": false
      },
      "duration_ms": null,
      "success": true,
      "error_message": null
    },
    {
      "event_id": "evt-790",
      "memory_id": "abc123",
      "user_id": "alice",
      "event_type": "searched",
      "severity": "debug",
      "timestamp": "2025-11-08T10:35:00Z",
      "metadata": {
        "query": "project status",
        "score": 0.92,
        "rank": 1
      },
      "duration_ms": null,
      "success": true,
      "error_message": null
    },
    {
      "event_id": "evt-791",
      "memory_id": "abc123",
      "user_id": "alice",
      "event_type": "updated",
      "severity": "info",
      "timestamp": "2025-11-08T11:00:00Z",
      "metadata": {
        "updated_fields": ["importance", "tags"]
      },
      "duration_ms": null,
      "success": true,
      "error_message": null
    }
  ]
}
```

---

### 2. Get Memory Statistics

**Endpoint**: `POST /v1/monitoring/stats`

Get comprehensive statistics for a user's memories.

**Request Body**:
```json
{
  "user_id": "alice"
}
```

**Example Request**:
```bash
curl -X POST http://localhost:8000/v1/monitoring/stats \
  -H "Content-Type: application/json" \
  -d '{"user_id": "alice"}'
```

**Example Response**:
```json
{
  "user_id": "alice",
  "total_events": 1250,
  "success_rate": 0.987,
  "event_counts": {
    "created": 45,
    "updated": 12,
    "deleted": 3,
    "searched": 890,
    "retrieved": 250,
    "consolidated": 5,
    "conflict_detected": 2,
    "conflict_resolved": 2
  },
  "total_memories_tracked": 42,
  "total_accesses": 1140,
  "avg_operation_duration_ms": 45.3,
  "most_accessed_memories": [
    {
      "memory_id": "mem-123",
      "access_count": 85,
      "last_accessed": "2025-11-08T10:30:00Z"
    },
    {
      "memory_id": "mem-456",
      "access_count": 67,
      "last_accessed": "2025-11-08T09:15:00Z"
    }
  ]
}
```

---

### 3. Get Access Pattern

**Endpoint**: `POST /v1/monitoring/access-pattern`

Get detailed access pattern for a specific memory.

**Request Body**:
```json
{
  "memory_id": "abc123",
  "user_id": "alice"
}
```

**Example Request**:
```bash
curl -X POST http://localhost:8000/v1/monitoring/access-pattern \
  -H "Content-Type: application/json" \
  -d '{
    "memory_id": "abc123",
    "user_id": "alice"
  }'
```

**Example Response**:
```json
{
  "memory_id": "abc123",
  "user_id": "alice",
  "access_count": 85,
  "last_accessed": "2025-11-08T10:30:00Z",
  "first_accessed": "2025-10-15T08:00:00Z",
  "search_hits": 65,
  "direct_retrievals": 20,
  "access_frequency": 3.5,
  "access_sources": {
    "/v1/recall": 65,
    "/v1/memories/{id}": 20
  }
}
```

**Fields**:
- `access_count`: Total number of times memory was accessed
- `last_accessed`: Most recent access timestamp
- `first_accessed`: First access timestamp
- `search_hits`: Number of times memory appeared in search results
- `direct_retrievals`: Number of times memory was directly fetched by ID
- `access_frequency`: Average accesses per day
- `access_sources`: Breakdown of access by endpoint

---

### 4. Get All Access Patterns

**Endpoint**: `GET /v1/monitoring/access-patterns/{user_id}`

Get access patterns for all memories belonging to a user.

**Example Request**:
```bash
curl http://localhost:8000/v1/monitoring/access-patterns/alice
```

**Example Response**:
```json
{
  "user_id": "alice",
  "total_memories": 42,
  "patterns": [
    {
      "memory_id": "mem-123",
      "user_id": "alice",
      "access_count": 85,
      "last_accessed": "2025-11-08T10:30:00Z",
      "first_accessed": "2025-10-15T08:00:00Z",
      "search_hits": 65,
      "direct_retrievals": 20,
      "access_frequency": 3.5,
      "access_sources": {}
    },
    {
      "memory_id": "mem-456",
      "user_id": "alice",
      "access_count": 67,
      "last_accessed": "2025-11-08T09:15:00Z",
      "first_accessed": "2025-09-20T14:30:00Z",
      "search_hits": 50,
      "direct_retrievals": 17,
      "access_frequency": 1.2,
      "access_sources": {}
    }
  ]
}
```

---

### 5. Get Health History

**Endpoint**: `POST /v1/monitoring/health-history`

Get health snapshots for a memory over time.

**Request Body**:
```json
{
  "memory_id": "abc123",
  "user_id": "alice",
  "limit": 50
}
```

**Example Request**:
```bash
curl -X POST http://localhost:8000/v1/monitoring/health-history \
  -H "Content-Type: application/json" \
  -d '{
    "memory_id": "abc123",
    "user_id": "alice",
    "limit": 10
  }'
```

**Example Response**:
```json
{
  "memory_id": "abc123",
  "user_id": "alice",
  "total_snapshots": 8,
  "history": [
    {
      "memory_id": "abc123",
      "user_id": "alice",
      "timestamp": "2025-11-08T10:00:00Z",
      "health_score": 85.5,
      "staleness_score": 0.15,
      "freshness_score": 0.92,
      "access_frequency": 3.5,
      "duplicate_likelihood": 0.05,
      "issues": []
    },
    {
      "memory_id": "abc123",
      "user_id": "alice",
      "timestamp": "2025-11-07T10:00:00Z",
      "health_score": 88.2,
      "staleness_score": 0.10,
      "freshness_score": 0.95,
      "access_frequency": 3.8,
      "duplicate_likelihood": 0.03,
      "issues": []
    }
  ]
}
```

---

## Usage Examples

### Python Client Library

```python
from hippocampai import MemoryClient

# Initialize client
client = MemoryClient()

# Create a memory (automatically tracked)
memory = client.remember(
    text="I'm working on Project Phoenix",
    user_id="alice",
    type="fact"
)

# The creation event is automatically logged:
# Event: memory_id=<id>, event_type=created, user_id=alice
```

### Query Events via API

```python
import requests

# Get all events for a user
response = requests.get(
    "http://localhost:8000/v1/monitoring/events",
    params={"user_id": "alice", "limit": 100}
)
events = response.json()

print(f"Total events: {events['total']}")
for event in events["events"]:
    print(f"  {event['timestamp']}: {event['event_type']} - {event['memory_id']}")
```

### Get User Statistics

```python
import requests

response = requests.post(
    "http://localhost:8000/v1/monitoring/stats",
    json={"user_id": "alice"}
)
stats = response.json()

print(f"Total memories tracked: {stats['total_memories_tracked']}")
print(f"Total accesses: {stats['total_accesses']}")
print(f"Success rate: {stats['success_rate']:.1%}")
print(f"\nMost accessed memories:")
for mem in stats["most_accessed_memories"]:
    print(f"  {mem['memory_id']}: {mem['access_count']} accesses")
```

### Monitor Access Patterns

```python
import requests

# Get access pattern for a specific memory
response = requests.post(
    "http://localhost:8000/v1/monitoring/access-pattern",
    json={
        "memory_id": "abc123",
        "user_id": "alice"
    }
)
pattern = response.json()

print(f"Memory: {pattern['memory_id']}")
print(f"Total accesses: {pattern['access_count']}")
print(f"Access frequency: {pattern['access_frequency']:.1f} per day")
print(f"Search hits: {pattern['search_hits']}")
print(f"Direct retrievals: {pattern['direct_retrievals']}")
```

---

## Integration with Grafana

The memory tracking system integrates seamlessly with Grafana dashboards:

### Memory Event Panels

**Events Over Time**:
```promql
# Events per minute by type
sum(rate(hippocampai_memory_events_total[1m])) by (event_type)
```

**Top Accessed Memories**:
```promql
# Most frequently accessed memories
topk(10, hippocampai_memory_access_count)
```

**Memory Health Trends**:
```promql
# Average health score over time
avg(hippocampai_memory_health_score) by (user_id)
```

---

## Best Practices

### 1. Event Retention
- Events are stored in-memory by default (last 10k events)
- Configure Redis backend for persistent storage
- Set retention policies based on compliance requirements

```python
from hippocampai.monitoring.memory_tracker import initialize_tracker

# Initialize with Redis backend
tracker = initialize_tracker(storage_backend="redis")
```

### 2. Query Optimization
- Use specific filters (memory_id, user_id, event_type) to reduce response size
- Set appropriate limits based on UI needs
- Cache frequently accessed patterns

### 3. Access Pattern Analysis
- Monitor `access_frequency` to identify hot memories
- Use `search_hits` vs `direct_retrievals` to understand usage patterns
- Track `last_accessed` to identify stale memories

### 4. Health Monitoring
- Set up alerts for low health scores
- Monitor `staleness_score` to trigger consolidation
- Track `duplicate_likelihood` to prevent redundancy

### 5. Event Streaming
- Use event types to filter noise (e.g., exclude `searched` for high-level views)
- Severity levels help prioritize issues (ERROR, WARNING, INFO, DEBUG)
- Metadata provides context for debugging

---

## Troubleshooting

### Events Not Appearing

```bash
# 1. Check if tracker is initialized
curl http://localhost:8000/v1/monitoring/events?limit=1

# 2. Check logs for tracking errors
docker-compose logs hippocampai-api | grep "Failed to track"

# 3. Verify operations are being performed
curl -X POST http://localhost:8000/v1/memories \
  -H "Content-Type: application/json" \
  -d '{"text": "Test memory", "user_id": "test"}'
```

### Missing Access Patterns

Access patterns are only created after memories are accessed. Generate some activity:

```bash
# Create a memory
curl -X POST http://localhost:8000/v1/memories \
  -H "Content-Type: application/json" \
  -d '{"text": "Test", "user_id": "alice"}'

# Search for it (creates search event)
curl "http://localhost:8000/v1/recall?query=test&user_id=alice"

# Check access pattern
curl -X POST http://localhost:8000/v1/monitoring/access-pattern \
  -H "Content-Type: application/json" \
  -d '{"memory_id": "<id>", "user_id": "alice"}'
```

### High Memory Usage

If the in-memory event buffer grows too large:

```python
from hippocampai.monitoring.memory_tracker import get_tracker

# Clear old events (keep last 30 days)
tracker = get_tracker()
tracker.clear_old_events(days=30)
```

---

## Production Recommendations

1. **Use Redis Backend**: Configure Redis for persistent event storage
2. **Set Retention Policies**: Clear old events periodically (e.g., 90 days)
3. **Monitor Performance**: Track tracking overhead (should be < 1ms per event)
4. **Enable Sampling**: For high-traffic systems, sample search events (e.g., track every 10th search)
5. **Set Up Alerts**: Alert on high error rates or low health scores
6. **Export Events**: Periodically export events to data warehouse for analytics
7. **Index Events**: If using database backend, index on `user_id`, `memory_id`, `event_type`, `timestamp`

---

## Summary

✅ **Memory Lifecycle Tracking** - Track every event from creation to deletion
✅ **Access Pattern Analysis** - Understand how memories are being used
✅ **Health Monitoring** - Track memory health over time
✅ **Real-time Events** - All events logged in real-time
✅ **API Access** - Query events, stats, and patterns via REST API
✅ **Grafana Integration** - Visualize memory metrics in dashboards
✅ **Production Ready** - Scalable and configurable

**Access Points**:
- Events: `GET /v1/monitoring/events`
- Stats: `POST /v1/monitoring/stats`
- Access Pattern: `POST /v1/monitoring/access-pattern`
- All Patterns: `GET /v1/monitoring/access-patterns/{user_id}`
- Health History: `POST /v1/monitoring/health-history`

**Key Benefits**:
1. **Complete Visibility**: See exactly what's happening with your memories
2. **Performance Insights**: Identify hot and cold memories
3. **Health Tracking**: Monitor memory quality over time
4. **Compliance**: Full audit trail for all operations
5. **Debugging**: Detailed event logs for troubleshooting

---

**Documentation Version**: 1.0
**Date**: 2025-11-08
**Status**: Production Ready ✅
