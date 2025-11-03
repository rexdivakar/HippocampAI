# Telemetry & Observability Guide

HippocampAI includes built-in telemetry for tracking memory operations, similar to Mem0's platform.

**Note:** Telemetry data is accessed via library functions only, not through REST API endpoints.

## Overview

The telemetry system provides:

- **Operation tracing** - Track every remember/recall/extract operation
- **Performance metrics** - Monitor latency, throughput, success rates
- **Detailed breakdowns** - See score components, retrieval paths
- **Export capabilities** - Send traces to external tools (OpenTelemetry, Prometheus, etc.)
- **Library-based access** - All data accessed through Python API, not HTTP endpoints

## Basic Usage

### Enable Telemetry

```python
from hippocampai import MemoryClient, get_telemetry

# Initialize client (telemetry enabled by default)
client = MemoryClient()

# Get telemetry instance
telemetry = get_telemetry()
```

### Track Operations

Telemetry automatically tracks all memory operations:

```python
# These operations are automatically traced
client.remember(text="I love coffee", user_id="alice", type="preference")
client.recall(query="What does Alice like?", user_id="alice")
client.extract_from_conversation(conversation="...", user_id="alice")
```

### View Metrics via Client

```python
# Access telemetry through the client
metrics = client.get_telemetry_metrics()

print(f"Remember operations: {metrics['remember_duration']['count']}")
print(f"Average recall time: {metrics['recall_duration']['avg']:.2f}ms")
print(f"P95 recall time: {metrics['recall_duration']['p95']:.2f}ms")
print(f"P99 recall time: {metrics['recall_duration']['p99']:.2f}ms")
```

### View Metrics via Global Instance

```python
# Or access through global telemetry instance
from hippocampai import get_telemetry

telemetry = get_telemetry()
metrics = telemetry.get_metrics_summary()

print(f"Remember operations: {metrics['remember_duration']['count']}")
print(f"Average recall time: {metrics['recall_duration']['avg']:.2f}ms")
```

### View Recent Traces

```python
# Via client
operations = client.get_recent_operations(limit=10)

for op in operations:
    print(f"Operation: {op.operation.value}")
    print(f"User: {op.user_id}")
    print(f"Duration: {op.duration_ms:.2f}ms")
    print(f"Status: {op.status}")
    print(f"Events: {len(op.events)}")
    print()

# Or via global telemetry instance
traces = telemetry.get_recent_traces(limit=10)

for trace in traces:
    print(f"Operation: {trace.operation}")
    print(f"User: {trace.user_id}")
    print(f"Duration: {trace.duration_ms:.2f}ms")
    print(f"Status: {trace.status}")
    print(f"Events: {len(trace.events)}")
    print()
```

### Get Specific Trace

```python
# Get a specific trace by ID
trace_id = "..."  # From previous operation
trace = telemetry.get_trace(trace_id)

if trace:
    print(f"Operation: {trace.operation.value}")
    print(f"Started: {trace.start_time}")
    print(f"Ended: {trace.end_time}")
    print(f"Duration: {trace.duration_ms:.2f}ms")

    print("\nEvents:")
    for event in trace.events:
        print(f"  - {event.timestamp}: {event.status}")
        print(f"    Metadata: {event.metadata}")
```

## Advanced Features

### Filter by Operation Type

```python
# Via client (simpler)
recall_ops = client.get_recent_operations(limit=20, operation="recall")
remember_ops = client.get_recent_operations(limit=20, operation="remember")

# Or via telemetry instance (with enum)
from hippocampai import OperationType

recall_traces = telemetry.get_recent_traces(
    limit=20,
    operation=OperationType.RECALL
)

remember_traces = telemetry.get_recent_traces(
    limit=20,
    operation=OperationType.REMEMBER
)
```

### Export Traces

Export traces for external monitoring tools:

```python
# Via client (recommended)
exported = client.export_telemetry()

# Export specific traces
exported = client.export_telemetry(trace_ids=["trace_1", "trace_2"])

# Or via telemetry instance
exported = telemetry.export_traces()
exported_specific = telemetry.export_traces(trace_ids=["trace_1", "trace_2"])

# Save to file
import json
with open("traces.json", "w") as f:
    json.dump(exported, f, indent=2)
```

### Clear Old Traces

Prevent memory buildup by clearing old traces:

```python
# Clear all traces
count = telemetry.clear_traces()
print(f"Cleared {count} traces")

# Clear traces older than 60 minutes
count = telemetry.clear_traces(older_than_minutes=60)
print(f"Cleared {count} old traces")
```

## Custom Tracing

### Manual Trace Creation

```python
from hippocampai.telemetry import get_telemetry, OperationType

telemetry = get_telemetry()

# Start a trace
trace_id = telemetry.start_trace(
    operation=OperationType.RECALL,
    user_id="alice",
    session_id="session_123",
    custom_field="custom_value"
)

# Add events
telemetry.add_event(
    trace_id,
    event_name="vector_search",
    status="success",
    duration_ms=45.2,
    candidates=100
)

telemetry.add_event(
    trace_id,
    event_name="reranking",
    status="success",
    duration_ms=12.3,
    reranked=20
)

# End trace
telemetry.end_trace(
    trace_id,
    status="success",
    result={"retrieved": 5}
)
```

### Decorator-Based Tracing

```python
from hippocampai.telemetry import traced, OperationType

@traced(operation=OperationType.RECALL, capture_result=True)
def my_custom_recall(query: str, user_id: str):
    # Your custom logic
    results = ...
    return results

# Automatically traced
results = my_custom_recall(query="...", user_id="alice")
```

## Metrics Reference

### Available Metrics

| Metric | Description |
|--------|-------------|
| `remember_duration` | Time to store a memory (ms) |
| `recall_duration` | Time to retrieve memories (ms) |
| `extract_duration` | Time to extract from conversation (ms) |
| `retrieval_count` | Number of memories retrieved |

### Metric Statistics

For each metric, the following statistics are available:

- `count` - Total number of operations
- `avg` - Average duration
- `min` - Minimum duration
- `max` - Maximum duration
- `p50` - 50th percentile (median)
- `p95` - 95th percentile
- `p99` - 99th percentile

## Integration with External Tools

### OpenTelemetry

```python
from hippocampai.telemetry import get_telemetry
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider

# Setup OpenTelemetry
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

# Get HippocampAI telemetry
hippocampai_telemetry = get_telemetry()

# Perform operations
client.remember(...)

# Export to OpenTelemetry
exported_traces = hippocampai_telemetry.export_traces()

for trace_data in exported_traces:
    with tracer.start_as_current_span(trace_data["operation"]) as span:
        span.set_attribute("user_id", trace_data["user_id"])
        span.set_attribute("duration_ms", trace_data["duration_ms"])
        # Add more attributes...
```

### Prometheus

```python
from prometheus_client import Counter, Histogram, start_http_server
from hippocampai.telemetry import get_telemetry

# Define Prometheus metrics
remember_counter = Counter('hippocampai_remember_total', 'Total remember operations')
recall_histogram = Histogram('hippocampai_recall_duration_seconds', 'Recall operation duration')

# Get telemetry
telemetry = get_telemetry()

# Periodic export
import time
while True:
    metrics = telemetry.get_metrics_summary()

    # Update Prometheus
    remember_counter.inc(metrics['remember_duration']['count'])
    recall_histogram.observe(metrics['recall_duration']['avg'] / 1000)  # Convert to seconds

    time.sleep(60)  # Every minute
```

### Grafana Dashboard

Sample queries for Grafana:

```promql
# Average recall latency
rate(hippocampai_recall_duration_seconds_sum[5m]) /
rate(hippocampai_recall_duration_seconds_count[5m])

# P95 recall latency
histogram_quantile(0.95, hippocampai_recall_duration_seconds_bucket)

# Remember operations per second
rate(hippocampai_remember_total[1m])
```

## Best Practices

### 1. Regular Cleanup

```python
import schedule

def cleanup_old_traces():
    telemetry = get_telemetry()
    count = telemetry.clear_traces(older_than_minutes=120)  # Keep last 2 hours
    print(f"Cleared {count} old traces")

# Run every hour
schedule.every().hour.do(cleanup_old_traces)
```

### 2. Monitor Performance

```python
def check_performance_degradation():
    telemetry = get_telemetry()
    metrics = telemetry.get_metrics_summary()

    # Alert if P95 > 500ms
    if metrics['recall_duration']['p95'] > 500:
        print("WARNING: Recall latency degradation!")
        # Send alert...
```

### 3. Debug Slow Operations

```python
def find_slow_operations(threshold_ms=1000):
    telemetry = get_telemetry()
    traces = telemetry.get_recent_traces(limit=100)

    slow_ops = [t for t in traces if t.duration_ms > threshold_ms]

    for op in slow_ops:
        print(f"Slow operation: {op.operation}")
        print(f"Duration: {op.duration_ms:.2f}ms")
        print(f"User: {op.user_id}")
        print(f"Metadata: {op.metadata}")
```

## Disable Telemetry

If you don't need telemetry (e.g., in tests):

```python
from hippocampai.telemetry import get_telemetry

# Disable globally
telemetry = get_telemetry(enabled=False)
```

Or via environment variable:

```bash
HIPPOCAMPAI_TELEMETRY_ENABLED=false
```

## Example: Complete Monitoring Setup

```python
from hippocampai import MemoryClient
from hippocampai.telemetry import get_telemetry, OperationType
import json

# Initialize
client = MemoryClient()
telemetry = get_telemetry()

# Perform operations
for i in range(10):
    client.remember(
        text=f"Fact {i}",
        user_id="alice",
        type="fact"
    )

# Get metrics
metrics = telemetry.get_metrics_summary()
print("Metrics Summary:")
print(json.dumps(metrics, indent=2))

# Get traces
traces = telemetry.get_recent_traces(
    operation=OperationType.REMEMBER,
    limit=10
)

print(f"\nRecent Traces: {len(traces)}")
for trace in traces:
    print(f"  {trace.operation}: {trace.duration_ms:.2f}ms ({trace.status})")

# Export
exported = telemetry.export_traces()
with open("telemetry_export.json", "w") as f:
    json.dump(exported, f, indent=2)

print("\nExported telemetry to telemetry_export.json")
```

## Next Steps

- See [CONFIGURATION.md](CONFIGURATION.md) for telemetry configuration options
- Check [ARCHITECTURE.md](ARCHITECTURE.md) for system internals
- Read [API_REFERENCE.md](API_REFERENCE.md) for complete API reference
