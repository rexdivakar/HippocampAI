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
- **LLM usage tracing** - Every model call, including every retry and fallback attempt behind it, with tokens/cost/latency/attribution (see [LLM Usage Tracing](#llm-usage-tracing) below)

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

## LLM Usage Tracing

Every LLM call made through `OpenAILLM`, `AnthropicLLM`, `GroqLLM`, and `OllamaLLM`
is automatically traced at two levels:

1. **Logical invocation** (`LLMInvocation`): the model call as your code requested
   it (one `llm.chat(...)` or `llm.generate(...)` call).
2. **Upstream attempt** (`UpstreamAttempt`): every real request made to the
   provider behind that one logical call. A logical call can have several
   attempts: a rate-limit retry, a timeout retry, or a fallback to a
   different model/provider all produce a new attempt.

This distinction matters because **cost and latency belong to the logical
call, not the last attempt**. A call that times out twice before succeeding
still cost you two attempts worth of latency (and, if the failed attempts
returned partial usage, tokens); the logical invocation aggregates all of
it.

```
Workflow
  → Agent
    → Workflow step
      → Logical LLM invocation
        → Attempt 1: provider timeout       (failed, 8.2s)
        → Attempt 2: model fallback         (failed, 1.1s)
        → Attempt 3: success                (312ms)

invocation.total_attempts   == 3
invocation.total_retries    == 2
invocation.total_fallbacks  == 1
invocation.latency_ms       == ~9.6s   (all three attempts, not just the 312ms winner)
invocation.estimated_cost   == sum of every attempt that reported usage
invocation.status           == "success"
invocation.selected_provider / invocation.selected_model  == whichever attempt finished it
```

### Reading invocations

```python
from hippocampai import MemoryClient

client = MemoryClient()
client.llm.chat([{"role": "user", "content": "Summarize this."}])

# Most recent logical LLM calls, each with its full attempt history
for invocation in client.get_recent_llm_invocations(limit=5):
    print(invocation.provider, invocation.requested_model, invocation.status)
    print(f"  {invocation.total_attempts} attempts, {invocation.total_retries} retries")
    cost = "unknown" if invocation.estimated_cost is None else f"${invocation.estimated_cost:.6f}"
    print(f"  {invocation.total_tokens} tokens, {cost}")
    for attempt in invocation.attempts:
        print(f"    attempt {attempt.attempt_number}: {attempt.provider}/{attempt.selected_model}"
              f" -> {attempt.status} ({attempt.latency_ms:.0f}ms)")

# Aggregate by any attribution dimension
by_workflow = client.get_llm_usage_summary(group_by="workflow_id")
by_agent = client.get_llm_usage_summary(group_by="agent_id")
by_tool = client.get_llm_usage_summary(group_by="tool_name")
by_provider = client.get_llm_usage_summary(group_by="provider")

# Which attempt caused a latency or cost spike?
slowest = client.telemetry.get_slowest_llm_attempts(limit=5)
costliest = client.telemetry.get_costliest_llm_attempts(limit=5)

# Ship it somewhere else
export = client.export_llm_invocations()
```

`get_llm_usage_summary` accepts any `LLMInvocation` field as `group_by`, so
the same call answers "which workflow used the most tokens", "which agent
generated the most retries", "which tenant/workspace/user generated the
usage", etc.; just group by `workflow_id`, `agent_id`, `tenant_id`,
`workspace_id`, or `user_id`.

Each summary bucket reports `estimated_cost` as the sum of invocations with
known pricing and `unknown_cost_invocations` as the number excluded from that
sum. When every invocation in a bucket has unknown pricing, `estimated_cost`
is `None`; a known zero-dollar cost remains `0.0`.

### Attaching attribution: workflow, agent, tool, tenant, ...

HippocampAI has no built-in workflow/agent orchestration engine. Attribution
is attached by *your* application code via `llm_trace_context`, an immutable
context object propagated with `contextvars` (safe across threads and
asyncio tasks, never leaks between concurrent calls):

```python
from hippocampai import llm_trace_context

with llm_trace_context(
    tenant_id="acme-corp",
    workspace_id="ws-42",
    user_id="user-99",
    session_id="sess-1",
    conversation_id="conv-1",
    workflow_id="research-pipeline",
    workflow_name="Research Pipeline",     # optional human-readable label
    workflow_step_id="draft-outline",
    workflow_step_name="Draft Outline",    # optional human-readable label
    agent_id="research-agent",
    agent_name="Research Agent",           # optional human-readable label
    tool_name="web_search",       # set when the call originated from a tool
    feature_name="draft_outline", # becomes the traced operation name
    billing_bucket="team-growth",
    metadata={"experiment": "prompt-v3"},
):
    client.llm.chat(messages)  # every field above is attached automatically

# Contexts nest and merge: inner blocks add to, not replace, outer fields:
with llm_trace_context(workflow_id="wf-1", agent_id="agent-1"):
    with llm_trace_context(workflow_step_id="step-1"):
        client.llm.chat(messages)  # tagged with workflow_id, agent_id, AND workflow_step_id
```

Nothing needs to be threaded through function signatures, and calls made
outside any `llm_trace_context` block simply have `None` for every
attribution field; existing callers are unaffected.

### Retry and fallback tracing

Every provider adapter records one `UpstreamAttempt` per real request, each
with its own unique `attempt_id` and 1-based `attempt_number`; retries are
never collapsed into a single record. A retry (rate limit, timeout,
transient connection error) produces attempt 2, 3, ... under the same
invocation automatically, nothing extra to configure:

```python
with llm_trace_context(workflow_id="wf-1"):
    result = client.llm.chat(messages)  # e.g. times out once, then succeeds

invocation = client.get_recent_llm_invocations(limit=1)[0]
assert invocation.total_attempts == 2
assert invocation.attempts[0].attempt_id != invocation.attempts[1].attempt_id
assert invocation.attempts[0].status == "error"
assert invocation.attempts[0].error_type == "timeout"
assert invocation.attempts[1].status == "success"
```

For a complete, runnable, offline proof of this (workflow → step → logical
invocation → retryable failure → provider fallback → success, with full
sanitized JSON output), see `scripts/validate_llm_tracing.py`:

```bash
python scripts/validate_llm_tracing.py
```

Model/provider **fallback** is not something the four built-in adapters do
today (each is pinned to one provider/model), but the trace contract fully
supports it for a router or wrapper that does. Record a fallback attempt by
passing `fallback_reason`/`routing_reason`/`selected_model` to `llm_attempt`:

```python
from hippocampai.telemetry import llm_invocation, llm_attempt, UpstreamMetadata

with llm_invocation(provider="openai", requested_model="gpt-4o", operation="chat") as inv_id:
    try:
        with llm_attempt(provider="openai", requested_model="gpt-4o"):
            raise TimeoutError("primary provider timed out")
    except TimeoutError:
        pass  # fall back to a secondary provider

    with llm_attempt(
        provider="anthropic",
        requested_model="gpt-4o",
        selected_model="claude-3-haiku-20240307",
        fallback_reason="provider_unavailable",
        routing_reason="primary_timeout",
    ) as attempt:
        # ... make the real fallback request, then:
        attempt.input_tokens, attempt.output_tokens = 8, 4
```

### Cost estimation

Cost is computed from per-token pricing you configure. HippocampAI ships
**no built-in price catalog**, so unpriced usage estimates as `None` rather
than a guessed number:

```bash
# Dollars per 1M tokens, keyed by "<provider>:<model>" (or bare "<model>")
LLM_PRICING='{"openai:gpt-4o-mini": {"input": 0.15, "output": 0.6, "cached_input": 0.075}, "anthropic:claude-3-5-sonnet-20241022": {"input": 3.0, "output": 15.0}}'
LLM_COST_ESTIMATION_ENABLED=true   # default
LLM_TELEMETRY_MAX_METADATA_BYTES=8192  # custom metadata is truncated beyond this
```

The metadata limit is a strict JSON-encoded byte boundary with a minimum of
two bytes. Oversized metadata is replaced by a bounded truncation summary;
for a budget too small to hold that marker, the stored metadata is `{}`.

`rates` may also include `"reasoning"` (falls back to the `output` rate if
omitted). `invocation.estimated_cost` is the sum of every attempt's cost
(so a failed attempt that still burned tokens before erroring is included);
it is `None` whenever no rate is configured for that provider/model, and
`None` (not summed as zero) when every attempt itself has unknown cost.

### OpenAI-compatible upstream metadata (e.g. NovaRouteAI or your own router)

If you route requests through an OpenAI-compatible proxy or a router such as
NovaRouteAI, `UpstreamMetadata` normalizes its response the same way the
built-in adapters do; HippocampAI never depends on any specific router:

```python
from hippocampai.telemetry import UpstreamMetadata, llm_attempt

with llm_attempt(provider="my-router", requested_model="gpt-4o-mini") as attempt:
    response = my_openai_compatible_client.chat.completions.create(...)
    # Works for any OpenAI-compatible response shape (choices/usage/id/model),
    # plus optional headers the router adds (x-request-id, routing hints, ...).
    attempt.apply_metadata(
        UpstreamMetadata.from_openai_response(response, headers=dict(response_headers))
    )
```

All `UpstreamMetadata` fields are optional. A router that doesn't expose a
`provider_request_id` or per-token cache/reasoning counts simply leaves
those fields unset; nothing breaks.

### Privacy

By default, telemetry stores **operational metadata only** - identifiers,
token counts, latency, status, error classification. It never stores raw
prompt or response text. Error messages are sanitized before being stored:
API keys, `Authorization`/`Bearer` headers, and common provider key formats
are redacted, and messages are truncated to 500 characters.

### Disabling

LLM usage tracing shares the same enabled flag as the rest of telemetry:

```python
client = MemoryClient(enable_telemetry=False)
```

When disabled, provider calls behave identically but no `LLMInvocation` or
`UpstreamAttempt` is recorded - the overhead is a handful of `is-disabled`
checks per call.

### Streaming

No provider adapter in this repository implements streaming today (`chat`/
`generate` return a plain `str`). The trace contract already carries
`streaming` and `time_to_first_token_ms` fields on `LLMInvocation` for when
a streaming adapter is added - a caller wrapping a stream can set
`invocation.time_to_first_token_ms` on first-chunk receipt without
consuming or altering the stream itself.

## Next Steps

- See [CONFIGURATION.md](CONFIGURATION.md) for telemetry configuration options
- Check [ARCHITECTURE.md](ARCHITECTURE.md) for system internals
- Read [API_REFERENCE.md](API_REFERENCE.md) for complete API reference
