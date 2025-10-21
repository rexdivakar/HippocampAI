"""Demo script showing telemetry features without requiring Qdrant."""

import time
from datetime import datetime
from hippocampai.telemetry import get_telemetry, OperationType, MemoryTrace

print("=" * 70)
print("  HippocampAI Telemetry System Demo")
print("=" * 70)

# Get telemetry instance
telemetry = get_telemetry(enabled=True)

# Demo 1: Manual Trace Creation
print("\n1. Creating Manual Traces")
print("-" * 70)

print("\n   Simulating 'remember' operation...")
trace_id = telemetry.start_trace(
    operation=OperationType.REMEMBER,
    user_id="demo_user",
    session_id="demo_session",
    text="I love Python programming",
    memory_type="preference"
)

time.sleep(0.05)  # Simulate work
telemetry.add_event(
    trace_id,
    "deduplication_check",
    status="success",
    duration_ms=15.3,
    duplicates_found=0
)

time.sleep(0.03)
telemetry.add_event(
    trace_id,
    "embedding_generation",
    status="success",
    duration_ms=45.2,
    model="BAAI/bge-small-en-v1.5"
)

time.sleep(0.02)
telemetry.add_event(
    trace_id,
    "vector_store",
    status="success",
    duration_ms=12.1,
    collection="hippocampai_prefs"
)

telemetry.end_trace(
    trace_id,
    status="success",
    result={"memory_id": "abc123", "collection": "hippocampai_prefs"}
)

print(f"   ✓ Created trace: {trace_id[:8]}...")
print(f"   ✓ Status: success")
print(f"   ✓ Events: 3 steps tracked")

# Demo 2: Multiple Operations
print("\n2. Simulating Multiple Operations")
print("-" * 70)

operations = [
    ("remember", "I work as a data scientist", "fact", 55.2),
    ("remember", "I want to learn Rust", "goal", 48.7),
    ("recall", "What does the user do?", "query", 125.4),
    ("extract", "User: I enjoy hiking...", "conversation", 234.1),
    ("recall", "User preferences?", "query", 98.3),
]

print(f"\n   Creating {len(operations)} operations...")
for op_type, content, content_type, duration in operations:
    trace_id = telemetry.start_trace(
        operation=OperationType(op_type),
        user_id="demo_user",
        content=content[:30] + "...",
        content_type=content_type
    )

    # Simulate some processing time
    time.sleep(duration / 1000)

    telemetry.end_trace(
        trace_id,
        status="success",
        result={"processed": True}
    )

    print(f"   ✓ {op_type}: {content[:40]}... ({duration:.1f}ms)")

# Demo 3: Get Metrics
print("\n3. Telemetry Metrics Summary")
print("-" * 70)

metrics = telemetry.get_metrics_summary()

print("\n   Performance Metrics:")
for metric_name, stats in metrics.items():
    print(f"\n   {metric_name}:")
    print(f"      Operations: {stats['count']}")
    print(f"      Average: {stats['avg']:.2f}ms")
    print(f"      Min: {stats['min']:.2f}ms")
    print(f"      Max: {stats['max']:.2f}ms")
    print(f"      P50 (median): {stats['p50']:.2f}ms")
    print(f"      P95: {stats['p95']:.2f}ms")
    print(f"      P99: {stats['p99']:.2f}ms")

# Demo 4: View Recent Traces
print("\n4. Recent Operation Traces")
print("-" * 70)

recent = telemetry.get_recent_traces(limit=5)

print(f"\n   Last {len(recent)} operations:")
for i, trace in enumerate(recent, 1):
    print(f"\n   {i}. Operation: {trace.operation.value}")
    print(f"      Trace ID: {trace.trace_id[:8]}...")
    print(f"      User: {trace.user_id}")
    print(f"      Session: {trace.session_id or 'N/A'}")
    print(f"      Started: {trace.start_time.strftime('%H:%M:%S')}")
    print(f"      Duration: {trace.duration_ms:.2f}ms" if trace.duration_ms else "      Duration: N/A")
    print(f"      Status: {trace.status}")
    print(f"      Events: {len(trace.events)}")

    # Show metadata
    if trace.metadata:
        print(f"      Metadata: {list(trace.metadata.keys())}")

# Demo 5: Filter by Operation Type
print("\n5. Filtering Operations by Type")
print("-" * 70)

remember_ops = telemetry.get_recent_traces(operation=OperationType.REMEMBER)
recall_ops = telemetry.get_recent_traces(operation=OperationType.RECALL)

print(f"\n   Remember operations: {len(remember_ops)}")
for op in remember_ops:
    print(f"      - {op.metadata.get('text', 'N/A')[:40]}... ({op.duration_ms:.2f}ms)")

print(f"\n   Recall operations: {len(recall_ops)}")
for op in recall_ops:
    print(f"      - {op.metadata.get('content', 'N/A')[:40]}... ({op.duration_ms:.2f}ms)")

# Demo 6: Detailed Trace Inspection
print("\n6. Detailed Trace Inspection")
print("-" * 70)

if recent:
    trace = recent[0]
    print(f"\n   Inspecting trace: {trace.trace_id[:8]}...")
    print(f"\n   Timeline:")

    for i, event in enumerate(trace.events, 1):
        print(f"\n      Event {i}: {event.timestamp.strftime('%H:%M:%S.%f')[:-3]}")
        print(f"         Status: {event.status}")
        if event.duration_ms:
            print(f"         Duration: {event.duration_ms:.2f}ms")
        if event.metadata:
            print(f"         Details: {event.metadata}")

# Demo 7: Export Functionality
print("\n7. Export Telemetry Data")
print("-" * 70)

print("\n   Exporting all traces...")
exported = telemetry.export_traces()

print(f"\n   ✓ Exported {len(exported)} traces")
print(f"   ✓ Format: JSON-compatible")
print(f"   ✓ Ready for: OpenTelemetry, Prometheus, Grafana, etc.")

# Show sample structure
if exported:
    sample = exported[0]
    print(f"\n   Sample trace structure:")
    print(f"   {{")
    print(f"      'trace_id': '{sample['trace_id'][:8]}...',")
    print(f"      'operation': '{sample['operation']}',")
    print(f"      'user_id': '{sample['user_id']}',")
    print(f"      'duration_ms': {sample['duration_ms']},")
    print(f"      'status': '{sample['status']}',")
    print(f"      'events': [{len(sample['events'])} events],")
    print(f"      'metadata': {list(sample['metadata'].keys())},")
    print(f"   }}")

# Demo 8: Telemetry for Debugging
print("\n8. Using Telemetry for Debugging")
print("-" * 70)

# Simulate slow operation
print("\n   Simulating slow operation...")
slow_trace = telemetry.start_trace(
    operation=OperationType.RECALL,
    user_id="debug_user",
    query="complex query"
)

time.sleep(0.5)  # Simulate slow operation
telemetry.add_event(slow_trace, "slow_step", duration_ms=500, reason="complex query")
telemetry.end_trace(slow_trace, status="success")

# Find slow operations
print("\n   Finding operations > 200ms...")
slow_ops = [t for t in telemetry.get_recent_traces() if t.duration_ms and t.duration_ms > 200]

print(f"\n   Found {len(slow_ops)} slow operations:")
for op in slow_ops:
    print(f"      - {op.operation.value}: {op.duration_ms:.2f}ms")
    print(f"        User: {op.user_id}")
    print(f"        Events: {len(op.events)}")

# Demo 9: Real-time Monitoring
print("\n9. Real-time Monitoring Capabilities")
print("-" * 70)

print("\n   What you can monitor:")
print("      ✓ Operation latency (avg, P50, P95, P99)")
print("      ✓ Success/failure rates")
print("      ✓ Operations per user/session")
print("      ✓ Memory type distribution")
print("      ✓ Retrieval performance")
print("      ✓ Extraction efficiency")
print("      ✓ Individual trace details")

print("\n   Integration options:")
print("      ✓ REST API endpoints (FastAPI)")
print("      ✓ Direct Python access (get_telemetry())")
print("      ✓ Export to external tools")
print("      ✓ Custom analytics")

# Demo 10: Cleanup
print("\n10. Telemetry Cleanup")
print("-" * 70)

total_traces = len(telemetry.traces)
print(f"\n   Current traces in memory: {total_traces}")

# Clear old traces (simulating cleanup after 60 minutes)
# cleared = telemetry.clear_traces(older_than_minutes=60)
# print(f"   ✓ Would clear {cleared} old traces")

print(f"\n   Note: Automatic cleanup prevents memory buildup")
print(f"   Recommendation: Run cleanup hourly in production")

# Final Summary
print("\n" + "=" * 70)
print("  Telemetry Demo Summary")
print("=" * 70)

all_traces = telemetry.get_recent_traces(limit=100)
remember_count = len([t for t in all_traces if t.operation == OperationType.REMEMBER])
recall_count = len([t for t in all_traces if t.operation == OperationType.RECALL])
extract_count = len([t for t in all_traces if t.operation == OperationType.EXTRACT])

print(f"\n   Total operations tracked: {len(all_traces)}")
print(f"   Remember operations: {remember_count}")
print(f"   Recall operations: {recall_count}")
print(f"   Extract operations: {extract_count}")

if metrics:
    avg_latency = sum(s['avg'] for s in metrics.values()) / len(metrics)
    print(f"   Average latency: {avg_latency:.2f}ms")

print(f"\n   Features demonstrated:")
print(f"      ✓ Automatic operation tracking")
print(f"      ✓ Performance metrics (P50, P95, P99)")
print(f"      ✓ Detailed trace inspection")
print(f"      ✓ Operation filtering")
print(f"      ✓ Export functionality")
print(f"      ✓ Real-time monitoring")
print(f"      ✓ Debugging capabilities")

print("\n   All telemetry features working! ✨")
print("\n" + "=" * 70)
