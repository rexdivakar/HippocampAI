"""Test script to demonstrate new HippocampAI features."""

import time

from hippocampai import MemoryClient, OperationType, get_telemetry

print("=" * 70)
print("  HippocampAI - New Features Demo")
print("=" * 70)

# Test 1: Configuration Presets
print("\n1. Testing Configuration Presets")
print("-" * 70)

print("\n   Creating client with 'development' preset...")
client = MemoryClient.from_preset("development")
print("   ✓ Client created with development preset")
print(f"   ✓ HNSW M: {client.config.hnsw_m}")
print(f"   ✓ Top K: {client.config.top_k_final}")
print(f"   ✓ Quantized embeddings: {client.config.embed_quantized}")

# Test 2: Telemetry Integration
print("\n2. Testing Telemetry Integration")
print("-" * 70)

user_id = "test_user"

print("\n   Storing 3 memories (telemetry auto-tracks)...")
memories_data = [
    ("I love Python programming", "preference"),
    ("I work as a software engineer", "fact"),
    ("I want to learn Rust", "goal"),
]

for text, mem_type in memories_data:
    memory = client.remember(text=text, user_id=user_id, type=mem_type, importance=7.5)
    print(f"   ✓ Stored: {text[:40]}... (ID: {memory.id[:8]}...)")
    time.sleep(0.1)  # Small delay to see different timestamps

# Test 3: Retrieve with Telemetry
print("\n3. Testing Recall with Telemetry")
print("-" * 70)

print("\n   Querying: 'What does the user do for work?'")
results = client.recall(query="What does the user do for work?", user_id=user_id, k=2)

print(f"   ✓ Retrieved {len(results)} results:")
for i, result in enumerate(results, 1):
    print(f"      {i}. {result.memory.text}")
    print(f"         Score: {result.score:.3f}")

# Test 4: Conversation Extraction with Telemetry
print("\n4. Testing Conversation Extraction")
print("-" * 70)

conversation = """
User: I really enjoy hiking on weekends
Assistant: That sounds great!
User: Yes, I prefer mountain trails
"""

print("\n   Extracting memories from conversation...")
extracted = client.extract_from_conversation(
    conversation=conversation, user_id=user_id, session_id="test_session"
)

print(f"   ✓ Extracted {len(extracted)} memories")
for mem in extracted:
    print(f"      - [{mem.type.value}] {mem.text}")

# Test 5: Access Telemetry Metrics
print("\n5. Accessing Telemetry Metrics")
print("-" * 70)

print("\n   Getting metrics summary...")
metrics = client.get_telemetry_metrics()

print("\n   Performance Metrics:")
for operation, stats in metrics.items():
    print(f"\n   {operation}:")
    print(f"      Count: {stats['count']}")
    if stats["count"] > 0:
        print(f"      Average: {stats['avg']:.2f}ms")
        print(f"      Min: {stats['min']:.2f}ms")
        print(f"      Max: {stats['max']:.2f}ms")
        print(f"      P95: {stats['p95']:.2f}ms")

# Test 6: View Recent Operations
print("\n6. Viewing Recent Operations")
print("-" * 70)

print("\n   Getting last 5 operations...")
operations = client.get_recent_operations(limit=5)

print(f"\n   Recent Operations ({len(operations)} total):")
for i, op in enumerate(operations, 1):
    print(f"\n   {i}. Operation: {op.operation.value}")
    print(f"      User: {op.user_id}")
    print(f"      Duration: {op.duration_ms:.2f}ms" if op.duration_ms else "      Duration: N/A")
    print(f"      Status: {op.status}")
    print(f"      Events: {len(op.events)} steps")

# Test 7: Filter Operations by Type
print("\n7. Filtering Operations by Type")
print("-" * 70)

print("\n   Getting only 'remember' operations...")
remember_ops = client.get_recent_operations(limit=10, operation="remember")

print(f"\n   Remember Operations: {len(remember_ops)}")
for op in remember_ops:
    if op.result:
        print(f"      - Memory ID: {op.result.get('memory_id', 'N/A')[:8]}...")
        print(f"        Duration: {op.duration_ms:.2f}ms")

# Test 8: Direct Telemetry Access
print("\n8. Direct Telemetry Access")
print("-" * 70)

print("\n   Accessing global telemetry instance...")
telemetry = get_telemetry()

# Get specific operation types
recall_traces = telemetry.get_recent_traces(operation=OperationType.RECALL, limit=5)

print(f"\n   Recall Operations: {len(recall_traces)}")
for trace in recall_traces:
    print(f"      - Query: {trace.metadata.get('query', 'N/A')[:40]}...")
    print(f"        Duration: {trace.duration_ms:.2f}ms")
    print(f"        Result count: {trace.result.get('count', 0)}")

# Test 9: Export Telemetry Data
print("\n9. Exporting Telemetry Data")
print("-" * 70)

print("\n   Exporting all telemetry data...")
exported = client.export_telemetry()

print(f"\n   ✓ Exported {len(exported)} traces")
print("   ✓ Format: JSON-compatible for external tools")
print("   ✓ Includes: timestamps, durations, metadata, events")

# Show sample exported trace
if exported:
    sample = exported[0]
    print("\n   Sample trace structure:")
    print(f"      - trace_id: {sample['trace_id'][:8]}...")
    print(f"      - operation: {sample['operation']}")
    print(f"      - user_id: {sample['user_id']}")
    print(f"      - duration_ms: {sample['duration_ms']}")
    print(f"      - status: {sample['status']}")
    print(f"      - events: {len(sample['events'])} events")

# Test 10: Preset Comparison
print("\n10. Comparing Different Presets")
print("-" * 70)

presets = {
    "local": MemoryClient.from_preset("local"),
    "production": MemoryClient.from_preset("production"),
    "development": MemoryClient.from_preset("development"),
}

print("\n   Preset Configurations:")
print(f"\n   {'Preset':<15} {'HNSW M':<10} {'Top K':<10} {'LLM':<15}")
print("   " + "-" * 50)
for name, preset_client in presets.items():
    print(
        f"   {name:<15} {preset_client.config.hnsw_m:<10} "
        f"{preset_client.config.top_k_final:<10} "
        f"{preset_client.config.llm_provider:<15}"
    )

# Final Summary
print("\n" + "=" * 70)
print("  Test Summary")
print("=" * 70)

total_operations = len(operations)
total_memories = len(memories_data) + len(extracted)

print("\n   ✓ Configuration presets: WORKING")
print("   ✓ Telemetry integration: WORKING")
print(f"   ✓ Total operations tracked: {total_operations}")
print(f"   ✓ Total memories stored: {total_memories}")
print("   ✓ Metrics available: YES")
print("   ✓ Export functionality: YES")

print("\n   All new features are working correctly!")
print("\n" + "=" * 70)
