#!/usr/bin/env python
"""
Full integration test for all HippocampAI features.
Run with: python scripts/test_all_features.py
"""

import os
import sys
from uuid import uuid4

# Default Qdrant URL
QDRANT_URL = "http://100.113.229.40:6333"
os.environ["QDRANT_URL"] = QDRANT_URL

# Track collections for cleanup
_test_collections: list[str] = []


def test_bitemporal():
    """Test bi-temporal fact tracking."""
    print("\n=== Testing Bi-Temporal Facts ===")

    from hippocampai import MemoryClient
    from hippocampai.models.bitemporal import FactStatus

    collection_facts = f"test_facts_{uuid4().hex[:8]}"
    collection_prefs = f"test_prefs_{uuid4().hex[:8]}"
    _test_collections.extend([collection_facts, collection_prefs])

    client = MemoryClient(
        qdrant_url=QDRANT_URL,
        collection_facts=collection_facts,
        collection_prefs=collection_prefs,
    )

    user_id = f"test_user_{uuid4().hex[:8]}"

    # Store a fact
    fact = client.store_bitemporal_fact(
        text="Alice works at Google",
        user_id=user_id,
        entity_id="alice",
        property_name="employer",
    )
    print(f"✓ Stored fact: {fact.text}")
    assert fact.status == FactStatus.ACTIVE

    # Query facts
    result = client.query_bitemporal_facts(user_id=user_id)
    print(f"✓ Queried facts: {len(result.facts)} found")
    assert len(result.facts) >= 1

    # Revise fact
    revised = client.revise_bitemporal_fact(
        original_fact_id=fact.id,
        new_text="Alice works at Microsoft",
        user_id=user_id,
    )
    print(f"✓ Revised fact: {revised.text}")
    assert revised.supersedes == fact.id

    # Get history
    history = client.get_bitemporal_fact_history(fact_id=fact.fact_id)
    print(f"✓ Got history: {len(history)} versions")
    assert len(history) >= 2

    print("✓ Bi-temporal tests PASSED")
    return True


def test_context_assembly():
    """Test automated context assembly."""
    print("\n=== Testing Context Assembly ===")

    from hippocampai import MemoryClient

    collection_facts = f"test_facts_{uuid4().hex[:8]}"
    collection_prefs = f"test_prefs_{uuid4().hex[:8]}"
    _test_collections.extend([collection_facts, collection_prefs])

    client = MemoryClient(
        qdrant_url=QDRANT_URL,
        collection_facts=collection_facts,
        collection_prefs=collection_prefs,
    )

    user_id = f"test_user_{uuid4().hex[:8]}"

    # Add memories
    client.remember("Alice is a software engineer", user_id=user_id, type="fact")
    client.remember("Alice prefers Python", user_id=user_id, type="preference")
    client.remember("Alice works remotely", user_id=user_id, type="fact")
    print("✓ Added 3 memories")

    # Assemble context
    pack = client.assemble_context(
        user_id=user_id,
        query="What does Alice do?",
        token_budget=1000,
    )

    print(f"✓ Assembled context: {len(pack.selected_items)} items")
    print(f"  Total tokens: {pack.total_tokens}")
    if pack.final_context_text:
        preview = pack.final_context_text[:100]
        print(f"  Context: {preview}...")

    assert len(pack.selected_items) > 0
    assert pack.total_tokens > 0

    print("✓ Context assembly tests PASSED")
    return True


def test_custom_schema():
    """Test custom schema support."""
    print("\n=== Testing Custom Schema ===")

    from hippocampai.schema.models import (
        AttributeDefinition,
        EntityTypeDefinition,
        SchemaDefinition,
    )
    from hippocampai.schema.registry import SchemaRegistry
    from hippocampai.schema.validator import SchemaValidator

    # Create schema
    person_type = EntityTypeDefinition(
        name="Person",
        description="A person",
        attributes=[
            AttributeDefinition(name="name", type="string", required=True),
            AttributeDefinition(name="age", type="integer", required=False),
        ],
    )

    schema = SchemaDefinition(
        name="test_schema",
        version="1.0",
        entity_types=[person_type],
        relationship_types=[],
    )
    print("✓ Created schema")

    # Validate entity
    validator = SchemaValidator(schema)

    # Valid entity
    result = validator.validate_entity("Person", {"name": "Alice", "age": 30})
    assert result.valid
    print("✓ Valid entity passed validation")

    # Invalid entity
    result = validator.validate_entity("Person", {"age": 30})
    assert not result.valid
    print("✓ Invalid entity failed validation (as expected)")

    # Registry
    registry = SchemaRegistry()
    registry.register_schema(schema)
    registry.set_active_schema("test_schema")
    print("✓ Registered schema in registry")

    print("✓ Custom schema tests PASSED")
    return True


def test_benchmarks():
    """Test benchmark suite."""
    print("\n=== Testing Benchmarks ===")

    import sys
    from pathlib import Path

    # Add bench to path if not already
    bench_path = Path(__file__).parent.parent / "bench"
    if str(bench_path.parent) not in sys.path:
        sys.path.insert(0, str(bench_path.parent))

    from bench.data_generator import generate_memories, generate_queries
    from bench.runner import run_benchmark, BenchmarkResult

    # Generate data
    memories = list(generate_memories(count=10, num_users=2))
    queries = list(generate_queries(count=5))

    print(f"✓ Generated {len(memories)} memories")
    print(f"✓ Generated {len(queries)} queries")

    # Run benchmark
    import time

    def sample_op():
        time.sleep(0.001)
        return True

    result = run_benchmark("test", sample_op, iterations=5)

    print("✓ Benchmark completed:")
    print(f"  P50 latency: {result.latency_p50_ms:.2f}ms")
    print(f"  Throughput: {result.ops_per_second:.2f} ops/sec")

    assert result.latency_p50_ms > 0
    assert result.ops_per_second > 0

    print("✓ Benchmark tests PASSED")
    return True


def test_api_endpoints():
    """Test API endpoints via HTTP."""
    print("\n=== Testing API Endpoints ===")

    import json

    import httpx

    base_url = "http://100.113.229.40:8000"
    user_id = f"test_user_{uuid4().hex[:8]}"

    # Health check
    try:
        resp = httpx.get(f"{base_url}/healthz", timeout=5)
        assert resp.status_code == 200
        print("✓ Health check passed")
    except Exception as e:
        print(f"✗ API not reachable: {e}")
        print("  Make sure docker compose is running: docker compose up -d")
        return False

    # Store bi-temporal fact
    resp = httpx.post(
        f"{base_url}/v1/bitemporal/facts:store",
        json={
            "text": "Bob works at Amazon",
            "user_id": user_id,
            "entity_id": "bob",
            "property_name": "employer",
        },
        timeout=10,
    )
    assert resp.status_code == 200
    fact_data = resp.json()
    print(f"✓ Stored bi-temporal fact via API: {fact_data['text']}")

    # Remember memory
    resp = httpx.post(
        f"{base_url}/v1/memories:remember",
        json={
            "text": "Bob likes coffee",
            "user_id": user_id,
            "type": "preference",
        },
        timeout=10,
    )
    assert resp.status_code == 200
    print("✓ Stored memory via API")

    # Assemble context
    resp = httpx.post(
        f"{base_url}/v1/context:assemble",
        json={
            "user_id": user_id,
            "query": "What does Bob like?",
            "max_tokens": 500,
        },
        timeout=30,
    )
    assert resp.status_code == 200
    context_data = resp.json()
    print(f"✓ Assembled context via API: {len(context_data['selected_items'])} items")

    print("✓ API endpoint tests PASSED")
    return True


def cleanup_test_collections():
    """Delete all test collections created during the test run."""
    print("\n=== Cleaning Up Test Collections ===")
    
    try:
        from qdrant_client import QdrantClient
        
        client = QdrantClient(url=QDRANT_URL)
        
        for collection_name in _test_collections:
            try:
                client.delete_collection(collection_name)
                print(f"✓ Deleted collection: {collection_name}")
            except Exception as e:
                print(f"✗ Failed to delete {collection_name}: {e}")
        
        print(f"✓ Cleanup complete ({len(_test_collections)} collections)")
    except Exception as e:
        print(f"✗ Cleanup failed: {e}")


def main():
    """Run all tests."""
    print("=" * 50)
    print("HippocampAI Feature Integration Tests")
    print(f"Qdrant URL: {QDRANT_URL}")
    print("=" * 50)

    results = []

    try:
        results.append(("Bi-Temporal", test_bitemporal()))
    except Exception as e:
        print(f"✗ Bi-temporal tests FAILED: {e}")
        results.append(("Bi-Temporal", False))

    try:
        results.append(("Context Assembly", test_context_assembly()))
    except Exception as e:
        print(f"✗ Context assembly tests FAILED: {e}")
        results.append(("Context Assembly", False))

    try:
        results.append(("Custom Schema", test_custom_schema()))
    except Exception as e:
        print(f"✗ Custom schema tests FAILED: {e}")
        results.append(("Custom Schema", False))

    try:
        results.append(("Benchmarks", test_benchmarks()))
    except Exception as e:
        print(f"✗ Benchmark tests FAILED: {e}")
        results.append(("Benchmarks", False))

    try:
        results.append(("API Endpoints", test_api_endpoints()))
    except Exception as e:
        print(f"✗ API endpoint tests FAILED: {e}")
        results.append(("API Endpoints", False))

    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)

    all_passed = True
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    # Cleanup test collections on success
    if all_passed:
        cleanup_test_collections()

    print("=" * 50)
    if all_passed:
        print("All tests PASSED!")
        sys.exit(0)
    else:
        print("Some tests FAILED!")
        print("Note: Test collections were NOT cleaned up for debugging")
        sys.exit(1)


if __name__ == "__main__":
    main()
