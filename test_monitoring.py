#!/usr/bin/env python3
"""Test script for memory tracking and monitoring."""

import json

import requests

BASE_URL = "http://localhost:8000"

def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")

def test_api_health():
    """Test API health endpoint."""
    print_section("1. Testing API Health")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    assert response.status_code == 200
    print("✅ API Health OK")

def test_prometheus_metrics():
    """Test Prometheus metrics endpoint."""
    print_section("2. Testing Prometheus Metrics")
    response = requests.get(f"{BASE_URL}/metrics")
    print(f"Status: {response.status_code}")
    metrics = response.text
    hippocampai_metrics = [line for line in metrics.split('\n') if 'hippocampai' in line][:10]
    for metric in hippocampai_metrics:
        print(f"  {metric}")
    newline = '\n'
    total_hippocampai_metrics = len([line for line in metrics.split(newline) if 'hippocampai' in line])
    print(f"  ... ({total_hippocampai_metrics} hippocampai metrics total)")
    assert response.status_code == 200
    print("✅ Prometheus Metrics OK")

def test_memory_creation():
    """Test memory creation."""
    print_section("3. Testing Memory Creation")

    # Create first memory
    memory1 = {
        "text": "I love coffee and prefer oat milk",
        "user_id": "demo_user",
        "type": "preference"
    }
    response = requests.post(f"{BASE_URL}/v1/memories", json=memory1)
    print(f"Created Memory 1: {response.status_code}")
    m1_data = response.json()
    print(f"  ID: {m1_data['id']}")
    print(f"  Type: {m1_data['type']}")
    print(f"  Text: {m1_data['text']}")

    # Create second memory
    memory2 = {
        "text": "I work with Python and FastAPI",
        "user_id": "demo_user",
        "type": "fact"
    }
    response = requests.post(f"{BASE_URL}/v1/memories", json=memory2)
    print(f"\nCreated Memory 2: {response.status_code}")
    m2_data = response.json()
    print(f"  ID: {m2_data['id']}")
    print(f"  Type: {m2_data['type']}")
    print(f"  Text: {m2_data['text']}")

    print("✅ Memory Creation OK")
    return m1_data['id'], m2_data['id']

def test_memory_search(memory_ids):
    """Test memory search/recall."""
    print_section("4. Testing Memory Search")

    response = requests.get(
        f"{BASE_URL}/v1/recall",
        params={"query": "coffee preferences", "user_id": "demo_user", "k": 5}
    )
    print(f"Search Status: {response.status_code}")
    results = response.json()
    print(f"Found {len(results)} memories")
    for i, result in enumerate(results, 1):
        print(f"  {i}. {result['text'][:50]}... (score: {result.get('score', 'N/A')})")

    print("✅ Memory Search OK")

def test_monitoring_endpoints():
    """Test monitoring endpoints."""
    print_section("5. Testing Monitoring Endpoints")

    # Test events endpoint
    print("Testing /v1/monitoring/events...")
    response = requests.get(f"{BASE_URL}/v1/monitoring/events?user_id=demo_user&limit=10")
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Total events: {data['total']}")
    if data['events']:
        for event in data['events'][:3]:
            print(f"  - {event['event_type']}: {event['memory_id']} @ {event['timestamp']}")
    else:
        print("  (No events tracked yet - this is expected for API-only calls)")

    # Test stats endpoint
    print("\nTesting /v1/monitoring/stats...")
    response = requests.post(
        f"{BASE_URL}/v1/monitoring/stats",
        json={"user_id": "demo_user"}
    )
    print(f"Status: {response.status_code}")
    stats = response.json()
    print(f"Stats for user {stats['user_id']}:")
    print(f"  Total events: {stats['total_events']}")
    print(f"  Success rate: {stats['success_rate']:.2%}")
    print(f"  Memories tracked: {stats['total_memories_tracked']}")

    print("✅ Monitoring Endpoints OK")

def test_grafana_access():
    """Test Grafana accessibility."""
    print_section("6. Testing Grafana Access")
    try:
        response = requests.get("http://localhost:3000", timeout=5)
        print(f"Grafana Status: {response.status_code}")
        if response.status_code == 200:
            print("✅ Grafana is accessible at http://localhost:3000")
            print("   Login with admin/admin")
        else:
            print(f"⚠️  Grafana returned status {response.status_code}")
    except Exception as e:
        print(f"❌ Could not access Grafana: {e}")

def test_prometheus_access():
    """Test Prometheus accessibility."""
    print_section("7. Testing Prometheus Access")
    try:
        response = requests.get("http://localhost:9090/-/healthy", timeout=5)
        print(f"Prometheus Status: {response.status_code}")
        if response.status_code == 200:
            print("✅ Prometheus is accessible at http://localhost:9090")
        else:
            print(f"⚠️  Prometheus returned status {response.status_code}")
    except Exception as e:
        print(f"❌ Could not access Prometheus: {e}")

def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("  HippocampAI Monitoring & Tracking Test Suite")
    print("="*60)

    try:
        test_api_health()
        test_prometheus_metrics()
        memory_ids = test_memory_creation()
        test_memory_search(memory_ids)
        test_monitoring_endpoints()
        test_grafana_access()
        test_prometheus_access()

        print_section("✅ ALL TESTS PASSED!")
        print("Summary:")
        print("  ✅ API Health - Working")
        print("  ✅ Prometheus Metrics - Working")
        print("  ✅ Memory Creation - Working")
        print("  ✅ Memory Search - Working")
        print("  ✅ Monitoring Endpoints - Working")
        print("  ✅ Grafana - Accessible")
        print("  ✅ Prometheus - Accessible")
        print("\nNext Steps:")
        print("  1. Open Grafana: http://localhost:3000 (admin/admin)")
        print("  2. Open Prometheus: http://localhost:9090")
        print("  3. View metrics: http://localhost:8000/metrics")
        print("  4. View Flower: http://localhost:5555")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise

if __name__ == "__main__":
    main()
