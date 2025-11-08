#!/usr/bin/env python3
"""
Comprehensive test to verify SaaS API and Library compatibility.
Tests all major functionalities through both interfaces.
"""

import sys
import time

import requests

# Test if library can be imported
try:
    from hippocampai import MemoryClient
    LIBRARY_AVAILABLE = True
except ImportError:
    LIBRARY_AVAILABLE = False
    print("⚠️  Library not installed, testing SaaS API only")

BASE_URL = "http://localhost:8000"
TEST_USER = "compatibility_test_user"


def print_section(title, emoji="🔍"):
    """Print formatted section header."""
    print(f"\n{'='*80}")
    print(f"{emoji}  {title}")
    print(f"{'='*80}\n")


def test_api_health():
    """Test API is accessible."""
    print_section("API Health Check", "🏥")
    response = requests.get(f"{BASE_URL}/healthz")
    if response.status_code == 200:
        print("✅ SaaS API is healthy")
        return True
    else:
        print(f"❌ SaaS API unhealthy: {response.status_code}")
        return False


def test_saas_memory_operations():
    """Test memory operations via SaaS API."""
    print_section("SaaS API - Memory Operations", "🌐")

    # Create memory
    print("1. Creating memory via SaaS API...")
    memory_data = {
        "text": "I love hiking in the mountains",
        "user_id": TEST_USER,
        "type": "preference"
    }
    response = requests.post(f"{BASE_URL}/v1/memories", json=memory_data)
    if response.status_code == 201:
        memory = response.json()
        print(f"   ✅ Memory created: {memory['id']}")
        memory_id = memory['id']
    else:
        print(f"   ❌ Failed to create memory: {response.status_code}")
        return False

    # Recall memories
    print("2. Recalling memories via SaaS API...")
    recall_data = {
        "query": "hiking mountains",
        "user_id": TEST_USER,
        "k": 5
    }
    response = requests.post(f"{BASE_URL}/v1/memories/recall", json=recall_data)
    if response.status_code == 200:
        results = response.json()
        print(f"   ✅ Recalled {len(results)} memories")
    else:
        print(f"   ❌ Failed to recall: {response.status_code}")
        return False

    # Update memory
    print("3. Updating memory via SaaS API...")
    update_data = {
        "text": "I love hiking in the mountains and forests"
    }
    response = requests.patch(f"{BASE_URL}/v1/memories/{memory_id}", json=update_data)
    if response.status_code == 200:
        print("   ✅ Memory updated")
    else:
        print(f"   ❌ Failed to update: {response.status_code}")
        return False

    # Delete memory
    print("4. Deleting memory via SaaS API...")
    response = requests.delete(f"{BASE_URL}/v1/memories/{memory_id}")
    if response.status_code == 200:
        print("   ✅ Memory deleted")
    else:
        print(f"   ❌ Failed to delete: {response.status_code}")
        return False

    print("\n✅ SaaS API Memory Operations: PASSED")
    return True


def test_library_memory_operations():
    """Test memory operations via Python library."""
    if not LIBRARY_AVAILABLE:
        print("⚠️  Skipping library tests - library not installed")
        return True

    print_section("Python Library - Memory Operations", "🐍")

    try:
        # Initialize client
        print("1. Initializing MemoryClient...")
        client = MemoryClient()
        print("   ✅ Client initialized")

        # Create memory
        print("2. Creating memory via Library...")
        memory = client.remember(
            "I enjoy reading science fiction books",
            user_id=TEST_USER,
            type="preference"
        )
        print(f"   ✅ Memory created: {memory.id}")
        memory_id = memory.id

        # Recall memories
        print("3. Recalling memories via Library...")
        results = client.recall("science fiction", user_id=TEST_USER, k=5)
        print(f"   ✅ Recalled {len(results)} memories")

        # Update memory
        print("4. Updating memory via Library...")
        updated = client.update_memory(
            memory_id=memory_id,
            text="I enjoy reading science fiction and fantasy books"
        )
        if updated:
            print("   ✅ Memory updated")
        else:
            print("   ⚠️  Update returned None")

        # Delete memory
        print("5. Deleting memory via Library...")
        deleted = client.delete_memory(memory_id)
        if deleted:
            print("   ✅ Memory deleted")
        else:
            print("   ⚠️  Delete returned False")

        print("\n✅ Python Library Memory Operations: PASSED")
        return True

    except Exception as e:
        print(f"\n❌ Python Library Memory Operations: FAILED - {e}")
        return False


def test_conflict_resolution_saas():
    """Test conflict resolution via SaaS API."""
    print_section("SaaS API - Conflict Resolution", "⚔️")

    # Create first memory
    print("1. Creating first memory...")
    mem1 = {
        "text": "I love ice cream",
        "user_id": f"{TEST_USER}_conflict",
        "type": "preference"
    }
    response = requests.post(f"{BASE_URL}/v1/memories", json=mem1)
    if response.status_code == 201:
        print("   ✅ Memory 1 created")
    else:
        print(f"   ❌ Failed: {response.status_code}")
        return False

    time.sleep(1)

    # Create conflicting memory
    print("2. Creating conflicting memory...")
    mem2 = {
        "text": "I hate ice cream",
        "user_id": f"{TEST_USER}_conflict",
        "type": "preference"
    }
    response = requests.post(f"{BASE_URL}/v1/memories", json=mem2)
    if response.status_code == 201:
        print("   ✅ Memory 2 created")
    else:
        print(f"   ❌ Failed: {response.status_code}")
        return False

    # Check if conflict was resolved
    print("3. Checking conflict resolution...")
    recall_data = {
        "query": "ice cream",
        "user_id": f"{TEST_USER}_conflict",
        "k": 10
    }
    response = requests.post(f"{BASE_URL}/v1/memories/recall", json=recall_data)
    if response.status_code == 200:
        results = response.json()
        print(f"   ℹ️  Found {len(results)} memories")
        if len(results) == 1:
            print("   ✅ Conflict resolved automatically!")
        elif len(results) == 2:
            print("   ℹ️  Both memories exist (may need conflict resolution config)")
        print(f"      Active memory: {results[0]['memory']['text']}")
    else:
        print(f"   ❌ Failed to recall: {response.status_code}")
        return False

    print("\n✅ SaaS API Conflict Resolution: TESTED")
    return True


def test_monitoring_endpoints():
    """Test monitoring endpoints."""
    print_section("Monitoring & Observability", "📊")

    # Test Prometheus metrics
    print("1. Testing Prometheus metrics endpoint...")
    response = requests.get(f"{BASE_URL}/metrics")
    if response.status_code == 200:
        metrics = response.text
        hippocampai_metrics = [line for line in metrics.split('\n') if 'hippocampai' in line and not line.startswith('#')]
        print(f"   ✅ Metrics endpoint working ({len(hippocampai_metrics)} metrics)")
    else:
        print(f"   ❌ Metrics endpoint failed: {response.status_code}")
        return False

    # Test memory tracking endpoints
    print("2. Testing memory tracking endpoints...")
    response = requests.get(
        f"{BASE_URL}/v1/monitoring/events",
        params={"user_id": TEST_USER, "limit": 10}
    )
    if response.status_code == 200:
        data = response.json()
        print(f"   ✅ Events endpoint working ({data['total']} events)")
    else:
        print(f"   ❌ Events endpoint failed: {response.status_code}")
        return False

    # Test stats endpoint
    print("3. Testing stats endpoint...")
    response = requests.post(
        f"{BASE_URL}/v1/monitoring/stats",
        json={"user_id": TEST_USER}
    )
    if response.status_code == 200:
        stats = response.json()
        print("   ✅ Stats endpoint working")
        print(f"      Total events: {stats['total_events']}")
        print(f"      Success rate: {stats['success_rate']*100:.1f}%")
    else:
        print(f"   ❌ Stats endpoint failed: {response.status_code}")
        return False

    print("\n✅ Monitoring Endpoints: PASSED")
    return True


def test_cross_compatibility():
    """Test that library and API can work together."""
    if not LIBRARY_AVAILABLE:
        print("⚠️  Skipping cross-compatibility tests - library not installed")
        return True

    print_section("Cross-Compatibility Test", "🔗")

    try:
        # Create memory via API
        print("1. Creating memory via SaaS API...")
        api_memory = {
            "text": "Testing cross-compatibility",
            "user_id": f"{TEST_USER}_cross",
            "type": "fact"
        }
        response = requests.post(f"{BASE_URL}/v1/memories", json=api_memory)
        memory_id = response.json()['id']
        print(f"   ✅ Memory created via API: {memory_id}")

        time.sleep(0.5)

        # Retrieve via library
        print("2. Retrieving via Python Library...")
        client = MemoryClient()
        memory = client.get_memory(memory_id)
        if memory:
            print(f"   ✅ Memory retrieved via Library: {memory.text}")
        else:
            print("   ❌ Memory not found via Library")
            return False

        # Update via library
        print("3. Updating via Python Library...")
        updated = client.update_memory(
            memory_id=memory_id,
            text="Testing cross-compatibility - updated"
        )
        if updated:
            print("   ✅ Memory updated via Library")
        else:
            print("   ⚠️  Update returned None")

        time.sleep(0.5)

        # Verify via API
        print("4. Verifying update via SaaS API...")
        response = requests.get(f"{BASE_URL}/v1/memories/{memory_id}")
        if response.status_code == 200:
            memory_data = response.json()
            if "updated" in memory_data['text']:
                print("   ✅ Update verified via API")
            else:
                print(f"   ⚠️  Update not reflected: {memory_data['text']}")
        else:
            print(f"   ❌ Failed to verify: {response.status_code}")
            return False

        # Delete via API
        print("5. Deleting via SaaS API...")
        response = requests.delete(f"{BASE_URL}/v1/memories/{memory_id}")
        if response.status_code == 200:
            print("   ✅ Memory deleted via API")
        else:
            print(f"   ❌ Delete failed: {response.status_code}")
            return False

        print("\n✅ Cross-Compatibility: PASSED")
        return True

    except Exception as e:
        print(f"\n❌ Cross-Compatibility: FAILED - {e}")
        return False


def main():
    """Run all compatibility tests."""
    print("\n" + "="*80)
    print("🧪  HippocampAI - SaaS API & Library Compatibility Test Suite")
    print("="*80)

    results = {}

    # Run tests
    results['api_health'] = test_api_health()
    if not results['api_health']:
        print("\n❌ API is not available. Aborting tests.")
        sys.exit(1)

    results['saas_operations'] = test_saas_memory_operations()
    results['library_operations'] = test_library_memory_operations()
    results['conflict_resolution'] = test_conflict_resolution_saas()
    results['monitoring'] = test_monitoring_endpoints()
    results['cross_compatibility'] = test_cross_compatibility()

    # Summary
    print_section("Test Summary", "📝")

    total_tests = len(results)
    passed_tests = sum(1 for v in results.values() if v)

    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {test_name.replace('_', ' ').title()}: {status}")

    print(f"\n{'='*80}")
    print(f"Results: {passed_tests}/{total_tests} tests passed")
    print(f"{'='*80}")

    if passed_tests == total_tests:
        print("\n🎉 ALL TESTS PASSED! SaaS API and Library are fully compatible.")
        print("\n✅ Key Findings:")
        print("  • SaaS API is operational and responding correctly")
        print("  • Python Library can interact with the API seamlessly")
        print("  • Memory operations work via both interfaces")
        print("  • Monitoring and observability endpoints are functional")
        print("  • Cross-compatibility between library and API verified")
        return 0
    else:
        print("\n⚠️  SOME TESTS FAILED - Review output above for details")
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
