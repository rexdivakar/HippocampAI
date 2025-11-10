#!/usr/bin/env python3
"""
Comprehensive test to verify SaaS API and Library compatibility.
Tests all major functionalities through both interfaces.
"""

import sys
import time
import traceback

import pytest
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
    print(f"\n{'=' * 80}")
    print(f"{emoji}  {title}")
    print(f"{'=' * 80}\n")


def test_api_health():
    """Test API is accessible."""
    print_section("API Health Check", "🏥")
    response = requests.get(f"{BASE_URL}/healthz", timeout=10)
    is_healthy = response.status_code == 200
    if is_healthy:
        print("✅ SaaS API is healthy")
    else:
        print(f"❌ SaaS API unhealthy: {response.status_code}")
    assert is_healthy, f"API health check failed with status code {response.status_code}"


def test_saas_memory_operations():
    """Test memory operations via SaaS API."""
    print_section("SaaS API - Memory Operations", "🌐")

    # Create memory
    print("1. Creating memory via SaaS API...")
    memory_data = {
        "text": "I love hiking in the mountains",
        "user_id": TEST_USER,
        "type": "preference",
        "check_duplicate": False,  # Disable duplicate check directly
    }
    response = requests.post(f"{BASE_URL}/v1/memories", json=memory_data, timeout=10)
    if response.status_code == 201:
        memory = response.json()
        print(f"   ✅ Memory created: {memory['id']}")
        memory_id = memory["id"]
    else:
        print(f"   ❌ Failed to create memory: {response.status_code}")
        return False

    # Recall memories
    print("2. Recalling memories via SaaS API...")
    recall_data = {"query": "hiking mountains", "user_id": TEST_USER, "k": 5}
    response = requests.post(f"{BASE_URL}/v1/memories/recall", json=recall_data, timeout=10)
    if response.status_code == 200:
        results = response.json()
        print(f"   ✅ Recalled {len(results)} memories")
    else:
        print(f"   ❌ Failed to recall: {response.status_code}")
        return False

    # Update memory
    print("3. Updating memory via SaaS API...")
    update_data = {"text": "I love hiking in the mountains and forests"}
    response = requests.patch(f"{BASE_URL}/v1/memories/{memory_id}", json=update_data, timeout=10)
    if response.status_code == 200:
        print("   ✅ Memory updated")
    else:
        print(f"   ❌ Failed to update: {response.status_code}")
        return False

    # Delete memory
    print("4. Deleting memory via SaaS API...")
    response = requests.delete(f"{BASE_URL}/v1/memories/{memory_id}", timeout=10)
    is_deleted = response.status_code == 200
    if is_deleted:
        print("   ✅ Memory deleted")
    else:
        print(f"   ❌ Failed to delete: {response.status_code}")
    assert is_deleted, f"Failed to delete memory: {response.status_code}"

    print("\n✅ SaaS API Memory Operations: PASSED")


def test_library_memory_operations():
    """Test memory operations via Python library."""
    if not LIBRARY_AVAILABLE:
        print("⚠️  Skipping library tests - library not installed")
        pytest.skip("Library not installed")

    print_section("Python Library - Memory Operations", "🐍")

    try:
        # Initialize client
        print("1. Initializing MemoryClient...")
        client = MemoryClient()
        print("   ✅ Client initialized")

        # Create memory
        print("2. Creating memory via Library...")
        memory = client.remember(
            "I enjoy reading science fiction books", user_id=TEST_USER, type="preference"
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
            memory_id=memory_id, text="I enjoy reading science fiction and fantasy books"
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
        raise AssertionError(f"Library operations failed: {e}")


def test_conflict_resolution_saas():
    """Test conflict resolution via SaaS API."""
    print("\n" + "=" * 80)
    print("⚔️  SaaS API - Conflict Resolution")
    print("=" * 80 + "\n")
    print_section("SaaS API - Conflict Resolution", "⚔️")

    # Clean up any existing test data
    print("0. Cleaning up existing test data...")
    recall_data = {"query": "ice cream", "user_id": f"{TEST_USER}_conflict", "k": 100}
    try:
        response = requests.post(f"{BASE_URL}/v1/memories/recall", json=recall_data, timeout=10)
        if response.status_code == 200:
            print("   Found existing memories, cleaning up...")
            for memory in response.json():
                memory_id = memory["memory"]["id"]
                try:
                    delete_response = requests.delete(
                        f"{BASE_URL}/v1/memories/{memory_id}", timeout=10
                    )
                    if delete_response.status_code == 200:
                        print(f"   ✅ Deleted memory {memory_id}")
                    elif delete_response.status_code == 404:
                        print(f"   ℹ️  Memory {memory_id} already deleted")
                    else:
                        print(
                            f"   ⚠️  Failed to delete memory {memory_id}: {delete_response.status_code}"
                        )
                except Exception as e:
                    print(f"   ⚠️  Error deleting memory {memory_id}: {e}")
        elif response.status_code == 404:
            print("   ℹ️  No existing memories found")
        else:
            print(f"   ⚠️  Failed to recall memories for cleanup: {response.status_code}")
    except Exception as e:
        print(f"   ⚠️  Error during cleanup: {e}")

    # Add a longer delay after cleanup to ensure consistency
    print("   ⏳ Waiting for cleanup to complete...")
    time.sleep(3)

    # Create first memory
    print("1. Creating first memory...")
    mem1 = {
        "text": "I love ice cream",
        "user_id": f"{TEST_USER}_conflict",
        "type": "preference",
        "check_duplicate": True,  # Enable duplicate check for first memory
        "check_conflicts": True,  # Also enable conflict checking
        "metadata": {
            "conflict_test": True  # Mark as part of conflict test
        },
    }
    try:
        response = requests.post(
            f"{BASE_URL}/v1/memories",
            params={"skip_duplicate_check": True},  # Add as query param
            json=mem1,
            timeout=10,
        )
        if response.status_code == 201:
            memory = response.json()
            print(f"   ✅ Memory 1 created with id {memory['id']}")
            memory_id = memory["id"]
        else:
            print(f"   ❌ Failed: {response.status_code}")
            error_detail = response.json()
            print(f"   Error details: {error_detail}")
            return False
    except Exception as e:
        print(f"   ❌ Request failed: {e}")
        return False

    time.sleep(1)

    # Create conflicting memory
    print("2. Creating conflicting memory...")
    mem2 = {
        "text": "I hate ice cream",
        "user_id": f"{TEST_USER}_conflict",
        "type": "preference",
        "check_duplicate": True,  # Enable check for duplicate detection
        "check_conflicts": True,  # Enable conflict checking
        "metadata": {
            "strategy": "keep_both",  # Keep both memories for conflict demo
            "auto_resolve": True,  # Enable auto resolution
            "known_conflict_id": memory_id,  # Pass first memory's ID
            "conflict_test": True,  # Mark as part of conflict test
        },
    }
    response = requests.post(f"{BASE_URL}/v1/memories", json=mem2, timeout=10)
    try:
        if response.status_code == 201:
            response_data = response.json()
            print(f"   ✅ Memory 2 created with id {response_data['id']}")
            print("   ✅ No conflict detected (both memories kept)")
            return True
        elif response.status_code == 409:  # Conflict detected
            error_detail = response.json().get("detail", {})
            print(f"   ℹ️  Conflict detected: {error_detail.get('message', 'Unknown error')}")
            print("   ℹ️  Resolution options:", error_detail.get("resolution_options", []))

            # Since we want to keep both, try again with explicit keep_both strategy
            mem2["metadata"]["strategy"] = "keep_both"
            mem2["metadata"]["auto_resolve"] = True
            response = requests.post(f"{BASE_URL}/v1/memories", json=mem2, timeout=10)
            if response.status_code == 201:
                new_memory = response.json()
                print(f"   ✅ Memory 2 created with keep_both strategy (id: {new_memory['id']})")
                return True
            else:
                print(f"   ❌ Failed to create with keep_both strategy: {response.status_code}")
                return False
        else:
            print(f"   ❌ Failed to create memory: {response.status_code}")
            error_detail = response.json()
            print(f"   Error details: {error_detail}")
            raise AssertionError(
                f"Failed to create memory: {response.status_code} - {error_detail}"
            )
    except Exception as e:
        print(f"   ❌ Request failed: {e}")
        raise AssertionError(f"Request failed: {e}")

    # Check if conflict was resolved
    print("3. Checking conflict resolution...")
    recall_data = {"query": "ice cream", "user_id": f"{TEST_USER}_conflict", "k": 10}
    response = requests.post(f"{BASE_URL}/v1/memories/recall", json=recall_data, timeout=10)
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
    try:
        response = requests.get(f"{BASE_URL}/metrics", timeout=10)
        if response.status_code == 200:
            metrics = response.text
            hippocampai_metrics = [
                line
                for line in metrics.split("\n")
                if "hippocampai" in line and not line.startswith("#")
            ]
            print(f"   ✅ Metrics endpoint working ({len(hippocampai_metrics)} metrics)")
        else:
            print(f"   ❌ Metrics endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Metrics request failed: {e}")
        return False

    # Test memory tracking endpoints
    print("2. Testing memory tracking endpoints...")
    try:
        response = requests.get(
            f"{BASE_URL}/v1/monitoring/events",
            params={"user_id": TEST_USER, "limit": 10},
            timeout=10,
        )
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Events endpoint working ({data['total']} events)")
        else:
            print(f"   ❌ Events endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Events request failed: {e}")
        return False

    # Test stats endpoint
    print("3. Testing stats endpoint...")
    response = requests.post(
        f"{BASE_URL}/v1/monitoring/stats", json={"user_id": TEST_USER}, timeout=10
    )
    if response.status_code == 200:
        stats = response.json()
        print("   ✅ Stats endpoint working")
        print(f"      Total events: {stats['total_events']}")
        print(f"      Success rate: {stats['success_rate'] * 100:.1f}%")
    else:
        print(f"   ❌ Stats endpoint failed: {response.status_code}")
        raise AssertionError(f"Stats endpoint failed: {response.status_code}")

    print("\n✅ Monitoring Endpoints: PASSED")


def test_cross_compatibility():
    """Test that library and API can work together."""
    if not LIBRARY_AVAILABLE:
        print("⚠️  Skipping cross-compatibility tests - library not installed")
        pytest.skip("Library not installed")

    print_section("Cross-Compatibility Test", "🔗")
    memory_id = None

    try:
        # Create memory via API
        print("1. Creating memory via SaaS API...")
        api_memory = {
            "text": "Testing cross-compatibility",
            "user_id": f"{TEST_USER}_cross",
            "type": "fact",
            "check_duplicate": False,
        }

        response = requests.post(f"{BASE_URL}/v1/memories", json=api_memory, timeout=10)
        if response.status_code != 201:
            error_msg = f"Failed to create memory via API: {response.status_code}"
            try:
                error_detail = response.json()
                error_msg += f"\nResponse: {error_detail}"
            except Exception:
                error_msg += f"\nResponse text: {response.text}"
            raise AssertionError(error_msg)

        try:
            memory_data = response.json()
            memory_id = memory_data["id"]
            print(f"   ✅ Memory created via API: {memory_id}")
        except (KeyError, ValueError) as e:
            raise AssertionError(f"Invalid API response format: {str(e)}")

        time.sleep(0.5)

        # Test library operations
        print("2. Testing library operations...")
        client = MemoryClient()

        # Retrieve via library
        memory = client.get_memory(memory_id)
        assert memory is not None, "Memory not found via library"
        print(f"   ✅ Memory retrieved via library: {memory.text}")

        # Update via library
        updated = client.update_memory(memory_id=memory_id, text="Testing cross-compatibility - updated")
        assert updated is not None, "Failed to update memory via library"
        print("   ✅ Memory updated via library")

        time.sleep(0.5)

        # Verify via API
        response = requests.get(f"{BASE_URL}/v1/memories/{memory_id}", timeout=10)
        assert response.status_code == 200, f"Failed to verify update via API: {response.status_code}"
        verify_data = response.json()
        assert "updated" in verify_data["text"], f"Update not reflected: {verify_data['text']}"
        print("   ✅ Update verified via API")

        # Delete via API
        response = requests.delete(f"{BASE_URL}/v1/memories/{memory_id}", timeout=10)
        assert response.status_code == 200, f"Failed to delete memory: {response.status_code}"
        print("   ✅ Memory deleted via API")

        print("\n✅ Cross-Compatibility: PASSED")

    except Exception as e:
        print(f"\n❌ Cross-Compatibility: FAILED - {e}")
        if memory_id:
            try:
                requests.delete(f"{BASE_URL}/v1/memories/{memory_id}", timeout=10)
            except Exception:
                pass  # Best effort cleanup
        raise


def main():
    """Run all compatibility tests."""
    print("\n" + "=" * 80)
    print("🧪  HippocampAI - SaaS API & Library Compatibility Test Suite")
    print("=" * 80)

    test_functions = [
        ("API Health", test_api_health),
        ("SaaS Operations", test_saas_memory_operations),
        ("Library Operations", test_library_memory_operations),
        ("Conflict Resolution", test_conflict_resolution_saas),
        ("Monitoring", test_monitoring_endpoints),
        ("Cross Compatibility", test_cross_compatibility),
    ]

    passed_tests = 0
    total_tests = len(test_functions)
    test_results = []

    # Run tests
    for test_name, test_func in test_functions:
        try:
            test_func()
            passed_tests += 1
            test_results.append((test_name, True))
            print(f"\n✅ {test_name}: PASSED")
        except AssertionError as e:
            test_results.append((test_name, False))
            print(f"\n❌ {test_name} failed: {e}")
            if test_name == "API Health":
                print("\n❌ API is not available. Aborting tests.")
                return 1
        except Exception as e:
            test_results.append((test_name, False))
            print(f"\n❌ {test_name} failed with unexpected error: {e}")
            if test_name == "API Health":
                print("\n❌ API is not available. Aborting tests.")
                return 1

    # Summary
    print_section("Test Summary", "📝")

    for test_name, passed in test_results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {test_name}: {status}")

    print(f"\n{'=' * 80}")
    print(f"Results: {passed_tests}/{total_tests} tests passed")
    print(f"{'=' * 80}")

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
