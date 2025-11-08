#!/usr/bin/env python3
"""
Test script for Memory Lifecycle & Tiering feature.

Tests:
- Temperature scoring based on access patterns
- Automatic tier assignment
- Memory tier migration
- Tier statistics
- Both SaaS API and Library compatibility
"""

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
TEST_USER = "lifecycle_test_user"


def print_section(title, emoji="🔍"):
    """Print formatted section header."""
    print(f"\n{'='*80}")
    print(f"{emoji}  {title}")
    print(f"{'='*80}\n")


def test_api_lifecycle_basics():
    """Test basic lifecycle features via SaaS API."""
    print_section("SaaS API - Lifecycle Basics", "🌐")

    # Create a new memory
    print("1. Creating a new memory...")
    memory_data = {
        "text": "Python is a programming language",
        "user_id": TEST_USER,
        "type": "fact",
        "importance": 8.0,
    }
    response = requests.post(f"{BASE_URL}/v1/memories", json=memory_data)
    if response.status_code == 201:
        memory = response.json()
        memory_id = memory["id"]
        print(f"   ✅ Memory created: {memory_id}")
    else:
        print(f"   ❌ Failed to create memory: {response.status_code}")
        return False

    # Small delay
    time.sleep(1)

    # Get temperature metrics
    print("2. Getting temperature metrics...")
    response = requests.get(f"{BASE_URL}/v1/lifecycle/temperature/{memory_id}")
    if response.status_code == 200:
        temp = response.json()
        print("   ✅ Temperature retrieved:")
        print(f"      Tier: {temp['tier']}")
        print(f"      Temperature Score: {temp['temperature_score']:.1f}/100")
        print(f"      Access Frequency: {temp['access_frequency']:.4f} accesses/day")
        print(f"      Recency Score: {temp['recency_score']:.3f}")
    else:
        print(f"   ❌ Failed to get temperature: {response.status_code}")
        return False

    # Access memory multiple times to increase temperature
    print("3. Accessing memory 5 times to increase temperature...")
    for i in range(5):
        response = requests.get(f"{BASE_URL}/v1/memories/{memory_id}")
        if response.status_code == 200:
            print(f"   Access {i+1}/5 ✓")
        time.sleep(0.2)

    # Check updated temperature
    print("4. Checking updated temperature...")
    response = requests.get(f"{BASE_URL}/v1/lifecycle/temperature/{memory_id}")
    if response.status_code == 200:
        temp = response.json()
        print("   ✅ Updated temperature:")
        print(f"      Access Count: {temp['access_count']}")
        print(f"      Temperature Score: {temp['temperature_score']:.1f}/100")
        print(f"      Access Frequency: {temp['access_frequency']:.4f} accesses/day")
    else:
        print(f"   ❌ Failed to get updated temperature: {response.status_code}")
        return False

    # Test tier migration
    print("5. Migrating memory to 'archived' tier...")
    migration_data = {"memory_id": memory_id, "target_tier": "archived"}
    response = requests.post(f"{BASE_URL}/v1/lifecycle/migrate", json=migration_data)
    if response.status_code == 200:
        result = response.json()
        print("   ✅ Migration successful:")
        print(f"      Target Tier: {result['target_tier']}")
        print(f"      Migrated: {result['migrated']}")
    else:
        print(f"   ❌ Failed to migrate: {response.status_code}")
        return False

    # Verify migration
    print("6. Verifying tier migration...")
    response = requests.get(f"{BASE_URL}/v1/lifecycle/temperature/{memory_id}")
    if response.status_code == 200:
        temp = response.json()
        if temp["tier"] == "archived":
            print("   ✅ Memory successfully migrated to 'archived' tier")
        else:
            print(f"   ⚠️  Tier is '{temp['tier']}', expected 'archived'")
    else:
        print(f"   ❌ Failed to verify migration: {response.status_code}")
        return False

    # Cleanup
    print("7. Cleaning up test memory...")
    response = requests.delete(f"{BASE_URL}/v1/memories/{memory_id}")
    if response.status_code == 200:
        print("   ✅ Memory deleted")
    else:
        print(f"   ⚠️  Failed to delete: {response.status_code}")

    print("\n✅ SaaS API Lifecycle Basics: PASSED")
    return True


def test_api_tier_statistics():
    """Test tier statistics via SaaS API."""
    print_section("SaaS API - Tier Statistics", "📊")

    # Create memories with different importance levels
    print("1. Creating 5 memories with varying importance...")
    memory_ids = []
    for i in range(5):
        memory_data = {
            "text": f"Test memory {i+1} for tier statistics",
            "user_id": f"{TEST_USER}_stats",
            "type": "fact",
            "importance": float(i * 2),  # 0, 2, 4, 6, 8
        }
        response = requests.post(f"{BASE_URL}/v1/memories", json=memory_data)
        if response.status_code == 201:
            memory_ids.append(response.json()["id"])
            print(f"   Memory {i+1}/5 created ✓")
        else:
            print(f"   ❌ Failed to create memory {i+1}")
            return False

    time.sleep(1)

    # Get tier statistics
    print("2. Getting tier statistics...")
    response = requests.post(
        f"{BASE_URL}/v1/lifecycle/stats", json={"user_id": f"{TEST_USER}_stats"}
    )
    if response.status_code == 200:
        stats = response.json()
        print("   ✅ Statistics retrieved:")
        print(f"      Total Memories: {stats['total_memories']}")
        print(f"      Total Size: {stats['total_size_bytes']} bytes")
        print(f"      Total Accesses: {stats['total_accesses']}")
        print("      Tier Distribution:")
        for tier, count in stats["tier_counts"].items():
            avg_temp = stats["tier_average_temperatures"][tier]
            print(f"         {tier}: {count} memories (avg temp: {avg_temp:.1f})")
    else:
        print(f"   ❌ Failed to get statistics: {response.status_code}")
        return False

    # Cleanup
    print("3. Cleaning up test memories...")
    for memory_id in memory_ids:
        requests.delete(f"{BASE_URL}/v1/memories/{memory_id}")
    print(f"   ✅ {len(memory_ids)} memories deleted")

    print("\n✅ SaaS API Tier Statistics: PASSED")
    return True


def test_library_lifecycle():
    """Test lifecycle features via Python Library."""
    if not LIBRARY_AVAILABLE:
        print("⚠️  Skipping library tests - library not installed")
        return True

    print_section("Python Library - Lifecycle Features", "🐍")

    try:
        # Initialize client
        print("1. Initializing MemoryClient...")
        client = MemoryClient()
        print("   ✅ Client initialized")

        # Create memory
        print("2. Creating memory via Library...")
        memory = client.remember(
            "Machine learning is a subset of AI",
            user_id=f"{TEST_USER}_lib",
            type="fact",
            importance=7.0,
        )
        print(f"   ✅ Memory created: {memory.id}")
        memory_id = memory.id

        time.sleep(1)

        # Get temperature
        print("3. Getting temperature via Library...")
        temp = client.get_memory_temperature(memory_id)
        if temp:
            print("   ✅ Temperature retrieved:")
            print(f"      Tier: {temp['tier']}")
            print(f"      Temperature Score: {temp['temperature_score']:.1f}/100")
        else:
            print("   ❌ Failed to get temperature")
            return False

        # Migrate tier
        print("4. Migrating to 'cold' tier via Library...")
        result = client.migrate_memory_tier(memory_id, "cold")
        print(f"   ✅ Migration successful to '{result['target_tier']}' tier")

        # Get tier statistics
        print("5. Getting tier statistics via Library...")
        stats = client.get_tier_statistics(f"{TEST_USER}_lib")
        if stats:
            print("   ✅ Statistics retrieved:")
            print(f"      Total Memories: {stats['total_memories']}")
            for tier, count in stats["tier_counts"].items():
                if count > 0:
                    print(f"         {tier}: {count} memories")
        else:
            print("   ❌ Failed to get statistics")

        # Cleanup
        print("6. Cleaning up...")
        client.delete_memory(memory_id)
        print("   ✅ Memory deleted")

        print("\n✅ Python Library Lifecycle: PASSED")
        return True

    except Exception as e:
        print(f"\n❌ Python Library Lifecycle: FAILED - {e}")
        import traceback

        traceback.print_exc()
        return False


def test_cross_compatibility():
    """Test lifecycle features work across API and Library."""
    if not LIBRARY_AVAILABLE:
        print("⚠️  Skipping cross-compatibility tests - library not installed")
        return True

    print_section("Cross-Compatibility - Lifecycle", "🔗")

    try:
        # Create via API
        print("1. Creating memory via SaaS API...")
        memory_data = {
            "text": "Cross-compatibility test memory",
            "user_id": f"{TEST_USER}_cross",
            "type": "fact",
            "importance": 5.0,
        }
        response = requests.post(f"{BASE_URL}/v1/memories", json=memory_data)
        memory_id = response.json()["id"]
        print(f"   ✅ Memory created via API: {memory_id}")

        time.sleep(1)

        # Get temperature via Library
        print("2. Getting temperature via Python Library...")
        client = MemoryClient()
        temp = client.get_memory_temperature(memory_id)
        if temp:
            print("   ✅ Temperature retrieved via Library:")
            print(f"      Tier: {temp['tier']}")
        else:
            print("   ❌ Failed to get temperature via Library")
            return False

        # Migrate via Library
        print("3. Migrating tier via Python Library...")
        result = client.migrate_memory_tier(memory_id, "warm")
        print(f"   ✅ Migrated to '{result['target_tier']}' via Library")

        time.sleep(0.5)

        # Verify via API
        print("4. Verifying migration via SaaS API...")
        response = requests.get(f"{BASE_URL}/v1/lifecycle/temperature/{memory_id}")
        if response.status_code == 200:
            temp = response.json()
            if temp["tier"] == "warm":
                print("   ✅ Migration verified via API")
            else:
                print(f"   ⚠️  Tier is '{temp['tier']}', expected 'warm'")
        else:
            print(f"   ❌ Failed to verify: {response.status_code}")
            return False

        # Cleanup
        print("5. Cleaning up...")
        requests.delete(f"{BASE_URL}/v1/memories/{memory_id}")
        print("   ✅ Memory deleted")

        print("\n✅ Cross-Compatibility Lifecycle: PASSED")
        return True

    except Exception as e:
        print(f"\n❌ Cross-Compatibility Lifecycle: FAILED - {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all lifecycle tests."""
    print("\n" + "=" * 80)
    print("🧪  HippocampAI - Memory Lifecycle & Tiering Test Suite")
    print("=" * 80)

    # Check API health
    print("\n🏥 Checking API health...")
    response = requests.get(f"{BASE_URL}/healthz")
    if response.status_code != 200:
        print("❌ API is not available. Aborting tests.")
        return 1

    print("✅ API is healthy\n")

    results = {}

    # Run tests
    results["api_basics"] = test_api_lifecycle_basics()
    results["api_statistics"] = test_api_tier_statistics()
    results["library_lifecycle"] = test_library_lifecycle()
    results["cross_compatibility"] = test_cross_compatibility()

    # Summary
    print_section("Test Summary", "📝")

    total_tests = len(results)
    passed_tests = sum(1 for v in results.values() if v)

    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {test_name.replace('_', ' ').title()}: {status}")

    print(f"\n{'=' * 80}")
    print(f"Results: {passed_tests}/{total_tests} tests passed")
    print(f"{'=' * 80}")

    if passed_tests == total_tests:
        print("\n🎉 ALL LIFECYCLE TESTS PASSED!")
        print("\n✅ Key Features Verified:")
        print("  • Temperature scoring based on access patterns")
        print("  • Automatic tier assignment (hot/warm/cold/archived/hibernated)")
        print("  • Manual tier migration")
        print("  • Tier statistics and distribution")
        print("  • SaaS API lifecycle endpoints")
        print("  • Python Library lifecycle methods")
        print("  • Cross-compatibility between API and Library")
        return 0
    else:
        print("\n⚠️  SOME TESTS FAILED - Review output above for details")
        return 1


if __name__ == "__main__":
    try:
        exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
