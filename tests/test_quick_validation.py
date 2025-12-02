"""Quick validation test for all HippocampAI features."""

import os
import sys
import time
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from hippocampai.client import MemoryClient
from hippocampai.multiagent.collaboration import CollaborationManager
from hippocampai.pipeline.predictive_analytics import PredictiveAnalyticsEngine
from hippocampai.pipeline.temporal_analytics import TemporalAnalytics
from hippocampai.pipeline.auto_healing import AutoHealingEngine
from hippocampai.monitoring.memory_health import MemoryHealthMonitor
from hippocampai.embed.embedder import Embedder
from hippocampai.models.agent import PermissionType
from hippocampai.models.healing import AutoHealingConfig
from hippocampai.models.prediction import ForecastMetric, ForecastHorizon


def test(name):
    """Test decorator."""
    def decorator(func):
        def wrapper():
            print(f"\n{'='*70}")
            print(f"Testing: {name}")
            print('='*70)
            try:
                start = time.time()
                result = func()
                duration = time.time() - start
                if result:
                    print(f"✅ PASSED ({duration:.2f}s)")
                    return True
                else:
                    print(f"❌ FAILED ({duration:.2f}s)")
                    return False
            except Exception as e:
                print(f"❌ ERROR: {str(e)}")
                import traceback
                traceback.print_exc()
                return False
        return wrapper
    return decorator


# Test user ID
USER_ID = "test_validation_user"


@test("1. Core Memory Operations")
def test_core_operations():
    """Test basic memory operations."""
    client = MemoryClient()

    # Create
    memory = client.remember(
        "Test memory for validation",
        user_id=USER_ID,
        type="fact",
        importance=7.0,
        tags=["test"]
    )
    print(f"  ✓ Created memory: {memory.id[:8]}...")

    # Read
    results = client.recall("validation", user_id=USER_ID, k=5)
    print(f"  ✓ Recalled {len(results)} memories")

    # Update
    updated = client.update_memory(
        memory_id=memory.id,
        user_id=USER_ID,
        text="Updated validation memory",
        importance=8.0
    )
    print(f"  ✓ Updated memory importance: {updated.importance}")

    # Delete
    deleted = client.delete_memory(memory.id, user_id=USER_ID)
    print(f"  ✓ Deleted memory: {deleted}")

    return True


@test("2. Session Management")
def test_sessions():
    """Test session creation and tracking."""
    client = MemoryClient()

    # Create session
    session = client.create_session(
        user_id=USER_ID,
        title="Test Session"
    )
    print(f"  ✓ Created session: {session.id[:8]}...")

    # Track messages
    for i in range(3):
        client.track_session_message(
            session_id=session.id,
            user_id=USER_ID,
            message=f"Message {i}",
            role="user"
        )
    print(f"  ✓ Tracked 3 messages")

    # Get session
    retrieved = client.get_session(session.id, user_id=USER_ID)
    print(f"  ✓ Retrieved session with {retrieved.message_count} messages")

    # Complete
    completed = client.complete_session(session.id, user_id=USER_ID)
    print(f"  ✓ Completed session: {completed.status.value}")

    return True


@test("3. Collaboration - Shared Spaces")
def test_collaboration():
    """Test collaboration features."""
    client = MemoryClient()
    collab = CollaborationManager()

    # Create agents
    agent1 = client.create_agent(
        name="Agent 1",
        user_id=USER_ID,
        role="assistant"
    )
    agent2 = client.create_agent(
        name="Agent 2",
        user_id=USER_ID,
        role="assistant"
    )
    print(f"  ✓ Created 2 agents")

    # Create space
    space = collab.create_space(
        name="Test Space",
        owner_agent_id=agent1.id
    )
    print(f"  ✓ Created shared space: {space.id[:8]}...")

    # Add collaborator
    collab.add_collaborator(
        space_id=space.id,
        agent_id=agent2.id,
        permissions=[PermissionType.READ, PermissionType.WRITE],
        inviter_id=agent1.id
    )
    print(f"  ✓ Added collaborator with READ, WRITE permissions")

    # Add memory to space
    memory = client.remember(
        "Shared memory",
        user_id=USER_ID,
        agent_id=agent1.id
    )
    collab.add_memory_to_space(space.id, memory.id, agent1.id)
    print(f"  ✓ Added memory to shared space")

    # Get events
    events = collab.get_space_events(space.id)
    print(f"  ✓ Retrieved {len(events)} collaboration events")

    # Get notifications
    notifs = collab.get_notifications(agent2.id)
    print(f"  ✓ Agent 2 has {len(notifs)} notification(s)")

    return True


@test("4. Predictive Analytics - Patterns")
def test_predictions():
    """Test predictive analytics."""
    client = MemoryClient()
    temporal = TemporalAnalytics()
    predictive = PredictiveAnalyticsEngine(temporal)

    # Create memories with pattern
    for i in range(10):
        date = datetime.now() - timedelta(days=i)
        client.remember(
            f"Pattern memory {i}",
            user_id=USER_ID,
            type="habit",
            importance=6.0
        )
    print(f"  ✓ Created 10 memories for pattern detection")

    # Detect patterns
    memories = client.get_memories(user_id=USER_ID)
    patterns = temporal.detect_temporal_patterns(memories, min_occurrences=3)
    print(f"  ✓ Detected {len(patterns)} temporal pattern(s)")

    # Generate recommendations
    recommendations = predictive.generate_recommendations(
        user_id=USER_ID,
        memories=memories,
        max_recommendations=5
    )
    print(f"  ✓ Generated {len(recommendations)} recommendation(s)")

    # Forecast activity
    forecast = predictive.forecast_metric(
        user_id=USER_ID,
        memories=memories,
        metric=ForecastMetric.ACTIVITY_LEVEL,
        horizon=ForecastHorizon.NEXT_WEEK
    )
    print(f"  ✓ Forecasted activity: {forecast.predicted_value:.1f} memories/day")

    return True


@test("5. Auto-Healing - Health Monitoring")
def test_autohealing():
    """Test auto-healing features."""
    client = MemoryClient()
    embedder = Embedder(model="all-MiniLM-L6-v2")
    health_monitor = MemoryHealthMonitor(embedder)
    healing_engine = AutoHealingEngine(health_monitor, embedder)

    # Create diverse memories
    for i in range(15):
        client.remember(
            f"Health test memory {i}",
            user_id=USER_ID,
            type="fact",
            importance=5.0 + (i % 5),
            tags=[f"tag_{i % 3}"],
            confidence=0.7 + (i % 3) * 0.1
        )
    print(f"  ✓ Created 15 diverse memories")

    # Check health
    memories = client.get_memories(user_id=USER_ID)
    health = health_monitor.calculate_health_score(memories, detailed=True)
    print(f"  ✓ Health score: {health.overall_score:.1f}/100 ({health.status.value})")
    print(f"    - Freshness: {health.freshness_score:.1f}/100")
    print(f"    - Diversity: {health.diversity_score:.1f}/100")
    print(f"    - Consistency: {health.consistency_score:.1f}/100")

    # Configure auto-healing
    config = AutoHealingConfig(
        user_id=USER_ID,
        auto_cleanup_enabled=True,
        auto_dedup_enabled=True,
        max_actions_per_run=20
    )

    # Run cleanup (dry run)
    report = healing_engine.auto_cleanup(
        user_id=USER_ID,
        memories=memories,
        config=config,
        dry_run=True
    )
    print(f"  ✓ Auto-cleanup: {len(report.actions_recommended)} actions recommended")

    # Detect duplicates
    clusters = health_monitor.detect_duplicate_clusters(memories)
    print(f"  ✓ Found {len(clusters)} duplicate cluster(s)")

    return True


@test("6. Multi-User Isolation")
def test_multiuser():
    """Test multi-user isolation."""
    client = MemoryClient()

    user1_id = "test_user_1"
    user2_id = "test_user_2"

    # User 1 creates memory
    mem1 = client.remember("User 1 private memory", user_id=user1_id)
    print(f"  ✓ User 1 created memory: {mem1.id[:8]}...")

    # User 2 creates memory
    mem2 = client.remember("User 2 private memory", user_id=user2_id)
    print(f"  ✓ User 2 created memory: {mem2.id[:8]}...")

    # User 1 should only see their memories
    user1_mems = client.get_memories(user_id=user1_id)
    user1_ids = [m.id for m in user1_mems]
    assert mem1.id in user1_ids
    assert mem2.id not in user1_ids
    print(f"  ✓ User 1 isolation verified")

    # User 2 should only see their memories
    user2_mems = client.get_memories(user_id=user2_id)
    user2_ids = [m.id for m in user2_mems]
    assert mem2.id in user2_ids
    assert mem1.id not in user2_ids
    print(f"  ✓ User 2 isolation verified")

    return True


@test("7. Performance - Basic Benchmarks")
def test_performance():
    """Test basic performance."""
    client = MemoryClient()
    perf_user = "test_perf_user"

    # Create benchmark
    start = time.time()
    for i in range(50):
        client.remember(f"Perf memory {i}", user_id=perf_user, type="fact")
    create_time = time.time() - start
    create_ops = 50 / create_time
    print(f"  ✓ Create: {create_ops:.1f} ops/sec")

    # Recall benchmark
    start = time.time()
    for i in range(10):
        client.recall("perf", user_id=perf_user, k=10)
    recall_time = time.time() - start
    recall_ops = 10 / recall_time
    print(f"  ✓ Recall: {recall_ops:.1f} ops/sec")

    # Get benchmark
    start = time.time()
    for i in range(20):
        client.get_memories(user_id=perf_user)
    get_time = time.time() - start
    get_ops = 20 / get_time
    print(f"  ✓ Get: {get_ops:.1f} ops/sec")

    return True


def run_all_tests():
    """Run all validation tests."""
    print("\n" + "="*70)
    print("HIPPOCAMPAI QUICK VALIDATION TEST SUITE")
    print("="*70)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    tests = [
        test_core_operations,
        test_sessions,
        test_collaboration,
        test_predictions,
        test_autohealing,
        test_multiuser,
        test_performance,
    ]

    results = []
    for test_func in tests:
        result = test_func()
        results.append(result)

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    passed = sum(results)
    total = len(results)
    pass_rate = (passed / total * 100) if total > 0 else 0

    print(f"\nTotal Tests: {total}")
    print(f"✅ Passed: {passed} ({pass_rate:.1f}%)")
    print(f"❌ Failed: {total - passed}")

    print(f"\nEnd: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    return all(results)


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
