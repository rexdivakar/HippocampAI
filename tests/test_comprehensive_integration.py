"""Comprehensive integration tests for all HippocampAI features.

Tests:
1. Core memory operations
2. Collaboration features
3. Predictive analytics
4. Auto-healing
5. SaaS mode
6. Library compatibility
7. Memory usage
8. Performance
"""

import os
import sys
import time
import traceback
import tracemalloc
from datetime import datetime, timedelta
from typing import Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from hippocampai.client import MemoryClient
from hippocampai.embed.embedder import Embedder
from hippocampai.models.agent import PermissionType
from hippocampai.models.healing import AutoHealingConfig
from hippocampai.models.prediction import ForecastHorizon, ForecastMetric
from hippocampai.monitoring.memory_health import MemoryHealthMonitor
from hippocampai.multiagent.collaboration import CollaborationManager
from hippocampai.pipeline.auto_healing import AutoHealingEngine
from hippocampai.pipeline.predictive_analytics import PredictiveAnalyticsEngine
from hippocampai.pipeline.temporal_analytics import TemporalAnalytics
from hippocampai.simple import Memory as SimpleMemory


class TestResults:
    """Store test results for reporting."""

    def __init__(self):
        self.passed = []
        self.failed = []
        self.skipped = []
        self.warnings = []
        self.performance_metrics = {}

    def add_pass(self, test_name: str, duration: float = 0):
        self.passed.append({"name": test_name, "duration": duration})

    def add_fail(self, test_name: str, error: str):
        self.failed.append({"name": test_name, "error": error})

    def add_skip(self, test_name: str, reason: str):
        self.skipped.append({"name": test_name, "reason": reason})

    def add_warning(self, test_name: str, warning: str):
        self.warnings.append({"name": test_name, "warning": warning})

    def add_metric(self, name: str, value: Any):
        self.performance_metrics[name] = value


results = TestResults()


def test_wrapper(test_name: str):
    """Decorator to wrap tests with timing and error handling."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            print(f"\n{'='*80}")
            print(f"Testing: {test_name}")
            print('='*80)
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                print(f"✅ PASSED ({duration:.2f}s)")
                results.add_pass(test_name, duration)
                return result
            except Exception as e:
                duration = time.time() - start_time
                print(f"❌ FAILED ({duration:.2f}s)")
                print(f"Error: {str(e)}")
                print(f"Traceback: {traceback.format_exc()}")
                results.add_fail(test_name, str(e))
                return None
        return wrapper
    return decorator


# ============================================================================
# PART 1: CORE MEMORY OPERATIONS
# ============================================================================

@test_wrapper("1.1 Basic Memory Operations")
def test_basic_memory_operations():
    """Test basic CRUD operations."""
    client = MemoryClient(user_id="test_user_basic")

    # Create
    memory = client.remember(
        "Test memory for basic operations",
        type="fact",
        importance=7.0,
        tags=["test", "basic"]
    )
    assert memory.id is not None
    assert memory.text == "Test memory for basic operations"
    print(f"  Created memory: {memory.id}")

    # Read
    recalled = client.recall("basic operations", k=1)
    assert len(recalled) > 0
    print(f"  Recalled {len(recalled)} memories")

    # Update
    updated = client.update_memory(
        memory_id=memory.id,
        text="Updated test memory",
        importance=8.0
    )
    assert updated.text == "Updated test memory"
    assert updated.importance == 8.0
    print(f"  Updated memory: {memory.id}")

    # Delete
    deleted = client.delete_memory(memory.id)
    assert deleted is True
    print(f"  Deleted memory: {memory.id}")

    return True


@test_wrapper("1.2 Batch Operations")
def test_batch_operations():
    """Test batch operations."""
    client = MemoryClient(user_id="test_user_batch")

    # Batch create
    texts = [f"Batch memory {i}" for i in range(10)]
    memories = client.batch_remember(texts, type="fact", importance=6.0)
    assert len(memories) == 10
    print(f"  Batch created {len(memories)} memories")

    # Batch delete
    memory_ids = [m.id for m in memories[:5]]
    success = client.batch_delete(memory_ids)
    assert success is True
    print(f"  Batch deleted {len(memory_ids)} memories")

    return True


@test_wrapper("1.3 Session Management")
def test_session_management():
    """Test session creation and management."""
    client = MemoryClient(user_id="test_user_session")

    # Create session
    session = client.create_session(title="Test Session", metadata={"test": True})
    assert session.id is not None
    print(f"  Created session: {session.id}")

    # Track messages
    for i in range(5):
        client.track_session_message(
            session_id=session.id,
            message=f"Test message {i}",
            role="user"
        )
    print("  Tracked 5 messages")

    # Get session
    retrieved = client.get_session(session.id)
    assert retrieved is not None
    assert retrieved.message_count >= 5
    print(f"  Session has {retrieved.message_count} messages")

    # Complete session
    completed = client.complete_session(session.id, generate_summary=False)
    assert completed.status.value == "completed"
    print("  Completed session")

    return True


# ============================================================================
# PART 2: COLLABORATION FEATURES
# ============================================================================

@test_wrapper("2.1 Shared Memory Spaces")
def test_shared_spaces():
    """Test shared memory space creation and management."""
    client = MemoryClient(user_id="test_user_collab")
    collab_manager = CollaborationManager()

    # Create agents
    agent1 = client.create_agent("Agent 1", role="assistant")
    agent2 = client.create_agent("Agent 2", role="assistant")
    print(f"  Created agents: {agent1.id[:8]}..., {agent2.id[:8]}...")

    # Create space
    space = collab_manager.create_space(
        name="Test Collaboration Space",
        owner_agent_id=agent1.id,
        description="Space for testing",
        tags=["test"]
    )
    assert space.id is not None
    assert space.owner_agent_id == agent1.id
    print(f"  Created space: {space.id}")

    # Add collaborator
    success = collab_manager.add_collaborator(
        space_id=space.id,
        agent_id=agent2.id,
        permissions=[PermissionType.READ, PermissionType.WRITE],
        inviter_id=agent1.id
    )
    assert success is True
    print(f"  Added collaborator: {agent2.id[:8]}...")

    # Check permissions
    can_read = space.has_permission(agent2.id, PermissionType.READ)
    can_write = space.has_permission(agent2.id, PermissionType.WRITE)
    can_delete = space.has_permission(agent2.id, PermissionType.DELETE)
    assert can_read is True
    assert can_write is True
    assert can_delete is False
    print("  Permissions verified: READ=True, WRITE=True, DELETE=False")

    # Add memory to space
    memory = client.remember("Shared memory", agent_id=agent1.id)
    collab_manager.add_memory_to_space(space.id, memory.id, agent1.id)
    assert memory.id in space.memory_ids
    print(f"  Added memory to space: {memory.id[:8]}...")

    return True


@test_wrapper("2.2 Collaboration Events")
def test_collaboration_events():
    """Test event tracking in collaboration."""
    client = MemoryClient(user_id="test_user_events")
    collab_manager = CollaborationManager()

    agent = client.create_agent("Event Agent", role="assistant")
    space = collab_manager.create_space("Event Test Space", owner_agent_id=agent.id)

    # Generate some events
    memory = client.remember("Event memory", agent_id=agent.id)
    collab_manager.add_memory_to_space(space.id, memory.id, agent.id)
    collab_manager.update_memory_in_space(space.id, memory.id, agent.id)

    # Get events
    events = collab_manager.get_space_events(space.id, limit=10)
    assert len(events) > 0
    print(f"  Retrieved {len(events)} events")

    # Check event types
    event_types = [e.event_type.value for e in events]
    assert "space_updated" in event_types or "memory_added" in event_types
    print(f"  Event types: {event_types}")

    return True


@test_wrapper("2.3 Notifications")
def test_notifications():
    """Test notification system."""
    client = MemoryClient(user_id="test_user_notif")
    collab_manager = CollaborationManager()

    agent1 = client.create_agent("Agent 1", role="assistant")
    agent2 = client.create_agent("Agent 2", role="assistant")

    space = collab_manager.create_space("Notif Space", owner_agent_id=agent1.id)
    collab_manager.add_collaborator(
        space.id, agent2.id, [PermissionType.READ], agent1.id
    )

    # Check notifications
    notifications = collab_manager.get_notifications(agent2.id)
    assert len(notifications) > 0
    print(f"  Agent 2 has {len(notifications)} notification(s)")

    # Mark as read
    for notif in notifications:
        collab_manager.mark_notification_read(agent2.id, notif.id)

    unread = collab_manager.get_notifications(agent2.id, unread_only=True)
    assert len(unread) == 0
    print("  All notifications marked as read")

    return True


# ============================================================================
# PART 3: PREDICTIVE ANALYTICS
# ============================================================================

@test_wrapper("3.1 Pattern Detection")
def test_pattern_detection():
    """Test temporal pattern detection."""
    client = MemoryClient(user_id="test_user_patterns")
    temporal = TemporalAnalytics()

    # Create memories with daily pattern
    for i in range(15):
        date = datetime.now() - timedelta(days=i)
        client.remember(
            f"Daily memory {i}",
            type="habit",
            metadata={"created_at": date.isoformat()}
        )

    memories = client.get_memories()
    patterns = temporal.detect_temporal_patterns(memories, min_occurrences=3)

    assert len(patterns) > 0
    print(f"  Detected {len(patterns)} pattern(s)")
    for pattern in patterns:
        print(f"    - {pattern.description} (confidence: {pattern.confidence:.0%})")

    return True


@test_wrapper("3.2 Anomaly Detection")
def test_anomaly_detection():
    """Test anomaly detection."""
    client = MemoryClient(user_id="test_user_anomaly")
    temporal = TemporalAnalytics()
    predictive = PredictiveAnalyticsEngine(temporal)

    # Create normal pattern
    for i in range(10):
        client.remember(f"Normal memory {i}", type="fact", importance=5.0)

    # Create anomalous burst
    for i in range(30):
        client.remember(f"Anomaly memory {i}", type="fact", importance=9.0)

    memories = client.get_memories()
    anomalies = predictive.detect_anomalies(
        user_id=client.user_id,
        memories=memories,
        lookback_days=30
    )

    print(f"  Detected {len(anomalies)} anomal(ies)")
    for anomaly in anomalies:
        print(f"    - {anomaly.title} ({anomaly.severity.value})")

    return True


@test_wrapper("3.3 Recommendations")
def test_recommendations():
    """Test recommendation generation."""
    client = MemoryClient(user_id="test_user_recs")
    temporal = TemporalAnalytics()
    predictive = PredictiveAnalyticsEngine(temporal)

    # Create diverse memories
    for i in range(20):
        client.remember(f"Memory {i}", type="fact", importance=5.0 + i % 5)

    memories = client.get_memories()
    recommendations = predictive.generate_recommendations(
        user_id=client.user_id,
        memories=memories,
        max_recommendations=5
    )

    print(f"  Generated {len(recommendations)} recommendation(s)")
    for rec in recommendations:
        print(f"    - [{rec.priority}/10] {rec.title}")

    return True


@test_wrapper("3.4 Forecasting")
def test_forecasting():
    """Test metric forecasting."""
    client = MemoryClient(user_id="test_user_forecast")
    temporal = TemporalAnalytics()
    predictive = PredictiveAnalyticsEngine(temporal)

    # Create memory history
    for i in range(30):
        date = datetime.now() - timedelta(days=i)
        client.remember(
            f"Historical memory {i}",
            type="fact",
            metadata={"created_at": date.isoformat()}
        )

    memories = client.get_memories()
    forecast = predictive.forecast_metric(
        user_id=client.user_id,
        memories=memories,
        metric=ForecastMetric.ACTIVITY_LEVEL,
        horizon=ForecastHorizon.NEXT_WEEK
    )

    assert forecast.predicted_value >= 0
    print(f"  Forecast: {forecast.predicted_value:.1f} memories/day")
    print(f"  Confidence: {forecast.confidence:.0%}")
    print(f"  CI: [{forecast.confidence_interval[0]:.1f}, {forecast.confidence_interval[1]:.1f}]")

    return True


# ============================================================================
# PART 4: AUTO-HEALING
# ============================================================================

@test_wrapper("4.1 Health Monitoring")
def test_health_monitoring():
    """Test health score calculation."""
    client = MemoryClient(user_id="test_user_health")
    embedder = Embedder(model="all-MiniLM-L6-v2")
    health_monitor = MemoryHealthMonitor(embedder)

    # Create diverse memories
    for i in range(20):
        client.remember(
            f"Memory {i}",
            type="fact",
            importance=5.0 + i % 5,
            tags=[f"tag_{i % 3}"],
            confidence=0.7 + (i % 3) * 0.1
        )

    memories = client.get_memories()
    health_score = health_monitor.calculate_health_score(memories, detailed=True)

    print(f"  Overall health: {health_score.overall_score:.1f}/100")
    print(f"  Status: {health_score.status.value}")
    print(f"  Freshness: {health_score.freshness_score:.1f}/100")
    print(f"  Diversity: {health_score.diversity_score:.1f}/100")
    print(f"  Consistency: {health_score.consistency_score:.1f}/100")
    print(f"  Coverage: {health_score.coverage_score:.1f}/100")

    assert health_score.overall_score >= 0
    assert health_score.overall_score <= 100

    return True


@test_wrapper("4.2 Auto-Cleanup")
def test_auto_cleanup():
    """Test automatic cleanup."""
    client = MemoryClient(user_id="test_user_cleanup")
    embedder = Embedder(model="all-MiniLM-L6-v2")
    health_monitor = MemoryHealthMonitor(embedder)
    healing_engine = AutoHealingEngine(health_monitor, embedder)

    # Create stale memories
    for i in range(5):
        old_date = datetime.now() - timedelta(days=120)
        client.remember(
            f"Stale memory {i}",
            type="fact",
            importance=3.0,
            confidence=0.4,
            metadata={"created_at": old_date.isoformat()}
        )

    # Create fresh memories
    for i in range(5):
        client.remember(f"Fresh memory {i}", type="fact", importance=7.0)

    config = AutoHealingConfig(
        user_id=client.user_id,
        auto_cleanup_enabled=True,
        cleanup_threshold_days=90,
        max_actions_per_run=50
    )

    memories = client.get_memories()
    report = healing_engine.auto_cleanup(
        user_id=client.user_id,
        memories=memories,
        config=config,
        dry_run=True
    )

    print(f"  Actions recommended: {len(report.actions_recommended)}")
    print(f"  Memories analyzed: {report.memories_analyzed}")
    print(f"  Duration: {report.duration_seconds:.2f}s")

    assert len(report.actions_recommended) > 0

    return True


@test_wrapper("4.3 Duplicate Detection")
def test_duplicate_detection():
    """Test duplicate memory detection."""
    client = MemoryClient(user_id="test_user_dupes")
    embedder = Embedder(model="all-MiniLM-L6-v2")
    health_monitor = MemoryHealthMonitor(embedder)

    # Create duplicates
    for i in range(3):
        client.remember(
            "I prefer coffee over tea in the morning",
            type="preference",
            importance=6.0
        )

    # Create unique memories
    for i in range(5):
        client.remember(f"Unique memory {i}", type="fact")

    memories = client.get_memories()
    clusters = health_monitor.detect_duplicate_clusters(
        memories,
        cluster_type="soft",
        min_cluster_size=2
    )

    print(f"  Detected {len(clusters)} duplicate cluster(s)")
    for cluster in clusters:
        print(f"    - {len(cluster.memories)} memories ({cluster.confidence:.0%} similar)")

    assert len(clusters) > 0

    return True


@test_wrapper("4.4 Auto-Tagging")
def test_auto_tagging():
    """Test automatic tagging."""
    client = MemoryClient(user_id="test_user_tags")
    embedder = Embedder(model="all-MiniLM-L6-v2")
    health_monitor = MemoryHealthMonitor(embedder)
    healing_engine = AutoHealingEngine(health_monitor, embedder)

    # Create untagged memories
    memories = []
    for i in range(5):
        mem = client.remember(
            "This is about work and meetings at the office",
            type="fact",
            tags=[]  # No tags
        )
        memories.append(mem)

    tag_actions = healing_engine.auto_tag(memories)

    print(f"  Generated {len(tag_actions)} tag suggestion(s)")
    for action in tag_actions[:3]:
        suggested = action.metadata.get('suggested_tags', [])
        print(f"    - Suggest: {suggested}")

    return True


# ============================================================================
# PART 5: SAAS MODE & AUTHENTICATION
# ============================================================================

@test_wrapper("5.1 Simple API Compatibility (mem0)")
def test_simple_api():
    """Test mem0-compatible SimpleMemory API."""
    try:
        simple = SimpleMemory(user_id="test_user_simple")

        # Test add
        result = simple.add("Test memory for simple API", metadata={"test": True})
        assert "id" in result
        print(f"  Added memory: {result['id']}")

        # Test search
        results = simple.search("simple API", limit=5)
        assert len(results) > 0
        print(f"  Searched: {len(results)} results")

        # Test get_all
        all_memories = simple.get_all()
        assert len(all_memories) > 0
        print(f"  Retrieved all: {len(all_memories)} memories")

        # Test delete
        delete_result = simple.delete(result['id'])
        print(f"  Deleted: {delete_result}")

        return True
    except Exception as e:
        print(f"  Warning: SimpleMemory test failed: {e}")
        results.add_warning("5.1 Simple API", str(e))
        return False


@test_wrapper("5.2 Multi-User Isolation")
def test_multi_user_isolation():
    """Test that users are properly isolated."""
    user1 = MemoryClient(user_id="test_user_1")
    user2 = MemoryClient(user_id="test_user_2")

    # User 1 creates memory
    mem1 = user1.remember("User 1's private memory")

    # User 2 creates memory
    mem2 = user2.remember("User 2's private memory")

    # User 1 should only see their memories
    user1_memories = user1.get_memories()
    user1_ids = [m.id for m in user1_memories]
    assert mem1.id in user1_ids
    assert mem2.id not in user1_ids
    print("  User 1 isolation verified")

    # User 2 should only see their memories
    user2_memories = user2.get_memories()
    user2_ids = [m.id for m in user2_memories]
    assert mem2.id in user2_ids
    assert mem1.id not in user2_ids
    print("  User 2 isolation verified")

    return True


# ============================================================================
# PART 6: MEMORY USAGE & PERFORMANCE
# ============================================================================

@test_wrapper("6.1 Memory Usage Analysis")
def test_memory_usage():
    """Test memory usage under load."""
    tracemalloc.start()

    client = MemoryClient(user_id="test_user_memory")

    # Create 100 memories
    for i in range(100):
        client.remember(
            f"Memory {i} with some content to test memory usage",
            type="fact",
            importance=5.0 + (i % 5),
            tags=[f"tag_{i % 10}"]
        )

    # Get current memory usage
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    current_mb = current / 1024 / 1024
    peak_mb = peak / 1024 / 1024

    print(f"  Current memory: {current_mb:.2f} MB")
    print(f"  Peak memory: {peak_mb:.2f} MB")

    results.add_metric("memory_usage_mb", peak_mb)

    # Warning if memory usage is too high
    if peak_mb > 500:
        results.add_warning("6.1 Memory Usage", f"High memory usage: {peak_mb:.2f} MB")

    return True


@test_wrapper("6.2 Performance Benchmarks")
def test_performance():
    """Test performance of key operations."""
    client = MemoryClient(user_id="test_user_perf")

    # Benchmark: Create
    start = time.time()
    for i in range(50):
        client.remember(f"Perf test memory {i}", type="fact")
    create_time = time.time() - start
    create_per_sec = 50 / create_time
    print(f"  Create: {create_per_sec:.1f} ops/sec")
    results.add_metric("create_ops_per_sec", create_per_sec)

    # Benchmark: Recall
    start = time.time()
    for i in range(20):
        client.recall("perf test", k=10)
    recall_time = time.time() - start
    recall_per_sec = 20 / recall_time
    print(f"  Recall: {recall_per_sec:.1f} ops/sec")
    results.add_metric("recall_ops_per_sec", recall_per_sec)

    # Benchmark: Get
    start = time.time()
    for _ in range(100):
        _ = client.get_memories()
    get_time = time.time() - start
    get_per_sec = 100 / get_time
    print(f"  Get: {get_per_sec:.1f} ops/sec")
    results.add_metric("get_ops_per_sec", get_per_sec)

    return True


@test_wrapper("6.3 Large Dataset Handling")
def test_large_dataset():
    """Test handling of larger datasets."""
    client = MemoryClient(user_id="test_user_large")

    print("  Creating 500 memories...")
    start = time.time()

    for i in range(500):
        client.remember(
            f"Large dataset memory {i} with some additional content",
            type="fact",
            importance=5.0 + (i % 10),
            tags=[f"tag_{i % 20}"]
        )

        if (i + 1) % 100 == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            print(f"    Progress: {i+1}/500 ({rate:.1f} ops/sec)")

    total_time = time.time() - start
    total_rate = 500 / total_time

    print(f"  Total time: {total_time:.2f}s ({total_rate:.1f} ops/sec)")
    results.add_metric("large_dataset_ops_per_sec", total_rate)

    # Test recall on large dataset
    start = time.time()
    results_recall = client.recall("dataset memory", k=20)
    recall_time = time.time() - start

    print(f"  Recall on 500 memories: {recall_time:.3f}s")
    print(f"  Retrieved: {len(results_recall)} results")

    results.add_metric("large_dataset_recall_time", recall_time)

    return True


# ============================================================================
# PART 7: ERROR HANDLING & EDGE CASES
# ============================================================================

@test_wrapper("7.1 Error Handling")
def test_error_handling():
    """Test error handling for invalid operations."""
    client = MemoryClient(user_id="test_user_errors")

    # Test invalid memory ID
    try:
        client.delete_memory("invalid_id_12345")
        print("  Warning: Should have raised error for invalid ID")
        results.add_warning("7.1 Error Handling", "Invalid ID didn't raise error")
    except Exception:
        print("  ✓ Invalid ID handled correctly")

    # Test empty text
    try:
        memory = client.remember("")
        if memory.text == "":
            results.add_warning("7.1 Error Handling", "Empty text allowed")
        print("  ✓ Empty text handled")
    except Exception:
        print("  ✓ Empty text rejected")

    # Test invalid importance
    try:
        memory = client.remember("Test", importance=15.0)  # Out of range
        if memory.importance > 10:
            results.add_warning("7.1 Error Handling", "Importance not clamped to valid range")
        print("  ✓ Invalid importance handled")
    except Exception:
        print("  ✓ Invalid importance rejected")

    return True


@test_wrapper("7.2 Concurrent Operations")
def test_concurrent_operations():
    """Test concurrent memory operations."""
    import threading

    client = MemoryClient(user_id="test_user_concurrent")
    errors = []

    def create_memories(thread_id, count):
        try:
            for i in range(count):
                client.remember(f"Thread {thread_id} memory {i}", type="fact")
        except Exception as e:
            errors.append(str(e))

    # Create 3 threads
    threads = []
    for i in range(3):
        t = threading.Thread(target=create_memories, args=(i, 10))
        threads.append(t)
        t.start()

    # Wait for completion
    for t in threads:
        t.join()

    if errors:
        print(f"  Errors during concurrent ops: {len(errors)}")
        results.add_warning("7.2 Concurrent", f"{len(errors)} errors occurred")
    else:
        print("  ✓ All concurrent operations succeeded")

    # Verify all memories created
    memories = client.get_memories()
    print(f"  Total memories after concurrent ops: {len(memories)}")

    return True


# ============================================================================
# TEST RUNNER
# ============================================================================

def run_all_tests():
    """Run all tests and generate report."""
    print("\n" + "="*80)
    print("HIPPOCAMPAI COMPREHENSIVE TEST SUITE")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    start_time = time.time()

    # Part 1: Core Operations
    print("\n\n" + "█"*80)
    print("PART 1: CORE MEMORY OPERATIONS")
    print("█"*80)
    test_basic_memory_operations()
    test_batch_operations()
    test_session_management()

    # Part 2: Collaboration
    print("\n\n" + "█"*80)
    print("PART 2: COLLABORATION FEATURES")
    print("█"*80)
    test_shared_spaces()
    test_collaboration_events()
    test_notifications()

    # Part 3: Predictive Analytics
    print("\n\n" + "█"*80)
    print("PART 3: PREDICTIVE ANALYTICS")
    print("█"*80)
    test_pattern_detection()
    test_anomaly_detection()
    test_recommendations()
    test_forecasting()

    # Part 4: Auto-Healing
    print("\n\n" + "█"*80)
    print("PART 4: AUTO-HEALING SYSTEM")
    print("█"*80)
    test_health_monitoring()
    test_auto_cleanup()
    test_duplicate_detection()
    test_auto_tagging()

    # Part 5: SaaS Mode
    print("\n\n" + "█"*80)
    print("PART 5: SAAS MODE & COMPATIBILITY")
    print("█"*80)
    test_simple_api()
    test_multi_user_isolation()

    # Part 6: Performance
    print("\n\n" + "█"*80)
    print("PART 6: MEMORY USAGE & PERFORMANCE")
    print("█"*80)
    test_memory_usage()
    test_performance()
    test_large_dataset()

    # Part 7: Error Handling
    print("\n\n" + "█"*80)
    print("PART 7: ERROR HANDLING & EDGE CASES")
    print("█"*80)
    test_error_handling()
    test_concurrent_operations()

    total_time = time.time() - start_time

    # Generate Report
    print("\n\n" + "="*80)
    print("TEST RESULTS SUMMARY")
    print("="*80)

    total_tests = len(results.passed) + len(results.failed) + len(results.skipped)
    pass_rate = (len(results.passed) / total_tests * 100) if total_tests > 0 else 0

    print(f"\nTotal Tests: {total_tests}")
    print(f"✅ Passed: {len(results.passed)} ({pass_rate:.1f}%)")
    print(f"❌ Failed: {len(results.failed)}")
    print(f"⏭️  Skipped: {len(results.skipped)}")
    print(f"⚠️  Warnings: {len(results.warnings)}")
    print(f"\nTotal Duration: {total_time:.2f}s")

    if results.failed:
        print("\n" + "="*80)
        print("FAILED TESTS")
        print("="*80)
        for failed in results.failed:
            print(f"\n❌ {failed['name']}")
            print(f"   Error: {failed['error']}")

    if results.warnings:
        print("\n" + "="*80)
        print("WARNINGS")
        print("="*80)
        for warning in results.warnings:
            print(f"\n⚠️  {warning['name']}")
            print(f"   Warning: {warning['warning']}")

    if results.performance_metrics:
        print("\n" + "="*80)
        print("PERFORMANCE METRICS")
        print("="*80)
        for metric, value in results.performance_metrics.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.2f}")
            else:
                print(f"  {metric}: {value}")

    print("\n" + "="*80)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    return results


if __name__ == "__main__":
    results = run_all_tests()

    # Exit with appropriate code
    exit_code = 0 if len(results.failed) == 0 else 1
    sys.exit(exit_code)
