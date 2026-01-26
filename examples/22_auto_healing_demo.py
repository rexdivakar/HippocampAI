"""Demo: Auto-healing and automatic memory maintenance."""

from datetime import datetime, timedelta

from hippocampai.client import MemoryClient
from hippocampai.embed.embedder import Embedder
from hippocampai.models.healing import (
    AutoHealingConfig,
    ConsolidationStrategy,
)
from hippocampai.monitoring.memory_health import MemoryHealthMonitor
from hippocampai.pipeline.auto_healing import AutoHealingEngine


def main():
    """Demonstrate auto-healing capabilities."""

    # Initialize components
    print("=" * 80)
    print("AUTO-HEALING & MEMORY MAINTENANCE DEMO")
    print("=" * 80)

    # Create client
    client = MemoryClient(user_id="user_healing_demo")

    # Initialize healing components
    embedder = Embedder(model="all-MiniLM-L6-v2")
    health_monitor = MemoryHealthMonitor(embedder)
    healing_engine = AutoHealingEngine(health_monitor, embedder)

    # 1. Create diverse memory set
    print("\n" + "=" * 80)
    print("1. Creating Sample Memory Set")
    print("=" * 80)

    # Fresh, high-quality memories
    for i in range(5):
        client.remember(
            f"Recent important project update #{i + 1}",
            type="fact",
            importance=8.0,
            tags=["project", "recent"],
            confidence=0.9,
        )
    print("✓ Created 5 fresh, high-quality memories")

    # Stale memories (old with no access)
    for i in range(8):
        old_date = datetime.now() - timedelta(days=120 + i * 5)
        client.remember(
            f"Old memory from 4 months ago #{i + 1}",
            type="fact",
            importance=5.0,
            tags=["old", "stale"],
            confidence=0.7,
            metadata={"created_at": old_date.isoformat()},
        )
    print("✓ Created 8 stale memories (120+ days old)")

    # Duplicate/similar memories
    for i in range(3):
        client.remember(
            "I prefer coffee over tea in the morning",
            type="preference",
            importance=6.0,
            tags=["beverage"],
            confidence=0.85,
        )
    print("✓ Created 3 duplicate memories (same content)")

    # Low confidence memories
    for i in range(4):
        client.remember(
            f"Uncertain information about topic {i}",
            type="fact",
            importance=4.0,
            tags=["uncertain"],
            confidence=0.2,  # Very low
        )
    print("✓ Created 4 low-confidence memories")

    # Untagged memories
    for i in range(6):
        client.remember(
            f"Memory without proper tags - item {i}",
            type="context",
            importance=5.0,
            tags=[],  # No tags
        )
    print("✓ Created 6 untagged memories")

    memories = client.get_memories()
    print(f"\n✓ Total memories created: {len(memories)}")

    # 2. Initial health assessment
    print("\n" + "=" * 80)
    print("2. Initial Health Assessment")
    print("=" * 80)

    health_score = health_monitor.calculate_health_score(memories, detailed=True)

    print(f"✓ Overall Health Score: {health_score.overall_score:.1f}/100")
    print(f"  Status: {health_score.status.value}")
    print(f"  Healthy memories: {health_score.healthy_memories}")
    print(f"  Stale memories: {health_score.stale_memories}")
    print(f"  Duplicate clusters: {health_score.duplicate_clusters}")
    print(f"  Low quality memories: {health_score.low_quality_memories}")
    print("\n  Component Scores:")
    print(f"  - Freshness: {health_score.freshness_score:.1f}/100")
    print(f"  - Diversity: {health_score.diversity_score:.1f}/100")
    print(f"  - Consistency: {health_score.consistency_score:.1f}/100")
    print(f"  - Coverage: {health_score.coverage_score:.1f}/100")

    print("\n  📋 Recommendations:")
    for rec in health_score.recommendations:
        print(f"    {rec}")

    # 3. Auto-healing configuration
    print("\n" + "=" * 80)
    print("3. Configuring Auto-Healing")
    print("=" * 80)

    config = AutoHealingConfig(
        user_id=client.user_id,
        enabled=True,
        auto_cleanup_enabled=True,
        auto_dedup_enabled=True,
        auto_consolidate_enabled=True,
        auto_tag_enabled=True,
        cleanup_threshold_days=90,
        dedup_similarity_threshold=0.88,
        consolidate_threshold=3,
        require_user_approval=False,  # Auto-apply for demo
        max_actions_per_run=50,
    )

    print("✓ Auto-healing configuration:")
    print(f"  - Auto cleanup: {config.auto_cleanup_enabled}")
    print(f"  - Auto deduplication: {config.auto_dedup_enabled}")
    print(f"  - Auto consolidation: {config.auto_consolidate_enabled}")
    print(f"  - Auto tagging: {config.auto_tag_enabled}")
    print(f"  - Cleanup threshold: {config.cleanup_threshold_days} days")
    print(f"  - Dedup similarity: {config.dedup_similarity_threshold:.0%}")
    print(f"  - Max actions per run: {config.max_actions_per_run}")

    # 4. Run cleanup (dry run first)
    print("\n" + "=" * 80)
    print("4. Running Auto-Cleanup (Dry Run)")
    print("=" * 80)

    cleanup_report = healing_engine.auto_cleanup(
        user_id=client.user_id, memories=memories, config=config, dry_run=True
    )

    print("✓ Cleanup analysis completed:")
    print(f"  Duration: {cleanup_report.duration_seconds:.2f}s")
    print(f"  Memories analyzed: {cleanup_report.memories_analyzed}")
    print(f"  Actions recommended: {len(cleanup_report.actions_recommended)}")
    print(f"  Actions applied: {len(cleanup_report.actions_applied)}")

    print("\n  Recommended actions:")
    for i, action in enumerate(cleanup_report.actions_recommended[:5], 1):
        print(f"  {i}. {action.action_type.value}")
        print(f"     Reason: {action.reason}")
        print(f"     Impact: {action.impact}")
        print(f"     Auto-applicable: {action.auto_applicable}")

    # 5. Run deduplication
    print("\n" + "=" * 80)
    print("5. Running Auto-Deduplication")
    print("=" * 80)

    dedup_report = healing_engine.auto_consolidate(
        user_id=client.user_id,
        memories=memories,
        config=config,
        strategy=ConsolidationStrategy.SEMANTIC_MERGE,
        dry_run=True,
    )

    print("✓ Deduplication analysis completed:")
    print(f"  Duration: {dedup_report.duration_seconds:.2f}s")
    print(f"  Duplicate clusters found: {len(dedup_report.actions_recommended)}")
    print(f"  Merges recommended: {len(dedup_report.actions_recommended)}")

    if dedup_report.actions_recommended:
        print("\n  Duplicate clusters:")
        for i, action in enumerate(dedup_report.actions_recommended[:3], 1):
            print(f"  {i}. Cluster with {len(action.memory_ids)} memories")
            print(f"     Reason: {action.reason}")
            print(f"     Strategy: {action.metadata.get('strategy', 'N/A')}")

    # 6. Auto-tagging
    print("\n" + "=" * 80)
    print("6. Running Auto-Tagging")
    print("=" * 80)

    tag_actions = healing_engine.auto_tag(memories)

    print("✓ Auto-tagging analysis completed:")
    print(f"  Tag suggestions: {len(tag_actions)}")

    if tag_actions:
        print("\n  Sample suggestions:")
        for i, action in enumerate(tag_actions[:5], 1):
            print(f"  {i}. Memory: {action.memory_ids[0][:8]}...")
            print(f"     Suggested tags: {', '.join(action.metadata.get('suggested_tags', []))}")

    # 7. Importance adjustment
    print("\n" + "=" * 80)
    print("7. Running Importance Adjustment")
    print("=" * 80)

    # First, simulate some access patterns
    for memory in memories[:10]:
        memory.access_count = 5 + (hash(memory.id) % 10)  # Random access counts

    importance_actions = healing_engine.auto_importance_adjustment(
        memories=memories[:10], access_weight=0.5
    )

    print("✓ Importance adjustment analysis:")
    print(f"  Adjustments recommended: {len(importance_actions)}")

    if importance_actions:
        print("\n  Sample adjustments:")
        for i, action in enumerate(importance_actions[:5], 1):
            print(f"  {i}. {action.reason}")
            print(f"     {action.impact}")

    # 8. Full health check
    print("\n" + "=" * 80)
    print("8. Running Full Health Check")
    print("=" * 80)

    full_report = healing_engine.run_full_health_check(
        user_id=client.user_id, memories=memories, config=config, dry_run=True
    )

    print("✓ Full health check completed:")
    print(f"  Duration: {full_report.duration_seconds:.2f}s")
    print(f"  Total actions recommended: {len(full_report.actions_recommended)}")
    print(f"  Total actions applied: {len(full_report.actions_applied)}")
    print(f"  Health before: {full_report.health_before:.1f}/100")
    print(f"  Health after: {full_report.health_after:.1f}/100")

    print("\n  Improvements:")
    for metric, value in full_report.improvements.items():
        print(f"    - {metric}: {value}")

    print(f"\n  Summary: {full_report.summary}")

    # 9. Consolidation example
    print("\n" + "=" * 80)
    print("9. Memory Consolidation Example")
    print("=" * 80)

    # Find duplicate memories
    duplicate_memories = [m for m in memories if "coffee over tea" in m.text]

    if len(duplicate_memories) >= 2:
        consolidation = healing_engine.create_consolidation(
            user_id=client.user_id,
            memories=duplicate_memories,
            strategy=ConsolidationStrategy.SEMANTIC_MERGE,
        )

        if consolidation:
            print("✓ Consolidation created:")
            print(f"  Original memories: {len(consolidation.original_memory_ids)}")
            print(f"  Strategy: {consolidation.strategy.value}")
            print(f"  Content preserved: {consolidation.content_preserved:.0%}")
            print(f"  Summary: {consolidation.summary}")
            print(f"  Reversible: {consolidation.reversible}")

    # 10. Action breakdown by type
    print("\n" + "=" * 80)
    print("10. Action Breakdown")
    print("=" * 80)

    action_counts = {}
    for action in full_report.actions_recommended:
        action_type = action.action_type.value
        action_counts[action_type] = action_counts.get(action_type, 0) + 1

    print("✓ Recommended actions by type:")
    for action_type, count in sorted(action_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {action_type}: {count}")

    # 11. Final health assessment
    print("\n" + "=" * 80)
    print("11. Final Assessment")
    print("=" * 80)

    print("✓ Auto-healing demo completed!")
    print(f"  - Total memories: {len(memories)}")
    print(f"  - Initial health: {health_score.overall_score:.1f}/100 ({health_score.status.value})")
    print(f"  - Potential health: {full_report.health_after:.1f}/100 (after healing)")
    print(f"  - Total actions: {len(full_report.actions_recommended)}")
    print(f"  - Analysis time: {full_report.duration_seconds:.2f}s")

    print("\n💡 Key Features Demonstrated:")
    print("  1. Automatic health monitoring and scoring")
    print("  2. Smart cleanup of stale memories")
    print("  3. Duplicate detection and consolidation")
    print("  4. Automatic tagging suggestions")
    print("  5. Importance score adjustment based on usage")
    print("  6. Comprehensive health check reports")
    print("  7. Configurable auto-healing policies")
    print("  8. Dry-run mode for safe testing")

    print("\n📋 Next Steps:")
    print("  1. Apply recommended actions (set dry_run=False)")
    print("  2. Schedule automatic maintenance tasks")
    print("  3. Monitor health improvements over time")
    print("  4. Customize healing policies per use case")


if __name__ == "__main__":
    main()
