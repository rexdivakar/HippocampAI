#!/usr/bin/env python3
"""Example: Library User Controlling SaaS Features.

This example shows how library users can configure and control
SaaS automation features programmatically.
"""

import os
import sys

try:
    from hippocampai import (
        AutomationController,
        AutomationPolicy,
        AutomationSchedule,
        MemoryClient,
        PolicyType,
    )
    from hippocampai.adapters import GroqLLM
    from hippocampai.embed.embedder import Embedder
    from hippocampai.saas import TaskManager, TaskPriority
except ImportError:
    print("ERROR: hippocampai not installed. Install with: pip install -e .")
    sys.exit(1)


def main():
    """Demonstrate unified SaaS control from library."""

    print("\n" + "=" * 70)
    print("  🚀 HippocampAI - Unified Library & SaaS Control Demo")
    print("=" * 70)

    # 1. Initialize components
    print("\n📦 Step 1: Initialize Components")
    print("-" * 70)

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("ERROR: GROQ_API_KEY environment variable not set")
        sys.exit(1)

    llm = GroqLLM(api_key=api_key, model="llama-3.1-8b-instant")
    client = MemoryClient(llm_provider="groq", llm_model="llama-3.1-8b-instant")
    embedder = Embedder(model_name="BAAI/bge-small-en-v1.5")

    print("✅ Memory client initialized")

    # Initialize automation controller
    automation = AutomationController(memory_service=client, llm=llm, embedder=embedder)

    print("✅ Automation controller initialized")

    # Initialize task manager (can use "celery", "rq", or "inline")
    task_manager = TaskManager(automation_controller=automation, backend="inline")

    print("✅ Task manager initialized (backend: inline)")

    # 2. Create automation policy
    print("\n⚙️  Step 2: Create Automation Policy")
    print("-" * 70)

    policy = AutomationPolicy(
        user_id="demo_user",
        policy_type=PolicyType.THRESHOLD,
        # Enable features
        auto_summarization=True,
        auto_consolidation=True,
        auto_compression=False,
        importance_decay=True,
        health_monitoring=True,
        conflict_resolution=True,
        # Configure thresholds
        summarization_threshold=50,  # Low threshold for demo
        consolidation_threshold=30,
        summarization_age_days=7,  # Short window for demo
        consolidation_similarity=0.85,
        decay_half_life_days=90,
        # Optional: Configure schedules (for SaaS cron)
        summarization_schedule=AutomationSchedule(
            enabled=True,
            cron_expression="0 2 * * *",  # 2 AM daily
        ),
        health_check_schedule=AutomationSchedule(
            enabled=True,
            interval_hours=24,  # Every 24 hours
        ),
        # Alert settings
        health_alert_threshold=60.0,
        conflict_strategy="temporal",
        auto_resolve_conflicts=True,
    )

    automation.create_policy(policy)

    print(f"✅ Created policy for: {policy.user_id}")
    print(f"   Policy type: {policy.policy_type}")
    print("   Features enabled:")
    print(f"     • Auto-summarization: {policy.auto_summarization}")
    print(f"     • Auto-consolidation: {policy.auto_consolidation}")
    print(f"     • Auto-compression: {policy.auto_compression}")
    print(f"     • Importance decay: {policy.importance_decay}")
    print(f"     • Health monitoring: {policy.health_monitoring}")
    print(f"     • Conflict resolution: {policy.conflict_resolution}")

    # 3. Add some memories
    print("\n💾 Step 3: Add Memories")
    print("-" * 70)

    memories_to_add = [
        "I love coffee in the morning",
        "My favorite color is blue",
        "I work as a software engineer",
        "I prefer Python over JavaScript",
        "I enjoy hiking on weekends",
        "I'm learning machine learning",
        "Coffee helps me focus",  # Related to first memory
        "I like blue shirts",  # Related to color preference
    ]

    for i, text in enumerate(memories_to_add, 1):
        client.remember(text=text, user_id="demo_user", type="preference")
        print(f"  {i}. Added: {text}")

    # 4. Check if optimization should run
    print("\n🔍 Step 4: Check Optimization Triggers")
    print("-" * 70)

    stats = client.get_memory_statistics(user_id="demo_user")
    memory_count = stats.get("total_memories", 0)

    print(f"Current memory count: {memory_count}")
    print(f"Summarization threshold: {policy.summarization_threshold}")
    print(f"Consolidation threshold: {policy.consolidation_threshold}")

    should_summarize = automation.should_run_summarization("demo_user")
    should_consolidate = automation.should_run_consolidation("demo_user")

    print(f"\nShould run summarization? {should_summarize}")
    print(f"Should run consolidation? {should_consolidate}")

    # 5. Run optimizations
    print("\n⚡ Step 5: Run Optimizations")
    print("-" * 70)

    # Run health check first
    print("\n🏥 Running health check...")
    health_result = automation.run_health_check("demo_user", force=True)

    if health_result["status"] == "success":
        print(f"✅ Health Score: {health_result['health_score']:.1f}/100")
        print(f"   Status: {health_result['health_status']}")
        print(f"   Alert needed: {health_result['alert_needed']}")

        if health_result["recommendations"]:
            print("   Recommendations:")
            for rec in health_result["recommendations"][:3]:
                print(f"     • {rec}")

    # Run consolidation (force for demo)
    print("\n🔄 Running consolidation...")
    consolidation_result = automation.run_consolidation("demo_user", force=True)

    if consolidation_result["status"] == "success":
        print("✅ Consolidation complete:")
        print(f"   Memories processed: {consolidation_result['memories_processed']}")
        print(f"   Clusters found: {consolidation_result['clusters_found']}")
        print(f"   Consolidated: {consolidation_result['consolidated_memories']}")

    # 6. Submit background tasks
    print("\n📋 Step 6: Submit Background Tasks")
    print("-" * 70)

    # Submit various tasks
    task1 = task_manager.submit_task(
        user_id="demo_user",
        task_type="health_check",
        priority=TaskPriority.HIGH,
        async_execution=False,  # Synchronous for demo
    )

    print(f"✅ Task 1 (health_check): {task1.status}")
    if task1.result:
        print(f"   Result: {task1.result}")

    task2 = task_manager.submit_task(
        user_id="demo_user",
        task_type="decay",
        priority=TaskPriority.NORMAL,
        async_execution=False,
    )

    print(f"✅ Task 2 (decay): {task2.status}")
    if task2.result:
        print(f"   Memories updated: {task2.result.get('memories_updated', 0)}")

    # 7. Get comprehensive statistics
    print("\n📊 Step 7: Get Statistics")
    print("-" * 70)

    stats = automation.get_user_statistics("demo_user")

    print(f"Total memories: {stats.get('total_memories', 0)}")
    print(f"Memory types: {stats.get('memory_by_type', {})}")
    print(f"Automation enabled: {stats.get('automation_enabled', False)}")

    if "automation_features" in stats:
        print("Automation features:")
        for feature, enabled in stats["automation_features"].items():
            print(f"  • {feature}: {enabled}")

    # 8. Update policy
    print("\n🔧 Step 8: Update Policy")
    print("-" * 70)

    # Get current policy
    current_policy = automation.get_policy("demo_user")

    if current_policy is None:
        print("❌ No policy found for user")
        return

    # Modify settings
    current_policy.auto_compression = True  # Enable compression
    current_policy.compression_threshold = 100

    # Update
    automation.create_policy(current_policy)

    print("✅ Policy updated:")
    print(f"   Compression enabled: {current_policy.auto_compression}")
    print(f"   Compression threshold: {current_policy.compression_threshold}")

    # 9. Get task history
    print("\n📜 Step 9: Get Task History")
    print("-" * 70)

    tasks = task_manager.get_user_tasks("demo_user", limit=10)

    print(f"Found {len(tasks)} tasks:")
    for i, task in enumerate(tasks, 1):
        print(f"  {i}. {task.task_type} - {task.status}")
        if task.completed_at:
            duration = (task.completed_at - task.created_at).total_seconds()
            print(f"     Duration: {duration:.2f}s")

    # Summary
    print("\n" + "=" * 70)
    print("  ✅ Demo Complete!")
    print("=" * 70)
    print("\nYou've seen how library users can:")
    print("  1. Configure automation policies programmatically")
    print("  2. Control feature toggles and thresholds")
    print("  3. Run optimizations immediately or in background")
    print("  4. Monitor task execution and results")
    print("  5. Update policies on the fly")
    print("\nThe same code works for:")
    print("  • Direct library usage (immediate execution)")
    print("  • SaaS deployment (background workers)")
    print("  • Hybrid mode (mix of both)")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
