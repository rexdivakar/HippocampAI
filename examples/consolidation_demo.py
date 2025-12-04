"""
End-to-end example of Memory Consolidation (Sleep Phase).

This script demonstrates the complete consolidation pipeline:
1. Create sample memories
2. Run consolidation
3. Show before/after state
"""

import json
import logging
from datetime import datetime, timedelta, timezone
from pprint import pprint

from hippocampai.consolidation.models import ConsolidationRun
from hippocampai.consolidation.policy import (
    ConsolidationPolicy,
    ConsolidationPolicyEngine,
    apply_consolidation_decisions,
)
from hippocampai.consolidation.prompts import (
    EXAMPLE_MEMORIES_DATA,
    build_consolidation_prompt,
    get_example_consolidation_response,
)
from hippocampai.models import Memory, MemoryType

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ================================================================
# SAMPLE DATA
# ================================================================

# Sample user
USER_ID = "user-demo-123"

# Sample memories (representing a day's activities)
SAMPLE_MEMORIES = [
    {
        "id": "mem-1",
        "text": "Had morning coffee at 9am while reading emails",
        "type": "event",
        "importance": 2.5,
        "tags": ["routine", "morning"],
        "access_count": 0,
        "created_at": datetime.now(timezone.utc) - timedelta(hours=8),
    },
    {
        "id": "mem-2",
        "text": "Q4 roadmap planning meeting with product team. Discussed AI features, analytics dashboard, and mobile app improvements. Team consensus: prioritize AI.",
        "type": "event",
        "importance": 7.5,
        "tags": ["work", "Q4", "planning", "meeting"],
        "access_count": 2,
        "created_at": datetime.now(timezone.utc) - timedelta(hours=7),
    },
    {
        "id": "mem-3",
        "text": "Strategic decision: Prioritize AI-powered features over mobile improvements for Q4 launch",
        "type": "fact",
        "importance": 8.5,
        "tags": ["decision", "strategy", "Q4", "AI"],
        "access_count": 3,
        "created_at": datetime.now(timezone.utc) - timedelta(hours=6, minutes=30),
    },
    {
        "id": "mem-4",
        "text": "Lunch break at 12:30pm - grabbed sandwich from cafe downstairs",
        "type": "event",
        "importance": 1.0,
        "tags": ["routine"],
        "access_count": 0,
        "created_at": datetime.now(timezone.utc) - timedelta(hours=4),
    },
    {
        "id": "mem-5",
        "text": "Goal: Hire 2 senior ML engineers for AI team by end of Q3",
        "type": "goal",
        "importance": 7.5,
        "tags": ["hiring", "AI", "team", "Q3"],
        "access_count": 1,
        "created_at": datetime.now(timezone.utc) - timedelta(hours=3),
    },
    {
        "id": "mem-6",
        "text": "Daily standup: All team members on track with sprint goals. No blockers reported.",
        "type": "event",
        "importance": 4.0,
        "tags": ["team", "standup", "agile"],
        "access_count": 0,
        "created_at": datetime.now(timezone.utc) - timedelta(hours=2),
    },
    {
        "id": "mem-7",
        "text": "Sprint retrospective feedback: Team wants better async communication tools (Slack threads too noisy)",
        "type": "preference",
        "importance": 6.5,
        "tags": ["team", "feedback", "tools", "communication"],
        "access_count": 1,
        "created_at": datetime.now(timezone.utc) - timedelta(hours=1, minutes=30),
    },
    {
        "id": "mem-8",
        "text": "Coffee break at 3pm",
        "type": "event",
        "importance": 1.5,
        "tags": ["routine"],
        "access_count": 0,
        "created_at": datetime.now(timezone.utc) - timedelta(hours=1),
    },
]


def create_sample_memories(user_id: str) -> list[Memory]:
    """Convert sample data to Memory objects."""
    memories = []
    for data in SAMPLE_MEMORIES:
        mem = Memory(
            id=data["id"],
            text=data["text"],
            user_id=user_id,
            type=MemoryType(data["type"]),
            importance=data["importance"],
            tags=data.get("tags", []),
            access_count=data.get("access_count", 0),
            created_at=data["created_at"],
            updated_at=data["created_at"],
        )
        memories.append(mem)
    return memories


# ================================================================
# STEP 1: PREPARE MEMORIES
# ================================================================


def step1_prepare_memories():
    """Create and display sample memories."""
    logger.info("=" * 70)
    logger.info("STEP 1: PREPARE SAMPLE MEMORIES")
    logger.info("=" * 70)

    memories = create_sample_memories(USER_ID)

    print(f"\nCreated {len(memories)} sample memories for user {USER_ID}:\n")
    for mem in memories:
        print(
            f"  [{mem.id}] {mem.type.value:12s} | importance={mem.importance:3.1f} | "
            f"accessed={mem.access_count}x | {mem.text[:60]}..."
        )

    return memories


# ================================================================
# STEP 2: GENERATE LLM PROMPT
# ================================================================


def step2_generate_prompt(memories: list[Memory]):
    """Generate the consolidation prompt for LLM."""
    logger.info("\n" + "=" * 70)
    logger.info("STEP 2: GENERATE LLM CONSOLIDATION PROMPT")
    logger.info("=" * 70)

    # User context for the prompt
    user_context = {
        "name": "Alex",
        "role": "Product Manager",
        "current_goals": ["Launch Q4 AI features", "Build high-performing team"],
        "recent_topics": ["Q4 roadmap", "hiring", "team dynamics"],
    }

    # Build prompt
    prompt = build_consolidation_prompt(
        memories=memories,
        user_context=user_context,
        cluster_theme="Daily Work Activities",
    )

    print("\n" + "─" * 70)
    print("PROMPT (first 1000 chars):")
    print("─" * 70)
    print(prompt[:1000] + "...\n")

    return prompt


# ================================================================
# STEP 3: SIMULATE LLM RESPONSE
# ================================================================


def step3_simulate_llm_response():
    """Simulate LLM response with consolidation decisions."""
    logger.info("=" * 70)
    logger.info("STEP 3: SIMULATE LLM CONSOLIDATION DECISIONS")
    logger.info("=" * 70)

    # Get example response (what LLM would return)
    llm_response = {
        "promoted_facts": [
            {
                "id": "mem-3",
                "reason": "Strategic product decision with long-term impact",
                "new_importance": 9.0,
            },
            {
                "id": "mem-5",
                "reason": "Critical hiring goal tied to product roadmap",
                "new_importance": 8.0,
            },
        ],
        "low_value_memory_ids": ["mem-1", "mem-4", "mem-8"],
        "updated_memories": [
            {
                "id": "mem-2",
                "new_text": "Q4 roadmap meeting: Decided to prioritize AI features (analytics dashboard, intelligent automation) over mobile improvements. Team aligned on vision.",
                "new_importance": 8.0,
                "merge_from_ids": ["mem-3"],
            }
        ],
        "synthetic_memories": [
            {
                "text": "Productive planning day: Finalized Q4 AI-first roadmap, identified need for 2 ML engineer hires, gathered team feedback on communication tools. Key strategic shifts made.",
                "type": "context",
                "importance": 7.5,
                "tags": ["work", "Q4", "planning", "AI", "hiring", "team", "summary"],
                "source_ids": ["mem-2", "mem-3", "mem-5", "mem-7"],
            }
        ],
        "reasoning": "User had a strategic planning day focused on Q4 roadmap. Promoted key decisions and goals. Archived routine/transient events (coffee breaks, lunch). Created high-level summary capturing the day's strategic work.",
    }

    print("\nLLM CONSOLIDATION DECISIONS:")
    print("─" * 70)
    print(json.dumps(llm_response, indent=2))
    print()

    return llm_response


# ================================================================
# STEP 4: APPLY POLICY VALIDATION
# ================================================================


def step4_apply_policy(memories: list[Memory], llm_decisions: dict):
    """Apply consolidation policy to validate LLM decisions."""
    logger.info("=" * 70)
    logger.info("STEP 4: APPLY POLICY VALIDATION")
    logger.info("=" * 70)

    # Create policy with default settings
    policy = ConsolidationPolicy(
        min_importance_to_keep=3.0,
        min_age_days_for_deletion=0,  # Allow immediate deletion for demo
        never_delete_if_accessed=True,
        promotion_multiplier=1.3,
    )

    # Apply decisions
    actions = apply_consolidation_decisions(
        memories=memories,
        llm_decisions=llm_decisions,
        policy=policy,
        dry_run=False,  # Actually modify memory objects
    )

    print("\nPOLICY-VALIDATED ACTIONS:")
    print("─" * 70)
    print(f"✓ To Delete:     {len(actions['to_delete'])} memories")
    for action in actions["to_delete"]:
        print(f"    - {action['id']}: {action['reason']}")

    print(f"\n✓ To Archive:    {len(actions['to_archive'])} memories")
    for action in actions["to_archive"]:
        print(f"    - {action['id']}: {action['reason']}")

    print(f"\n✓ To Promote:    {len(actions['to_promote'])} memories")
    for action in actions["to_promote"]:
        print(f"    - {action['id']}: {action['old_importance']:.1f} → {action['new_importance']:.1f} ({action['reason']})")

    print(f"\n✓ To Update:     {len(actions['to_update'])} memories")
    for action in actions["to_update"]:
        print(f"    - {action['id']}: Updated text and importance")

    print(f"\n✓ To Synthesize: {len(actions['to_create'])} new memories")
    for action in actions["to_create"]:
        print(f"    - New synthetic memory: {action['text'][:80]}...")

    print(f"\n⚠ Blocked:       {len(actions['blocked'])} actions")
    for action in actions["blocked"]:
        print(f"    - {action['id']}: {action['action']} blocked - {action['reason']}")

    print("\n" + "─" * 70)
    print("STATISTICS:")
    print("─" * 70)
    for key, value in actions["stats"].items():
        print(f"  {key:15s}: {value}")
    print()

    return actions


# ================================================================
# STEP 5: SHOW BEFORE/AFTER
# ================================================================


def step5_show_before_after(memories: list[Memory], actions: dict):
    """Display memory state before and after consolidation."""
    logger.info("=" * 70)
    logger.info("STEP 5: BEFORE/AFTER COMPARISON")
    logger.info("=" * 70)

    # Build memory lookup
    memory_map = {m.id: m for m in memories}

    print("\nBEFORE CONSOLIDATION:")
    print("─" * 70)
    print(f"Total memories: {len(memories)}")
    print(f"Avg importance: {sum(m.importance for m in memories) / len(memories):.2f}")
    print("\nMEMORIES:")
    for mem in sorted(memories, key=lambda x: x.importance, reverse=True):
        status = ""
        if mem.id in [a["id"] for a in actions["to_delete"]]:
            status = "❌ DELETED"
        elif mem.id in [a["id"] for a in actions["to_promote"]]:
            status = "⬆ PROMOTED"
        elif mem.id in [a["id"] for a in actions["to_update"]]:
            status = "✏ UPDATED"

        print(f"  [{mem.id}] importance={mem.importance:4.1f} {status:12s} | {mem.text[:50]}...")

    print("\nAFTER CONSOLIDATION:")
    print("─" * 70)

    # Calculate remaining memories
    deleted_ids = {a["id"] for a in actions["to_delete"]}
    archived_ids = {a["id"] for a in actions["to_archive"]}
    remaining = [m for m in memories if m.id not in deleted_ids and m.id not in archived_ids]
    remaining_count = len(remaining) + len(actions["to_create"])  # Add synthetics

    print(f"Total memories: {remaining_count}")
    print(f"Avg importance: {sum(m.importance for m in remaining) / len(remaining):.2f}" if remaining else "N/A")
    print("\nMEMORIES:")

    # Show remaining memories (sorted by importance)
    for mem in sorted(remaining, key=lambda x: x.importance, reverse=True):
        print(f"  [{mem.id}] importance={mem.importance:4.1f}           | {mem.text[:50]}...")

    # Show synthetic memories
    for syn in actions["to_create"]:
        print(f"  [NEW-SYNTH] importance={syn['importance']:4.1f} 🔄 SYNTHETIC | {syn['text'][:50]}...")

    print("\nCHANGES SUMMARY:")
    print("─" * 70)
    print(f"  Memories removed:     {len(deleted_ids) + len(archived_ids)}")
    print(f"  Memories promoted:    {len(actions['to_promote'])}")
    print(f"  Memories synthesized: {len(actions['to_create'])}")
    print(f"  Net change:           {remaining_count - len(memories):+d}")
    print()


# ================================================================
# STEP 6: GENERATE DREAM REPORT
# ================================================================


def step6_generate_dream_report(actions: dict):
    """Generate a human-readable dream report."""
    logger.info("=" * 70)
    logger.info("STEP 6: DREAM REPORT")
    logger.info("=" * 70)

    report = f"""
╔══════════════════════════════════════════════════════════════════════╗
║                    NIGHTLY CONSOLIDATION REPORT                      ║
║                       User: {USER_ID:36s} ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  📊 STATISTICS                                                       ║
║  ────────────────────────────────────────────────────────────────   ║
║    Memories reviewed:      {actions['stats']['reviewed']:3d}                                      ║
║    Memories deleted:       {actions['stats']['deleted']:3d}                                      ║
║    Memories archived:      {actions['stats']['archived']:3d}                                      ║
║    Memories promoted:      {actions['stats']['promoted']:3d}                                      ║
║    Memories updated:       {actions['stats']['updated']:3d}                                      ║
║    Memories synthesized:   {actions['stats']['synthesized']:3d}                                      ║
║                                                                      ║
║  💡 KEY INSIGHTS                                                     ║
║  ────────────────────────────────────────────────────────────────   ║
║    • Productive strategic planning session focused on Q4            ║
║    • Major decision: AI-first roadmap for Q4 launch                 ║
║    • Identified hiring needs: 2 ML engineers                        ║
║    • Team feedback: Need better async communication                 ║
║    • Cleared 3 low-value transient events (routine activities)      ║
║                                                                      ║
║  🎯 PROMOTED MEMORIES                                                ║
║  ────────────────────────────────────────────────────────────────   ║
"""

    for action in actions["to_promote"]:
        report += f"║    • [{action['id']}] {action['old_importance']:.1f}→{action['new_importance']:.1f}: {action['reason'][:35]:35s} ║\n"

    report += """║                                                                      ║
║  🗑️  CLEANED UP                                                     ║
║  ────────────────────────────────────────────────────────────────   ║
"""

    for action in actions["to_delete"][:3]:  # Show first 3
        report += f"║    • {action['id']}: {action['reason'][:49]:49s} ║\n"

    report += """║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
"""

    print(report)


# ================================================================
# MAIN: RUN COMPLETE DEMO
# ================================================================


def main():
    """Run the complete consolidation demo."""
    logger.info("╔═══════════════════════════════════════════════════════════════╗")
    logger.info("║   HIPPOCAMPAI SLEEP PHASE CONSOLIDATION - END-TO-END DEMO    ║")
    logger.info("╚═══════════════════════════════════════════════════════════════╝\n")

    # Run all steps
    memories = step1_prepare_memories()
    prompt = step2_generate_prompt(memories)
    llm_response = step3_simulate_llm_response()
    actions = step4_apply_policy(memories, llm_response)
    step5_show_before_after(memories, actions)
    step6_generate_dream_report(actions)

    logger.info("✅ Demo completed successfully!\n")

    print("\n" + "=" * 70)
    print("NEXT STEPS:")
    print("=" * 70)
    print("1. Integrate actual LLM API calls in tasks.py (call_llm_for_consolidation)")
    print("2. Implement database persistence (persist_consolidation_changes)")
    print("3. Add Celery Beat schedule to run nightly at 3 AM")
    print("4. Enable ACTIVE_CONSOLIDATION_ENABLED=true in .env")
    print("5. Monitor logs and metrics for first few runs")
    print("6. Fine-tune thresholds based on your use case")
    print("=" * 70)


if __name__ == "__main__":
    main()
