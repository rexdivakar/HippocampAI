"""Demonstration of memory consolidation scheduler.

This example shows how to use the background scheduler for
automatic memory maintenance tasks like consolidation and decay.

Run this with:
    python examples/08_scheduler_demo.py
"""

import time

from hippocampai import MemoryClient
from hippocampai.config import Config


def print_section(title: str):
    """Print a section header."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def main():
    """Run the scheduler demonstration."""

    # === 1. INITIALIZE CLIENT WITH SCHEDULER ===
    print_section("1. Initialize Client with Scheduler")

    config = Config(
        enable_scheduler=True,
        consolidate_cron="0 3 * * 0",  # Weekly on Sunday at 3am
        decay_cron="0 2 * * *",  # Daily at 2am
        snapshot_cron="0 * * * *",  # Hourly
    )

    client = MemoryClient(config=config, enable_telemetry=True)
    print("✓ Client initialized with scheduler enabled")
    print(f"  Consolidation: {config.consolidate_cron}")
    print(f"  Decay: {config.decay_cron}")
    print(f"  Snapshots: {config.snapshot_cron}")

    # === 2. START SCHEDULER ===
    print_section("2. Start Background Scheduler")

    client.start_scheduler()
    print("✓ Scheduler started")

    # Get scheduler status
    status = client.get_scheduler_status()
    print(f"\nScheduler status: {status['status']}")
    print(f"Registered jobs: {len(status['jobs'])}")

    for job in status["jobs"]:
        print(f"\n  Job: {job['name']}")
        print(f"    ID: {job['id']}")
        print(f"    Next run: {job['next_run']}")
        print(f"    Trigger: {job['trigger']}")

    # === 3. CREATE SAMPLE MEMORIES ===
    print_section("3. Create Sample Memories")

    user_id = "demo_user"

    # Create similar memories (candidates for consolidation)
    print("Creating similar memories about Python...")
    memories = [
        "I love Python programming",
        "Python is my favorite programming language",
        "I really enjoy coding in Python",
        "Python is great for data science",
        "I use Python for machine learning projects",
    ]

    for text in memories:
        client.remember(text=text, user_id=user_id, type="preference", importance=8.0)

    print(f"✓ Created {len(memories)} memories")

    # === 4. MANUAL CONSOLIDATION TRIGGER ===
    print_section("4. Manual Consolidation Trigger")

    print("Manually triggering consolidation job...")
    initial_count = len(client.get_memories(user_id=user_id))
    print(f"Initial memory count: {initial_count}")

    client.scheduler.trigger_consolidation_now()

    # Wait a bit for consolidation to complete
    time.sleep(1)

    final_count = len(client.get_memories(user_id=user_id))
    print(f"Final memory count: {final_count}")

    if final_count < initial_count:
        print(f"✓ Consolidated {initial_count - final_count} memories into clusters")
    else:
        print("✓ No consolidation needed (memories not similar enough)")

    # === 5. IMPORTANCE DECAY ===
    print_section("5. Importance Decay")

    print("Creating memories with high importance...")
    for i in range(3):
        client.remember(
            text=f"Important fact {i}",
            user_id=user_id,
            type="fact",
            importance=9.0,
        )

    print("Manually triggering importance decay...")
    decayed = client.apply_importance_decay()
    print(f"✓ Processed {decayed} memories")

    # Note: New memories won't decay much, this is just a demonstration

    # === 6. SNAPSHOT CREATION ===
    print_section("6. Snapshot Creation")

    print("Manually triggering snapshot creation...")
    client.scheduler.trigger_snapshot_now()
    print("✓ Snapshots created for both collections")

    # === 7. SCHEDULER STATUS ===
    print_section("7. Scheduler Status Check")

    status = client.get_scheduler_status()
    print(f"Status: {status['status']}")
    print(f"Active jobs: {len(status['jobs'])}")

    # === 8. CLEANUP ===
    print_section("8. Cleanup")

    print("Stopping scheduler...")
    client.stop_scheduler()

    status = client.get_scheduler_status()
    print(f"✓ Scheduler stopped (status: {status['status']})")

    # Delete test memories
    memories = client.get_memories(user_id=user_id)
    memory_ids = [m.id for m in memories]
    if memory_ids:
        deleted = client.delete_memories(memory_ids, user_id)
        print(f"✓ Cleaned up {deleted} test memories")

    # === USAGE NOTES ===
    print_section("Usage Notes")

    print("""
Scheduler Configuration:

1. Enable in config:
   config = Config(enable_scheduler=True)

2. Set cron schedules:
   - consolidate_cron: "0 3 * * 0" (weekly)
   - decay_cron: "0 2 * * *" (daily)
   - snapshot_cron: "0 * * * *" (hourly)

3. Control scheduler:
   - client.start_scheduler()
   - client.stop_scheduler()
   - client.get_scheduler_status()

4. Manual triggers:
   - client.scheduler.trigger_consolidation_now()
   - client.scheduler.trigger_decay_now()
   - client.scheduler.trigger_snapshot_now()

5. Direct calls (without scheduler):
   - client.consolidate_all_memories(similarity_threshold=0.85)
   - client.apply_importance_decay()
   - client.create_snapshot("facts")

Cron Format:
  minute hour day_of_month month day_of_week
  Example: "0 3 * * 0" = Every Sunday at 3:00 AM
""")

    print("\n" + "=" * 60)
    print("  Scheduler demonstration completed!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
