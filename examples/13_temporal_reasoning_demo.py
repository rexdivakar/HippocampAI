"""
Temporal Reasoning Demo

This example demonstrates HippocampAI's temporal reasoning capabilities:
- Time-based memory queries
- Chronological narrative construction
- Event sequence analysis
- Memory timeline creation
- Future memory scheduling
- Temporal statistics and summaries

Features demonstrated:
1. Time-range based queries (last week, last month, etc.)
2. Custom time range queries
3. Building chronological narratives
4. Creating memory timelines
5. Analyzing event sequences
6. Scheduling future memories with recurrence
7. Getting temporal summaries
"""

from datetime import datetime, timedelta, timezone
from hippocampai import MemoryClient, TimeRange
import time


def print_section(title: str):
    """Print a section header."""
    print(f"\n{'=' * 80}")
    print(f"{title:^80}")
    print(f"{'=' * 80}\n")


def demo_time_range_queries(client: MemoryClient, user_id: str):
    """Demonstrate time-range based memory queries."""
    print_section("1. TIME-RANGE BASED QUERIES")

    # Get memories from last week
    print("📅 Retrieving memories from LAST_WEEK...")
    last_week_memories = client.get_memories_by_time_range(
        user_id=user_id,
        time_range=TimeRange.LAST_WEEK
    )
    print(f"Found {len(last_week_memories)} memories from last week")
    for mem in last_week_memories[:3]:
        print(f"  • {mem.text[:60]}... (created: {mem.created_at.strftime('%Y-%m-%d %H:%M')})")

    # Get memories from today
    print("\n📅 Retrieving memories from TODAY...")
    today_memories = client.get_memories_by_time_range(
        user_id=user_id,
        time_range=TimeRange.TODAY
    )
    print(f"Found {len(today_memories)} memories from today")

    # Get memories from this month
    print("\n📅 Retrieving memories from THIS_MONTH...")
    this_month_memories = client.get_memories_by_time_range(
        user_id=user_id,
        time_range=TimeRange.THIS_MONTH
    )
    print(f"Found {len(this_month_memories)} memories from this month")


def demo_custom_time_range(client: MemoryClient, user_id: str):
    """Demonstrate custom time range queries."""
    print_section("2. CUSTOM TIME RANGE QUERIES")

    # Get memories from last 3 days
    print("📅 Retrieving memories from last 3 days (custom range)...")
    start = datetime.now(timezone.utc) - timedelta(days=3)
    end = datetime.now(timezone.utc)

    custom_memories = client.get_memories_by_time_range(
        user_id=user_id,
        start_time=start,
        end_time=end
    )
    print(f"Found {len(custom_memories)} memories in custom range")
    print(f"Range: {start.strftime('%Y-%m-%d %H:%M')} to {end.strftime('%Y-%m-%d %H:%M')}")

    for mem in custom_memories[:5]:
        age = datetime.now(timezone.utc) - mem.created_at
        print(f"  • {mem.text[:60]}... ({age.days} days, {age.seconds//3600} hours ago)")


def demo_chronological_narrative(client: MemoryClient, user_id: str):
    """Demonstrate building chronological narratives."""
    print_section("3. CHRONOLOGICAL NARRATIVES")

    print("📖 Building narrative from last week's memories...")
    narrative = client.build_memory_narrative(
        user_id=user_id,
        time_range=TimeRange.LAST_WEEK,
        title="My Week in Review"
    )

    print("\nGenerated Narrative:")
    print("-" * 80)
    print(narrative)
    print("-" * 80)


def demo_memory_timeline(client: MemoryClient, user_id: str):
    """Demonstrate creating memory timelines."""
    print_section("4. MEMORY TIMELINES")

    print("🗓️  Creating timeline for last month...")
    timeline = client.create_memory_timeline(
        user_id=user_id,
        title="Last Month's Journey",
        time_range=TimeRange.LAST_MONTH
    )

    print(f"\nTimeline: {timeline.title}")
    print(f"Events: {len(timeline.events)}")
    print(f"Time span: {timeline.start_time.strftime('%Y-%m-%d')} to {timeline.end_time.strftime('%Y-%m-%d')}")
    print(f"Duration: {timeline.get_duration()}")

    print("\nTop Events:")
    for event in timeline.events[:5]:
        print(f"  📌 {event.timestamp.strftime('%Y-%m-%d %H:%M')} - {event.event_type}")
        print(f"     {event.text[:70]}...")
        if event.participants:
            print(f"     Participants: {', '.join(event.participants)}")
        if event.location:
            print(f"     Location: {event.location}")


def demo_event_sequences(client: MemoryClient, user_id: str):
    """Demonstrate analyzing event sequences."""
    print_section("5. EVENT SEQUENCE ANALYSIS")

    print("🔗 Analyzing event sequences (max gap: 24 hours)...")
    sequences = client.analyze_event_sequences(user_id=user_id, max_gap_hours=24)

    print(f"\nFound {len(sequences)} event sequences")

    for i, sequence in enumerate(sequences[:3], 1):
        print(f"\n  Sequence {i}: {len(sequence)} related events")
        print(f"  Timespan: {sequence[0].created_at.strftime('%Y-%m-%d %H:%M')} to {sequence[-1].created_at.strftime('%Y-%m-%d %H:%M')}")
        print("  Events:")
        for mem in sequence[:3]:
            print(f"    • {mem.text[:60]}...")
        if len(sequence) > 3:
            print(f"    ... and {len(sequence) - 3} more events")


def demo_memory_scheduling(client: MemoryClient, user_id: str):
    """Demonstrate scheduling future memories."""
    print_section("6. MEMORY SCHEDULING")

    # Schedule a one-time memory
    print("⏰ Scheduling a one-time memory for tomorrow...")
    tomorrow = datetime.now(timezone.utc) + timedelta(days=1)
    scheduled1 = client.schedule_memory(
        text="Follow up on project proposal",
        user_id=user_id,
        scheduled_for=tomorrow,
        type="fact",
        tags=["reminder", "work"]
    )
    print(f"  ✓ Scheduled: {scheduled1.text}")
    print(f"    Due: {scheduled1.scheduled_for.strftime('%Y-%m-%d %H:%M')}")
    print(f"    ID: {scheduled1.id}")

    # Schedule a recurring memory
    print("\n⏰ Scheduling a recurring memory (daily)...")
    tomorrow_morning = datetime.now(timezone.utc).replace(
        hour=9, minute=0, second=0, microsecond=0
    ) + timedelta(days=1)
    scheduled2 = client.schedule_memory(
        text="Morning standup meeting",
        user_id=user_id,
        scheduled_for=tomorrow_morning,
        type="event",
        tags=["meeting", "daily"],
        recurrence="daily",
        reminder_offset=15  # 15 minutes before
    )
    print(f"  ✓ Scheduled: {scheduled2.text}")
    print(f"    Due: {scheduled2.scheduled_for.strftime('%Y-%m-%d %H:%M')}")
    print(f"    Recurrence: {scheduled2.recurrence}")
    print(f"    Reminder: {scheduled2.reminder_offset} minutes before")

    # Schedule a weekly recurring memory
    print("\n⏰ Scheduling a weekly recurring memory...")
    next_week = datetime.now(timezone.utc) + timedelta(days=7)
    scheduled3 = client.schedule_memory(
        text="Weekly team retrospective",
        user_id=user_id,
        scheduled_for=next_week,
        type="event",
        tags=["meeting", "weekly"],
        recurrence="weekly"
    )
    print(f"  ✓ Scheduled: {scheduled3.text}")
    print(f"    Due: {scheduled3.scheduled_for.strftime('%Y-%m-%d %H:%M')}")
    print(f"    Recurrence: {scheduled3.recurrence}")

    # Check for due scheduled memories
    print("\n🔔 Checking for due scheduled memories...")
    due_memories = client.get_due_scheduled_memories()
    print(f"Found {len(due_memories)} due memories")

    if due_memories:
        for due in due_memories[:3]:
            print(f"  • {due.text}")
            print(f"    Due: {due.scheduled_for.strftime('%Y-%m-%d %H:%M')}")
            print(f"    Recurrence: {due.recurrence or 'None'}")


def demo_temporal_summary(client: MemoryClient, user_id: str):
    """Demonstrate getting temporal summaries."""
    print_section("7. TEMPORAL SUMMARIES")

    print("📊 Getting temporal summary for user...")
    summary = client.get_temporal_summary(user_id=user_id)

    print("\nTemporal Statistics:")
    print(f"  Total memories: {summary.get('total_memories', 0)}")
    print(f"  Time span: {summary.get('time_span_days', 0)} days")
    print(f"  First memory: {summary.get('first_memory', 'N/A')}")
    print(f"  Most recent: {summary.get('most_recent', 'N/A')}")

    if 'peak_activity_hour' in summary:
        print(f"\n  Peak activity hour: {summary['peak_activity_hour']}:00")

    if 'daily_distribution' in summary:
        print("\n  Daily distribution:")
        for day, count in list(summary['daily_distribution'].items())[:7]:
            bar = '█' * (count // 5) if count > 0 else ''
            print(f"    {day}: {count:3d} {bar}")

    if 'memory_type_distribution' in summary:
        print("\n  Memory type distribution:")
        for mem_type, count in summary['memory_type_distribution'].items():
            print(f"    {mem_type}: {count}")


def create_sample_memories(client: MemoryClient, user_id: str):
    """Create sample memories with different timestamps for demo."""
    print_section("SETUP: Creating Sample Memories")

    print("Creating sample memories across different time periods...")

    # Memories from today
    today_memories = [
        "Had breakfast at 8 AM",
        "Morning team meeting at 9 AM",
        "Worked on project proposal",
        "Lunch with colleagues at noon",
    ]

    # Memories from yesterday
    yesterday_memories = [
        "Completed code review",
        "Attended client presentation",
        "Evening workout at gym",
    ]

    # Memories from last week
    last_week_memories = [
        "Started new project on Monday",
        "Team building event on Wednesday",
        "Finished quarterly report on Friday",
    ]

    # Memories from last month
    last_month_memories = [
        "Company all-hands meeting",
        "Launched new feature",
        "Received positive client feedback",
    ]

    # Create today's memories
    for text in today_memories:
        client.remember(text, user_id, tags=["daily"])
        time.sleep(0.1)

    # Create yesterday's memories (simulate older memories)
    for text in yesterday_memories:
        client.remember(text, user_id, tags=["daily"])
        # Note: In production, you'd need database access to backdate timestamps
        time.sleep(0.1)

    # Create last week's memories
    for text in last_week_memories:
        client.remember(text, user_id, tags=["weekly"])
        time.sleep(0.1)

    # Create last month's memories
    for text in last_month_memories:
        client.remember(text, user_id, tags=["monthly"])
        time.sleep(0.1)

    print(f"✓ Created {len(today_memories) + len(yesterday_memories) + len(last_week_memories) + len(last_month_memories)} sample memories")


def main():
    """Run all temporal reasoning demos."""
    print("\n" + "=" * 80)
    print("HIPPOCAMPAI TEMPORAL REASONING DEMO".center(80))
    print("=" * 80)

    # Initialize client
    print("\n🚀 Initializing HippocampAI client...")
    client = MemoryClient(collection_name="temporal_demo")
    user_id = "demo_user_temporal"

    # Create sample memories
    create_sample_memories(client, user_id)

    # Run demos
    demo_time_range_queries(client, user_id)
    demo_custom_time_range(client, user_id)
    demo_chronological_narrative(client, user_id)
    demo_memory_timeline(client, user_id)
    demo_event_sequences(client, user_id)
    demo_memory_scheduling(client, user_id)
    demo_temporal_summary(client, user_id)

    print_section("DEMO COMPLETE")
    print("✓ All temporal reasoning features demonstrated successfully!")
    print("\nKey Takeaways:")
    print("  • Time-range queries enable efficient temporal filtering")
    print("  • Chronological narratives provide human-readable summaries")
    print("  • Timelines organize events in temporal context")
    print("  • Event sequences reveal patterns in related activities")
    print("  • Memory scheduling supports future reminders with recurrence")
    print("  • Temporal summaries provide statistical insights")
    print("\nFor more information, visit: https://github.com/rexdivakar/HippocampAI")


if __name__ == "__main__":
    main()
