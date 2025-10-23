"""
Cross-Session Insights Demo

This example demonstrates HippocampAI's cross-session insight capabilities:
- Pattern detection across sessions
- Behavioral change tracking
- Preference drift analysis
- Habit formation detection
- Long-term trend analysis

Features demonstrated:
1. Detecting recurring, sequential, and correlational patterns
2. Tracking behavioral changes between time periods
3. Analyzing how preferences evolve over time
4. Identifying habit formation and breaking
5. Analyzing long-term trends in behavior
"""

from hippocampai import MemoryClient
import time


def print_section(title: str):
    """Print a section header."""
    print(f"\n{'=' * 80}")
    print(f"{title:^80}")
    print(f"{'=' * 80}\n")


def demo_pattern_detection(client: MemoryClient, user_id: str):
    """Demonstrate pattern detection across memories and sessions."""
    print_section("1. PATTERN DETECTION")

    print("🔍 Detecting patterns in user behavior...")
    patterns = client.detect_patterns(user_id=user_id)

    print(f"\nFound {len(patterns)} patterns")

    # Show top patterns by confidence
    top_patterns = sorted(patterns, key=lambda p: p.confidence, reverse=True)[:5]

    print("\nTop Patterns by Confidence:")
    for i, pattern in enumerate(top_patterns, 1):
        print(f"\n{i}. Pattern Type: {pattern.pattern_type.upper()}")
        print(f"   Description: {pattern.description}")
        print(f"   Confidence: {pattern.confidence:.2f}")
        print(f"   Occurrences: {pattern.occurrences}")
        print(f"   First seen: {pattern.first_seen.strftime('%Y-%m-%d %H:%M')}")
        print(f"   Last seen: {pattern.last_seen.strftime('%Y-%m-%d %H:%M')}")
        if pattern.frequency:
            print(f"   Frequency: {pattern.frequency}")
        print(f"   Related memories: {len(pattern.memory_ids)}")

    # Group patterns by type
    print("\n\nPatterns by Type:")
    pattern_types = {}
    for p in patterns:
        pattern_types[p.pattern_type] = pattern_types.get(p.pattern_type, 0) + 1

    for ptype, count in pattern_types.items():
        print(f"  • {ptype}: {count} patterns")


def demo_behavior_changes(client: MemoryClient, user_id: str):
    """Demonstrate tracking behavioral changes."""
    print_section("2. BEHAVIORAL CHANGE TRACKING")

    print("📊 Tracking behavior changes (comparing last 30 days vs older)...")
    changes = client.track_behavior_changes(
        user_id=user_id,
        comparison_days=30
    )

    print(f"\nFound {len(changes)} behavioral changes")

    if changes:
        print("\nDetected Changes:")
        for i, change in enumerate(changes[:8], 1):
            print(f"\n{i}. Change Type: {change.change_type.value.upper()}")
            print(f"   Description: {change.description}")
            print(f"   Confidence: {change.confidence:.2f}")

            if change.before_value and change.after_value:
                print(f"   Before: {change.before_value}")
                print(f"   After: {change.after_value}")

            if change.change_magnitude:
                print(f"   Magnitude: {change.change_magnitude:.2f}")

            print(f"   Evidence: {len(change.evidence_memory_ids)} memories")
            print(f"   Detected: {change.detected_at.strftime('%Y-%m-%d %H:%M')}")

        # Summarize by change type
        print("\n\nChanges by Type:")
        change_types = {}
        for c in changes:
            change_types[c.change_type.value] = change_types.get(c.change_type.value, 0) + 1

        for ctype, count in change_types.items():
            print(f"  • {ctype}: {count} changes")
    else:
        print("  No significant behavioral changes detected in this period.")


def demo_preference_drift(client: MemoryClient, user_id: str):
    """Demonstrate analyzing preference drift."""
    print_section("3. PREFERENCE DRIFT ANALYSIS")

    print("🎯 Analyzing how preferences have evolved over time...")
    drifts = client.analyze_preference_drift(user_id=user_id)

    print(f"\nFound {len(drifts)} preference drifts")

    if drifts:
        # Show drifts with highest drift score
        top_drifts = sorted(drifts, key=lambda d: d.drift_score, reverse=True)[:5]

        print("\nTop Preference Drifts:")
        for i, drift in enumerate(top_drifts, 1):
            print(f"\n{i}. Category: {drift.category}")
            print(f"   Original Preference: {drift.original_preference}")
            print(f"   Current Preference: {drift.current_preference}")
            print(f"   Drift Score: {drift.drift_score:.2f} (0=stable, 1=complete change)")
            print(f"   First recorded: {drift.first_recorded.strftime('%Y-%m-%d')}")
            print(f"   Last updated: {drift.last_updated.strftime('%Y-%m-%d')}")
            print(f"   Supporting evidence: {len(drift.memory_ids)} memories")

            if drift.timeline:
                print(f"   Evolution timeline ({len(drift.timeline)} points):")
                for timestamp, value in drift.timeline[:3]:
                    print(f"     • {timestamp.strftime('%Y-%m-%d')}: {value}")
                if len(drift.timeline) > 3:
                    print(f"     ... and {len(drift.timeline) - 3} more timeline points")

        # Analyze drift by category
        print("\n\nDrift by Category:")
        categories = {}
        for d in drifts:
            categories[d.category] = categories.get(d.category, 0) + 1

        for category, count in categories.items():
            print(f"  • {category}: {count} drifts")
    else:
        print("  No significant preference drift detected.")


def demo_habit_detection(client: MemoryClient, user_id: str):
    """Demonstrate detecting habit formation."""
    print_section("4. HABIT FORMATION DETECTION")

    print("🔄 Detecting habits from behavioral patterns...")
    habits = client.detect_habits(
        user_id=user_id,
        min_occurrences=3  # Lower threshold for demo
    )

    print(f"\nFound {len(habits)} potential habits")

    if habits:
        # Habits are already sorted by habit_score in descending order
        print("\nTop Habits by Score:")
        for i, habit in enumerate(habits[:8], 1):
            print(f"\n{i}. Behavior: {habit.behavior}")
            print(f"   Habit Score: {habit.habit_score:.2f} (0=not a habit, 1=strong habit)")
            print(f"   Status: {habit.status.upper()}")
            print(f"   Frequency: {habit.frequency} occurrences")
            print(f"   Consistency: {habit.consistency:.2f}")
            print(f"   Recency: {habit.recency:.2f}")
            print(f"   Duration: {habit.duration} days")
            print(f"   Supporting memories: {len(habit.memory_ids)}")

            # Show a few occurrence timestamps
            if habit.occurrences:
                print(f"   Recent occurrences:")
                for timestamp in habit.occurrences[-3:]:
                    print(f"     • {timestamp.strftime('%Y-%m-%d %H:%M')}")

        # Summarize by status
        print("\n\nHabits by Status:")
        statuses = {}
        for h in habits:
            statuses[h.status] = statuses.get(h.status, 0) + 1

        for status, count in statuses.items():
            print(f"  • {status}: {count} habits")

        # Show habit quality breakdown
        print("\n\nHabit Quality Breakdown:")
        strong = sum(1 for h in habits if h.habit_score >= 0.7)
        moderate = sum(1 for h in habits if 0.4 <= h.habit_score < 0.7)
        weak = sum(1 for h in habits if h.habit_score < 0.4)

        print(f"  • Strong habits (≥0.7): {strong}")
        print(f"  • Moderate habits (0.4-0.7): {moderate}")
        print(f"  • Weak habits (<0.4): {weak}")
    else:
        print("  No habits detected yet. Keep tracking behaviors!")


def demo_trend_analysis(client: MemoryClient, user_id: str):
    """Demonstrate analyzing long-term trends."""
    print_section("5. LONG-TERM TREND ANALYSIS")

    print("📈 Analyzing long-term trends in behavior...")
    trends = client.analyze_trends(
        user_id=user_id,
        window_days=30
    )

    print(f"\nFound {len(trends)} trends")

    if trends:
        print("\nDetected Trends:")
        for i, trend in enumerate(trends[:8], 1):
            print(f"\n{i}. Category: {trend.category}")
            print(f"   Trend Type: {trend.trend_type}")
            print(f"   Direction: {trend.direction.upper()}")
            print(f"   Strength: {trend.strength:.2f}")
            print(f"   Confidence: {trend.confidence:.2f}")
            print(f"   Description: {trend.description}")
            print(f"   Data points: {len(trend.data_points)}")
            print(f"   Detected: {trend.detected_at.strftime('%Y-%m-%d %H:%M')}")

            # Show visual representation
            if trend.direction == "up":
                visual = "📈 ↗️"
            elif trend.direction == "down":
                visual = "📉 ↘️"
            else:
                visual = "📊 ➡️"
            print(f"   Visualization: {visual}")

        # Summarize by direction
        print("\n\nTrends by Direction:")
        directions = {}
        for t in trends:
            directions[t.direction] = directions.get(t.direction, 0) + 1

        for direction, count in directions.items():
            print(f"  • {direction}: {count} trends")

        # Summarize by trend type
        print("\n\nTrends by Type:")
        types = {}
        for t in trends:
            types[t.trend_type] = types.get(t.trend_type, 0) + 1

        for ttype, count in types.items():
            print(f"  • {ttype}: {count} trends")
    else:
        print("  No significant trends detected in this period.")


def create_diverse_memories(client: MemoryClient, user_id: str):
    """Create diverse sample memories for insights demo."""
    print_section("SETUP: Creating Sample Memories")

    print("Creating diverse sample memories to demonstrate insights...")

    # Create memories showing various patterns and behaviors
    memory_sets = [
        # Work patterns
        ("Morning standup meeting at 9 AM", ["work", "meeting", "daily"]),
        ("Code review session", ["work", "development"]),
        ("Lunch break at noon", ["daily", "routine"]),
        ("Afternoon focus time for coding", ["work", "development"]),

        # Exercise patterns (habit forming)
        ("Morning run for 30 minutes", ["exercise", "health"]),
        ("Evening gym session", ["exercise", "health"]),
        ("Morning run again, feeling great", ["exercise", "health"]),
        ("Skipped gym today, too tired", ["exercise", "health"]),
        ("Back to morning runs, building consistency", ["exercise", "health"]),

        # Food preferences (showing drift)
        ("Had sushi for lunch, loved it", ["food", "preference"]),
        ("Tried Indian food, not my favorite", ["food", "preference"]),
        ("Sushi again, my go-to choice", ["food", "preference"]),
        ("Experimenting with vegan options", ["food", "preference"]),
        ("Really enjoying plant-based meals now", ["food", "preference"]),

        # Learning behaviors (showing trend)
        ("Started learning Python", ["learning", "tech"]),
        ("Python tutorial - day 2", ["learning", "tech"]),
        ("Building first Python project", ["learning", "tech"]),
        ("Python is becoming second nature", ["learning", "tech"]),
        ("Advanced Python concepts mastered", ["learning", "tech"]),

        # Social patterns
        ("Coffee with Sarah", ["social", "friends"]),
        ("Team lunch with colleagues", ["social", "work"]),
        ("Weekend brunch with friends", ["social", "friends"]),
        ("Coffee with Sarah again", ["social", "friends"]),

        # Goals and achievements
        ("Set goal to read 12 books this year", ["goals", "personal"]),
        ("Finished first book of the year", ["achievement", "reading"]),
        ("Started meditation practice", ["goals", "health"]),
        ("30 days of consistent meditation!", ["achievement", "health"]),

        # Project work (sequential pattern)
        ("Started project planning phase", ["work", "project"]),
        ("Completed requirements gathering", ["work", "project"]),
        ("Design phase underway", ["work", "project"]),
        ("Development sprint started", ["work", "project"]),
        ("Testing and QA in progress", ["work", "project"]),
    ]

    for text, tags in memory_sets:
        client.remember(text, user_id, tags=tags)
        time.sleep(0.1)

    print(f"✓ Created {len(memory_sets)} diverse sample memories")
    print("  Memory categories:")
    print("    • Work patterns and routines")
    print("    • Exercise habits (forming)")
    print("    • Food preferences (drifting)")
    print("    • Learning behaviors (trending up)")
    print("    • Social interactions")
    print("    • Goals and achievements")


def main():
    """Run all cross-session insights demos."""
    print("\n" + "=" * 80)
    print("HIPPOCAMPAI CROSS-SESSION INSIGHTS DEMO".center(80))
    print("=" * 80)

    # Initialize client
    print("\n🚀 Initializing HippocampAI client...")
    client = MemoryClient(collection_name="insights_demo")
    user_id = "demo_user_insights"

    # Create diverse sample memories
    create_diverse_memories(client, user_id)

    # Run demos
    demo_pattern_detection(client, user_id)
    demo_behavior_changes(client, user_id)
    demo_preference_drift(client, user_id)
    demo_habit_detection(client, user_id)
    demo_trend_analysis(client, user_id)

    print_section("DEMO COMPLETE")
    print("✓ All cross-session insight features demonstrated successfully!")
    print("\nKey Takeaways:")
    print("  • Pattern detection reveals recurring behaviors and correlations")
    print("  • Behavioral change tracking identifies shifts in user actions")
    print("  • Preference drift analysis shows how tastes evolve over time")
    print("  • Habit detection scores behaviors by consistency and frequency")
    print("  • Trend analysis identifies long-term directional changes")
    print("\nThese insights enable:")
    print("  ✓ Personalized recommendations based on behavioral patterns")
    print("  ✓ Proactive interventions when habits break or goals stall")
    print("  ✓ Adaptive experiences that evolve with user preferences")
    print("  ✓ Long-term user understanding beyond individual sessions")
    print("\nFor more information, visit: https://github.com/rexdivakar/HippocampAI")


if __name__ == "__main__":
    main()
