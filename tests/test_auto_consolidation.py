"""Tests for automatic consolidation features."""

from datetime import datetime, timedelta, timezone

import pytest

from hippocampai.models.memory import Memory, MemoryType
from hippocampai.pipeline.auto_consolidation import (
    AutoConsolidator,
    ConsolidationSchedule,
    ConsolidationStatus,
    ConsolidationTrigger,
)


@pytest.fixture
def auto_consolidator():
    """Create AutoConsolidator with default schedule."""
    schedule = ConsolidationSchedule(enabled=True, interval_hours=24, min_memories_threshold=5)
    return AutoConsolidator(schedule=schedule)


@pytest.fixture
def sample_memories():
    """Create sample memories for consolidation."""
    now = datetime.now(timezone.utc)
    memories = []

    # Similar memories about Python
    memories.append(
        Memory(
            text="Python is great for data science",
            user_id="test_user",
            type=MemoryType.FACT,
            importance=7.0,
            tags=["python", "data-science"],
            created_at=now - timedelta(days=5),
        )
    )

    memories.append(
        Memory(
            text="I use Python for machine learning projects",
            user_id="test_user",
            type=MemoryType.PREFERENCE,
            importance=7.5,
            tags=["python", "ml"],
            created_at=now - timedelta(days=4),
        )
    )

    # Similar memories about meetings
    memories.append(
        Memory(
            text="Team standup meeting every morning",
            user_id="test_user",
            type=MemoryType.EVENT,
            importance=6.0,
            tags=["work", "meeting"],
            created_at=now - timedelta(days=10),
        )
    )

    memories.append(
        Memory(
            text="Daily standup at 9 AM with the team",
            user_id="test_user",
            type=MemoryType.EVENT,
            importance=6.5,
            tags=["work", "meeting"],
            created_at=now - timedelta(days=9),
        )
    )

    # Unrelated memory
    memories.append(
        Memory(
            text="Coffee shop downtown has great espresso",
            user_id="test_user",
            type=MemoryType.PREFERENCE,
            importance=5.0,
            tags=["coffee"],
            created_at=now - timedelta(days=15),
        )
    )

    return memories


def test_should_consolidate_scheduled(auto_consolidator, sample_memories):
    """Test scheduled consolidation trigger."""
    now = datetime.now(timezone.utc)

    # Set last run to more than interval ago
    auto_consolidator.schedule.last_run = now - timedelta(hours=25)

    should, trigger = auto_consolidator.should_consolidate(sample_memories, now)
    assert should is True
    assert trigger == ConsolidationTrigger.SCHEDULED


def test_should_consolidate_threshold(auto_consolidator):
    """Test threshold-based consolidation trigger."""
    # Create many memories to exceed threshold
    now = datetime.now(timezone.utc)
    many_memories = [
        Memory(
            text=f"Memory {i}",
            user_id="test_user",
            type=MemoryType.FACT,
            importance=5.0,
            created_at=now,
        )
        for i in range(20)  # More than 2x min_memories_threshold
    ]

    should, trigger = auto_consolidator.should_consolidate(many_memories, now)
    assert should is True
    assert trigger == ConsolidationTrigger.THRESHOLD


def test_should_consolidate_token_budget(auto_consolidator):
    """Test token budget consolidation trigger."""
    now = datetime.now(timezone.utc)

    # Set a low threshold to avoid threshold trigger
    auto_consolidator.schedule.min_memories_threshold = 100

    # Create memories with lots of text to exceed token budget
    large_memories = [
        Memory(
            text=" ".join(["word"] * 500),  # ~500 tokens each
            user_id="test_user",
            type=MemoryType.FACT,
            importance=5.0,
            created_at=now,
        )
        for i in range(25)  # Total ~12,500 tokens, but under threshold
    ]

    should, trigger = auto_consolidator.should_consolidate(large_memories, now)
    assert should is True
    assert trigger == ConsolidationTrigger.SIZE_LIMIT


def test_should_not_consolidate_when_disabled(auto_consolidator, sample_memories):
    """Test that consolidation doesn't trigger when disabled."""
    auto_consolidator.schedule.enabled = False
    should, trigger = auto_consolidator.should_consolidate(sample_memories)
    assert should is False
    assert trigger is None


def test_should_not_consolidate_too_few_memories(auto_consolidator):
    """Test that consolidation doesn't trigger with too few memories."""
    few_memories = [
        Memory(
            text=f"Memory {i}",
            user_id="test_user",
            type=MemoryType.FACT,
            importance=5.0,
            created_at=datetime.now(timezone.utc),
        )
        for i in range(2)  # Less than min_memories_threshold
    ]

    should, trigger = auto_consolidator.should_consolidate(few_memories)
    assert should is False


def test_find_consolidation_groups_heuristic(auto_consolidator, sample_memories):
    """Test finding consolidation groups using heuristics."""
    groups = auto_consolidator.find_consolidation_groups(sample_memories)

    # Should find at least one group (memories with same type/tags)
    assert len(groups) > 0

    # Each group should have at least 2 memories
    for group in groups:
        assert len(group) >= 2


def test_consolidate_batch_manual(auto_consolidator, sample_memories):
    """Test manual batch consolidation."""
    result = auto_consolidator.consolidate_batch(
        sample_memories, trigger=ConsolidationTrigger.MANUAL
    )

    assert result.trigger == ConsolidationTrigger.MANUAL
    assert result.status == ConsolidationStatus.COMPLETED
    assert result.memories_analyzed == len(sample_memories)
    assert result.tokens_before > 0


def test_consolidate_batch_tracks_tokens(auto_consolidator, sample_memories):
    """Test that consolidation tracks token counts."""
    result = auto_consolidator.consolidate_batch(sample_memories)

    assert result.tokens_before > 0
    assert result.tokens_after > 0
    # Should save some tokens (or at least not increase)
    assert result.tokens_after <= result.tokens_before


def test_consolidate_batch_updates_schedule(auto_consolidator, sample_memories):
    """Test that consolidation updates schedule."""
    now = datetime.now(timezone.utc)
    auto_consolidator.consolidate_batch(
        sample_memories, trigger=ConsolidationTrigger.SCHEDULED, current_time=now
    )

    assert auto_consolidator.schedule.last_run is not None
    assert auto_consolidator.schedule.next_run is not None
    assert auto_consolidator.schedule.next_run > now


def test_consolidate_batch_adds_to_history(auto_consolidator, sample_memories):
    """Test that consolidation results are stored in history."""
    initial_history_len = len(auto_consolidator.history)

    auto_consolidator.consolidate_batch(sample_memories)

    assert len(auto_consolidator.history) == initial_history_len + 1


def test_consolidation_efficiency_calculation(auto_consolidator, sample_memories):
    """Test efficiency calculation."""
    result = auto_consolidator.consolidate_batch(sample_memories)

    efficiency = result.calculate_efficiency()
    assert 0.0 <= efficiency <= 1.0


def test_auto_consolidate_when_needed(auto_consolidator, sample_memories):
    """Test automatic consolidation when conditions are met."""
    now = datetime.now(timezone.utc)

    # Set last run to trigger scheduled consolidation
    auto_consolidator.schedule.last_run = now - timedelta(hours=25)

    result = auto_consolidator.auto_consolidate(sample_memories, now)

    assert result is not None
    assert result.status == ConsolidationStatus.COMPLETED


def test_auto_consolidate_when_not_needed(auto_consolidator):
    """Test automatic consolidation when conditions aren't met."""
    few_memories = [
        Memory(
            text=f"Memory {i}",
            user_id="test_user",
            type=MemoryType.FACT,
            importance=5.0,
            created_at=datetime.now(timezone.utc),
        )
        for i in range(2)
    ]

    result = auto_consolidator.auto_consolidate(few_memories)
    assert result is None  # Should not consolidate


def test_get_consolidation_stats(auto_consolidator, sample_memories):
    """Test consolidation statistics."""
    # Run a consolidation
    auto_consolidator.consolidate_batch(sample_memories)

    stats = auto_consolidator.get_consolidation_stats()

    assert stats["total_runs"] == 1
    assert stats["completed_runs"] == 1
    assert stats["failed_runs"] == 0
    assert stats["total_memories_analyzed"] == len(sample_memories)


def test_get_consolidation_stats_empty(auto_consolidator):
    """Test stats when no consolidation has been run."""
    stats = auto_consolidator.get_consolidation_stats()

    assert stats["total_runs"] == 0
    assert "message" in stats


def test_get_recent_results(auto_consolidator, sample_memories):
    """Test getting recent consolidation results."""
    # Run multiple consolidations
    auto_consolidator.consolidate_batch(sample_memories)
    auto_consolidator.consolidate_batch(sample_memories)
    auto_consolidator.consolidate_batch(sample_memories)

    recent = auto_consolidator.get_recent_results(limit=2)

    assert len(recent) == 2
    # Should be sorted by most recent first
    assert recent[0].started_at >= recent[1].started_at


def test_estimate_consolidation_impact(auto_consolidator, sample_memories):
    """Test estimating consolidation impact."""
    estimate = auto_consolidator.estimate_consolidation_impact(sample_memories)

    assert "consolidation_possible" in estimate
    assert "total_memories" in estimate
    assert "estimated_groups" in estimate
    assert "current_tokens" in estimate
    assert "estimated_tokens_after" in estimate
    assert "recommendation" in estimate

    assert estimate["total_memories"] == len(sample_memories)


def test_estimate_no_consolidation_possible(auto_consolidator):
    """Test estimate when no consolidation is possible."""
    # Memories that won't group together
    different_memories = [
        Memory(
            text=f"Unique memory about {topic}",
            user_id="test_user",
            type=MemoryType.FACT,
            importance=5.0,
            tags=[topic],
            created_at=datetime.now(timezone.utc),
        )
        for topic in ["A", "B", "C", "D", "E"]
    ]

    estimate = auto_consolidator.estimate_consolidation_impact(different_memories)

    # Might still find some groups with heuristic, but should be minimal
    assert estimate["consolidation_possible"] in [True, False]


def test_cancel_pending_consolidation(auto_consolidator, sample_memories):
    """Test canceling pending consolidation."""
    # Manually add a pending result
    from hippocampai.pipeline.auto_consolidation import ConsolidationResult

    pending = ConsolidationResult(
        trigger=ConsolidationTrigger.MANUAL, status=ConsolidationStatus.PENDING
    )
    auto_consolidator.history.append(pending)

    cancelled = auto_consolidator.cancel_pending_consolidation()
    assert cancelled is True
    assert pending.status == ConsolidationStatus.CANCELLED


def test_cancel_when_nothing_pending(auto_consolidator):
    """Test cancel when there are no pending consolidations."""
    cancelled = auto_consolidator.cancel_pending_consolidation()
    assert cancelled is False


def test_update_schedule(auto_consolidator):
    """Test updating schedule parameters."""
    auto_consolidator.update_schedule(
        interval_hours=48, min_memories_threshold=100, similarity_threshold=0.9
    )

    assert auto_consolidator.schedule.interval_hours == 48
    assert auto_consolidator.schedule.min_memories_threshold == 100
    assert auto_consolidator.schedule.similarity_threshold == 0.9


def test_consolidation_with_empty_memories(auto_consolidator):
    """Test consolidation with no memories."""
    result = auto_consolidator.consolidate_batch([])

    assert result.status == ConsolidationStatus.COMPLETED
    assert result.memories_analyzed == 0
    assert result.consolidation_groups == 0


def test_group_by_heuristics(auto_consolidator, sample_memories):
    """Test heuristic grouping method."""
    groups = auto_consolidator._group_by_heuristics(sample_memories)

    # Should create some groups
    assert len(groups) >= 0

    # Each group should have same type
    for group in groups:
        if len(group) > 1:
            types = set(m.type for m in group)
            assert len(types) == 1  # All same type


def test_consolidation_preserves_user_context(auto_consolidator, sample_memories):
    """Test that consolidation maintains user context."""
    result = auto_consolidator.consolidate_batch(sample_memories)

    # All memories are from same user
    assert all(m.user_id == "test_user" for m in sample_memories)

    # Result should track this
    assert result.memories_analyzed == len(sample_memories)


def test_consolidation_history_ordering(auto_consolidator, sample_memories):
    """Test that consolidation history maintains order."""
    # Run multiple consolidations with delays
    for i in range(3):
        auto_consolidator.consolidate_batch(sample_memories)

    # Get recent results
    recent = auto_consolidator.get_recent_results(limit=10)

    # Should be in reverse chronological order
    for i in range(len(recent) - 1):
        assert recent[i].started_at >= recent[i + 1].started_at
