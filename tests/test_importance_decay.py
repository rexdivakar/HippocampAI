"""Tests for importance decay and pruning features."""

from datetime import datetime, timedelta, timezone

import pytest

from hippocampai.models.memory import Memory, MemoryType
from hippocampai.pipeline.importance_decay import (
    DecayConfig,
    DecayFunction,
    ImportanceDecayEngine,
    PruningStrategy,
)


@pytest.fixture
def decay_engine():
    """Create decay engine with default config."""
    return ImportanceDecayEngine()


@pytest.fixture
def custom_decay_engine():
    """Create decay engine with custom config."""
    config = DecayConfig(
        decay_function=DecayFunction.LINEAR, min_importance=2.0, access_boost_factor=1.0
    )
    return ImportanceDecayEngine(config)


@pytest.fixture
def sample_memories():
    """Create sample memories with various ages and importance."""
    now = datetime.now(timezone.utc)
    memories = []

    # Recent high-importance memory
    memories.append(
        Memory(
            text="Critical project deadline next week",
            user_id="test_user",
            type=MemoryType.GOAL,
            importance=9.0,
            confidence=0.95,
            access_count=5,
            created_at=now - timedelta(days=2),
        )
    )

    # Medium age, medium importance
    memories.append(
        Memory(
            text="Python best practices for clean code",
            user_id="test_user",
            type=MemoryType.FACT,
            importance=7.0,
            confidence=0.85,
            access_count=3,
            created_at=now - timedelta(days=30),
        )
    )

    # Old, low importance, never accessed
    memories.append(
        Memory(
            text="Random note about weather last month",
            user_id="test_user",
            type=MemoryType.CONTEXT,
            importance=3.0,
            confidence=0.6,
            access_count=0,
            created_at=now - timedelta(days=90),
        )
    )

    # Very old, but frequently accessed
    memories.append(
        Memory(
            text="Core project documentation reference",
            user_id="test_user",
            type=MemoryType.FACT,
            importance=8.0,
            confidence=0.9,
            access_count=20,
            created_at=now - timedelta(days=120),
        )
    )

    return memories


def test_exponential_decay(decay_engine, sample_memories):
    """Test exponential decay function."""
    memory = sample_memories[1]  # 30 days old, fact type (30 day half-life)
    decayed = decay_engine.calculate_decayed_importance(memory)

    # After one half-life, importance should be ~50% of original (with adjustments for confidence/access)
    assert decayed < memory.importance
    assert decayed > memory.importance * 0.3  # Should be > 30% due to access boost


def test_linear_decay(custom_decay_engine, sample_memories):
    """Test linear decay function."""
    memory = sample_memories[1]
    decayed = custom_decay_engine.calculate_decayed_importance(memory)

    assert decayed < memory.importance
    assert decayed >= custom_decay_engine.config.min_importance


def test_decay_respects_min_importance(decay_engine):
    """Test that decay never goes below min_importance."""
    now = datetime.now(timezone.utc)
    old_memory = Memory(
        text="Very old memory",
        user_id="test_user",
        type=MemoryType.FACT,
        importance=2.0,
        confidence=0.5,
        access_count=0,
        created_at=now - timedelta(days=365),  # 1 year old
    )

    decayed = decay_engine.calculate_decayed_importance(old_memory)
    assert decayed >= decay_engine.config.min_importance


def test_access_boost_reduces_decay(decay_engine):
    """Test that frequent access reduces decay."""
    now = datetime.now(timezone.utc)

    # Two identical memories, different access counts
    low_access = Memory(
        text="Memory A",
        user_id="test_user",
        type=MemoryType.FACT,
        importance=8.0,
        confidence=0.9,
        access_count=1,
        created_at=now - timedelta(days=60),
    )

    high_access = Memory(
        text="Memory B",
        user_id="test_user",
        type=MemoryType.FACT,
        importance=8.0,
        confidence=0.9,
        access_count=15,
        created_at=now - timedelta(days=60),
    )

    decayed_low = decay_engine.calculate_decayed_importance(low_access)
    decayed_high = decay_engine.calculate_decayed_importance(high_access)

    # High access should have less decay
    assert decayed_high > decayed_low


def test_calculate_memory_health(decay_engine, sample_memories):
    """Test memory health calculation."""
    memory = sample_memories[0]  # Recent high-importance memory
    health = decay_engine.calculate_memory_health(memory)

    assert health.memory_id == memory.id
    assert 0.0 <= health.health_score <= 10.0
    assert health.recommendation in ["keep", "decay", "archive", "prune"]
    assert health.health_score >= 7.0  # Recent + high importance = healthy


def test_health_score_components(decay_engine, sample_memories):
    """Test that health score includes all components."""
    memory = sample_memories[1]
    health = decay_engine.calculate_memory_health(memory)

    assert hasattr(health, "importance_score")
    assert hasattr(health, "recency_score")
    assert hasattr(health, "access_score")
    assert hasattr(health, "confidence_score")
    assert "age_days" in health.factors
    assert "access_count" in health.factors


def test_health_recommendation_keep(decay_engine, sample_memories):
    """Test 'keep' recommendation for healthy memories."""
    memory = sample_memories[0]  # Recent, high importance
    health = decay_engine.calculate_memory_health(memory)
    assert health.recommendation == "keep"


def test_health_recommendation_prune(decay_engine):
    """Test 'prune' recommendation for unhealthy memories."""
    now = datetime.now(timezone.utc)
    bad_memory = Memory(
        text="Low value old memory",
        user_id="test_user",
        type=MemoryType.CONTEXT,
        importance=2.0,
        confidence=0.4,
        access_count=0,
        created_at=now - timedelta(days=180),
    )

    health = decay_engine.calculate_memory_health(bad_memory)
    assert health.recommendation in ["prune", "archive"]


def test_identify_pruning_candidates_comprehensive(decay_engine, sample_memories):
    """Test comprehensive pruning strategy."""
    result = decay_engine.identify_pruning_candidates(
        sample_memories, strategy=PruningStrategy.COMPREHENSIVE, min_health_threshold=5.0
    )

    assert "candidates" in result
    assert "stats" in result
    assert "recommendations" in result

    stats = result["stats"]
    assert stats["total_memories"] == len(sample_memories)
    assert stats["strategy"] == PruningStrategy.COMPREHENSIVE.value


def test_identify_pruning_candidates_importance_only(decay_engine, sample_memories):
    """Test importance-only pruning strategy."""
    result = decay_engine.identify_pruning_candidates(
        sample_memories, strategy=PruningStrategy.IMPORTANCE_ONLY
    )

    # Should identify low-importance memories
    candidates = result["candidates"]
    for candidate in candidates:
        assert candidate.factors["decayed_importance"] < decay_engine.config.min_importance * 2


def test_identify_pruning_candidates_age_based(decay_engine, sample_memories):
    """Test age-based pruning strategy."""
    result = decay_engine.identify_pruning_candidates(
        sample_memories, strategy=PruningStrategy.AGE_BASED
    )

    # All candidates should be old
    candidates = result["candidates"]
    for candidate in candidates:
        assert candidate.factors["age_days"] > 90


def test_identify_pruning_candidates_conservative(decay_engine, sample_memories):
    """Test conservative pruning strategy."""
    result = decay_engine.identify_pruning_candidates(
        sample_memories, strategy=PruningStrategy.CONSERVATIVE
    )

    # Conservative should prune very few
    assert result["stats"]["prune_candidates"] <= len(sample_memories) // 2


def test_apply_decay_batch(decay_engine, sample_memories):
    """Test batch decay application."""
    result = decay_engine.apply_decay_batch(sample_memories)

    assert "updates" in result
    assert "stats" in result

    stats = result["stats"]
    assert stats["total_memories"] == len(sample_memories)
    assert stats["updated_count"] >= 0
    assert stats["total_decay"] >= 0.0


def test_apply_decay_preserves_memory_id(decay_engine, sample_memories):
    """Test that decay updates track correct memory IDs."""
    result = decay_engine.apply_decay_batch(sample_memories)

    for memory_id, update in result["updates"].items():
        # Check memory_id exists in original memories
        assert any(m.id == memory_id for m in sample_memories)
        assert "original_importance" in update
        assert "new_importance" in update
        # Note: new_importance CAN be higher due to confidence/access boosts
        assert update["new_importance"] > 0


def test_generate_maintenance_report(decay_engine, sample_memories):
    """Test maintenance report generation."""
    report = decay_engine.generate_maintenance_report(sample_memories)

    assert "summary" in report
    assert "recommendations_by_category" in report
    assert "decay_analysis" in report
    assert "pruning_analysis" in report
    assert "actions" in report
    assert "health_distribution" in report

    summary = report["summary"]
    assert summary["total_memories"] == len(sample_memories)
    assert summary["average_health"] > 0.0


def test_maintenance_report_health_distribution(decay_engine, sample_memories):
    """Test health distribution in maintenance report."""
    report = decay_engine.generate_maintenance_report(sample_memories)

    distribution = report["health_distribution"]
    assert "excellent (8-10)" in distribution
    assert "good (6-8)" in distribution
    assert "fair (4-6)" in distribution
    assert "poor (2-4)" in distribution
    assert "critical (0-2)" in distribution

    # Total should equal number of memories
    total = sum(distribution.values())
    assert total == len(sample_memories)


def test_maintenance_report_actions(decay_engine, sample_memories):
    """Test action recommendations in maintenance report."""
    report = decay_engine.generate_maintenance_report(sample_memories)

    actions = report["actions"]
    assert "immediate" in actions
    assert "recommended" in actions
    assert "monitoring" in actions
    assert isinstance(actions["immediate"], list)


def test_decay_with_empty_memories(decay_engine):
    """Test decay operations with no memories."""
    result = decay_engine.apply_decay_batch([])
    assert result["stats"]["total_memories"] == 0


def test_pruning_with_target_count(decay_engine, sample_memories):
    """Test pruning with specific target count."""
    target = 2  # Keep only 2 memories
    result = decay_engine.identify_pruning_candidates(
        sample_memories, strategy=PruningStrategy.COMPREHENSIVE, target_count=target
    )

    candidates = result["candidates"]
    # Should prune enough to reach target
    assert len(sample_memories) - len(candidates) <= target


def test_hybrid_decay_function():
    """Test hybrid decay function with access patterns."""
    config = DecayConfig(decay_function=DecayFunction.HYBRID)
    engine = ImportanceDecayEngine(config)

    now = datetime.now(timezone.utc)

    # Memory with recent access
    recent_access = Memory(
        text="Recently accessed",
        user_id="test_user",
        type=MemoryType.FACT,
        importance=8.0,
        confidence=0.9,
        access_count=5,
        created_at=now - timedelta(days=60),
        updated_at=now - timedelta(days=1),  # Accessed yesterday
    )

    # Memory with old access
    old_access = Memory(
        text="Old access",
        user_id="test_user",
        type=MemoryType.FACT,
        importance=8.0,
        confidence=0.9,
        access_count=5,
        created_at=now - timedelta(days=60),
        updated_at=now - timedelta(days=50),  # Accessed 50 days ago
    )

    # Recent access should have less decay
    decayed_recent = engine.calculate_decayed_importance(recent_access)
    decayed_old = engine.calculate_decayed_importance(old_access)

    assert decayed_recent > decayed_old


def test_confidence_affects_decay(decay_engine):
    """Test that confidence affects decay calculation."""
    now = datetime.now(timezone.utc)

    # High confidence memory
    high_conf = Memory(
        text="High confidence",
        user_id="test_user",
        type=MemoryType.FACT,
        importance=7.0,
        confidence=0.95,
        access_count=2,
        created_at=now - timedelta(days=30),
    )

    # Low confidence memory
    low_conf = Memory(
        text="Low confidence",
        user_id="test_user",
        type=MemoryType.FACT,
        importance=7.0,
        confidence=0.4,
        access_count=2,
        created_at=now - timedelta(days=30),
    )

    decayed_high = decay_engine.calculate_decayed_importance(high_conf)
    decayed_low = decay_engine.calculate_decayed_importance(low_conf)

    # High confidence should decay less
    assert decayed_high > decayed_low
