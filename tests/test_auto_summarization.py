"""Tests for auto-summarization features."""

from datetime import datetime, timedelta, timezone

import pytest

from hippocampai.models.memory import Memory, MemoryType
from hippocampai.pipeline.auto_summarization import (
    AutoSummarizer,
    CompressionLevel,
    MemoryTier,
)


@pytest.fixture
def auto_summarizer():
    """Create AutoSummarizer instance."""
    return AutoSummarizer(llm=None)  # Test without LLM for simplicity


@pytest.fixture
def sample_memories():
    """Create sample memories with various ages."""
    now = datetime.now(timezone.utc)
    memories = []

    # Recent HOT memory
    memories.append(
        Memory(
            text="I love Python programming and use it daily for ML projects",
            user_id="test_user",
            type=MemoryType.PREFERENCE,
            importance=8.0,
            access_count=15,
            created_at=now - timedelta(days=3),
        )
    )

    # WARM memory
    memories.append(
        Memory(
            text="Attended team meeting about Q4 planning and roadmap",
            user_id="test_user",
            type=MemoryType.EVENT,
            importance=6.0,
            access_count=3,
            created_at=now - timedelta(days=20),
        )
    )

    # COLD memory
    memories.append(
        Memory(
            text="Learning about distributed systems and microservices architecture",
            user_id="test_user",
            type=MemoryType.FACT,
            importance=7.0,
            access_count=1,
            created_at=now - timedelta(days=60),
        )
    )

    # ARCHIVED memory
    memories.append(
        Memory(
            text="Visited coffee shop downtown last summer for project discussions",
            user_id="test_user",
            type=MemoryType.EVENT,
            importance=5.0,
            access_count=0,
            created_at=now - timedelta(days=120),
        )
    )

    return memories


def test_determine_memory_tier_hot(auto_summarizer, sample_memories):
    """Test HOT tier classification."""
    hot_memory = sample_memories[0]
    tier = auto_summarizer.determine_memory_tier(hot_memory)
    assert tier == MemoryTier.HOT


def test_determine_memory_tier_warm(auto_summarizer, sample_memories):
    """Test WARM tier classification."""
    warm_memory = sample_memories[1]
    tier = auto_summarizer.determine_memory_tier(warm_memory)
    assert tier == MemoryTier.WARM


def test_determine_memory_tier_cold(auto_summarizer, sample_memories):
    """Test COLD tier classification."""
    cold_memory = sample_memories[2]
    tier = auto_summarizer.determine_memory_tier(cold_memory)
    assert tier == MemoryTier.COLD


def test_determine_memory_tier_archived(auto_summarizer, sample_memories):
    """Test ARCHIVED tier classification."""
    archived_memory = sample_memories[3]
    tier = auto_summarizer.determine_memory_tier(archived_memory)
    assert tier == MemoryTier.ARCHIVED


def test_compression_level_for_tier(auto_summarizer):
    """Test compression level mapping."""
    assert auto_summarizer.get_compression_level_for_tier(MemoryTier.HOT) == CompressionLevel.NONE
    assert auto_summarizer.get_compression_level_for_tier(MemoryTier.WARM) == CompressionLevel.LIGHT
    assert (
        auto_summarizer.get_compression_level_for_tier(MemoryTier.COLD) == CompressionLevel.MEDIUM
    )
    assert (
        auto_summarizer.get_compression_level_for_tier(MemoryTier.ARCHIVED)
        == CompressionLevel.HEAVY
    )


def test_compress_memory_light(auto_summarizer, sample_memories):
    """Test light compression."""
    memory = sample_memories[0]
    compressed = auto_summarizer.compress_memory(memory, CompressionLevel.LIGHT)

    assert compressed is not None
    assert compressed.compression_level == CompressionLevel.LIGHT
    assert compressed.compressed_token_count < compressed.original_token_count
    assert compressed.compression_ratio > 0.5  # Should preserve most content


def test_compress_memory_heavy(auto_summarizer, sample_memories):
    """Test heavy compression."""
    memory = sample_memories[0]
    compressed = auto_summarizer.compress_memory(memory, CompressionLevel.HEAVY)

    assert compressed is not None
    assert compressed.compression_level == CompressionLevel.HEAVY
    assert compressed.compressed_token_count < compressed.original_token_count * 0.5


def test_compress_memory_none(auto_summarizer, sample_memories):
    """Test that NONE compression returns None."""
    memory = sample_memories[0]
    compressed = auto_summarizer.compress_memory(memory, CompressionLevel.NONE)
    assert compressed is None


def test_hierarchical_summary_single_level(auto_summarizer, sample_memories):
    """Test hierarchical summarization with single level."""
    summaries = auto_summarizer.create_hierarchical_summary(
        sample_memories, user_id="test_user", max_levels=1
    )

    # Should have level 0 (leaves) and level 1
    assert len(summaries) > len(sample_memories)

    # Check level 0 summaries exist
    level_0 = [s for s in summaries if s.level == 0]
    assert len(level_0) == len(sample_memories)

    # Check level 1 summaries exist
    level_1 = [s for s in summaries if s.level == 1]
    assert len(level_1) > 0


def test_hierarchical_summary_multiple_levels(auto_summarizer, sample_memories):
    """Test hierarchical summarization with multiple levels."""
    # Create more memories for better hierarchy
    memories = sample_memories * 3  # 12 memories

    summaries = auto_summarizer.create_hierarchical_summary(
        memories, user_id="test_user", max_levels=2
    )

    # Should have multiple levels
    levels = set(s.level for s in summaries)
    assert len(levels) >= 2


def test_sliding_window_compression(auto_summarizer, sample_memories):
    """Test sliding window compression."""
    result = auto_summarizer.sliding_window_compression(
        sample_memories, window_size=2, keep_recent=2, user_id="test_user"
    )

    assert "compressed_memories" in result
    assert "recent_memories" in result
    assert "stats" in result

    stats = result["stats"]
    assert stats["total_memories"] == len(sample_memories)
    assert stats["recent_uncompressed"] == 2
    assert stats["compression_ratio"] < 1.0  # Should save tokens
    assert stats["tokens_saved"] > 0


def test_sliding_window_keeps_recent(auto_summarizer, sample_memories):
    """Test that sliding window keeps recent memories uncompressed."""
    result = auto_summarizer.sliding_window_compression(
        sample_memories, window_size=2, keep_recent=2, user_id="test_user"
    )

    recent = result["recent_memories"]
    assert len(recent) == 2

    # Recent memories should be the newest ones
    assert all(isinstance(m, Memory) for m in recent)


def test_analyze_compression_opportunities(auto_summarizer, sample_memories):
    """Test compression opportunity analysis."""
    analysis = auto_summarizer.analyze_compression_opportunities(sample_memories)

    assert "total_memories" in analysis
    assert "opportunities" in analysis
    assert "potential_tokens_saved" in analysis
    assert "compression_ratio" in analysis

    assert analysis["total_memories"] == len(sample_memories)
    assert len(analysis["opportunities"]) > 0


def test_analyze_compression_empty_memories(auto_summarizer):
    """Test compression analysis with no memories."""
    analysis = auto_summarizer.analyze_compression_opportunities([])
    assert analysis["total_memories"] == 0


def test_heuristic_compression(auto_summarizer):
    """Test heuristic compression fallback."""
    text = "The quick brown fox jumps over the lazy dog in the park"
    compressed = auto_summarizer._compress_heuristic(text, target_ratio=0.5)

    assert len(compressed) < len(text)
    # Should remove filler words
    assert "the" not in compressed.lower() or compressed.count("the") < text.count("the")


def test_compression_preserves_user_id(auto_summarizer, sample_memories):
    """Test that compression preserves user_id."""
    memory = sample_memories[0]
    compressed = auto_summarizer.compress_memory(memory, CompressionLevel.MEDIUM)

    assert compressed is not None
    assert compressed.user_id == memory.user_id


def test_compression_ratio_calculation(auto_summarizer, sample_memories):
    """Test compression ratio calculation."""
    memory = sample_memories[0]
    compressed = auto_summarizer.compress_memory(memory, CompressionLevel.MEDIUM)

    assert compressed is not None
    assert 0.0 < compressed.compression_ratio < 1.0
    assert (
        compressed.compression_ratio
        == compressed.compressed_token_count / compressed.original_token_count
    )


def test_sliding_window_with_empty_memories(auto_summarizer):
    """Test sliding window with no memories."""
    result = auto_summarizer.sliding_window_compression([], window_size=10)
    assert result["stats"]["total_memories"] == 0
    assert len(result["compressed_memories"]) == 0


def test_hierarchical_summary_with_empty_memories(auto_summarizer):
    """Test hierarchical summary with no memories."""
    summaries = auto_summarizer.create_hierarchical_summary([], user_id="test_user")
    assert len(summaries) == 0


def test_memory_tier_updates_with_access(auto_summarizer):
    """Test that tier changes based on access patterns."""
    now = datetime.now(timezone.utc)

    # Old memory with no access - should be ARCHIVED
    old_memory = Memory(
        text="Old memory",
        user_id="test_user",
        type=MemoryType.FACT,
        importance=5.0,
        access_count=0,
        created_at=now - timedelta(days=100),
    )
    assert auto_summarizer.determine_memory_tier(old_memory) == MemoryTier.ARCHIVED

    # Same age but with high access - should be HOT
    accessed_memory = Memory(
        text="Accessed memory",
        user_id="test_user",
        type=MemoryType.FACT,
        importance=5.0,
        access_count=15,
        created_at=now - timedelta(days=100),
    )
    assert auto_summarizer.determine_memory_tier(accessed_memory) == MemoryTier.HOT
