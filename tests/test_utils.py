"""Tests for utility functions."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest  # noqa: F401

from hippocampai.utils.cache import get_cache
from hippocampai.utils.scoring import fuse_scores, normalize, recency_score
from hippocampai.utils.time import (
    ensure_utc,
    isoformat_utc,
    now_utc,
    parse_iso_datetime,
    timestamp_to_datetime,
)


class TestTimeUtils:
    """Test time utility functions."""

    def test_now_utc_returns_aware_datetime(self):
        """Test now_utc returns timezone-aware datetime."""
        now = now_utc()
        assert isinstance(now, datetime)
        assert now.tzinfo is not None
        assert now.tzinfo == UTC

    def test_ensure_utc_with_naive_datetime(self):
        """Test ensure_utc adds UTC timezone to naive datetime."""
        naive = datetime(2024, 1, 1, 12, 0, 0)
        aware = ensure_utc(naive)

        assert aware.tzinfo == UTC
        assert aware.year == 2024
        assert aware.month == 1
        assert aware.day == 1

    def test_ensure_utc_with_aware_datetime(self):
        """Test ensure_utc converts aware datetime to UTC."""
        # Create a datetime in a different timezone (simulated with offset)
        from datetime import timezone

        tz = timezone(timedelta(hours=5))
        aware = datetime(2024, 1, 1, 12, 0, 0, tzinfo=tz)
        utc = ensure_utc(aware)

        assert utc.tzinfo == UTC
        # Should be converted to UTC (12:00 +5 = 07:00 UTC)
        assert utc.hour == 7

    def test_timestamp_to_datetime(self):
        """Test converting Unix timestamp to datetime."""
        timestamp = 1704110400.0  # 2024-01-01 12:00:00 UTC
        dt = timestamp_to_datetime(timestamp)

        assert isinstance(dt, datetime)
        assert dt.tzinfo == UTC
        assert dt.year == 2024

    def test_parse_iso_datetime_with_z_suffix(self):
        """Test parsing ISO datetime with Z suffix."""
        iso_str = "2024-01-01T12:00:00Z"
        dt = parse_iso_datetime(iso_str)

        assert isinstance(dt, datetime)
        assert dt.tzinfo == UTC
        assert dt.year == 2024
        assert dt.month == 1
        assert dt.day == 1
        assert dt.hour == 12

    def test_parse_iso_datetime_with_offset(self):
        """Test parsing ISO datetime with timezone offset."""
        iso_str = "2024-01-01T12:00:00+00:00"
        dt = parse_iso_datetime(iso_str)

        assert dt.tzinfo == UTC
        assert dt.hour == 12

    def test_isoformat_utc_with_datetime(self):
        """Test converting datetime to ISO format."""
        dt = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
        iso_str = isoformat_utc(dt)

        assert isinstance(iso_str, str)
        assert "2024-01-01" in iso_str
        assert "12:00:00" in iso_str
        assert "+00:00" in iso_str

    def test_isoformat_utc_without_datetime(self):
        """Test isoformat_utc with no argument uses current time."""
        iso_str = isoformat_utc()

        assert isinstance(iso_str, str)
        assert "+00:00" in iso_str


class TestScoringUtils:
    """Test scoring utility functions."""

    def test_normalize_basic(self):
        """Test basic normalization."""
        assert normalize(5, 0, 10) == 0.5
        assert normalize(0, 0, 10) == 0.0
        assert normalize(10, 0, 10) == 1.0

    def test_normalize_different_ranges(self):
        """Test normalization with different ranges."""
        assert normalize(50, 0, 100) == 0.5
        assert normalize(75, 50, 100) == 0.5
        assert abs(normalize(2.5, 0, 5) - 0.5) < 0.01

    def test_normalize_equal_min_max(self):
        """Test normalization when min equals max."""
        result = normalize(5, 5, 5)
        assert result == 0.5  # Returns 0.5 when range is zero

    def test_normalize_negative_range(self):
        """Test normalization with negative values."""
        assert normalize(0, -10, 10) == 0.5
        assert normalize(-10, -10, 10) == 0.0
        assert normalize(10, -10, 10) == 1.0

    def test_recency_score_recent(self):
        """Test recency score for very recent memories."""
        now = now_utc()
        score = recency_score(now, half_life_days=30)

        # Very recent should be close to 1.0
        assert 0.95 < score <= 1.0

    def test_recency_score_at_half_life(self):
        """Test recency score at exactly half-life."""
        now = now_utc()
        half_life_ago = now - timedelta(days=30)
        score = recency_score(half_life_ago, half_life_days=30)

        # At half-life, score should be ~0.5
        assert 0.45 < score < 0.55

    def test_recency_score_old(self):
        """Test recency score for old memories."""
        now = now_utc()
        very_old = now - timedelta(days=365)
        score = recency_score(very_old, half_life_days=30)

        # Very old should be close to 0
        assert 0.0 <= score < 0.01

    def test_recency_score_different_half_lives(self):
        """Test recency score with different half-life values."""
        now = now_utc()
        old = now - timedelta(days=60)

        score_fast = recency_score(old, half_life_days=30)
        score_slow = recency_score(old, half_life_days=60)

        # Longer half-life should give higher score for same age
        assert score_slow > score_fast

    def test_fuse_scores_balanced(self):
        """Test score fusion with balanced weights."""
        weights = {"sim": 0.25, "rerank": 0.25, "recency": 0.25, "importance": 0.25}
        score = fuse_scores(sim=1.0, rerank=1.0, recency=1.0, importance=1.0, weights=weights)

        assert score == 1.0

    def test_fuse_scores_realistic(self):
        """Test score fusion with realistic values."""
        weights = {"sim": 0.55, "rerank": 0.20, "recency": 0.15, "importance": 0.10}
        score = fuse_scores(sim=0.9, rerank=0.8, recency=0.7, importance=0.6, weights=weights)

        # Manual calculation: 0.55*0.9 + 0.20*0.8 + 0.15*0.7 + 0.10*0.6
        # = 0.495 + 0.16 + 0.105 + 0.06 = 0.82
        assert abs(score - 0.82) < 0.01

    def test_fuse_scores_sim_dominant(self):
        """Test score fusion when similarity is dominant."""
        weights = {"sim": 0.80, "rerank": 0.10, "recency": 0.05, "importance": 0.05}
        score = fuse_scores(sim=0.9, rerank=0.3, recency=0.3, importance=0.3, weights=weights)

        # Should be heavily influenced by sim score
        assert score > 0.75

    def test_fuse_scores_weights_sum(self):
        """Test that weights should sum to 1.0 for expected behavior."""
        weights = {"sim": 0.55, "rerank": 0.20, "recency": 0.15, "importance": 0.10}
        total_weight = sum(weights.values())

        assert abs(total_weight - 1.0) < 0.01


class TestCacheUtils:
    """Test cache utility functions."""

    def test_get_cache_returns_cache(self):
        """Test get_cache returns a cache object."""
        from hippocampai.utils.cache import Cache

        cache = get_cache(maxsize=100)
        assert cache is not None
        assert isinstance(cache, Cache)
        assert hasattr(cache, "get")
        assert hasattr(cache, "set")
        assert hasattr(cache, "clear")

    def test_cache_basic_operations(self):
        """Test basic cache get/set operations."""
        from hippocampai.utils.cache import Cache

        cache = Cache(maxsize=10)

        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

        cache.set("key2", "value2")
        assert cache.get("key2") == "value2"

    def test_cache_size_limit(self):
        """Test cache respects size limit."""
        from hippocampai.utils.cache import Cache

        cache = Cache(maxsize=3)

        cache.set("k1", "v1")
        cache.set("k2", "v2")
        cache.set("k3", "v3")

        # Cache should have 3 items
        assert len(cache.cache) == 3

        # Adding 4th item with TTL cache
        cache.set("k4", "v4")

        # Should still have max 3 items (oldest evicted)
        assert len(cache.cache) <= 3

    def test_invalidate_cache(self):
        """Test cache invalidation."""
        from hippocampai.utils.cache import Cache

        cache = Cache(maxsize=10)
        cache.set("test", "value")

        assert cache.get("test") == "value"
        cache.clear()

        # After invalidation, cache should be empty
        assert len(cache.cache) == 0

    def test_cache_independence(self):
        """Test that different cache instances are independent."""
        from hippocampai.utils.cache import Cache

        cache1 = Cache(maxsize=10)
        cache2 = Cache(maxsize=10)

        cache1.set("key", "value1")
        cache2.set("key", "value2")

        # Each cache should have its own value
        assert cache1.get("key") == "value1"
        assert cache2.get("key") == "value2"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_normalize_with_floats(self):
        """Test normalize with float precision."""
        result = normalize(0.333333, 0, 1)
        assert 0.333 < result < 0.334

    def test_recency_score_with_future_date(self):
        """Test recency score with future date (should handle gracefully)."""
        future = now_utc() + timedelta(days=30)
        score = recency_score(future, half_life_days=30)

        # Future date should give score > 1.0
        assert score > 1.0

    def test_fuse_scores_with_zeros(self):
        """Test score fusion with all zeros."""
        weights = {"sim": 0.25, "rerank": 0.25, "recency": 0.25, "importance": 0.25}
        score = fuse_scores(sim=0, rerank=0, recency=0, importance=0, weights=weights)

        assert score == 0.0

    def test_fuse_scores_with_mixed_values(self):
        """Test score fusion with very different values."""
        weights = {"sim": 0.25, "rerank": 0.25, "recency": 0.25, "importance": 0.25}
        score = fuse_scores(sim=1.0, rerank=0.0, recency=0.5, importance=0.8, weights=weights)

        # Should be average: (1.0 + 0.0 + 0.5 + 0.8) / 4 = 0.575
        assert abs(score - 0.575) < 0.01
