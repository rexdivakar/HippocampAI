"""Tests for retrieval components."""

from datetime import timezone, datetime, timedelta

from hippocampai.retrieval.rrf import reciprocal_rank_fusion
from hippocampai.utils.scoring import fuse_scores, normalize, recency_score


def test_rrf_fusion():
    """Test RRF fusion."""
    rank1 = [("doc1", 0.9), ("doc2", 0.8), ("doc3", 0.7)]
    rank2 = [("doc2", 0.95), ("doc1", 0.85), ("doc4", 0.75)]

    fused = reciprocal_rank_fusion([rank1, rank2], k=60, c=60)

    assert len(fused) == 4
    assert fused[0][0] in ["doc1", "doc2"]  # Top docs should be 1 or 2


def test_normalize():
    """Test normalization."""
    assert normalize(5, 0, 10) == 0.5
    assert normalize(0, 0, 10) == 0.0
    assert normalize(10, 0, 10) == 1.0


def test_recency_score():
    """Test recency decay."""
    now = datetime.now(UTC)
    old = now - timedelta(days=30)

    score_now = recency_score(now, half_life_days=30)
    score_old = recency_score(old, half_life_days=30)

    assert score_now > score_old
    assert 0.4 < score_old < 0.6  # ~0.5 at half-life


def test_fuse_scores():
    """Test score fusion."""
    weights = {"sim": 0.55, "rerank": 0.20, "recency": 0.15, "importance": 0.10}
    score = fuse_scores(sim=0.9, rerank=0.8, recency=0.7, importance=0.6, weights=weights)

    assert 0.7 < score < 0.9
    assert abs(score - 0.82) < 0.05
