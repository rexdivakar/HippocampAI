"""Scoring utilities for memory fusion."""

import math
from datetime import datetime

from hippocampai.utils.time import ensure_utc, now_utc


def normalize(value: float, min_val: float, max_val: float) -> float:
    """Normalize to [0,1]."""
    if max_val == min_val:
        return 0.5
    return (value - min_val) / (max_val - min_val)


def recency_score(created_at: datetime, half_life_days: int) -> float:
    """Exponential decay based on half-life."""
    created_utc = ensure_utc(created_at)
    age_days = (now_utc() - created_utc).total_seconds() / 86400
    return math.exp(-0.693 * age_days / half_life_days)


def fuse_scores(
    sim: float,
    rerank: float,
    recency: float,
    importance: float,
    weights: dict[str, float],
    graph: float = 0.0,
    feedback: float = 0.0,
) -> float:
    """
    Weighted fusion of scores with auto-normalization.

    Args:
        sim: Similarity score [0,1]
        rerank: Rerank score [0,1]
        recency: Recency score [0,1]
        importance: Importance score [0,1]
        weights: {"sim": 0.55, "rerank": 0.20, "graph": 0.0, "feedback": 0.1, ...}
        graph: Graph retrieval score [0,1]
        feedback: Feedback score [0,1]

    Returns:
        Fused score [0,1]
    """
    raw_total = (
        weights.get("sim", 0) * sim
        + weights.get("rerank", 0) * rerank
        + weights.get("recency", 0) * recency
        + weights.get("importance", 0) * importance
        + weights.get("graph", 0) * graph
        + weights.get("feedback", 0) * feedback
    )
    weight_sum = sum(
        weights.get(k, 0)
        for k in ("sim", "rerank", "recency", "importance", "graph", "feedback")
    )
    if weight_sum <= 0:
        return 0.0
    return raw_total / weight_sum
