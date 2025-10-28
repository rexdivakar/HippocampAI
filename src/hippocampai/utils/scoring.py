"""Scoring utilities for memory fusion."""

import math
from datetime import datetime
from typing import Dict

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
    sim: float, rerank: float, recency: float, importance: float, weights: dict[str, float]
) -> float:
    """
    Weighted fusion of scores.

    Args:
        sim: Similarity score [0,1]
        rerank: Rerank score [0,1]
        recency: Recency score [0,1]
        importance: Importance score [0,1]
        weights: {"sim": 0.55, "rerank": 0.20, ...}

    Returns:
        Fused score [0,1]
    """
    return (
        weights["sim"] * sim
        + weights["rerank"] * rerank
        + weights["recency"] * recency
        + weights["importance"] * importance
    )
