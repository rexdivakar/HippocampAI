"""Retrieval-quality metrics.

All functions take an ordered list of retrieved ids (best first) and a set of
relevant/expected ids. Binary relevance is assumed, which matches how LOCOMO /
LongMemEval supply evidence.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from statistics import mean


def recall_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Fraction of relevant ids that appear in the top-k retrieved."""
    rel = set(relevant)
    if not rel:
        return 0.0
    top = set(retrieved[:k])
    return len(top & rel) / len(rel)


def precision_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Fraction of the top-k retrieved that are relevant."""
    if k <= 0:
        return 0.0
    rel = set(relevant)
    top = retrieved[:k]
    if not top:
        return 0.0
    hits = sum(1 for r in top if r in rel)
    return hits / len(top)


def reciprocal_rank(retrieved: Sequence[str], relevant: Iterable[str]) -> float:
    """1 / rank of the first relevant id (0 if none retrieved)."""
    rel = set(relevant)
    for idx, r in enumerate(retrieved, start=1):
        if r in rel:
            return 1.0 / idx
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Normalized discounted cumulative gain at k (binary relevance)."""
    rel = set(relevant)
    if not rel:
        return 0.0
    dcg = 0.0
    for idx, r in enumerate(retrieved[:k], start=1):
        if r in rel:
            dcg += 1.0 / math.log2(idx + 1)
    ideal_hits = min(len(rel), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def aggregate(values: list[float]) -> float:
    """Mean of a list, 0.0 when empty."""
    return mean(values) if values else 0.0
