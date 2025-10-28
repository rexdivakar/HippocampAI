"""Reciprocal Rank Fusion."""

from typing import Dict, List, Set, Tuple


def reciprocal_rank_fusion(
    rankings: list[list[tuple[str, float]]], k: int = 60, c: int = 60
) -> list[tuple[str, float]]:
    """
    Fuse multiple ranked lists using RRF.

    Args:
        rankings: List of ranked results [(id, score), ...]
        k: Number of top results to consider from each ranking
        c: RRF constant (default 60)

    Returns:
        Fused ranking [(id, rrf_score), ...]
    """
    rrf_scores: dict[str, float] = {}
    all_ids: set[str] = set()

    for ranking in rankings:
        for rank, (doc_id, _) in enumerate(ranking[:k], start=1):
            all_ids.add(doc_id)
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (rank + c)

    fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return fused
