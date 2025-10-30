"""CrossEncoder reranker with caching."""

import hashlib
import logging

from sentence_transformers import CrossEncoder

from hippocampai.utils.cache import get_cache

logger = logging.getLogger(__name__)


class Reranker:
    def __init__(
        self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", cache_ttl: int = 86400
    ):
        self.model = CrossEncoder(model_name)
        self.cache = get_cache(ttl=cache_ttl)
        logger.info(f"Loaded reranker: {model_name}")

    def rerank(
        self, query: str, candidates: list[tuple[str, str, float]], top_k: int = 20
    ) -> list[tuple[str, str, float, float]]:
        """
        Rerank candidates using CrossEncoder.

        Args:
            query: Search query
            candidates: [(id, text, orig_score), ...]
            top_k: Number of top results to return

        Returns:
            [(id, text, orig_score, rerank_score), ...]
        """
        if not candidates:
            return []

        query_hash = hashlib.md5(query.encode()).hexdigest()
        results = []

        # Try cache first
        cached_scores = {}
        uncached_pairs = []
        uncached_indices = []

        for idx, (doc_id, text, orig_score) in enumerate(candidates):
            cache_key = f"{query_hash}:{doc_id}"
            cached = self.cache.get(cache_key)
            if cached is not None:
                cached_scores[idx] = cached
            else:
                uncached_pairs.append([query, text])
                uncached_indices.append(idx)

        # Compute uncached scores
        if uncached_pairs:
            scores = self.model.predict(uncached_pairs)
            for i, score in enumerate(scores):
                idx = uncached_indices[i]
                doc_id = candidates[idx][0]
                cache_key = f"{query_hash}:{doc_id}"
                self.cache.set(cache_key, float(score))
                cached_scores[idx] = float(score)

        # Build results
        for idx, (doc_id, text, orig_score) in enumerate(candidates):
            rerank_score = cached_scores[idx]
            results.append((doc_id, text, orig_score, rerank_score))

        # Sort by rerank score
        results.sort(key=lambda x: x[3], reverse=True)
        return results[:top_k]

    def rerank_single(self, text1: str, text2: str) -> float:
        """
        Compute similarity score between two texts using CrossEncoder.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score (higher is more similar)
        """
        # Create cache key from both texts
        cache_key = hashlib.md5(f"{text1}:{text2}".encode()).hexdigest()
        cached = self.cache.get(cache_key)
        if cached is not None:
            return float(cached)

        # Compute score
        score = float(self.model.predict([[text1, text2]])[0])
        self.cache.set(cache_key, score)
        return score
