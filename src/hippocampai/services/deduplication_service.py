"""Unified deduplication service for memory management.

This module consolidates duplicate detection logic from multiple implementations
into a single service with consistent thresholds and modes.

Usage:
    from hippocampai.services.deduplication_service import get_deduplication_service

    service = get_deduplication_service()

    # Check a single new memory
    action, duplicates = service.check_duplicate(memory, user_id)

    # Detect all duplicate clusters
    clusters = service.detect_clusters(memories)

    # Find near-duplicate pairs
    pairs = service.detect_near_duplicates(memories)
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, Optional

from hippocampai.models.memory import Memory, MemoryType

logger = logging.getLogger(__name__)


# Default thresholds (standardized across all deduplication methods)
DEFAULT_SIMILARITY_THRESHOLD = 0.88  # Near-duplicate threshold
EXACT_DUPLICATE_THRESHOLD = 0.95  # Exact duplicate threshold
CLUSTER_THRESHOLD = 0.70  # Minimum for cluster membership


class DeduplicationMode(Enum):
    """Deduplication detection modes."""

    EXACT = "exact"  # Exact duplicates only (>= 0.95 similarity)
    NEAR = "near"  # Near duplicates (>= 0.88 similarity)
    CLUSTER = "cluster"  # Soft clustering (>= 0.70 similarity)


@dataclass
class DuplicateCheckResult:
    """Result of checking a single memory for duplicates."""

    action: Literal["store", "skip", "update"]
    duplicate_ids: list[str] = field(default_factory=list)
    highest_similarity: float = 0.0
    reasoning: str = ""


@dataclass
class DuplicateCluster:
    """A cluster of duplicate or similar memories."""

    cluster_id: str
    memory_ids: list[str]
    similarity_score: float
    suggested_action: Literal["merge", "review", "keep_all"]
    canonical_memory_id: Optional[str] = None
    representative_text: Optional[str] = None


@dataclass
class NearDuplicatePair:
    """A pair of near-duplicate memories."""

    memory_id_1: str
    memory_id_2: str
    similarity: float
    text_preview_1: str = ""
    text_preview_2: str = ""
    merge_suggestion: Optional[str] = None


class DeduplicationService:
    """Unified service for memory deduplication.

    This service provides multiple deduplication modes:
    - EXACT: Only exact duplicates (>= 95% similarity)
    - NEAR: Near duplicates (>= 88% similarity)
    - CLUSTER: Soft clustering for finding related memories

    Example:
        service = DeduplicationService()

        # Check if a new memory is duplicate
        result = service.check_duplicate(memory, user_id)
        if result.action == "skip":
            print(f"Memory is duplicate of {result.duplicate_ids[0]}")

        # Find all clusters in user's memories
        clusters = service.detect_clusters(memories)
    """

    def __init__(
        self,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        exact_threshold: float = EXACT_DUPLICATE_THRESHOLD,
        cluster_threshold: float = CLUSTER_THRESHOLD,
        embedder: Optional[Any] = None,
        reranker: Optional[Any] = None,
        qdrant_store: Optional[Any] = None,
    ) -> None:
        """Initialize the deduplication service.

        Args:
            similarity_threshold: Threshold for near-duplicate detection.
            exact_threshold: Threshold for exact duplicate detection.
            cluster_threshold: Threshold for cluster membership.
            embedder: Optional embedder for vector-based similarity.
            reranker: Optional reranker for refined similarity scoring.
            qdrant_store: Optional Qdrant store for vector search.
        """
        self.similarity_threshold = similarity_threshold
        self.exact_threshold = exact_threshold
        self.cluster_threshold = cluster_threshold
        self.embedder = embedder
        self.reranker = reranker
        self.qdrant = qdrant_store

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts.

        Uses embeddings if available, otherwise falls back to simple text similarity.
        """
        if self.embedder is not None:
            try:
                import numpy as np
                vec1 = self.embedder.encode_single(text1)
                vec2 = self.embedder.encode_single(text2)
                # Cosine similarity
                similarity = float(
                    np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                )
                return similarity
            except Exception as e:
                logger.debug(f"Embedding-based similarity failed: {e}")

        # Fallback: simple normalized Jaccard similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if not words1 or not words2:
            return 0.0
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        return intersection / union if union > 0 else 0.0

    def check_duplicate(
        self,
        new_memory: Memory,
        user_id: str,
        mode: DeduplicationMode = DeduplicationMode.NEAR,
    ) -> DuplicateCheckResult:
        """Check if a new memory is a duplicate of existing memories.

        Args:
            new_memory: The new memory to check.
            user_id: User identifier.
            mode: Deduplication mode (EXACT, NEAR, or CLUSTER).

        Returns:
            DuplicateCheckResult with action and duplicate IDs.
        """
        threshold = self._get_threshold_for_mode(mode)

        # If we have a vector store, use vector search
        if self.qdrant is not None and self.embedder is not None:
            return self._check_duplicate_vector(new_memory, user_id, threshold)

        # Otherwise return store action (no comparison possible)
        logger.debug("No vector store available, skipping duplicate check")
        return DuplicateCheckResult(
            action="store",
            reasoning="No vector store available for comparison",
        )

    def _check_duplicate_vector(
        self,
        new_memory: Memory,
        user_id: str,
        threshold: float,
    ) -> DuplicateCheckResult:
        """Check duplicates using vector search."""
        # Type guards for required components
        if self.qdrant is None or self.embedder is None:
            return DuplicateCheckResult(
                action="store",
                reasoning="Vector store or embedder not available",
            )

        collection = new_memory.collection_name(
            self.qdrant.collection_facts, self.qdrant.collection_prefs
        )

        # Vector search for candidates
        vector = self.embedder.encode_single(new_memory.text)
        candidates = self.qdrant.search(
            collection_name=collection,
            vector=vector,
            limit=10,
            filters={"user_id": user_id},
        )

        if not candidates:
            return DuplicateCheckResult(
                action="store",
                reasoning="No similar memories found",
            )

        # Find duplicates above threshold
        duplicates = []
        highest_score = 0.0

        for cand in candidates:
            score = cand["score"]
            if score > highest_score:
                highest_score = score
            if score >= threshold:
                duplicates.append(cand["id"])

        if not duplicates:
            return DuplicateCheckResult(
                action="store",
                highest_similarity=highest_score,
                reasoning=f"Highest similarity {highest_score:.2f} below threshold {threshold}",
            )

        # Exact duplicate - skip
        if highest_score >= self.exact_threshold:
            return DuplicateCheckResult(
                action="skip",
                duplicate_ids=duplicates,
                highest_similarity=highest_score,
                reasoning=f"Exact duplicate found (similarity: {highest_score:.2f})",
            )

        # Near duplicate - update existing
        return DuplicateCheckResult(
            action="update",
            duplicate_ids=duplicates,
            highest_similarity=highest_score,
            reasoning=f"Near duplicate found (similarity: {highest_score:.2f})",
        )

    def detect_clusters(
        self,
        memories: list[Memory],
        mode: DeduplicationMode = DeduplicationMode.NEAR,
        min_cluster_size: int = 2,
        precomputed_similarities: Optional[dict[tuple[str, str], float]] = None,
    ) -> list[DuplicateCluster]:
        """Detect clusters of duplicate or similar memories.

        Args:
            memories: List of memories to analyze.
            mode: Deduplication mode for threshold selection.
            min_cluster_size: Minimum cluster size to report.
            precomputed_similarities: Optional precomputed similarity matrix.

        Returns:
            List of DuplicateCluster objects.
        """
        if len(memories) < min_cluster_size:
            return []

        threshold = self._get_threshold_for_mode(mode)

        # Build similarity matrix if not provided
        similarity_matrix = precomputed_similarities or {}
        if not similarity_matrix:
            similarity_matrix = self._compute_similarity_matrix(memories, threshold)

        # Build adjacency list
        adjacency: dict[str, list[tuple[str, float]]] = {}
        for memory in memories:
            adjacency[memory.id] = []

        for (id1, id2), similarity in similarity_matrix.items():
            if similarity >= threshold:
                adjacency.setdefault(id1, []).append((id2, similarity))
                adjacency.setdefault(id2, []).append((id1, similarity))

        # Find clusters using graph traversal
        clusters: list[DuplicateCluster] = []
        processed: set[str] = set()
        memory_lookup = {m.id: m for m in memories}

        for memory in memories:
            if memory.id in processed:
                continue

            # BFS to find cluster
            cluster_members = {memory.id}
            similarities: list[float] = []
            queue = [memory.id]

            while queue:
                current = queue.pop(0)
                if current in processed:
                    continue
                processed.add(current)

                for neighbor_id, sim in adjacency.get(current, []):
                    if neighbor_id not in processed and neighbor_id not in cluster_members:
                        cluster_members.add(neighbor_id)
                        queue.append(neighbor_id)
                        similarities.append(sim)

            # Create cluster if meets size threshold
            if len(cluster_members) >= min_cluster_size:
                avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0

                # Determine suggested action
                if avg_similarity >= self.exact_threshold:
                    action: Literal["merge", "review", "keep_all"] = "merge"
                elif avg_similarity >= self.similarity_threshold:
                    action = "review"
                else:
                    action = "keep_all"

                # Find canonical (highest importance)
                canonical = max(
                    cluster_members,
                    key=lambda mid: memory_lookup.get(mid, Memory(text="", user_id="", type=MemoryType.CONTEXT)).importance,
                )

                canonical_mem = memory_lookup.get(canonical)
                representative_text = canonical_mem.text[:200] if canonical_mem else None

                clusters.append(
                    DuplicateCluster(
                        cluster_id=f"cluster_{len(clusters) + 1}",
                        memory_ids=list(cluster_members),
                        similarity_score=avg_similarity,
                        suggested_action=action,
                        canonical_memory_id=canonical,
                        representative_text=representative_text,
                    )
                )

        return clusters

    def detect_near_duplicates(
        self,
        memories: list[Memory],
        generate_suggestions: bool = False,
    ) -> list[NearDuplicatePair]:
        """Detect pairs of near-duplicate memories.

        Args:
            memories: List of memories to analyze.
            generate_suggestions: Whether to generate merge suggestions.

        Returns:
            List of NearDuplicatePair objects.
        """
        pairs: list[NearDuplicatePair] = []

        for i, mem1 in enumerate(memories):
            for mem2 in memories[i + 1:]:
                similarity = self._calculate_text_similarity(mem1.text, mem2.text)

                # Near duplicate but not exact
                if self.similarity_threshold <= similarity < self.exact_threshold:
                    suggestion = None
                    if generate_suggestions:
                        suggestion = self._generate_merge_suggestion(mem1, mem2)

                    pairs.append(
                        NearDuplicatePair(
                            memory_id_1=mem1.id,
                            memory_id_2=mem2.id,
                            similarity=similarity,
                            text_preview_1=mem1.text[:200],
                            text_preview_2=mem2.text[:200],
                            merge_suggestion=suggestion,
                        )
                    )

        # Sort by similarity descending
        pairs.sort(key=lambda x: x.similarity, reverse=True)
        return pairs

    def count_exact_duplicates(self, memories: list[Memory]) -> int:
        """Count exact duplicate pairs in a list of memories.

        This is a fast method for dashboard/health statistics.

        Args:
            memories: List of memories to check.

        Returns:
            Count of exact duplicate pairs.
        """
        count = 0
        for i, mem1 in enumerate(memories):
            for mem2 in memories[i + 1:]:
                if mem1.text.lower().strip() == mem2.text.lower().strip():
                    count += 1
        return count

    def _get_threshold_for_mode(self, mode: DeduplicationMode) -> float:
        """Get similarity threshold for a deduplication mode."""
        if mode == DeduplicationMode.EXACT:
            return self.exact_threshold
        elif mode == DeduplicationMode.NEAR:
            return self.similarity_threshold
        else:  # CLUSTER
            return self.cluster_threshold

    def _compute_similarity_matrix(
        self,
        memories: list[Memory],
        threshold: float,
    ) -> dict[tuple[str, str], float]:
        """Compute similarity matrix for memories."""
        similarity_matrix: dict[tuple[str, str], float] = {}

        for i, mem1 in enumerate(memories):
            for mem2 in memories[i + 1:]:
                similarity = self._calculate_text_similarity(mem1.text, mem2.text)
                if similarity >= threshold:
                    similarity_matrix[(mem1.id, mem2.id)] = similarity

        return similarity_matrix

    def _generate_merge_suggestion(self, mem1: Memory, mem2: Memory) -> str:
        """Generate a merge suggestion for two similar memories."""
        # Keep the one with higher importance or newer
        if mem1.importance > mem2.importance:
            keep, merge = mem1, mem2
        elif mem2.importance > mem1.importance:
            keep, merge = mem2, mem1
        elif mem1.updated_at > mem2.updated_at:
            keep, merge = mem1, mem2
        else:
            keep, merge = mem2, mem1

        return f"Keep '{keep.text[:50]}...' (higher importance/newer), merge content from '{merge.text[:50]}...'"


# Global singleton instance
_deduplication_service: Optional[DeduplicationService] = None


def get_deduplication_service(
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    embedder: Optional[Any] = None,
    reranker: Optional[Any] = None,
    qdrant_store: Optional[Any] = None,
) -> DeduplicationService:
    """Get the global DeduplicationService instance.

    Args:
        similarity_threshold: Similarity threshold for near-duplicate detection.
        embedder: Optional embedder for vector similarity.
        reranker: Optional reranker for refined scoring.
        qdrant_store: Optional Qdrant store for vector search.

    Returns:
        The DeduplicationService instance.
    """
    global _deduplication_service
    if _deduplication_service is None:
        _deduplication_service = DeduplicationService(
            similarity_threshold=similarity_threshold,
            embedder=embedder,
            reranker=reranker,
            qdrant_store=qdrant_store,
        )
    return _deduplication_service


def check_duplicate(
    new_memory: Memory,
    user_id: str,
    mode: DeduplicationMode = DeduplicationMode.NEAR,
) -> DuplicateCheckResult:
    """Convenience function to check if a memory is duplicate.

    Args:
        new_memory: Memory to check.
        user_id: User identifier.
        mode: Deduplication mode.

    Returns:
        DuplicateCheckResult with action and duplicate IDs.
    """
    return get_deduplication_service().check_duplicate(new_memory, user_id, mode)


def detect_clusters(
    memories: list[Memory],
    mode: DeduplicationMode = DeduplicationMode.NEAR,
    min_cluster_size: int = 2,
) -> list[DuplicateCluster]:
    """Convenience function to detect duplicate clusters.

    Args:
        memories: List of memories to analyze.
        mode: Deduplication mode.
        min_cluster_size: Minimum cluster size.

    Returns:
        List of DuplicateCluster objects.
    """
    return get_deduplication_service().detect_clusters(memories, mode, min_cluster_size)


def detect_near_duplicates(
    memories: list[Memory],
    generate_suggestions: bool = False,
) -> list[NearDuplicatePair]:
    """Convenience function to detect near-duplicate pairs.

    Args:
        memories: List of memories to analyze.
        generate_suggestions: Whether to generate merge suggestions.

    Returns:
        List of NearDuplicatePair objects.
    """
    return get_deduplication_service().detect_near_duplicates(memories, generate_suggestions)
