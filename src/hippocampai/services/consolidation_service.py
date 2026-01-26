"""Unified consolidation service facade for memory management.

This module provides a facade over the consolidation pipeline components,
routing all consolidation operations through a single entry point.

Usage:
    from hippocampai.services.consolidation_service import get_consolidation_service

    service = get_consolidation_service()
    result = service.consolidate_memories(memories)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from hippocampai.adapters.llm_base import BaseLLM
from hippocampai.models.memory import Memory
from hippocampai.pipeline.consolidate import MemoryConsolidator

logger = logging.getLogger(__name__)


@dataclass
class ConsolidationResult:
    """Result of a consolidation operation."""

    success: bool
    consolidated_memory: Optional[Memory] = None
    source_memory_ids: list[str] = field(default_factory=list)
    method_used: str = ""
    error_message: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class BatchConsolidationResult:
    """Result of batch consolidation across multiple groups."""

    total_groups: int = 0
    groups_consolidated: int = 0
    memories_merged: int = 0
    memories_created: int = 0
    errors: list[str] = field(default_factory=list)
    results: list[ConsolidationResult] = field(default_factory=list)


class ConsolidationService:
    """Unified service for memory consolidation operations.

    This service provides a facade over the MemoryConsolidator and any
    additional consolidation logic, offering a consistent interface for
    all consolidation operations.

    Example:
        service = ConsolidationService(llm=llm_instance)

        # Consolidate a group of memories
        result = service.consolidate_memories(similar_memories)
        if result.success:
            print(f"Consolidated into: {result.consolidated_memory.text}")

        # Batch consolidate multiple groups
        batch_result = service.consolidate_groups(memory_groups)
    """

    def __init__(
        self,
        llm: Optional[BaseLLM] = None,
        min_group_size: int = 2,
        max_group_size: int = 10,
    ) -> None:
        """Initialize the consolidation service.

        Args:
            llm: Optional LLM for intelligent consolidation.
            min_group_size: Minimum memories required for consolidation.
            max_group_size: Maximum memories to consolidate at once.
        """
        self.llm = llm
        self.min_group_size = min_group_size
        self.max_group_size = max_group_size
        self._consolidator = MemoryConsolidator(llm=llm)

    def consolidate_memories(
        self,
        memories: list[Memory],
        use_llm: bool = True,
    ) -> ConsolidationResult:
        """Consolidate a group of similar memories into one.

        Args:
            memories: List of memories to consolidate.
            use_llm: Whether to use LLM for intelligent consolidation.

        Returns:
            ConsolidationResult with the consolidated memory or error.
        """
        if len(memories) < self.min_group_size:
            return ConsolidationResult(
                success=False,
                error_message=f"Need at least {self.min_group_size} memories to consolidate",
                source_memory_ids=[m.id for m in memories],
            )

        # Limit group size
        if len(memories) > self.max_group_size:
            logger.warning(
                f"Truncating consolidation group from {len(memories)} to {self.max_group_size}"
            )
            memories = memories[: self.max_group_size]

        try:
            # Use the underlying consolidator
            if use_llm and self._consolidator.llm:
                consolidated = self._consolidator._consolidate_llm(memories)
                method = "llm"
            else:
                consolidated = self._consolidator._consolidate_heuristic(memories)
                method = "heuristic"

            if consolidated is None:
                return ConsolidationResult(
                    success=False,
                    error_message="Consolidation produced no result",
                    source_memory_ids=[m.id for m in memories],
                    method_used=method,
                )

            return ConsolidationResult(
                success=True,
                consolidated_memory=consolidated,
                source_memory_ids=[m.id for m in memories],
                method_used=method,
            )

        except Exception as e:
            logger.error(f"Consolidation failed: {e}")
            return ConsolidationResult(
                success=False,
                error_message=str(e),
                source_memory_ids=[m.id for m in memories],
            )

    def consolidate_groups(
        self,
        memory_groups: list[list[Memory]],
        use_llm: bool = True,
    ) -> BatchConsolidationResult:
        """Consolidate multiple groups of memories.

        Args:
            memory_groups: List of memory groups, each to be consolidated.
            use_llm: Whether to use LLM for intelligent consolidation.

        Returns:
            BatchConsolidationResult with statistics and results.
        """
        batch_result = BatchConsolidationResult(total_groups=len(memory_groups))

        for group in memory_groups:
            result = self.consolidate_memories(group, use_llm=use_llm)
            batch_result.results.append(result)

            if result.success:
                batch_result.groups_consolidated += 1
                batch_result.memories_merged += len(group)
                batch_result.memories_created += 1
            else:
                batch_result.errors.append(result.error_message)

        return batch_result

    def should_consolidate(
        self,
        memories: list[Memory],
        similarity_threshold: float = 0.85,
    ) -> bool:
        """Determine if a group of memories should be consolidated.

        Args:
            memories: List of memories to evaluate.
            similarity_threshold: Minimum average similarity for consolidation.

        Returns:
            True if the group should be consolidated.
        """
        if len(memories) < self.min_group_size:
            return False

        # Check if all memories are from the same user
        user_ids = {m.user_id for m in memories}
        if len(user_ids) > 1:
            logger.warning("Cannot consolidate memories from different users")
            return False

        # Check if memories have similar types
        types = {m.type for m in memories}
        if len(types) > 2:  # Allow some type mixing
            return False

        return True


# Global singleton instance
_consolidation_service: Optional[ConsolidationService] = None


def get_consolidation_service(
    llm: Optional[BaseLLM] = None,
) -> ConsolidationService:
    """Get the global ConsolidationService instance.

    Args:
        llm: Optional LLM for intelligent consolidation.

    Returns:
        The ConsolidationService instance.
    """
    global _consolidation_service
    if _consolidation_service is None:
        _consolidation_service = ConsolidationService(llm=llm)
    return _consolidation_service


def consolidate_memories(
    memories: list[Memory],
    use_llm: bool = True,
) -> ConsolidationResult:
    """Convenience function to consolidate memories.

    Args:
        memories: List of memories to consolidate.
        use_llm: Whether to use LLM for consolidation.

    Returns:
        ConsolidationResult with the consolidated memory.
    """
    return get_consolidation_service().consolidate_memories(memories, use_llm)
