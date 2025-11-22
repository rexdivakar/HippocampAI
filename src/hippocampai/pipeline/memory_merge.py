"""
Memory Merge Tools.

This module provides intelligent memory merging capabilities:
- Similarity-based merge suggestions
- Conflict detection and resolution
- Information preservation from multiple sources
- Merge history tracking
- Undo/rollback capabilities
- Quality-aware merging

Features:
- Automatic duplicate detection for merging
- Smart conflict resolution strategies
- Information synthesis from multiple memories
- Source attribution and provenance tracking
- Merge preview and validation
- Batch merge operations
"""

import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class MergeStrategy(str, Enum):
    """Strategies for merging conflicting information."""

    NEWEST = "newest"  # Keep most recent information
    HIGHEST_CONFIDENCE = "highest_confidence"  # Keep most confident
    LONGEST = "longest"  # Keep longest/most detailed
    COMBINE = "combine"  # Synthesize both
    MANUAL = "manual"  # Require manual resolution


class MergeConflictType(str, Enum):
    """Types of conflicts during merge."""

    TEXT_DIFFERENCE = "text_difference"
    METADATA_CONFLICT = "metadata_conflict"
    TAG_MISMATCH = "tag_mismatch"
    TYPE_MISMATCH = "type_mismatch"
    IMPORTANCE_DIFFERENCE = "importance_difference"
    CONTRADICTORY_INFO = "contradictory_info"


class MergeConflict(BaseModel):
    """A conflict detected during merge."""

    conflict_type: MergeConflictType
    field: str
    source_values: list[Any]  # Values from each source memory
    suggested_resolution: Optional[Any] = None
    resolution_strategy: Optional[MergeStrategy] = None
    requires_manual: bool = False


class MergeCandidate(BaseModel):
    """A suggested merge between memories."""

    candidate_id: str = Field(default_factory=lambda: str(uuid4()))
    memory_ids: list[str]
    similarity_score: float = Field(ge=0.0, le=1.0)
    merge_confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence in merge suggestion"
    )
    conflicts: list[MergeConflict] = Field(default_factory=list)
    suggested_strategy: MergeStrategy = MergeStrategy.COMBINE
    preview_text: Optional[str] = None
    estimated_quality_gain: float = 0.0  # Expected improvement in quality score
    metadata: dict[str, Any] = Field(default_factory=dict)


class MergeResult(BaseModel):
    """Result of a merge operation."""

    merged_memory_id: str
    source_memory_ids: list[str]
    strategy_used: MergeStrategy
    conflicts_resolved: int
    manual_conflicts: int
    merged_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    merged_by: Optional[str] = None
    rollback_data: dict[str, Any] = Field(
        default_factory=dict, description="Data needed to undo merge"
    )
    quality_before: float = 0.0
    quality_after: float = 0.0
    success: bool = True
    error_message: Optional[str] = None


class MergeHistory(BaseModel):
    """History of merge operations."""

    merge_id: str
    result: MergeResult
    can_rollback: bool = True
    rolled_back: bool = False
    rollback_at: Optional[datetime] = None


class MemoryMergeEngine:
    """Engine for intelligent memory merging."""

    def __init__(
        self,
        llm: Optional[Any] = None,
        similarity_threshold: float = 0.75,
        auto_merge_threshold: float = 0.90,
        default_strategy: MergeStrategy = MergeStrategy.COMBINE,
    ):
        """
        Initialize merge engine.

        Args:
            llm: Optional LLM for intelligent merging
            similarity_threshold: Minimum similarity to suggest merge
            auto_merge_threshold: Threshold for automatic merging
            default_strategy: Default conflict resolution strategy
        """
        self.llm = llm
        self.similarity_threshold = similarity_threshold
        self.auto_merge_threshold = auto_merge_threshold
        self.default_strategy = default_strategy

        # Merge history (should be persisted)
        self.merge_history: dict[str, MergeHistory] = {}

    def suggest_merges(
        self,
        memories: list[dict[str, Any]],
        similarity_matrix: dict[tuple[str, str], float],
    ) -> list[MergeCandidate]:
        """
        Suggest memory merges based on similarity.

        Args:
            memories: List of memory dictionaries
            similarity_matrix: Precomputed similarity scores

        Returns:
            List of merge candidates
        """
        candidates: list[MergeCandidate] = []
        processed: set[str] = set()

        # Find similar pairs above threshold
        for (id1, id2), similarity in similarity_matrix.items():
            if similarity < self.similarity_threshold:
                continue

            if id1 in processed or id2 in processed:
                continue

            # Get memory data
            mem1 = next((m for m in memories if m["id"] == id1), None)
            mem2 = next((m for m in memories if m["id"] == id2), None)

            if not mem1 or not mem2:
                continue

            # Analyze merge
            conflicts = self._detect_conflicts(mem1, mem2)
            preview = self._generate_preview(mem1, mem2, conflicts)
            quality_gain = self._estimate_quality_gain(mem1, mem2, conflicts)

            # Determine merge confidence
            merge_confidence = self._calculate_merge_confidence(
                similarity, len(conflicts), mem1, mem2
            )

            # Determine strategy
            if similarity >= self.auto_merge_threshold and len(conflicts) == 0:
                strategy = MergeStrategy.COMBINE
            elif conflicts:
                strategy = self._suggest_strategy(conflicts, mem1, mem2)
            else:
                strategy = self.default_strategy

            candidate = MergeCandidate(
                memory_ids=[id1, id2],
                similarity_score=similarity,
                merge_confidence=merge_confidence,
                conflicts=conflicts,
                suggested_strategy=strategy,
                preview_text=preview,
                estimated_quality_gain=quality_gain,
            )

            candidates.append(candidate)
            processed.add(id1)
            processed.add(id2)

        logger.info(f"Generated {len(candidates)} merge candidates")

        return candidates

    def merge_memories(
        self,
        memories: list[dict[str, Any]],
        strategy: Optional[MergeStrategy] = None,
        user_id: Optional[str] = None,
        manual_resolutions: Optional[dict[str, Any]] = None,
    ) -> MergeResult:
        """
        Merge multiple memories into one.

        Args:
            memories: List of memories to merge (2+)
            strategy: Merge strategy to use
            user_id: User performing the merge
            manual_resolutions: Manual conflict resolutions

        Returns:
            MergeResult with merged memory and metadata
        """
        if len(memories) < 2:
            return MergeResult(
                merged_memory_id="",
                source_memory_ids=[],
                strategy_used=strategy or self.default_strategy,
                conflicts_resolved=0,
                manual_conflicts=0,
                success=False,
                error_message="Need at least 2 memories to merge",
            )

        strategy = strategy or self.default_strategy
        source_ids = [m["id"] for m in memories]

        try:
            # Detect conflicts
            all_conflicts = self._detect_multi_conflicts(memories)

            # Resolve conflicts
            resolved_data, manual_count = self._resolve_conflicts(
                memories, all_conflicts, strategy, manual_resolutions
            )

            # Create merged memory
            merged_memory = self._create_merged_memory(memories, resolved_data)

            # Calculate quality scores
            quality_before = sum(m.get("confidence", 0.5) for m in memories) / len(memories)
            quality_after = merged_memory.get("confidence", 0.5)

            # Create rollback data
            rollback_data = {
                "source_memories": memories,
                "merged_memory_id": merged_memory["id"],
            }

            result = MergeResult(
                merged_memory_id=merged_memory["id"],
                source_memory_ids=source_ids,
                strategy_used=strategy,
                conflicts_resolved=len(all_conflicts) - manual_count,
                manual_conflicts=manual_count,
                merged_by=user_id,
                rollback_data=rollback_data,
                quality_before=quality_before,
                quality_after=quality_after,
                success=True,
            )

            # Store in history
            merge_id = str(uuid4())
            self.merge_history[merge_id] = MergeHistory(
                merge_id=merge_id, result=result
            )

            logger.info(
                f"Merged {len(memories)} memories into {merged_memory['id']} "
                f"using {strategy.value} strategy"
            )

            return result

        except Exception as e:
            logger.error(f"Merge failed: {e}")
            return MergeResult(
                merged_memory_id="",
                source_memory_ids=source_ids,
                strategy_used=strategy,
                conflicts_resolved=0,
                manual_conflicts=0,
                success=False,
                error_message=str(e),
            )

    def preview_merge(
        self,
        memories: list[dict[str, Any]],
        strategy: Optional[MergeStrategy] = None,
    ) -> dict[str, Any]:
        """
        Preview what a merge would look like without executing.

        Args:
            memories: Memories to merge
            strategy: Strategy to use

        Returns:
            Dictionary with preview information
        """
        strategy = strategy or self.default_strategy

        # Detect conflicts
        conflicts = self._detect_multi_conflicts(memories)

        # Generate preview text
        preview_text = self._generate_multi_preview(memories, conflicts)

        # Estimate quality
        quality_gain = self._estimate_multi_quality_gain(memories, conflicts)

        return {
            "preview_text": preview_text,
            "conflicts": [c.model_dump() for c in conflicts],
            "conflict_count": len(conflicts),
            "manual_required": sum(1 for c in conflicts if c.requires_manual),
            "estimated_quality_gain": quality_gain,
            "strategy": strategy.value,
            "source_count": len(memories),
        }

    def rollback_merge(self, merge_id: str) -> bool:
        """
        Rollback a previous merge operation.

        Args:
            merge_id: Merge operation ID

        Returns:
            True if successful
        """
        if merge_id not in self.merge_history:
            logger.error(f"Merge {merge_id} not found in history")
            return False

        history = self.merge_history[merge_id]

        if history.rolled_back:
            logger.warning(f"Merge {merge_id} already rolled back")
            return False

        if not history.can_rollback:
            logger.error(f"Merge {merge_id} cannot be rolled back")
            return False

        try:
            # Mark as rolled back
            history.rolled_back = True
            history.rollback_at = datetime.now(timezone.utc)

            logger.info(f"Rolled back merge {merge_id}")
            return True

        except Exception as e:
            logger.error(f"Rollback failed for {merge_id}: {e}")
            return False

    # ========================================================================
    # HELPER METHODS
    # ========================================================================

    def _detect_conflicts(
        self, mem1: dict[str, Any], mem2: dict[str, Any]
    ) -> list[MergeConflict]:
        """Detect conflicts between two memories."""
        conflicts: list[MergeConflict] = []

        # Text difference
        if mem1["text"] != mem2["text"]:
            similarity = self._text_similarity(mem1["text"], mem2["text"])
            if similarity < 0.95:  # Not nearly identical
                conflicts.append(
                    MergeConflict(
                        conflict_type=MergeConflictType.TEXT_DIFFERENCE,
                        field="text",
                        source_values=[mem1["text"], mem2["text"]],
                        suggested_resolution=self._combine_texts(
                            mem1["text"], mem2["text"]
                        ),
                        resolution_strategy=MergeStrategy.COMBINE,
                    )
                )

        # Type mismatch
        if mem1.get("type") != mem2.get("type"):
            conflicts.append(
                MergeConflict(
                    conflict_type=MergeConflictType.TYPE_MISMATCH,
                    field="type",
                    source_values=[mem1.get("type"), mem2.get("type")],
                    requires_manual=True,
                )
            )

        # Importance difference
        imp1 = mem1.get("importance", 5.0)
        imp2 = mem2.get("importance", 5.0)
        if abs(imp1 - imp2) > 3.0:
            conflicts.append(
                MergeConflict(
                    conflict_type=MergeConflictType.IMPORTANCE_DIFFERENCE,
                    field="importance",
                    source_values=[imp1, imp2],
                    suggested_resolution=max(imp1, imp2),
                    resolution_strategy=MergeStrategy.HIGHEST_CONFIDENCE,
                )
            )

        # Tag mismatch
        tags1 = set(mem1.get("tags", []))
        tags2 = set(mem2.get("tags", []))
        if tags1 != tags2:
            conflicts.append(
                MergeConflict(
                    conflict_type=MergeConflictType.TAG_MISMATCH,
                    field="tags",
                    source_values=[list(tags1), list(tags2)],
                    suggested_resolution=list(tags1.union(tags2)),
                    resolution_strategy=MergeStrategy.COMBINE,
                )
            )

        return conflicts

    def _detect_multi_conflicts(
        self, memories: list[dict[str, Any]]
    ) -> list[MergeConflict]:
        """Detect conflicts among multiple memories."""
        conflicts: list[MergeConflict] = []

        # Pairwise conflict detection
        for i in range(len(memories)):
            for j in range(i + 1, len(memories)):
                pair_conflicts = self._detect_conflicts(memories[i], memories[j])
                for conflict in pair_conflicts:
                    # Check if similar conflict already exists
                    existing = next(
                        (
                            c
                            for c in conflicts
                            if c.field == conflict.field
                            and c.conflict_type == conflict.conflict_type
                        ),
                        None,
                    )
                    if not existing:
                        conflicts.append(conflict)

        return conflicts

    def _resolve_conflicts(
        self,
        memories: list[dict[str, Any]],
        conflicts: list[MergeConflict],
        strategy: MergeStrategy,
        manual_resolutions: Optional[dict[str, Any]],
    ) -> tuple[dict[str, Any], int]:
        """
        Resolve conflicts using strategy.

        Returns:
            (resolved_data, manual_conflict_count)
        """
        resolved = {}
        manual_count = 0

        for conflict in conflicts:
            # Check for manual resolution
            if manual_resolutions and conflict.field in manual_resolutions:
                resolved[conflict.field] = manual_resolutions[conflict.field]
                continue

            # Check if requires manual
            if conflict.requires_manual:
                manual_count += 1
                continue

            # Use suggested resolution if available
            if conflict.suggested_resolution is not None:
                resolved[conflict.field] = conflict.suggested_resolution
                continue

            # Apply strategy
            if strategy == MergeStrategy.NEWEST:
                # Get most recent value
                sorted_mems = sorted(
                    memories, key=lambda m: m.get("updated_at") or m.get("created_at") or "", reverse=True
                )
                resolved[conflict.field] = sorted_mems[0].get(conflict.field)

            elif strategy == MergeStrategy.HIGHEST_CONFIDENCE:
                # Get from most confident memory
                sorted_mems = sorted(
                    memories, key=lambda m: m.get("confidence", 0.5), reverse=True
                )
                resolved[conflict.field] = sorted_mems[0].get(conflict.field)

            elif strategy == MergeStrategy.LONGEST:
                # Get longest value (for text fields)
                values = [m.get(conflict.field, "") for m in memories]
                resolved[conflict.field] = max(values, key=lambda x: len(str(x)))

            elif strategy == MergeStrategy.COMBINE:
                # Combine values
                if conflict.field == "text":
                    texts = [m.get("text", "") for m in memories]
                    resolved["text"] = self._combine_multiple_texts(texts)
                elif conflict.field == "tags":
                    all_tags = set()
                    for m in memories:
                        all_tags.update(m.get("tags", []))
                    resolved["tags"] = list(all_tags)

        return resolved, manual_count

    def _create_merged_memory(
        self, memories: list[dict[str, Any]], resolved_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Create merged memory from sources and resolved conflicts."""
        # Start with first memory as base
        merged = memories[0].copy()

        # Update with resolved conflicts
        merged.update(resolved_data)

        # Generate new ID
        merged["id"] = str(uuid4())

        # Update timestamps
        merged["created_at"] = min(m.get("created_at", datetime.now(timezone.utc)) for m in memories)
        merged["updated_at"] = datetime.now(timezone.utc)

        # Combine metadata
        merged["metadata"] = merged.get("metadata", {})
        merged["metadata"]["merged_from"] = [m["id"] for m in memories]
        merged["metadata"]["merge_timestamp"] = datetime.now(timezone.utc).isoformat()

        # Average confidence
        merged["confidence"] = sum(m.get("confidence", 0.5) for m in memories) / len(
            memories
        )

        # Max importance
        merged["importance"] = max(m.get("importance", 5.0) for m in memories)

        return merged

    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity (simple Jaccard)."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return intersection / union if union > 0 else 0.0

    def _combine_texts(self, text1: str, text2: str) -> str:
        """Combine two texts intelligently."""
        # If one is substring of other, use longer
        if text1 in text2:
            return text2
        if text2 in text1:
            return text1

        # Otherwise concatenate with separator
        return f"{text1}. {text2}"

    def _combine_multiple_texts(self, texts: list[str]) -> str:
        """Combine multiple texts."""
        # Remove duplicates while preserving order
        unique_texts = []
        seen = set()

        for text in texts:
            text_lower = text.lower().strip()
            if text_lower not in seen:
                unique_texts.append(text)
                seen.add(text_lower)

        # Combine
        return ". ".join(unique_texts)

    def _generate_preview(
        self, mem1: dict[str, Any], mem2: dict[str, Any], conflicts: list[MergeConflict]
    ) -> str:
        """Generate merge preview text."""
        preview = "Merging 2 memories:\n\n"
        preview += f"Memory 1: {mem1['text'][:100]}...\n"
        preview += f"Memory 2: {mem2['text'][:100]}...\n\n"

        if conflicts:
            preview += f"Conflicts detected: {len(conflicts)}\n"
            for conflict in conflicts[:3]:  # Show first 3
                preview += f"  - {conflict.field}: {conflict.conflict_type.value}\n"

        return preview

    def _generate_multi_preview(
        self, memories: list[dict[str, Any]], conflicts: list[MergeConflict]
    ) -> str:
        """Generate preview for multi-memory merge."""
        preview = f"Merging {len(memories)} memories:\n\n"

        for i, mem in enumerate(memories[:3], 1):
            preview += f"Memory {i}: {mem['text'][:80]}...\n"

        if len(memories) > 3:
            preview += f"... and {len(memories) - 3} more\n"

        preview += f"\nConflicts: {len(conflicts)}\n"

        return preview

    def _estimate_quality_gain(
        self, mem1: dict[str, Any], mem2: dict[str, Any], conflicts: list[MergeConflict]
    ) -> float:
        """Estimate quality improvement from merge."""
        # Base gain from deduplication
        gain = 5.0

        # Penalty for conflicts
        gain -= len(conflicts) * 1.0

        # Bonus for combining information
        if mem1.get("text") and mem2.get("text"):
            if mem1["text"] != mem2["text"]:
                gain += 3.0  # Additional information

        return max(0.0, gain)

    def _estimate_multi_quality_gain(
        self, memories: list[dict[str, Any]], conflicts: list[MergeConflict]
    ) -> float:
        """Estimate quality gain for multi-memory merge."""
        gain = 5.0 * (len(memories) - 1)  # Base gain
        gain -= len(conflicts) * 2.0  # Conflict penalty
        return max(0.0, gain)

    def _calculate_merge_confidence(
        self,
        similarity: float,
        conflict_count: int,
        mem1: dict[str, Any],
        mem2: dict[str, Any],
    ) -> float:
        """Calculate confidence in merge suggestion."""
        confidence = similarity

        # Reduce for conflicts
        confidence -= conflict_count * 0.1

        # Bonus for same type
        if mem1.get("type") == mem2.get("type"):
            confidence += 0.05

        # Bonus for similar importance
        imp_diff = abs(mem1.get("importance", 5.0) - mem2.get("importance", 5.0))
        if imp_diff < 2.0:
            confidence += 0.05

        return max(0.0, min(1.0, confidence))

    def _suggest_strategy(
        self, conflicts: list[MergeConflict], mem1: dict[str, Any], mem2: dict[str, Any]
    ) -> MergeStrategy:
        """Suggest best merge strategy based on conflicts."""
        # If any requires manual, suggest manual
        if any(c.requires_manual for c in conflicts):
            return MergeStrategy.MANUAL

        # If mostly text conflicts, combine
        text_conflicts = sum(
            1 for c in conflicts if c.conflict_type == MergeConflictType.TEXT_DIFFERENCE
        )
        if text_conflicts > len(conflicts) / 2:
            return MergeStrategy.COMBINE

        # If one is much newer, use newest
        time1 = mem1.get("updated_at", mem1.get("created_at"))
        time2 = mem2.get("updated_at", mem2.get("created_at"))
        if time1 and time2:
            diff = abs((time1 - time2).total_seconds())
            if diff > 86400 * 30:  # 30 days
                return MergeStrategy.NEWEST

        # Default
        return self.default_strategy
