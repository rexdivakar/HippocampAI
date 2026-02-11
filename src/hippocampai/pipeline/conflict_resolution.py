"""Memory conflict resolution for detecting and resolving contradictory memories."""

import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field

from hippocampai.adapters.llm_base import BaseLLM
from hippocampai.embed.embedder import Embedder
from hippocampai.models.memory import Memory

logger = logging.getLogger(__name__)


class ConflictResolutionStrategy(str, Enum):
    """Strategy for resolving memory conflicts."""

    TEMPORAL = "temporal"  # Latest memory wins
    CONFIDENCE = "confidence"  # Higher confidence wins
    IMPORTANCE = "importance"  # Higher importance wins
    USER_REVIEW = "user_review"  # Flag for manual review
    AUTO_MERGE = "auto_merge"  # Attempt to merge contradicting info
    KEEP_BOTH = "keep_both"  # Keep both with conflict flag


class ConflictType(str, Enum):
    """Type of memory conflict detected."""

    DIRECT_CONTRADICTION = "direct_contradiction"  # "I love X" vs "I hate X"
    VALUE_CHANGE = "value_change"  # Age changed, preference updated
    FACTUAL_INCONSISTENCY = "factual_inconsistency"  # Conflicting facts
    TEMPORAL_INCONSISTENCY = "temporal_inconsistency"  # Timeline doesn't make sense


class MemoryConflict(BaseModel):
    """Represents a detected conflict between memories."""

    model_config = {"arbitrary_types_allowed": True}

    id: str = Field(default_factory=lambda: str(id(object())))
    memory_1: Memory
    memory_2: Memory
    conflict_type: ConflictType
    confidence_score: float = Field(ge=0.0, le=1.0)
    similarity_score: float = Field(ge=0.0, le=1.0)
    detected_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    resolution_strategy: Optional[ConflictResolutionStrategy] = None
    resolved: bool = False
    resolution_notes: str = ""
    winner_memory_id: Optional[str] = None


class ConflictResolution(BaseModel):
    """Result of conflict resolution."""

    conflict: MemoryConflict
    action: str  # "keep_first", "keep_second", "keep_both", "merge", "flag"
    updated_memory: Optional[Memory] = None
    deleted_memory_ids: list[str] = Field(default_factory=list)
    notes: str = ""


class MemoryConflictResolver:
    """
    Detects and resolves conflicts between memories.

    Features:
    - Contradiction detection using semantic similarity + LLM analysis
    - Multiple resolution strategies (temporal, confidence, importance, user review)
    - Automatic conflict flagging and tracking
    - Support for merging contradictory information
    """

    def __init__(
        self,
        embedder: Embedder,
        llm: Optional[BaseLLM] = None,
        default_strategy: ConflictResolutionStrategy = ConflictResolutionStrategy.TEMPORAL,
        similarity_threshold: float = 0.75,
        contradiction_threshold: float = 0.85,
    ):
        """
        Initialize conflict resolver.

        Args:
            embedder: Embedding model for semantic similarity
            llm: Optional LLM for contradiction analysis
            default_strategy: Default resolution strategy
            similarity_threshold: Minimum similarity to consider as related (0-1)
            contradiction_threshold: LLM confidence threshold for contradiction (0-1)
        """
        self.embedder = embedder
        self.llm = llm
        self.default_strategy = default_strategy
        self.similarity_threshold = similarity_threshold
        self.contradiction_threshold = contradiction_threshold

        # Contradiction indicators (simple keyword-based detection)
        self.contradiction_patterns = [
            ("love", "hate"),
            ("like", "dislike"),
            ("enjoy", "don't enjoy"),
            ("prefer", "don't prefer"),
            ("allergic to", "not allergic"),
            ("can eat", "cannot eat"),
            ("yes", "no"),
            ("always", "never"),
            ("vegetarian", "not vegetarian"),
            ("vegan", "not vegan"),
        ]

    def detect_conflicts(
        self, memory: Memory, existing_memories: list[Memory], check_llm: bool = True
    ) -> list[MemoryConflict]:
        """
        Detect conflicts between a new memory and existing memories.

        Args:
            memory: New memory to check
            existing_memories: List of existing memories to compare against
            check_llm: Whether to use LLM for deep contradiction analysis

        Returns:
            List of detected conflicts
        """
        conflicts: list[MemoryConflict] = []

        # Skip if no existing memories
        if not existing_memories:
            return conflicts

        # Embed the new memory
        new_embedding = self.embedder.encode_single(memory.text)

        for existing_mem in existing_memories:
            # Skip comparing memory with itself
            if memory.id == existing_mem.id:
                continue

            # Skip if memory types are different (preferences don't conflict with facts)
            if memory.type != existing_mem.type:
                continue

            # Get embedding for existing memory
            if existing_mem.embedding:
                existing_embedding = np.array(existing_mem.embedding)
            else:
                existing_embedding = self.embedder.encode_single(existing_mem.text)

            # Calculate semantic similarity
            similarity = self._calculate_similarity(new_embedding, existing_embedding)

            # Only check for conflicts if memories are semantically similar
            if similarity < self.similarity_threshold:
                continue

            # Check for contradictions
            conflict_type, confidence = self._detect_contradiction(
                memory, existing_mem, check_llm=check_llm
            )

            if conflict_type and confidence >= self.contradiction_threshold:
                conflict = MemoryConflict(
                    memory_1=existing_mem,
                    memory_2=memory,
                    conflict_type=conflict_type,
                    confidence_score=confidence,
                    similarity_score=similarity,
                )
                conflicts.append(conflict)
                logger.info(
                    f"Detected {conflict_type} between memories: '{existing_mem.text[:50]}' vs '{memory.text[:50]}'"
                )

        return conflicts

    def _calculate_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    def _detect_contradiction(
        self, memory1: Memory, memory2: Memory, check_llm: bool = True
    ) -> tuple[Optional[ConflictType], float]:
        """
        Detect if two memories contradict each other.

        Args:
            memory1: First memory
            memory2: Second memory
            check_llm: Whether to use LLM for deep analysis

        Returns:
            Tuple of (conflict_type, confidence_score)
        """
        text1 = memory1.text.lower()
        text2 = memory2.text.lower()

        # Pattern-based detection (fast but simple)
        pattern_confidence = self._check_contradiction_patterns(text1, text2)
        if pattern_confidence >= 0.7:
            return ConflictType.DIRECT_CONTRADICTION, pattern_confidence

        # Temporal inconsistency check
        if self._check_temporal_inconsistency(memory1, memory2):
            return ConflictType.TEMPORAL_INCONSISTENCY, 0.9

        # LLM-based detection (more accurate but slower)
        if check_llm and self.llm:
            llm_conflict_type, llm_confidence = self._llm_detect_contradiction(memory1, memory2)
            if llm_confidence >= self.contradiction_threshold:
                return llm_conflict_type, llm_confidence

        # No contradiction detected
        return None, 0.0

    def _check_contradiction_patterns(self, text1: str, text2: str) -> float:
        """Check for contradiction using simple patterns."""
        for positive, negative in self.contradiction_patterns:
            # Check if one has positive and other has negative
            has_positive_1 = positive in text1
            has_negative_1 = negative in text1
            has_positive_2 = positive in text2
            has_negative_2 = negative in text2

            # Direct contradiction: one says positive, other says negative
            if (has_positive_1 and has_negative_2) or (has_negative_1 and has_positive_2):
                return 0.85

        return 0.0

    def _check_temporal_inconsistency(self, memory1: Memory, memory2: Memory) -> bool:
        """Check if memories have temporal inconsistencies.

        Detects potential temporal conflicts when both memories mention ages
        but were created at significantly different times.
        """
        age_keywords = ["age", "years old", "born in"]

        text1_has_age = any(keyword in memory1.text.lower() for keyword in age_keywords)
        text2_has_age = any(keyword in memory2.text.lower() for keyword in age_keywords)

        # If both mention ages and were created more than 1 year apart, flag as potential inconsistency
        if text1_has_age and text2_has_age:
            time_diff_seconds = abs((memory1.created_at - memory2.created_at).total_seconds())
            # 1 year = 365.25 days * 24 hours * 3600 seconds
            one_year_seconds = 365.25 * 24 * 3600

            if time_diff_seconds > one_year_seconds:
                # Age mentions in memories created over a year apart may indicate inconsistency
                return True

        return False

    def _llm_detect_contradiction(
        self, memory1: Memory, memory2: Memory
    ) -> tuple[Optional[ConflictType], float]:
        """Use LLM to detect contradictions between memories."""
        if not self.llm:
            return None, 0.0

        prompt = f"""Analyze these two memories and determine if they contradict each other.

Memory 1: "{memory1.text}"
Memory 2: "{memory2.text}"

Consider:
- Direct contradictions (opposite statements about same topic)
- Value changes (different values for same attribute)
- Factual inconsistencies (conflicting facts)
- Temporal inconsistencies (timeline doesn't make sense)

Respond with JSON:
{{
    "is_contradiction": true/false,
    "conflict_type": "direct_contradiction|value_change|factual_inconsistency|temporal_inconsistency|none",
    "confidence": 0.0-1.0,
    "explanation": "brief explanation"
}}
"""

        try:
            import json

            response = self.llm.generate(prompt, max_tokens=200, temperature=0.1)

            # Try to extract JSON from response
            if "{" in response and "}" in response:
                json_start = response.index("{")
                json_end = response.rindex("}") + 1
                json_str = response[json_start:json_end]
                result = json.loads(json_str)

                if result.get("is_contradiction", False):
                    conflict_type_str = result.get("conflict_type", "direct_contradiction")
                    try:
                        conflict_type = ConflictType(conflict_type_str)
                    except ValueError:
                        conflict_type = ConflictType.DIRECT_CONTRADICTION

                    confidence = float(result.get("confidence", 0.8))
                    return conflict_type, confidence

        except Exception as e:
            logger.warning(f"LLM contradiction detection failed: {e}")

        return None, 0.0

    def resolve_conflict(
        self,
        conflict: MemoryConflict,
        strategy: Optional[ConflictResolutionStrategy] = None,
    ) -> ConflictResolution:
        """
        Resolve a detected conflict using specified strategy.

        Args:
            conflict: Detected conflict to resolve
            strategy: Resolution strategy (uses default if not specified)

        Returns:
            ConflictResolution with action to take
        """
        strategy = strategy or self.default_strategy

        if strategy == ConflictResolutionStrategy.TEMPORAL:
            return self._resolve_temporal(conflict)
        elif strategy == ConflictResolutionStrategy.CONFIDENCE:
            return self._resolve_confidence(conflict)
        elif strategy == ConflictResolutionStrategy.IMPORTANCE:
            return self._resolve_importance(conflict)
        elif strategy == ConflictResolutionStrategy.USER_REVIEW:
            return self._flag_for_review(conflict)
        elif strategy == ConflictResolutionStrategy.AUTO_MERGE:
            return self._auto_merge(conflict)
        else:  # KEEP_BOTH
            return self._keep_both(conflict)

    def _resolve_temporal(self, conflict: MemoryConflict) -> ConflictResolution:
        """Resolve conflict by keeping the most recent memory."""
        mem1_time = conflict.memory_1.updated_at or conflict.memory_1.created_at
        mem2_time = conflict.memory_2.updated_at or conflict.memory_2.created_at

        if mem2_time > mem1_time:
            # Keep memory 2 (newer), delete memory 1
            winner = conflict.memory_2
            deleted = [conflict.memory_1.id]
            notes = f"Kept newer memory from {mem2_time.isoformat()}"
        else:
            # Keep memory 1 (newer or equal), delete memory 2
            winner = conflict.memory_1
            deleted = [conflict.memory_2.id]
            notes = f"Kept newer memory from {mem1_time.isoformat()}"

        conflict.resolution_strategy = ConflictResolutionStrategy.TEMPORAL
        conflict.resolved = True
        conflict.winner_memory_id = winner.id
        conflict.resolution_notes = notes

        return ConflictResolution(
            conflict=conflict,
            action="keep_second" if mem2_time > mem1_time else "keep_first",
            updated_memory=winner,
            deleted_memory_ids=deleted,
            notes=notes,
        )

    def _resolve_confidence(self, conflict: MemoryConflict) -> ConflictResolution:
        """Resolve conflict by keeping the memory with higher confidence."""
        if conflict.memory_2.confidence > conflict.memory_1.confidence:
            winner = conflict.memory_2
            deleted = [conflict.memory_1.id]
            action = "keep_second"
            notes = f"Kept higher confidence memory (conf={conflict.memory_2.confidence})"
        elif conflict.memory_1.confidence > conflict.memory_2.confidence:
            winner = conflict.memory_1
            deleted = [conflict.memory_2.id]
            action = "keep_first"
            notes = f"Kept higher confidence memory (conf={conflict.memory_1.confidence})"
        else:
            # Equal confidence - fall back to temporal
            return self._resolve_temporal(conflict)

        conflict.resolution_strategy = ConflictResolutionStrategy.CONFIDENCE
        conflict.resolved = True
        conflict.winner_memory_id = winner.id
        conflict.resolution_notes = notes

        return ConflictResolution(
            conflict=conflict,
            action=action,
            updated_memory=winner,
            deleted_memory_ids=deleted,
            notes=notes,
        )

    def _resolve_importance(self, conflict: MemoryConflict) -> ConflictResolution:
        """Resolve conflict by keeping the memory with higher importance."""
        if conflict.memory_2.importance > conflict.memory_1.importance:
            winner = conflict.memory_2
            deleted = [conflict.memory_1.id]
            action = "keep_second"
            notes = f"Kept higher importance memory (imp={conflict.memory_2.importance})"
        elif conflict.memory_1.importance > conflict.memory_2.importance:
            winner = conflict.memory_1
            deleted = [conflict.memory_2.id]
            action = "keep_first"
            notes = f"Kept higher importance memory (imp={conflict.memory_1.importance})"
        else:
            # Equal importance - fall back to temporal
            return self._resolve_temporal(conflict)

        conflict.resolution_strategy = ConflictResolutionStrategy.IMPORTANCE
        conflict.resolved = True
        conflict.winner_memory_id = winner.id
        conflict.resolution_notes = notes

        return ConflictResolution(
            conflict=conflict,
            action=action,
            updated_memory=winner,
            deleted_memory_ids=deleted,
            notes=notes,
        )

    def _flag_for_review(self, conflict: MemoryConflict) -> ConflictResolution:
        """Flag conflict for manual user review."""
        # Add conflict flag to both memories' metadata
        conflict.memory_1.metadata["has_conflict"] = True
        conflict.memory_1.metadata["conflict_id"] = conflict.id
        conflict.memory_1.metadata["conflict_with"] = conflict.memory_2.id

        conflict.memory_2.metadata["has_conflict"] = True
        conflict.memory_2.metadata["conflict_id"] = conflict.id
        conflict.memory_2.metadata["conflict_with"] = conflict.memory_1.id

        conflict.resolution_strategy = ConflictResolutionStrategy.USER_REVIEW
        conflict.resolved = False
        conflict.resolution_notes = "Flagged for user review"

        return ConflictResolution(
            conflict=conflict,
            action="flag",
            updated_memory=None,
            deleted_memory_ids=[],
            notes="Both memories flagged for user review",
        )

    def _auto_merge(self, conflict: MemoryConflict) -> ConflictResolution:
        """Attempt to merge conflicting memories using LLM."""
        if not self.llm:
            logger.warning("Auto-merge requires LLM, falling back to user review")
            return self._flag_for_review(conflict)

        prompt = f"""Two conflicting memories need to be merged. Create a single memory that captures the evolution of information.

Memory 1 (from {conflict.memory_1.created_at.isoformat()}): "{conflict.memory_1.text}"
Memory 2 (from {conflict.memory_2.created_at.isoformat()}): "{conflict.memory_2.text}"

Create a merged memory that:
1. Acknowledges the change/evolution
2. Captures both perspectives if relevant
3. Indicates the most current information

Respond with just the merged memory text (no JSON, no explanation):"""

        try:
            merged_text = self.llm.generate(prompt, max_tokens=200, temperature=0.3).strip()

            # Create merged memory with properties from both
            merged_memory = Memory(
                text=merged_text,
                user_id=conflict.memory_1.user_id,
                session_id=conflict.memory_1.session_id,
                type=conflict.memory_1.type,
                importance=max(conflict.memory_1.importance, conflict.memory_2.importance),
                confidence=min(
                    conflict.memory_1.confidence, conflict.memory_2.confidence
                ),  # Lower confidence due to merge
                tags=list(set(conflict.memory_1.tags + conflict.memory_2.tags)),
                metadata={
                    "merged_from": [conflict.memory_1.id, conflict.memory_2.id],
                    "conflict_type": conflict.conflict_type,
                },
            )

            conflict.resolution_strategy = ConflictResolutionStrategy.AUTO_MERGE
            conflict.resolved = True
            conflict.winner_memory_id = merged_memory.id
            conflict.resolution_notes = "Automatically merged conflicting memories"

            return ConflictResolution(
                conflict=conflict,
                action="merge",
                updated_memory=merged_memory,
                deleted_memory_ids=[conflict.memory_1.id, conflict.memory_2.id],
                notes="Successfully merged conflicting memories",
            )

        except Exception as e:
            logger.error(f"Auto-merge failed: {e}, falling back to user review")
            return self._flag_for_review(conflict)

    def _keep_both(self, conflict: MemoryConflict) -> ConflictResolution:
        """Keep both memories but flag them as conflicting."""
        # Similar to flag_for_review but explicitly keeps both
        conflict.memory_1.metadata["has_conflict"] = True
        conflict.memory_1.metadata["conflict_id"] = conflict.id
        conflict.memory_1.metadata["conflict_with"] = conflict.memory_2.id
        conflict.memory_1.metadata["conflict_type"] = conflict.conflict_type

        conflict.memory_2.metadata["has_conflict"] = True
        conflict.memory_2.metadata["conflict_id"] = conflict.id
        conflict.memory_2.metadata["conflict_with"] = conflict.memory_1.id
        conflict.memory_2.metadata["conflict_type"] = conflict.conflict_type

        conflict.resolution_strategy = ConflictResolutionStrategy.KEEP_BOTH
        conflict.resolved = True
        conflict.resolution_notes = "Kept both memories with conflict flags"

        return ConflictResolution(
            conflict=conflict,
            action="keep_both",
            updated_memory=None,
            deleted_memory_ids=[],
            notes="Both memories kept with conflict markers",
        )

    def batch_detect_conflicts(
        self, memories: list[Memory], check_llm: bool = False
    ) -> list[MemoryConflict]:
        """
        Detect conflicts within a batch of memories.

        Args:
            memories: List of memories to check
            check_llm: Whether to use LLM for deep analysis (slower)

        Returns:
            List of all detected conflicts
        """
        all_conflicts = []
        processed_pairs = set()

        for i, memory in enumerate(memories):
            # Compare with all subsequent memories
            remaining_memories = memories[i + 1 :]
            conflicts = self.detect_conflicts(memory, remaining_memories, check_llm=check_llm)

            # Deduplicate conflicts
            for conflict in conflicts:
                pair_id = tuple(sorted([conflict.memory_1.id, conflict.memory_2.id]))
                if pair_id not in processed_pairs:
                    all_conflicts.append(conflict)
                    processed_pairs.add(pair_id)

        logger.info(f"Detected {len(all_conflicts)} conflicts in batch of {len(memories)} memories")
        return all_conflicts
