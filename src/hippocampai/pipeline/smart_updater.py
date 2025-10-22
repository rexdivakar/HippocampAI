"""Smart memory update and reconciliation logic.

This module provides intelligent memory management including:
- Merge vs update vs skip decisions
- Conflict resolution
- Memory reconciliation
- Confidence scoring evolution
- Quality refinement
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional, Literal
import logging
import re

from hippocampai.models.memory import Memory, MemoryType

logger = logging.getLogger(__name__)


class UpdateDecision:
    """Result of update decision logic."""

    def __init__(
        self,
        action: Literal["merge", "update", "skip", "keep_both"],
        reason: str,
        merged_memory: Optional[Memory] = None,
        confidence_adjustment: float = 0.0
    ):
        self.action = action
        self.reason = reason
        self.merged_memory = merged_memory
        self.confidence_adjustment = confidence_adjustment


class SmartMemoryUpdater:
    """Handles intelligent memory updates and conflict resolution."""

    def __init__(self, llm=None, similarity_threshold: float = 0.85):
        """Initialize smart updater.

        Args:
            llm: Language model for conflict resolution (optional)
            similarity_threshold: Threshold for considering memories similar
        """
        self.llm = llm
        self.similarity_threshold = similarity_threshold

    def should_update_memory(
        self,
        existing: Memory,
        new_text: str,
        new_metadata: Optional[Dict] = None
    ) -> UpdateDecision:
        """Decide whether to merge, update, skip, or keep both memories.

        Args:
            existing: Existing memory in database
            new_text: New memory text
            new_metadata: Optional metadata for new memory

        Returns:
            UpdateDecision with action and reasoning
        """
        # Calculate text similarity
        similarity = self._calculate_similarity(existing.text, new_text)

        # Case 1: Nearly identical - skip
        if similarity > 0.95:
            return UpdateDecision(
                action="skip",
                reason="Memory already exists with high similarity",
                confidence_adjustment=0.1  # Boost confidence (reinforcement)
            )

        # Case 2: Very similar - update or merge
        if similarity > self.similarity_threshold:
            # Check if new info adds value
            if len(new_text) > len(existing.text) * 1.2:  # 20% more content
                return self._decide_merge_or_update(existing, new_text)
            else:
                return UpdateDecision(
                    action="skip",
                    reason="Similar memory exists with comparable detail",
                    confidence_adjustment=0.05
                )

        # Case 3: Related but different - check for conflicts
        if similarity > 0.6:
            conflict = self._detect_conflict(existing.text, new_text)
            if conflict:
                return self._resolve_conflict(existing, new_text)
            else:
                # Related but complementary - keep both
                return UpdateDecision(
                    action="keep_both",
                    reason="Memories are related but contain different information"
                )

        # Case 4: Unrelated - keep both
        return UpdateDecision(
            action="keep_both",
            reason="Memories are unrelated"
        )

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings.

        Uses token-based Jaccard similarity with some normalization.
        """
        # Normalize
        t1 = text1.lower().strip()
        t2 = text2.lower().strip()

        # Exact match
        if t1 == t2:
            return 1.0

        # Tokenize
        tokens1 = set(re.findall(r'\w+', t1))
        tokens2 = set(re.findall(r'\w+', t2))

        if not tokens1 or not tokens2:
            return 0.0

        # Jaccard similarity
        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)

        return len(intersection) / len(union)

    def _detect_conflict(self, text1: str, text2: str) -> bool:
        """Detect if two memories conflict (contradict each other)."""
        # Keywords indicating negation or change
        negation_patterns = [
            r'\bnot\b', r'\bno\b', r'\bnever\b', r'\bdon\'t\b', r'\bdoesn\'t\b',
            r'\bno longer\b', r'\bused to\b', r'\bchanged\b', r'\bdifferent\b',
            r'\binstead\b', r'\bnow\b', r'\bactually\b'
        ]

        t1_lower = text1.lower()
        t2_lower = text2.lower()

        # Check if texts are similar but one contains negation
        similarity = self._calculate_similarity(text1, text2)
        if similarity > 0.6:
            for pattern in negation_patterns:
                if re.search(pattern, t2_lower) and not re.search(pattern, t1_lower):
                    return True

        # LLM-based conflict detection if available
        if self.llm and similarity > 0.7:
            return self._llm_detect_conflict(text1, text2)

        return False

    def _llm_detect_conflict(self, text1: str, text2: str) -> bool:
        """Use LLM to detect if two statements conflict."""
        prompt = f"""Analyze if these two statements contradict each other.
Answer only "YES" or "NO".

Statement 1: {text1}
Statement 2: {text2}

Do these statements contradict? Answer YES or NO:"""

        try:
            response = self.llm.generate(prompt, max_tokens=10)
            return "yes" in response.lower()
        except Exception as e:
            logger.warning(f"LLM conflict detection failed: {e}")
            return False

    def _resolve_conflict(self, existing: Memory, new_text: str) -> UpdateDecision:
        """Resolve conflicting memories using recency and confidence."""
        # Favor newer information by default (recency bias)
        # But consider confidence scores

        # Calculate age in days
        age_days = (datetime.now(timezone.utc) - existing.updated_at).days

        # If existing memory is recent (< 7 days) and high confidence
        if age_days < 7 and existing.confidence > 0.8:
            # Keep existing, but note the conflict in metadata
            return UpdateDecision(
                action="skip",
                reason=f"Recent memory conflicts with new info (age: {age_days}d, conf: {existing.confidence:.2f})",
                confidence_adjustment=-0.1  # Lower confidence due to conflict
            )

        # If existing memory is old or low confidence, update
        return UpdateDecision(
            action="update",
            reason=f"Updating with newer information (old age: {age_days}d, conf: {existing.confidence:.2f})",
            merged_memory=self._create_updated_memory(existing, new_text)
        )

    def _decide_merge_or_update(self, existing: Memory, new_text: str) -> UpdateDecision:
        """Decide whether to merge or replace similar memories."""
        # If new text contains existing text (expansion)
        if existing.text.lower() in new_text.lower():
            return UpdateDecision(
                action="update",
                reason="New memory expands on existing memory",
                merged_memory=self._create_updated_memory(existing, new_text)
            )

        # If existing contains new (new is subset)
        if new_text.lower() in existing.text.lower():
            return UpdateDecision(
                action="skip",
                reason="Existing memory already contains this information",
                confidence_adjustment=0.05
            )

        # Otherwise merge
        merged_text = self._merge_texts(existing.text, new_text)
        merged_memory = self._create_merged_memory(existing, merged_text)

        return UpdateDecision(
            action="merge",
            reason="Merging complementary information",
            merged_memory=merged_memory
        )

    def _merge_texts(self, text1: str, text2: str) -> str:
        """Intelligently merge two text strings."""
        # Simple merge: combine unique sentences
        sentences1 = set(s.strip() for s in text1.split('.') if s.strip())
        sentences2 = set(s.strip() for s in text2.split('.') if s.strip())

        all_sentences = sentences1.union(sentences2)
        return '. '.join(sorted(all_sentences, key=len, reverse=True)) + '.'

    def _create_updated_memory(self, existing: Memory, new_text: str) -> Memory:
        """Create updated version of memory."""
        updated = existing.model_copy(deep=True)
        updated.text = new_text
        updated.updated_at = datetime.now(timezone.utc)
        updated.access_count += 1
        updated.calculate_size_metrics()

        # Adjust confidence based on update
        updated.confidence = min(0.95, updated.confidence + 0.05)

        return updated

    def _create_merged_memory(self, existing: Memory, merged_text: str) -> Memory:
        """Create merged version of memory."""
        merged = existing.model_copy(deep=True)
        merged.text = merged_text
        merged.updated_at = datetime.now(timezone.utc)
        merged.access_count += 1
        merged.calculate_size_metrics()

        # Higher confidence for merged memories (more evidence)
        merged.confidence = min(0.95, existing.confidence + 0.1)

        return merged

    def refine_memory(self, memory: Memory, context: Optional[str] = None) -> Memory:
        """Refine memory quality using LLM.

        Args:
            memory: Memory to refine
            context: Optional context for refinement

        Returns:
            Refined memory
        """
        if not self.llm:
            return memory  # Can't refine without LLM

        prompt = f"""Refine this memory to be more clear, complete, and useful.
Keep it concise but ensure all important information is preserved.

Original memory: {memory.text}
Memory type: {memory.type}
{f"Context: {context}" if context else ""}

Refined memory:"""

        try:
            refined_text = self.llm.generate(prompt, max_tokens=100).strip()

            # Only update if refinement is significantly different and longer
            if len(refined_text) > len(memory.text) * 0.8:
                refined = memory.model_copy(deep=True)
                refined.text = refined_text
                refined.updated_at = datetime.now(timezone.utc)
                refined.calculate_size_metrics()
                refined.confidence = min(0.95, memory.confidence + 0.05)
                return refined

        except Exception as e:
            logger.warning(f"Memory refinement failed: {e}")

        return memory

    def update_confidence(
        self,
        memory: Memory,
        interaction_type: Literal["access", "validation", "conflict", "reinforcement"]
    ) -> Memory:
        """Update memory confidence based on interactions.

        Args:
            memory: Memory to update
            interaction_type: Type of interaction

        Returns:
            Memory with updated confidence
        """
        updated = memory.model_copy(deep=True)

        # Confidence adjustments by interaction type
        adjustments = {
            "access": 0.01,  # Small boost for being accessed
            "validation": 0.1,  # Large boost for being confirmed/used
            "conflict": -0.15,  # Penalty for conflicts
            "reinforcement": 0.05,  # Medium boost for being reinforced
        }

        adjustment = adjustments.get(interaction_type, 0.0)
        updated.confidence = max(0.1, min(1.0, memory.confidence + adjustment))

        # Decay confidence for very old memories without access
        age_days = (datetime.now(timezone.utc) - memory.updated_at).days
        if age_days > 90 and memory.access_count < 2:
            updated.confidence *= 0.95

        updated.updated_at = datetime.now(timezone.utc)
        return updated

    def reconcile_memories(
        self,
        memories: List[Memory],
        user_id: str
    ) -> List[Memory]:
        """Reconcile a group of related memories, resolving conflicts.

        Args:
            memories: List of related memories
            user_id: User ID

        Returns:
            Reconciled list of memories
        """
        if len(memories) <= 1:
            return memories

        reconciled = []
        processed = set()

        # Sort by confidence and recency
        sorted_memories = sorted(
            memories,
            key=lambda m: (m.confidence, m.updated_at),
            reverse=True
        )

        for i, mem1 in enumerate(sorted_memories):
            if i in processed:
                continue

            # Find conflicts with this memory
            has_conflict = False
            for j, mem2 in enumerate(sorted_memories[i+1:], start=i+1):
                if j in processed:
                    continue

                if self._detect_conflict(mem1.text, mem2.text):
                    has_conflict = True
                    # Resolve: keep higher confidence memory
                    if mem1.confidence >= mem2.confidence:
                        processed.add(j)
                        # Update confidence of winner
                        mem1 = self.update_confidence(mem1, "reinforcement")
                    else:
                        processed.add(i)
                        mem2 = self.update_confidence(mem2, "reinforcement")
                        reconciled.append(mem2)
                        break

            if not has_conflict or i not in processed:
                reconciled.append(mem1)
                processed.add(i)

        return reconciled
