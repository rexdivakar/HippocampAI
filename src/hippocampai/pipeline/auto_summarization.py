"""Automatic memory summarization and compression system.

This module provides:
- Recursive/hierarchical summarization at memory level
- Sliding window compression for long conversations
- Automatic memory consolidation with scheduling
- Importance decay and intelligent pruning
- Token budget management
"""

import logging
from collections import defaultdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from hippocampai.adapters.llm_base import BaseLLM
from hippocampai.models.memory import Memory

logger = logging.getLogger(__name__)


class CompressionLevel(str, Enum):
    """Memory compression levels."""

    NONE = "none"  # No compression
    LIGHT = "light"  # Minor compression (80% of original)
    MEDIUM = "medium"  # Moderate compression (50% of original)
    HEAVY = "heavy"  # Aggressive compression (25% of original)
    ARCHIVAL = "archival"  # Maximum compression (10% of original)


class MemoryTier(str, Enum):
    """Memory storage tiers based on access patterns."""

    HOT = "hot"  # Frequently accessed, kept verbatim
    WARM = "warm"  # Occasionally accessed, lightly compressed
    COLD = "cold"  # Rarely accessed, heavily compressed
    ARCHIVED = "archived"  # Very old, maximum compression


class SummarizedMemory(BaseModel):
    """Represents a summarized/compressed memory."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    original_memory_ids: list[str] = Field(default_factory=list)
    summary_text: str
    compression_level: CompressionLevel
    tier: MemoryTier
    original_token_count: int = 0
    compressed_token_count: int = 0
    compression_ratio: float = 0.0
    user_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    access_count: int = 0
    importance_score: float = 5.0
    metadata: dict[str, Any] = Field(default_factory=dict)

    def calculate_compression_ratio(self):
        """Calculate actual compression ratio."""
        if self.original_token_count > 0:
            self.compression_ratio = self.compressed_token_count / self.original_token_count


class HierarchicalSummary(BaseModel):
    """Hierarchical summary structure for recursive summarization."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    level: int  # 0 = leaf (original), 1+ = summary levels
    parent_id: Optional[str] = None
    child_ids: list[str] = Field(default_factory=list)
    content: str
    memory_ids: list[str] = Field(default_factory=list)
    user_id: str
    token_count: int = 0
    importance_score: float = 5.0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)


class AutoSummarizer:
    """Automatic memory summarization and compression engine."""

    def __init__(
        self,
        llm: Optional[BaseLLM] = None,
        hot_threshold_days: int = 7,
        warm_threshold_days: int = 30,
        cold_threshold_days: int = 90,
        hot_access_count: int = 10,
        max_tokens_per_summary: int = 150,
        hierarchical_batch_size: int = 5,
    ):
        """Initialize auto-summarizer.

        Args:
            llm: Language model for summarization
            hot_threshold_days: Days before HOT → WARM transition
            warm_threshold_days: Days before WARM → COLD transition
            cold_threshold_days: Days before COLD → ARCHIVED transition
            hot_access_count: Access count threshold for HOT tier
            max_tokens_per_summary: Maximum tokens for each summary
            hierarchical_batch_size: Number of memories to batch for hierarchical summarization
        """
        self.llm = llm
        self.hot_threshold_days = hot_threshold_days
        self.warm_threshold_days = warm_threshold_days
        self.cold_threshold_days = cold_threshold_days
        self.hot_access_count = hot_access_count
        self.max_tokens_per_summary = max_tokens_per_summary
        self.hierarchical_batch_size = hierarchical_batch_size

    def determine_memory_tier(
        self, memory: Memory, current_time: Optional[datetime] = None
    ) -> MemoryTier:
        """Determine appropriate storage tier for a memory.

        Args:
            memory: Memory to evaluate
            current_time: Current timestamp (defaults to now)

        Returns:
            MemoryTier classification
        """
        if current_time is None:
            current_time = datetime.now(timezone.utc)

        # Calculate age
        age_days = (current_time - memory.created_at).days

        # HOT tier: frequently accessed or very recent
        if memory.access_count >= self.hot_access_count or age_days <= self.hot_threshold_days:
            return MemoryTier.HOT

        # WARM tier: moderate age and access
        if age_days <= self.warm_threshold_days and memory.access_count >= 3:
            return MemoryTier.WARM

        # COLD tier: older but not ancient
        if age_days <= self.cold_threshold_days:
            return MemoryTier.COLD

        # ARCHIVED: very old
        return MemoryTier.ARCHIVED

    def get_compression_level_for_tier(self, tier: MemoryTier) -> CompressionLevel:
        """Map memory tier to compression level.

        Args:
            tier: Memory tier

        Returns:
            Appropriate compression level
        """
        tier_to_compression = {
            MemoryTier.HOT: CompressionLevel.NONE,
            MemoryTier.WARM: CompressionLevel.LIGHT,
            MemoryTier.COLD: CompressionLevel.MEDIUM,
            MemoryTier.ARCHIVED: CompressionLevel.HEAVY,
        }
        return tier_to_compression.get(tier, CompressionLevel.NONE)

    def compress_memory(
        self, memory: Memory, compression_level: CompressionLevel
    ) -> Optional[SummarizedMemory]:
        """Compress a single memory based on compression level.

        Args:
            memory: Memory to compress
            compression_level: Desired compression level

        Returns:
            SummarizedMemory or None if compression fails
        """
        if compression_level == CompressionLevel.NONE:
            return None

        # Get target token ratio
        target_ratios = {
            CompressionLevel.LIGHT: 0.8,
            CompressionLevel.MEDIUM: 0.5,
            CompressionLevel.HEAVY: 0.25,
            CompressionLevel.ARCHIVAL: 0.1,
        }
        target_ratio = target_ratios.get(compression_level, 0.5)

        original_tokens = Memory.estimate_tokens(memory.text)
        target_tokens = int(original_tokens * target_ratio)

        # Generate compressed version
        if self.llm:
            compressed_text = self._compress_with_llm(memory.text, target_tokens, compression_level)
        else:
            compressed_text = self._compress_heuristic(memory.text, target_ratio)

        compressed_tokens = Memory.estimate_tokens(compressed_text)

        # Determine tier
        tier = self.determine_memory_tier(memory)

        summarized = SummarizedMemory(
            original_memory_ids=[memory.id],
            summary_text=compressed_text,
            compression_level=compression_level,
            tier=tier,
            original_token_count=original_tokens,
            compressed_token_count=compressed_tokens,
            user_id=memory.user_id,
            importance_score=memory.importance,
            metadata={
                "original_type": memory.type.value,
                "tags": memory.tags,
                "session_id": memory.session_id,
            },
        )
        summarized.calculate_compression_ratio()

        return summarized

    def _compress_with_llm(self, text: str, target_tokens: int, level: CompressionLevel) -> str:
        """Compress text using LLM.

        Args:
            text: Text to compress
            target_tokens: Target token count
            level: Compression level

        Returns:
            Compressed text
        """
        if self.llm is None:
            return text

        level_instructions = {
            CompressionLevel.LIGHT: "lightly condense while keeping most details",
            CompressionLevel.MEDIUM: "summarize to half the length, keeping key information",
            CompressionLevel.HEAVY: "create a brief summary with only essential information",
            CompressionLevel.ARCHIVAL: "create an ultra-brief summary with core facts only",
        }

        instruction = level_instructions.get(level, "summarize")

        prompt = f"""Compress the following text. {instruction}. Target length: {target_tokens} tokens.

Original text:
{text[:2000]}

Compressed version:"""

        try:
            response = self.llm.generate(prompt, max_tokens=min(target_tokens * 2, 500))
            return response.strip() if response else text
        except Exception as e:
            logger.warning(f"LLM compression failed: {e}, using heuristic")
            return self._compress_heuristic(text, 0.5)

    def _compress_heuristic(self, text: str, target_ratio: float) -> str:
        """Heuristic compression by removing filler words and truncating.

        Args:
            text: Text to compress
            target_ratio: Target compression ratio (0.0-1.0)

        Returns:
            Compressed text
        """
        # Common filler words to remove
        filler_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "from",
            "as",
            "is",
            "was",
            "are",
            "were",
            "been",
            "be",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "should",
            "could",
            "may",
            "might",
            "can",
            "very",
            "really",
            "just",
            "quite",
        }

        words = text.split()
        target_word_count = int(len(words) * target_ratio)

        if target_word_count >= len(words):
            return text

        # Remove filler words first
        important_words = [w for w in words if w.lower() not in filler_words]

        # If still too long, truncate to target
        if len(important_words) > target_word_count:
            compressed = important_words[:target_word_count]
        else:
            compressed = important_words

        return " ".join(compressed)

    def create_hierarchical_summary(
        self, memories: list[Memory], user_id: str, max_levels: int = 3
    ) -> list[HierarchicalSummary]:
        """Create hierarchical summary structure from memories.

        Args:
            memories: List of memories to summarize
            user_id: User identifier
            max_levels: Maximum hierarchy levels

        Returns:
            List of hierarchical summaries at all levels
        """
        if not memories:
            return []

        all_summaries = []

        # Level 0: Original memories (leaf nodes)
        current_level_summaries = []
        for memory in memories:
            leaf_summary = HierarchicalSummary(
                level=0,
                content=memory.text,
                memory_ids=[memory.id],
                user_id=user_id,
                token_count=Memory.estimate_tokens(memory.text),
                importance_score=memory.importance,
                metadata={"original_memory_id": memory.id, "type": memory.type.value},
            )
            current_level_summaries.append(leaf_summary)
            all_summaries.append(leaf_summary)

        # Build higher levels
        for level in range(1, max_levels + 1):
            if len(current_level_summaries) <= 1:
                break

            next_level_summaries = []

            # Batch memories for summarization
            for i in range(0, len(current_level_summaries), self.hierarchical_batch_size):
                batch = current_level_summaries[i : i + self.hierarchical_batch_size]

                # Create summary for this batch
                batch_summary = self._summarize_batch(batch, level, user_id)
                if batch_summary:
                    # Link parent-child relationships
                    for child in batch:
                        child.parent_id = batch_summary.id
                        batch_summary.child_ids.append(child.id)

                    next_level_summaries.append(batch_summary)
                    all_summaries.append(batch_summary)

            current_level_summaries = next_level_summaries

        return all_summaries

    def _summarize_batch(
        self, summaries: list[HierarchicalSummary], level: int, user_id: str
    ) -> Optional[HierarchicalSummary]:
        """Summarize a batch of summaries into higher-level summary.

        Args:
            summaries: List of summaries to combine
            level: Current hierarchy level
            user_id: User identifier

        Returns:
            Higher-level summary or None
        """
        if not summaries:
            return None

        # Collect all content
        combined_content = "\n".join([s.content for s in summaries])
        combined_memory_ids = []
        for s in summaries:
            combined_memory_ids.extend(s.memory_ids)

        # Calculate average importance
        avg_importance = sum(s.importance_score for s in summaries) / len(summaries)

        # Generate summary
        if self.llm:
            summary_text = self._generate_hierarchical_summary_llm(
                summaries, level, combined_content
            )
        else:
            # Fallback: take key sentences from each
            summary_text = self._generate_hierarchical_summary_heuristic(summaries)

        token_count = Memory.estimate_tokens(summary_text)

        return HierarchicalSummary(
            level=level,
            content=summary_text,
            memory_ids=combined_memory_ids,
            user_id=user_id,
            token_count=token_count,
            importance_score=avg_importance,
            metadata={
                "num_children": len(summaries),
                "child_ids": [s.id for s in summaries],
            },
        )

    def _generate_hierarchical_summary_llm(
        self, summaries: list[HierarchicalSummary], level: int, combined_content: str
    ) -> str:
        """Generate hierarchical summary using LLM.

        Args:
            summaries: Summaries to combine
            level: Current level
            combined_content: Combined text from all summaries

        Returns:
            Summary text
        """
        if self.llm is None:
            return combined_content[:500]

        prompt = f"""Create a concise summary combining these {len(summaries)} memory summaries at level {level}.
Extract the most important information and maintain coherence.

Summaries to combine:
{combined_content[:2000]}

Combined summary (max {self.max_tokens_per_summary} tokens):"""

        try:
            response = self.llm.generate(
                prompt, max_tokens=self.max_tokens_per_summary, temperature=0.3
            )
            return response.strip() if response else combined_content[:500]
        except Exception as e:
            logger.warning(f"LLM hierarchical summarization failed: {e}")
            return self._generate_hierarchical_summary_heuristic(summaries)

    def _generate_hierarchical_summary_heuristic(self, summaries: list[HierarchicalSummary]) -> str:
        """Generate hierarchical summary using heuristics.

        Args:
            summaries: Summaries to combine

        Returns:
            Summary text
        """
        # Take first sentence from each summary
        sentences = []
        for summary in summaries:
            # Get first sentence
            first_sentence = summary.content.split(".")[0].strip()
            if first_sentence:
                sentences.append(first_sentence)

        combined = ". ".join(sentences[:5])  # Limit to 5 sentences
        return combined + "."

    def sliding_window_compression(
        self,
        memories: list[Memory],
        window_size: int = 10,
        keep_recent: int = 5,
        user_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Apply sliding window compression to memory list.

        Recent memories are kept verbatim, older ones are progressively compressed.

        Args:
            memories: Ordered list of memories (oldest first)
            window_size: Size of sliding window
            keep_recent: Number of recent memories to keep uncompressed
            user_id: User identifier (optional, inferred from memories if not provided)

        Returns:
            Dictionary with compressed memories and statistics
        """
        if not memories:
            return {
                "compressed_memories": [],
                "recent_memories": [],
                "stats": {
                    "total_memories": 0,
                    "compressed_windows": 0,
                    "recent_uncompressed": 0,
                    "original_tokens": 0,
                    "compressed_tokens": 0,
                    "compression_ratio": 1.0,
                    "tokens_saved": 0,
                },
            }

        # Sort by creation time (oldest first)
        sorted_memories = sorted(memories, key=lambda m: m.created_at)

        # Determine user_id
        if user_id is None and sorted_memories:
            user_id = sorted_memories[0].user_id

        # Keep recent memories verbatim
        recent_memories = sorted_memories[-keep_recent:]
        older_memories = sorted_memories[:-keep_recent]

        compressed_results = []
        original_tokens = 0
        compressed_tokens = 0

        # Process older memories in windows
        for i in range(0, len(older_memories), window_size):
            window = older_memories[i : i + window_size]
            original_tokens += sum(Memory.estimate_tokens(m.text) for m in window)

            # Determine compression level based on age
            oldest_in_window = window[0]
            tier = self.determine_memory_tier(oldest_in_window)
            compression_level = self.get_compression_level_for_tier(tier)

            # Create window summary
            window_summary = self._summarize_memory_window(
                window, compression_level, user_id or "unknown"
            )
            if window_summary:
                compressed_tokens += window_summary.compressed_token_count
                compressed_results.append(window_summary)

        # Add recent memories (uncompressed)
        recent_tokens = sum(Memory.estimate_tokens(m.text) for m in recent_memories)
        original_tokens += recent_tokens
        compressed_tokens += recent_tokens

        # Calculate statistics
        stats = {
            "total_memories": len(memories),
            "compressed_windows": len(compressed_results),
            "recent_uncompressed": len(recent_memories),
            "original_tokens": original_tokens,
            "compressed_tokens": compressed_tokens,
            "compression_ratio": (
                compressed_tokens / original_tokens if original_tokens > 0 else 1.0
            ),
            "tokens_saved": original_tokens - compressed_tokens,
        }

        return {
            "compressed_memories": compressed_results,
            "recent_memories": recent_memories,
            "stats": stats,
        }

    def _summarize_memory_window(
        self, memories: list[Memory], compression_level: CompressionLevel, user_id: str
    ) -> Optional[SummarizedMemory]:
        """Summarize a window of memories.

        Args:
            memories: Memories in window
            compression_level: Compression level to apply
            user_id: User identifier

        Returns:
            Summarized memory or None
        """
        if not memories:
            return None

        # Combine memory texts
        combined_text = " ".join([m.text for m in memories])
        memory_ids = [m.id for m in memories]

        # Calculate average importance
        avg_importance = sum(m.importance for m in memories) / len(memories)

        # Get compression ratio
        target_ratios = {
            CompressionLevel.NONE: 1.0,
            CompressionLevel.LIGHT: 0.8,
            CompressionLevel.MEDIUM: 0.5,
            CompressionLevel.HEAVY: 0.25,
            CompressionLevel.ARCHIVAL: 0.1,
        }
        target_ratio = target_ratios.get(compression_level, 0.5)

        original_tokens = Memory.estimate_tokens(combined_text)
        target_tokens = int(original_tokens * target_ratio)

        # Generate summary
        if self.llm:
            summary_text = self._compress_with_llm(combined_text, target_tokens, compression_level)
        else:
            summary_text = self._compress_heuristic(combined_text, target_ratio)

        compressed_tokens = Memory.estimate_tokens(summary_text)

        summarized = SummarizedMemory(
            original_memory_ids=memory_ids,
            summary_text=summary_text,
            compression_level=compression_level,
            tier=MemoryTier.COLD,  # Windows are typically older memories
            original_token_count=original_tokens,
            compressed_token_count=compressed_tokens,
            user_id=user_id,
            importance_score=avg_importance,
            metadata={
                "window_size": len(memories),
                "window_start": memories[0].created_at.isoformat(),
                "window_end": memories[-1].created_at.isoformat(),
            },
        )
        summarized.calculate_compression_ratio()

        return summarized

    def analyze_compression_opportunities(self, memories: list[Memory]) -> dict[str, Any]:
        """Analyze memories and identify compression opportunities.

        Args:
            memories: List of memories to analyze

        Returns:
            Analysis report with recommendations
        """
        if not memories:
            return {"total_memories": 0, "opportunities": []}

        # Group by tier
        tier_counts = defaultdict(int)
        tier_tokens = defaultdict(int)
        tier_memories = defaultdict(list)

        for memory in memories:
            tier = self.determine_memory_tier(memory)
            tier_counts[tier.value] += 1
            tier_tokens[tier.value] += Memory.estimate_tokens(memory.text)
            tier_memories[tier.value].append(memory)

        # Calculate potential savings
        opportunities = []
        total_tokens = sum(tier_tokens.values())
        potential_compressed_tokens = 0

        for tier_name, tier_value in [
            ("WARM", MemoryTier.WARM),
            ("COLD", MemoryTier.COLD),
            ("ARCHIVED", MemoryTier.ARCHIVED),
        ]:
            if tier_value.value not in tier_counts:
                continue

            compression_level = self.get_compression_level_for_tier(tier_value)
            target_ratios = {
                CompressionLevel.LIGHT: 0.8,
                CompressionLevel.MEDIUM: 0.5,
                CompressionLevel.HEAVY: 0.25,
                CompressionLevel.ARCHIVAL: 0.1,
            }
            ratio = target_ratios.get(compression_level, 1.0)

            tokens = tier_tokens[tier_value.value]
            compressed = int(tokens * ratio)
            potential_compressed_tokens += compressed
            savings = tokens - compressed

            opportunities.append(
                {
                    "tier": tier_name,
                    "memory_count": tier_counts[tier_value.value],
                    "current_tokens": tokens,
                    "compressed_tokens": compressed,
                    "tokens_saved": savings,
                    "compression_level": compression_level.value,
                    "recommended": savings > 100,  # Recommend if saves >100 tokens
                }
            )

        # Calculate overall potential
        hot_tokens = tier_tokens.get(MemoryTier.HOT.value, 0)
        total_after_compression = hot_tokens + potential_compressed_tokens
        overall_savings = total_tokens - total_after_compression

        return {
            "total_memories": len(memories),
            "total_tokens": total_tokens,
            "tier_distribution": dict(tier_counts),
            "opportunities": opportunities,
            "potential_tokens_after_compression": total_after_compression,
            "potential_tokens_saved": overall_savings,
            "compression_ratio": (
                total_after_compression / total_tokens if total_tokens > 0 else 1.0
            ),
            "recommended_action": (
                "Compress older memories" if overall_savings > 500 else "No compression needed yet"
            ),
        }
