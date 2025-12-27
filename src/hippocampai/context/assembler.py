"""Automated context assembly for LLM prompts.

This module provides intelligent context assembly that:
1. Retrieves relevant memories (vector/BM25/hybrid)
2. Re-ranks using cross-encoder
3. Deduplicates similar content
4. Applies temporal filters
5. Compresses/summarizes when token budget exceeded
6. Outputs a ready-to-use ContextPack
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Optional

from hippocampai.context.models import (
    ContextConstraints,
    ContextPack,
    DroppedItem,
    DropReason,
    SelectedItem,
)
from hippocampai.models.memory import Memory, RetrievalResult

if TYPE_CHECKING:
    from hippocampai.client import MemoryClient

logger = logging.getLogger(__name__)


class ContextAssembler:
    """Assembles context packs from memories for LLM prompts.

    This class provides automated context assembly that agents can use
    instead of manually picking memory items.
    """

    def __init__(
        self,
        client: "MemoryClient",
        default_constraints: Optional[ContextConstraints] = None,
    ) -> None:
        """Initialize context assembler.

        Args:
            client: MemoryClient instance
            default_constraints: Default constraints for assembly
        """
        self.client = client
        self.default_constraints = default_constraints or ContextConstraints()

    def assemble_context(
        self,
        query: str,
        user_id: str,
        session_id: Optional[str] = None,
        constraints: Optional[ContextConstraints] = None,
    ) -> ContextPack:
        """Assemble a context pack for the given query.

        This is the main entry point for context assembly. It:
        1. Retrieves candidate memories
        2. Re-ranks by relevance
        3. Deduplicates
        4. Applies filters
        5. Fits to token budget (with optional summarization)
        6. Returns a ready-to-use ContextPack

        Args:
            query: User query to find relevant context for
            user_id: User ID
            session_id: Optional session ID for session-scoped context
            constraints: Optional constraints (uses defaults if not provided)

        Returns:
            ContextPack with assembled context

        Example:
            ```python
            assembler = ContextAssembler(client)
            pack = assembler.assemble_context(
                query="What are my coffee preferences?",
                user_id="alice",
                constraints=ContextConstraints(token_budget=2000),
            )
            prompt = f"{pack.final_context_text}\\n\\nUser: {query}"
            ```
        """
        constraints = constraints or self.default_constraints

        # Step 1: Retrieve candidates
        candidates = self._retrieve_candidates(
            query=query,
            user_id=user_id,
            session_id=session_id,
            constraints=constraints,
        )

        if not candidates:
            return self._empty_pack(query, user_id, session_id, constraints)

        # Step 2: Apply filters
        filtered, dropped_filtered = self._apply_filters(candidates, constraints)

        # Step 3: Deduplicate
        if constraints.deduplicate:
            deduped, dropped_dedup = self._deduplicate(filtered)
        else:
            deduped = filtered
            dropped_dedup = []

        # Step 4: Apply recency bias and re-score
        scored = self._apply_recency_bias(deduped, constraints.recency_bias)

        # Step 5: Sort by final score
        scored.sort(key=lambda x: x[1], reverse=True)

        # Step 6: Fit to token budget
        selected, dropped_budget = self._fit_to_budget(
            scored, constraints, query
        )

        # Combine all dropped items
        all_dropped = dropped_filtered + dropped_dedup + dropped_budget

        # Step 7: Build context pack
        return self._build_pack(
            selected=selected,
            dropped=all_dropped,
            query=query,
            user_id=user_id,
            session_id=session_id,
            constraints=constraints,
        )

    def _retrieve_candidates(
        self,
        query: str,
        user_id: str,
        session_id: Optional[str],
        constraints: ContextConstraints,
    ) -> list[RetrievalResult]:
        """Retrieve candidate memories."""
        # Build filters
        filters: dict[str, Any] = {}

        if constraints.type_filter:
            filters["type"] = constraints.type_filter

        if constraints.time_range_days:
            # Note: This would need to be implemented in the retriever
            # For now, we'll filter post-retrieval
            pass

        # Retrieve more than needed to allow for filtering
        k = min(constraints.max_items * 3, 100)

        try:
            results = self.client.recall(
                query=query,
                user_id=user_id,
                session_id=session_id,
                k=k,
                filters=filters if filters else None,
            )
            # Ensure we return a list
            if isinstance(results, list):
                return results
            return []
        except Exception as e:
            logger.error(f"Failed to retrieve candidates: {e}")
            return []

    def _apply_filters(
        self,
        candidates: list[RetrievalResult],
        constraints: ContextConstraints,
    ) -> tuple[list[RetrievalResult], list[DroppedItem]]:
        """Apply filters to candidates."""
        filtered = []
        dropped = []

        now = datetime.now(timezone.utc)
        time_cutoff = None
        if constraints.time_range_days:
            time_cutoff = now - timedelta(days=constraints.time_range_days)

        for result in candidates:
            memory = result.memory

            # Check relevance threshold
            if result.score < constraints.min_relevance:
                dropped.append(DroppedItem(
                    memory_id=memory.id,
                    text_preview=memory.text[:100],
                    reason=DropReason.LOW_RELEVANCE,
                    relevance_score=result.score,
                    details=f"Score {result.score:.3f} < threshold {constraints.min_relevance}",
                ))
                continue

            # Check expiration
            if memory.is_expired():
                dropped.append(DroppedItem(
                    memory_id=memory.id,
                    text_preview=memory.text[:100],
                    reason=DropReason.EXPIRED,
                    relevance_score=result.score,
                ))
                continue

            # Check time range
            if time_cutoff and memory.created_at < time_cutoff:
                dropped.append(DroppedItem(
                    memory_id=memory.id,
                    text_preview=memory.text[:100],
                    reason=DropReason.FILTERED,
                    relevance_score=result.score,
                    details=f"Created {memory.created_at} before cutoff {time_cutoff}",
                ))
                continue

            # Check type filter
            if constraints.type_filter:
                mem_type = memory.type.value if hasattr(memory.type, 'value') else str(memory.type)
                if mem_type not in constraints.type_filter:
                    dropped.append(DroppedItem(
                        memory_id=memory.id,
                        text_preview=memory.text[:100],
                        reason=DropReason.FILTERED,
                        relevance_score=result.score,
                        details=f"Type {mem_type} not in {constraints.type_filter}",
                    ))
                    continue

            filtered.append(result)

        return filtered, dropped

    def _deduplicate(
        self,
        candidates: list[RetrievalResult],
    ) -> tuple[list[RetrievalResult], list[DroppedItem]]:
        """Remove duplicate/very similar memories."""
        if not candidates:
            return [], []

        deduped = []
        dropped = []
        seen_texts: set[str] = set()

        for result in candidates:
            # Simple text-based dedup (normalized)
            normalized = result.memory.text.lower().strip()

            # Check for exact or near-exact duplicates
            is_dup = False
            for seen in seen_texts:
                if self._is_similar(normalized, seen):
                    is_dup = True
                    break

            if is_dup:
                dropped.append(DroppedItem(
                    memory_id=result.memory.id,
                    text_preview=result.memory.text[:100],
                    reason=DropReason.DUPLICATE,
                    relevance_score=result.score,
                ))
            else:
                deduped.append(result)
                seen_texts.add(normalized)

        return deduped, dropped

    def _is_similar(self, text1: str, text2: str, threshold: float = 0.9) -> bool:
        """Check if two texts are similar using simple heuristics."""
        # Exact match
        if text1 == text2:
            return True

        # Length-based quick check
        len_ratio = min(len(text1), len(text2)) / max(len(text1), len(text2))
        if len_ratio < 0.5:
            return False

        # Word overlap
        words1 = set(text1.split())
        words2 = set(text2.split())

        if not words1 or not words2:
            return False

        intersection = len(words1 & words2)
        union = len(words1 | words2)
        jaccard = intersection / union

        return jaccard >= threshold

    def _apply_recency_bias(
        self,
        candidates: list[RetrievalResult],
        recency_bias: float,
    ) -> list[tuple[RetrievalResult, float]]:
        """Apply recency bias to scores."""
        if not candidates:
            return []

        now = datetime.now(timezone.utc)
        scored = []

        for result in candidates:
            # Calculate recency score (0-1)
            age_days = (now - result.memory.created_at).total_seconds() / 86400
            recency_score = 1.0 / (1.0 + age_days / 30)  # Half-life of 30 days

            # Combine relevance and recency
            final_score = (
                (1 - recency_bias) * result.score +
                recency_bias * recency_score
            )

            scored.append((result, final_score))

        return scored

    def _fit_to_budget(
        self,
        scored: list[tuple[RetrievalResult, float]],
        constraints: ContextConstraints,
        query: str,
    ) -> tuple[list[SelectedItem], list[DroppedItem]]:
        """Fit memories to token budget."""
        selected = []
        dropped = []
        current_tokens = 0

        for result, score in scored:
            memory = result.memory

            # Estimate tokens
            tokens = self._estimate_tokens(memory.text)

            # Check if we'd exceed budget
            if current_tokens + tokens > constraints.token_budget:
                # Try summarization if allowed
                if constraints.allow_summaries and self.client.llm:
                    summarized = self._try_summarize(memory, constraints.token_budget - current_tokens)
                    if summarized:
                        selected.append(summarized)
                        current_tokens += summarized.token_count
                        continue

                dropped.append(DroppedItem(
                    memory_id=memory.id,
                    text_preview=memory.text[:100],
                    reason=DropReason.TOKEN_BUDGET,
                    relevance_score=score,
                    details=f"Would exceed budget: {current_tokens + tokens} > {constraints.token_budget}",
                ))
                continue

            # Check max items
            if len(selected) >= constraints.max_items:
                dropped.append(DroppedItem(
                    memory_id=memory.id,
                    text_preview=memory.text[:100],
                    reason=DropReason.TOKEN_BUDGET,
                    relevance_score=score,
                    details=f"Max items reached: {constraints.max_items}",
                ))
                continue

            # Add to selected
            mem_type = memory.type.value if hasattr(memory.type, 'value') else str(memory.type)
            selected.append(SelectedItem(
                memory_id=memory.id,
                text=memory.text,
                memory_type=mem_type,
                relevance_score=score,
                importance=memory.importance,
                created_at=memory.created_at,
                token_count=tokens,
                tags=memory.tags,
                metadata=memory.metadata,
            ))
            current_tokens += tokens

        return selected, dropped

    def _try_summarize(
        self,
        memory: Memory,
        available_tokens: int,
    ) -> Optional[SelectedItem]:
        """Try to summarize a memory to fit in available tokens."""
        if available_tokens < 50:  # Too little space
            return None

        try:
            # Use existing summarizer if available
            if hasattr(self.client, 'summarizer') and self.client.summarizer:
                # Simple summarization prompt
                target_length = available_tokens * 4  # Rough chars estimate
                prompt = f"Summarize in under {target_length} characters: {memory.text}"

                if self.client.llm:
                    summary = self.client.llm.generate(prompt, max_tokens=available_tokens)
                    summary = summary.strip()

                    tokens = self._estimate_tokens(summary)
                    if tokens <= available_tokens:
                        mem_type = memory.type.value if hasattr(memory.type, 'value') else str(memory.type)
                        return SelectedItem(
                            memory_id=memory.id,
                            text=f"[Summarized] {summary}",
                            memory_type=mem_type,
                            relevance_score=0.0,  # Unknown after summarization
                            importance=memory.importance,
                            created_at=memory.created_at,
                            token_count=tokens,
                            tags=memory.tags,
                            metadata={**memory.metadata, "summarized": True},
                        )
        except Exception as e:
            logger.warning(f"Summarization failed: {e}")

        return None

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text.

        Uses a conservative heuristic: ~4 characters per token.
        For production, consider using tiktoken.
        """
        return len(text) // 4 + 1

    def _build_pack(
        self,
        selected: list[SelectedItem],
        dropped: list[DroppedItem],
        query: str,
        user_id: str,
        session_id: Optional[str],
        constraints: ContextConstraints,
    ) -> ContextPack:
        """Build the final context pack."""
        # Build context text
        lines = []
        for i, item in enumerate(selected, 1):
            lines.append(f"{i}. {item.text}")

        context_text = "\n".join(lines)

        # Build citations
        citations = [item.memory_id for item in selected] if constraints.include_citations else []

        # Calculate total tokens
        total_tokens = sum(item.token_count for item in selected)

        return ContextPack(
            final_context_text=context_text,
            citations=citations,
            selected_items=selected,
            dropped_items=dropped,
            total_tokens=total_tokens,
            query=query,
            user_id=user_id,
            session_id=session_id,
            constraints=constraints,
            metadata={
                "candidates_retrieved": len(selected) + len(dropped),
                "selected_count": len(selected),
                "dropped_count": len(dropped),
            },
        )

    def _empty_pack(
        self,
        query: str,
        user_id: str,
        session_id: Optional[str],
        constraints: ContextConstraints,
    ) -> ContextPack:
        """Create an empty context pack."""
        return ContextPack(
            final_context_text="",
            citations=[],
            selected_items=[],
            dropped_items=[],
            total_tokens=0,
            query=query,
            user_id=user_id,
            session_id=session_id,
            constraints=constraints,
            metadata={"empty": True, "reason": "no_candidates"},
        )
