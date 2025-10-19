"""Memory consolidator for periodic cleanup."""

import logging
from typing import List, Optional
from hippocampai.models.memory import Memory
from hippocampai.adapters.llm_base import BaseLLM

logger = logging.getLogger(__name__)


class MemoryConsolidator:
    """Consolidate similar memories using LLM or heuristics."""

    CONSOLIDATE_PROMPT = """Merge these similar memories into ONE concise memory. Return only the merged text.

Memories:
{memories}

Merged memory:"""

    def __init__(self, llm: Optional[BaseLLM] = None):
        self.llm = llm

    def consolidate(self, memories: List[Memory]) -> Optional[Memory]:
        """
        Consolidate multiple memories into one.

        Returns:
            Consolidated memory or None
        """
        if len(memories) < 2:
            return None

        if self.llm:
            return self._consolidate_llm(memories)
        else:
            return self._consolidate_heuristic(memories)

    def _consolidate_llm(self, memories: List[Memory]) -> Optional[Memory]:
        """LLM-based consolidation."""
        try:
            mem_texts = "\n".join([f"- {m.text}" for m in memories])
            prompt = self.CONSOLIDATE_PROMPT.format(memories=mem_texts[:1000])
            merged_text = self.llm.generate(prompt, max_tokens=256, temperature=0.0)

            if not merged_text:
                return None

            # Use first memory as template
            base = memories[0]
            return Memory(
                text=merged_text.strip(),
                user_id=base.user_id,
                session_id=base.session_id,
                type=base.type,
                importance=max(m.importance for m in memories),
                confidence=min(m.confidence for m in memories),
                tags=list(set(sum([m.tags for m in memories], []))),
                access_count=sum(m.access_count for m in memories),
                metadata={"consolidated_from": [m.id for m in memories]}
            )

        except Exception as e:
            logger.error(f"LLM consolidation failed: {e}")
            return None

    def _consolidate_heuristic(self, memories: List[Memory]) -> Optional[Memory]:
        """Simple heuristic: pick highest importance."""
        best = max(memories, key=lambda m: m.importance)
        return Memory(
            text=best.text,
            user_id=best.user_id,
            session_id=best.session_id,
            type=best.type,
            importance=max(m.importance for m in memories),
            confidence=min(m.confidence for m in memories),
            tags=list(set(sum([m.tags for m in memories], []))),
            access_count=sum(m.access_count for m in memories),
            metadata={"consolidated_from": [m.id for m in memories]}
        )
