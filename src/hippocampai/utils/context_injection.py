"""Context injection utilities for LLM prompts."""

import logging
from typing import Optional

from hippocampai.models.memory import Memory, RetrievalResult

logger = logging.getLogger(__name__)


class ContextInjector:
    """Helper for injecting memories into LLM prompts."""

    def __init__(
        self,
        max_tokens: int = 2000,
        template: str = "default",
        include_metadata: bool = False,
    ):
        """
        Initialize context injector.

        Args:
            max_tokens: Approximate maximum tokens for context
            template: Template style ('default', 'minimal', 'detailed')
            include_metadata: Include memory metadata in context
        """
        self.max_tokens = max_tokens
        self.template = template
        self.include_metadata = include_metadata

    def inject_memories(
        self,
        prompt: str,
        memories: list[Memory],
        position: str = "prefix",  # 'prefix' or 'suffix'
    ) -> str:
        """
        Inject memories into a prompt.

        Args:
            prompt: Original prompt
            memories: List of Memory objects
            position: Where to inject ('prefix' or 'suffix')

        Returns:
            Prompt with injected context
        """
        if not memories:
            return prompt

        context = self._format_memories(memories)

        if position == "prefix":
            return f"{context}\n\n{prompt}"
        return f"{prompt}\n\n{context}"

    def inject_retrieval_results(
        self,
        prompt: str,
        results: list[RetrievalResult],
        position: str = "prefix",
        include_scores: bool = False,
    ) -> str:
        """
        Inject retrieval results into a prompt.

        Args:
            prompt: Original prompt
            results: List of RetrievalResult objects
            position: Where to inject
            include_scores: Include relevance scores

        Returns:
            Prompt with injected context
        """
        if not results:
            return prompt

        context = self._format_retrieval_results(results, include_scores)

        if position == "prefix":
            return f"{context}\n\n{prompt}"
        return f"{prompt}\n\n{context}"

    def _format_memories(self, memories: list[Memory]) -> str:
        """Format memories into context string."""
        if self.template == "minimal":
            return self._format_minimal(memories)
        if self.template == "detailed":
            return self._format_detailed(memories)
        return self._format_default(memories)

    def _format_default(self, memories: list[Memory]) -> str:
        """Default formatting template."""
        lines = ["## Relevant Context\n"]

        for i, mem in enumerate(memories, 1):
            lines.append(f"{i}. {mem.text}")

            if self.include_metadata:
                metadata_parts = []
                if mem.tags:
                    metadata_parts.append(f"Tags: {', '.join(mem.tags)}")
                if mem.importance:
                    metadata_parts.append(f"Importance: {mem.importance:.1f}/10")

                if metadata_parts:
                    lines.append(f"   ({' | '.join(metadata_parts)})")

        return "\n".join(lines)

    def _format_minimal(self, memories: list[Memory]) -> str:
        """Minimal formatting - just the text."""
        return "Context: " + " | ".join([mem.text for mem in memories])

    def _format_detailed(self, memories: list[Memory]) -> str:
        """Detailed formatting with full metadata."""
        lines = ["## Relevant Context (Detailed)\n"]

        for i, mem in enumerate(memories, 1):
            lines.append(f"### Memory {i}")
            lines.append(f"**Text:** {mem.text}")
            lines.append(f"**Type:** {mem.type.value}")
            lines.append(f"**Importance:** {mem.importance:.1f}/10")

            if mem.tags:
                lines.append(f"**Tags:** {', '.join(mem.tags)}")

            if mem.created_at:
                lines.append(f"**Created:** {mem.created_at.strftime('%Y-%m-%d')}")

            lines.append("")  # Blank line between memories

        return "\n".join(lines)

    def _format_retrieval_results(
        self, results: list[RetrievalResult], include_scores: bool
    ) -> str:
        """Format retrieval results into context string."""
        lines = ["## Relevant Memories\n"]

        for i, result in enumerate(results, 1):
            mem = result.memory
            score_str = f" (relevance: {result.score:.2f})" if include_scores else ""
            lines.append(f"{i}. {mem.text}{score_str}")

            if self.include_metadata and mem.tags:
                lines.append(f"   Tags: {', '.join(mem.tags)}")

        return "\n".join(lines)

    def create_prompt_with_history(
        self,
        current_query: str,
        conversation_history: list[dict[str, str]],
        memories: list[Memory],
        max_history_turns: int = 5,
    ) -> str:
        """
        Create prompt with conversation history and memories.

        Args:
            current_query: Current user query
            conversation_history: List of {"role": "...", "content": "..."} dicts
            memories: Relevant memories
            max_history_turns: Maximum conversation turns to include

        Returns:
            Complete prompt string
        """
        # Add memories as context
        context = self._format_memories(memories)

        # Add conversation history
        history_lines = []
        recent_history = conversation_history[-max_history_turns:]

        for turn in recent_history:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            history_lines.append(f"{role.capitalize()}: {content}")

        history_str = "\n".join(history_lines) if history_lines else ""

        # Combine everything
        parts = []
        if context:
            parts.append(context)
        if history_str:
            parts.append(f"## Conversation History\n{history_str}")
        parts.append(f"## Current Query\n{current_query}")

        return "\n\n".join(parts)

    def estimate_tokens(self, text: str) -> int:
        """
        Rough token estimation (4 chars ≈ 1 token).

        For production, use tiktoken or similar.
        """
        return len(text) // 4

    def truncate_to_token_limit(self, memories: list[Memory], current_prompt: str) -> list[Memory]:
        """
        Truncate memories to fit within token limit.

        Args:
            memories: List of memories to truncate
            current_prompt: The prompt they'll be injected into

        Returns:
            Truncated list of memories
        """
        prompt_tokens = self.estimate_tokens(current_prompt)
        available_tokens = self.max_tokens - prompt_tokens

        if available_tokens <= 0:
            logger.warning("No tokens available for context injection")
            return []

        selected = []
        current_tokens = 0

        for mem in memories:
            mem_context = self._format_memories([mem])
            mem_tokens = self.estimate_tokens(mem_context)

            if current_tokens + mem_tokens <= available_tokens:
                selected.append(mem)
                current_tokens += mem_tokens
            else:
                break

        logger.debug(
            f"Selected {len(selected)}/{len(memories)} memories "
            f"({current_tokens}/{available_tokens} tokens)"
        )

        return selected


def inject_context(
    prompt: str,
    memories: list[Memory],
    max_memories: Optional[int] = None,
    template: str = "default",
) -> str:
    """
    Quick helper function for context injection.

    Args:
        prompt: Original prompt
        memories: Memories to inject
        max_memories: Maximum memories to include
        template: Formatting template

    Returns:
        Prompt with injected context
    """
    if max_memories:
        memories = memories[:max_memories]

    injector = ContextInjector(template=template)
    return injector.inject_memories(prompt, memories)
