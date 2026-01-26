"""Models for context assembly."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class DropReason(str, Enum):
    """Reasons why a memory was dropped from context."""

    TOKEN_BUDGET = "token_budget"
    LOW_RELEVANCE = "low_relevance"
    DUPLICATE = "duplicate"
    EXPIRED = "expired"
    FILTERED = "filtered"
    SUMMARIZED = "summarized"


class SelectedItem(BaseModel):
    """A memory item selected for the context pack."""

    memory_id: str
    text: str
    memory_type: str
    relevance_score: float
    importance: float
    created_at: datetime
    token_count: int
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DroppedItem(BaseModel):
    """A memory item that was dropped from context."""

    memory_id: str
    text_preview: str  # First 100 chars
    reason: DropReason
    relevance_score: Optional[float] = None
    details: str = ""


class ContextConstraints(BaseModel):
    """Constraints for context assembly.

    Attributes:
        token_budget: Maximum tokens for the context (default: 4000)
        max_items: Maximum number of memory items (default: 20)
        recency_bias: Weight for recent memories (0-1, default: 0.3)
        entity_focus: Optional list of entities to prioritize
        type_filter: Optional list of memory types to include
        min_relevance: Minimum relevance score (0-1, default: 0.1)
        allow_summaries: Allow summarization when budget exceeded (default: True)
        include_citations: Include memory IDs as citations (default: True)
        deduplicate: Remove duplicate/similar memories (default: True)
        time_range_days: Optional limit to memories from last N days
    """

    token_budget: int = Field(default=4000, ge=100, le=100000)
    max_items: int = Field(default=20, ge=1, le=100)
    recency_bias: float = Field(default=0.3, ge=0.0, le=1.0)
    entity_focus: Optional[list[str]] = None
    type_filter: Optional[list[str]] = None
    min_relevance: float = Field(default=0.1, ge=0.0, le=1.0)
    allow_summaries: bool = True
    include_citations: bool = True
    deduplicate: bool = True
    time_range_days: Optional[int] = None


class ContextPack(BaseModel):
    """Assembled context pack ready for LLM injection.

    Attributes:
        final_context_text: The assembled context string
        citations: List of memory IDs used (for attribution)
        selected_items: Structured list of selected memories
        dropped_items: Items that were dropped (for debugging)
        total_tokens: Estimated token count
        query: Original query used for assembly
        user_id: User ID
        session_id: Optional session ID
        constraints: Constraints used for assembly
        assembled_at: Timestamp of assembly
        metadata: Additional metadata
    """

    final_context_text: str
    citations: list[str] = Field(default_factory=list)
    selected_items: list[SelectedItem] = Field(default_factory=list)
    dropped_items: list[DroppedItem] = Field(default_factory=list)
    total_tokens: int = 0
    query: str
    user_id: str
    session_id: Optional[str] = None
    constraints: ContextConstraints = Field(default_factory=ContextConstraints)
    assembled_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)

    def get_context_for_prompt(self, include_header: bool = True) -> str:
        """Get context formatted for prompt injection.

        Args:
            include_header: Include a header section

        Returns:
            Formatted context string
        """
        if not include_header:
            return self.final_context_text

        header = "## Relevant Context\n\n"
        return header + self.final_context_text

    def get_citations_text(self) -> str:
        """Get citations as a formatted string."""
        if not self.citations:
            return ""
        return "Sources: " + ", ".join(self.citations)
