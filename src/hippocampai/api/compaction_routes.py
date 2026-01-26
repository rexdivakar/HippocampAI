"""API routes for conversation compaction."""

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/compaction", tags=["compaction"])


# ============================================
# COMPACTION HISTORY STORAGE
# ============================================


@dataclass
class CompactionHistoryEntry:
    """A single compaction history entry."""

    id: str
    user_id: str
    session_id: Optional[str]
    status: str
    started_at: datetime
    completed_at: Optional[datetime]
    input_memories: int
    output_memories: int
    compression_ratio: float
    tokens_saved: int
    dry_run: bool
    error: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


class CompactionHistoryStore:
    """In-memory store for compaction history with size limit."""

    MAX_ENTRIES_PER_USER = 100

    def __init__(self) -> None:
        self._history: dict[str, deque[CompactionHistoryEntry]] = {}

    def add(self, entry: CompactionHistoryEntry) -> None:
        """Add a compaction history entry."""
        if entry.user_id not in self._history:
            self._history[entry.user_id] = deque(maxlen=self.MAX_ENTRIES_PER_USER)
        self._history[entry.user_id].appendleft(entry)

    def get(self, user_id: str, limit: int = 20) -> list[CompactionHistoryEntry]:
        """Get compaction history for a user."""
        if user_id not in self._history:
            return []
        entries = list(self._history[user_id])
        return entries[:limit]

    def add_from_result(self, result: Any) -> None:
        """Add a compaction history entry from a CompactionResult."""
        entry = CompactionHistoryEntry(
            id=result.id,
            user_id=result.user_id,
            session_id=result.session_id,
            status=result.status,
            started_at=result.started_at,
            completed_at=result.completed_at,
            input_memories=result.metrics.input_memories,
            output_memories=result.metrics.output_memories,
            compression_ratio=result.metrics.compression_ratio,
            tokens_saved=result.metrics.tokens_saved,
            dry_run=result.dry_run,
            error=result.error,
        )
        self.add(entry)


# Global history store instance
_compaction_history = CompactionHistoryStore()


# ============================================
# REQUEST/RESPONSE MODELS
# ============================================


class CompactRequest(BaseModel):
    """Request to compact conversations."""

    user_id: str
    session_id: Optional[str] = None
    lookback_hours: int = 168  # 1 week
    min_memories: int = 5
    dry_run: bool = True
    memory_types: Optional[list[str]] = Field(
        default=None, description="Memory types to compact (e.g., ['fact', 'event', 'context'])"
    )


class CompactionMetricsResponse(BaseModel):
    """Compaction metrics."""

    input_memories: int
    input_tokens: int
    input_characters: int
    output_memories: int
    output_tokens: int
    output_characters: int
    compression_ratio: float
    tokens_saved: int
    memories_merged: int
    clusters_found: int
    llm_calls: int
    duration_seconds: float
    estimated_input_cost: float
    estimated_output_cost: float
    estimated_storage_saved_bytes: int = 0
    avg_memory_size_before: float = 0.0
    avg_memory_size_after: float = 0.0
    types_compacted: dict = {}
    key_facts_preserved: int = 0
    entities_preserved: int = 0
    context_retention_score: float = 0.0


class CompactionResponse(BaseModel):
    """Compaction result response."""

    id: str
    user_id: str
    session_id: Optional[str]
    status: str
    started_at: datetime
    completed_at: Optional[datetime]
    metrics: CompactionMetricsResponse
    actions: list[dict]
    summary: Optional[str]
    insights: list[str] = []
    preserved_facts: list[str] = []
    preserved_entities: list[str] = []
    config: dict = {}
    dry_run: bool
    error: Optional[str]


# ============================================
# ENDPOINTS
# ============================================


@router.post("/compact", response_model=CompactionResponse)
async def compact_conversations(request: CompactRequest) -> CompactionResponse:
    """
    Compact and consolidate conversation memories.

    This analyzes conversation memories, clusters them by topic/time,
    and creates concise summaries while preserving key facts.

    Use dry_run=true to preview what would happen without making changes.

    Args:
        request: Compaction request with user_id, options, and memory_types filter

    Returns:
        CompactionResponse with detailed metrics, insights, and preserved information
    """
    try:
        import os

        from hippocampai.consolidation.compactor import ConversationCompactor

        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        compactor = ConversationCompactor(qdrant_url=qdrant_url)

        result = compactor.compact_conversations(
            user_id=request.user_id,
            session_id=request.session_id,
            lookback_hours=request.lookback_hours,
            min_memories=request.min_memories,
            dry_run=request.dry_run,
            memory_types=request.memory_types,
        )

        # Store in compaction history
        _compaction_history.add_from_result(result)

        return CompactionResponse(
            id=result.id,
            user_id=result.user_id,
            session_id=result.session_id,
            status=result.status,
            started_at=result.started_at,
            completed_at=result.completed_at,
            metrics=CompactionMetricsResponse(
                input_memories=result.metrics.input_memories,
                input_tokens=result.metrics.input_tokens,
                input_characters=result.metrics.input_characters,
                output_memories=result.metrics.output_memories,
                output_tokens=result.metrics.output_tokens,
                output_characters=result.metrics.output_characters,
                compression_ratio=result.metrics.compression_ratio,
                tokens_saved=result.metrics.tokens_saved,
                memories_merged=result.metrics.memories_merged,
                clusters_found=result.metrics.clusters_found,
                llm_calls=result.metrics.llm_calls,
                duration_seconds=result.metrics.duration_seconds,
                estimated_input_cost=result.metrics.estimated_input_cost,
                estimated_output_cost=result.metrics.estimated_output_cost,
                estimated_storage_saved_bytes=result.metrics.estimated_storage_saved_bytes,
                avg_memory_size_before=result.metrics.avg_memory_size_before,
                avg_memory_size_after=result.metrics.avg_memory_size_after,
                types_compacted=result.metrics.types_compacted,
                key_facts_preserved=result.metrics.key_facts_preserved,
                entities_preserved=result.metrics.entities_preserved,
                context_retention_score=result.metrics.context_retention_score,
            ),
            actions=result.actions,
            summary=result.summary,
            insights=result.insights,
            preserved_facts=result.preserved_facts,
            preserved_entities=result.preserved_entities,
            config=result.config,
            dry_run=result.dry_run,
            error=result.error,
        )

    except Exception as e:
        logger.exception(f"Compaction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/preview")
async def preview_compaction(
    user_id: str = Query(..., description="User ID"),
    session_id: Optional[str] = Query(None, description="Session ID filter"),
    lookback_hours: int = Query(168, description="Hours to look back"),
    memory_types: Optional[str] = Query(None, description="Comma-separated memory types"),
) -> dict:
    """
    Preview what compaction would do without making changes.

    Returns metrics about the memories that would be compacted.
    """
    try:
        import os

        from hippocampai.consolidation.compactor import ConversationCompactor, estimate_tokens

        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        compactor = ConversationCompactor(qdrant_url=qdrant_url)

        # Parse memory types
        types_list = None
        if memory_types:
            types_list = [t.strip() for t in memory_types.split(",")]

        memories = compactor._collect_memories(user_id, session_id, lookback_hours, types_list)
        clusters = compactor._cluster_memories(memories)

        total_tokens = sum(estimate_tokens(m.get("text", "")) for m in memories)
        total_chars = sum(len(m.get("text", "")) for m in memories)

        # Group by type
        by_type = {}
        for m in memories:
            t = m.get("type", "unknown")
            by_type[t] = by_type.get(t, 0) + 1

        # Estimate output
        compactable_clusters = [c for c in clusters if len(c) >= 2]
        estimated_output = len(compactable_clusters)
        estimated_compression = (
            (1 - estimated_output / max(len(memories), 1)) * 100 if memories else 0
        )

        return {
            "user_id": user_id,
            "session_id": session_id,
            "lookback_hours": lookback_hours,
            "memory_types_filter": types_list,
            "total_memories": len(memories),
            "total_tokens": total_tokens,
            "total_characters": total_chars,
            "clusters": len(clusters),
            "compactable_clusters": len(compactable_clusters),
            "by_type": by_type,
            "available_types": list(by_type.keys()),
            "estimated_output_memories": estimated_output,
            "estimated_compression": f"{estimated_compression:.1f}%",
            "estimated_tokens_saved": int(total_tokens * estimated_compression / 100),
        }

    except Exception as e:
        logger.exception(f"Preview failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/types")
async def get_compactable_types() -> dict:
    """Get list of memory types that can be compacted."""
    from hippocampai.consolidation.compactor import ConversationCompactor

    return {
        "types": ConversationCompactor.COMPACTABLE_TYPES,
        "descriptions": {
            "fact": "Factual information about the user or world",
            "event": "Events and activities that occurred",
            "context": "Contextual information from conversations",
            "preference": "User preferences and likes/dislikes",
            "goal": "User goals and objectives",
            "habit": "User habits and routines",
        },
    }


@router.get("/history")
async def get_compaction_history(
    user_id: str = Query(..., description="User ID"),
    limit: int = Query(20, ge=1, le=100),
) -> list[dict[str, Any]]:
    """Get compaction history for a user.

    Returns a list of past compaction operations with their results.
    History is stored in-memory and will be cleared on server restart.
    """
    entries = _compaction_history.get(user_id, limit)

    return [
        {
            "id": entry.id,
            "user_id": entry.user_id,
            "session_id": entry.session_id,
            "status": entry.status,
            "started_at": entry.started_at.isoformat(),
            "completed_at": entry.completed_at.isoformat() if entry.completed_at else None,
            "input_memories": entry.input_memories,
            "output_memories": entry.output_memories,
            "compression_ratio": entry.compression_ratio,
            "tokens_saved": entry.tokens_saved,
            "dry_run": entry.dry_run,
            "error": entry.error,
        }
        for entry in entries
    ]
