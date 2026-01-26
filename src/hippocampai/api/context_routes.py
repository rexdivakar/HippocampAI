"""Context assembly API routes."""

import logging
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from hippocampai.api.deps import get_memory_client
from hippocampai.client import MemoryClient
from hippocampai.context.models import ContextPack

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/context", tags=["context"])


class AssembleContextRequest(BaseModel):
    """Request to assemble context."""

    query: str
    user_id: str
    session_id: Optional[str] = None
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


@router.post(":assemble", response_model=ContextPack)
def assemble_context(
    request: AssembleContextRequest,
    client: MemoryClient = Depends(get_memory_client),
) -> ContextPack:
    """Assemble a context pack for LLM prompts.

    This endpoint provides automated context assembly that:
    1. Retrieves relevant memories
    2. Re-ranks by relevance
    3. Deduplicates similar content
    4. Applies temporal filters
    5. Fits to token budget (with optional summarization)
    6. Returns a ready-to-use ContextPack

    The response includes:
    - final_context_text: Ready-to-use context string
    - citations: Memory IDs for attribution
    - selected_items: Structured list of selected memories
    - dropped_items: Items that were dropped (for debugging)
    """
    try:
        pack = client.assemble_context(
            query=request.query,
            user_id=request.user_id,
            session_id=request.session_id,
            token_budget=request.token_budget,
            max_items=request.max_items,
            recency_bias=request.recency_bias,
            entity_focus=request.entity_focus,
            type_filter=request.type_filter,
            min_relevance=request.min_relevance,
            allow_summaries=request.allow_summaries,
            include_citations=request.include_citations,
            deduplicate=request.deduplicate,
            time_range_days=request.time_range_days,
        )
        return pack
    except Exception as e:
        logger.error(f"Failed to assemble context: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(":assemble/text")
def assemble_context_text(
    request: AssembleContextRequest,
    client: MemoryClient = Depends(get_memory_client),
) -> dict[str, Any]:
    """Assemble context and return just the text (simplified response).

    Returns only the essential fields for quick integration.
    """
    try:
        pack = client.assemble_context(
            query=request.query,
            user_id=request.user_id,
            session_id=request.session_id,
            token_budget=request.token_budget,
            max_items=request.max_items,
            recency_bias=request.recency_bias,
            entity_focus=request.entity_focus,
            type_filter=request.type_filter,
            min_relevance=request.min_relevance,
            allow_summaries=request.allow_summaries,
            include_citations=request.include_citations,
            deduplicate=request.deduplicate,
            time_range_days=request.time_range_days,
        )
        return {
            "context": pack.final_context_text,
            "citations": pack.citations,
            "token_count": pack.total_tokens,
            "item_count": len(pack.selected_items),
        }
    except Exception as e:
        logger.error(f"Failed to assemble context: {e}")
        raise HTTPException(status_code=500, detail=str(e))
