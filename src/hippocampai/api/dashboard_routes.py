"""API routes for dashboard statistics and activity."""

import logging
from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from hippocampai.api.deps import get_memory_client
from hippocampai.client import MemoryClient

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


# ============================================
# RESPONSE MODELS
# ============================================


class DashboardStats(BaseModel):
    """Dashboard statistics summary."""

    total_memories: int
    total_entities: int
    total_concepts: int
    total_tags: int
    total_sleep_runs: int
    health_score: float
    top_tags: list[dict[str, Any]]
    recent_memories: list[dict[str, Any]]
    top_clusters: list[dict[str, Any]]
    total_connections: int
    potential_duplicates: int
    uncategorized_memories: int
    archived_memories: int


class ActivityItem(BaseModel):
    """Recent activity item."""

    type: str
    description: str
    timestamp: datetime
    metadata: Optional[dict[str, Any]] = None


class RecentActivity(BaseModel):
    """Recent activity feed."""

    activities: list[ActivityItem]


# ============================================
# API ENDPOINTS
# ============================================


@router.get("/stats")
async def get_dashboard_stats(
    user_id: str,
    client: MemoryClient = Depends(get_memory_client),
) -> DashboardStats:
    """
    Get comprehensive dashboard statistics.

    Returns aggregated metrics about memories, entities, concepts, and system health.
    """
    try:
        # Get all memories for the user
        memories = client.get_memories(user_id=user_id, limit=10000)
        total_memories = len(memories)

        # Get entities from knowledge graph
        try:
            all_entities = client.kg.get_all_entities(user_id=user_id)
            total_entities = len(all_entities)
        except Exception as e:
            logger.warning(f"Could not get entities: {e}")
            total_entities = 0

        # Get concepts (entities with type 'concept')
        try:
            concepts = [e for e in all_entities if e.type == "concept"]
            total_concepts = len(concepts)
        except Exception:
            total_concepts = 0

        # Count unique tags
        all_tags = set()
        for memory in memories:
            if memory.tags:
                all_tags.update(memory.tags)
        total_tags = len(all_tags)

        # Count tags frequency for top tags
        tag_counts: dict[str, int] = {}
        for memory in memories:
            if memory.tags:
                for tag in memory.tags:
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1

        # Get top 10 tags
        top_tags = [
            {"name": tag, "count": count}
            for tag, count in sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        ]

        # Get recent memories (last 10)
        recent_memories_list = sorted(memories, key=lambda m: m.created_at, reverse=True)[:10]
        recent_memories = [
            {
                "id": m.id,
                "text": m.text,
                "type": m.type,
                "created_at": m.created_at.isoformat(),
                "importance": m.importance,
            }
            for m in recent_memories_list
        ]

        # Get clusters
        try:
            clusters = client.get_clusters(user_id=user_id)
            top_clusters = [
                {
                    "id": c.id,
                    "label": c.label,
                    "size": len(c.memory_ids),
                    "coherence": c.coherence_score,
                }
                for c in sorted(clusters, key=lambda x: len(x.memory_ids), reverse=True)[:5]
            ]
        except Exception as e:
            logger.warning(f"Could not get clusters: {e}")
            top_clusters = []

        # Get connections count from knowledge graph
        try:
            relationships = client.kg.get_all_relationships(user_id=user_id)
            total_connections = len(relationships)
        except Exception as e:
            logger.warning(f"Could not get relationships: {e}")
            total_connections = 0

        # Check for potential duplicates
        try:
            # Use deduplicator to find potential duplicates
            potential_duplicates = 0
            # This is a simple heuristic - we'll improve it later
            for i, m1 in enumerate(memories):
                for m2 in memories[i + 1 :]:
                    # Simple text similarity check
                    if m1.text.lower().strip() == m2.text.lower().strip():
                        potential_duplicates += 1
        except Exception as e:
            logger.warning(f"Could not check duplicates: {e}")
            potential_duplicates = 0

        # Count uncategorized memories (no tags and no type)
        uncategorized = sum(
            1 for m in memories if (not m.tags or len(m.tags) == 0) and m.type == "context"
        )

        # Count archived memories
        archived = sum(
            1
            for m in memories
            if m.metadata and m.metadata.get("archived", False)
        )

        # Calculate health score (simple heuristic)
        health_score = 85.0  # Base score
        if total_memories > 0:
            # Adjust based on various factors
            if uncategorized / total_memories > 0.3:
                health_score -= 10
            if potential_duplicates > 5:
                health_score -= 5
            if total_tags < 5:
                health_score -= 5
            # Cap between 0-100
            health_score = max(0, min(100, health_score))

        # Get sleep runs count from consolidation routes storage
        from hippocampai.api.consolidation_routes import get_user_consolidation_runs

        try:
            sleep_runs = get_user_consolidation_runs(user_id, limit=1000)
            total_sleep_runs = len(sleep_runs)
        except Exception as e:
            logger.warning(f"Could not get sleep runs: {e}")
            total_sleep_runs = 0

        return DashboardStats(
            total_memories=total_memories,
            total_entities=total_entities,
            total_concepts=total_concepts,
            total_tags=total_tags,
            total_sleep_runs=total_sleep_runs,
            health_score=health_score,
            top_tags=top_tags,
            recent_memories=recent_memories,
            top_clusters=top_clusters,
            total_connections=total_connections,
            potential_duplicates=potential_duplicates,
            uncategorized_memories=uncategorized,
            archived_memories=archived,
        )

    except Exception as e:
        logger.exception(f"Failed to get dashboard stats for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get dashboard stats: {str(e)}")


@router.get("/recent-activity")
async def get_recent_activity(
    user_id: str,
    client: MemoryClient = Depends(get_memory_client),
    limit: int = 20,
) -> RecentActivity:
    """
    Get recent activity feed.

    Returns a timeline of recent actions: memories created, sleep runs, etc.
    """
    try:
        activities: list[ActivityItem] = []

        # Get recent memories
        memories = client.get_memories(user_id=user_id, limit=50)
        recent_memories = sorted(memories, key=lambda m: m.created_at, reverse=True)[:limit]

        for memory in recent_memories:
            activities.append(
                ActivityItem(
                    type="memory_created",
                    description=f"Created {memory.type} memory: {memory.text[:60]}...",
                    timestamp=memory.created_at,
                    metadata={
                        "memory_id": memory.id,
                        "memory_type": memory.type,
                        "importance": memory.importance,
                    },
                )
            )

        # Get recent sleep runs
        from hippocampai.api.consolidation_routes import get_user_consolidation_runs

        try:
            sleep_runs = get_user_consolidation_runs(user_id, limit=10)
            for run in sleep_runs:
                if run.completed_at:
                    activities.append(
                        ActivityItem(
                            type="sleep_cycle",
                            description=f"Sleep cycle completed: {run.memories_reviewed} memories reviewed, {run.memories_promoted} promoted",
                            timestamp=run.completed_at,
                            metadata={
                                "run_id": run.id,
                                "status": run.status.value,
                                "memories_reviewed": run.memories_reviewed,
                            },
                        )
                    )
        except Exception as e:
            logger.warning(f"Could not get sleep runs for activity: {e}")

        # Sort all activities by timestamp
        activities.sort(key=lambda a: a.timestamp, reverse=True)

        # Limit to requested number
        activities = activities[:limit]

        return RecentActivity(activities=activities)

    except Exception as e:
        logger.exception(f"Failed to get recent activity for user {user_id}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get recent activity: {str(e)}"
        )
