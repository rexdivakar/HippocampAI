"""API routes for dashboard statistics and activity."""

import logging
from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from hippocampai.api.deps import get_memory_client
from hippocampai.client import MemoryClient

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])


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
# HELPER FUNCTIONS
# ============================================


def _get_entity_counts(client: MemoryClient, user_id: str) -> tuple[int, int]:
    """Get total entities and concepts count."""
    try:
        stats = client.graph.get_graph_stats(user_id=user_id)
        total_entities = stats.get("num_nodes", 0)
        return total_entities, 0
    except Exception as e:
        logger.warning(f"Could not get entities: {e}")
        return 0, 0


def _get_tag_stats(memories: list[Any]) -> tuple[int, list[dict[str, Any]]]:
    """Get total tags count and top tags."""
    all_tags: set[str] = set()
    tag_counts: dict[str, int] = {}
    for memory in memories:
        if memory.tags:
            all_tags.update(memory.tags)
            for tag in memory.tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

    top_tags = [
        {"name": tag, "count": count}
        for tag, count in sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    ]
    return len(all_tags), top_tags


def _get_recent_memories(memories: list[Any], limit: int = 10) -> list[dict[str, Any]]:
    """Get recent memories list."""
    recent = sorted(memories, key=lambda m: m.created_at, reverse=True)[:limit]
    return [
        {
            "id": m.id,
            "text": m.text,
            "type": m.type,
            "created_at": m.created_at.isoformat(),
            "importance": m.importance,
        }
        for m in recent
    ]


def _get_cluster_stats(client: MemoryClient, user_id: str) -> list[dict[str, Any]]:
    """Get top clusters."""
    try:
        clusters = client.graph.get_clusters(user_id=user_id)
        return [
            {
                "id": idx,
                "size": len(cluster),
                "memory_ids": list(cluster)[:5],
            }
            for idx, cluster in enumerate(
                sorted(clusters, key=lambda x: len(x), reverse=True)[:5]
            )
        ]
    except Exception as e:
        logger.warning(f"Could not get clusters: {e}")
        return []


def _get_connection_count(client: MemoryClient, user_id: str) -> int:
    """Get total connections from knowledge graph."""
    try:
        stats = client.graph.get_graph_stats(user_id=user_id)
        return stats.get("num_edges", 0)
    except Exception as e:
        logger.warning(f"Could not get relationships: {e}")
        return 0


def _count_duplicates(memories: list[Any]) -> int:
    """Count potential duplicate memories."""
    try:
        count = 0
        for i, m1 in enumerate(memories):
            for m2 in memories[i + 1:]:
                if m1.text.lower().strip() == m2.text.lower().strip():
                    count += 1
        return count
    except Exception as e:
        logger.warning(f"Could not check duplicates: {e}")
        return 0


def _calculate_health_score(
    total_memories: int, uncategorized: int, potential_duplicates: int, total_tags: int
) -> float:
    """Calculate health score based on memory quality metrics."""
    health_score = 85.0
    if total_memories > 0:
        if uncategorized / total_memories > 0.3:
            health_score -= 10
        if potential_duplicates > 5:
            health_score -= 5
        if total_tags < 5:
            health_score -= 5
    return max(0.0, min(100.0, health_score))


# ============================================
# API ENDPOINTS
# ============================================


def _get_sleep_runs_count(user_id: str) -> int:
    """Get total sleep runs count."""
    from hippocampai.api.consolidation_routes import get_user_consolidation_runs

    try:
        sleep_runs = get_user_consolidation_runs(user_id, limit=1000)
        return len(sleep_runs)
    except Exception as e:
        logger.warning(f"Could not get sleep runs: {e}")
        return 0


def _count_uncategorized(memories: list[Any]) -> int:
    """Count uncategorized memories (no tags and type is context)."""
    return sum(
        1 for m in memories if (not m.tags or len(m.tags) == 0) and m.type == "context"
    )


def _count_archived(memories: list[Any]) -> int:
    """Count archived memories."""
    return sum(1 for m in memories if m.metadata and m.metadata.get("archived", False))


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
        memories = client.get_memories(
            user_id=user_id, filters={"session_id": user_id}, limit=10000
        )
        total_memories = len(memories)

        # Get entity and concept counts
        total_entities, total_concepts = _get_entity_counts(client, user_id)

        # Get tag statistics
        total_tags, top_tags = _get_tag_stats(memories)

        # Get recent memories
        recent_memories = _get_recent_memories(memories)

        # Get cluster statistics
        top_clusters = _get_cluster_stats(client, user_id)

        # Get connection count
        total_connections = _get_connection_count(client, user_id)

        # Check for potential duplicates
        potential_duplicates = _count_duplicates(memories)

        # Count uncategorized and archived memories
        uncategorized = _count_uncategorized(memories)
        archived = _count_archived(memories)

        # Calculate health score
        health_score = _calculate_health_score(
            total_memories, uncategorized, potential_duplicates, total_tags
        )

        # Get sleep runs count
        total_sleep_runs = _get_sleep_runs_count(user_id)

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

        # Get recent memories (include session_id filter to match by either field)
        memories = client.get_memories(user_id=user_id, filters={"session_id": user_id}, limit=50)
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
        raise HTTPException(status_code=500, detail=f"Failed to get recent activity: {str(e)}")
