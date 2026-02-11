"""REST API routes for auto-healing and health monitoring."""

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from hippocampai.client import MemoryClient
from hippocampai.embed.embedder import Embedder
from hippocampai.models.healing import AutoHealingConfig, HealingActionType
from hippocampai.monitoring.memory_health import MemoryHealthMonitor
from hippocampai.pipeline.auto_healing import AutoHealingEngine

router = APIRouter(prefix="/v1/healing", tags=["healing"])

# Initialize engines
embedder = Embedder(model_name="all-MiniLM-L6-v2")
health_monitor = MemoryHealthMonitor(embedder)
healing_engine = AutoHealingEngine(health_monitor, embedder)


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================


class HealthCheckRequest(BaseModel):
    user_id: str
    detailed: bool = True


class CleanupRequest(BaseModel):
    user_id: str
    dry_run: bool = True
    max_actions: int = 50


class DeduplicationRequest(BaseModel):
    user_id: str
    similarity_threshold: float = 0.90
    dry_run: bool = True


class ConsolidationRequest(BaseModel):
    user_id: str
    tag: Optional[str] = None
    max_memories: int = 10
    dry_run: bool = True


class TaggingRequest(BaseModel):
    user_id: str
    max_suggestions: int = 20
    dry_run: bool = True


class HealingConfigRequest(BaseModel):
    user_id: str
    enabled: bool = True
    auto_cleanup_enabled: bool = True
    auto_dedup_enabled: bool = True
    cleanup_threshold_days: int = 90
    dedup_similarity_threshold: float = 0.90
    require_user_approval: bool = True
    max_actions_per_run: int = 50


class ImportanceAdjustmentRequest(BaseModel):
    user_id: str
    dry_run: bool = True
    max_adjustments: int = 20


# ============================================================================
# HEALTH MONITORING
# ============================================================================


@router.post("/health")
async def check_health(request: HealthCheckRequest):
    """Calculate health score for user's memories."""
    try:
        client = MemoryClient()
        memories = client.get_memories(user_id=request.user_id)

        health = health_monitor.calculate_health_score(memories=memories, detailed=request.detailed)

        response = {
            "overall_score": health.overall_score,
            "status": health.status.value,
            "memory_count": len(memories),
        }

        if request.detailed:
            response.update(
                {
                    "freshness_score": health.freshness_score,
                    "diversity_score": health.diversity_score,
                    "consistency_score": health.consistency_score,
                    "coverage_score": health.coverage_score,
                    "engagement_score": health.engagement_score,
                    "issues": [
                        {
                            "severity": issue.severity.value,
                            "category": issue.category,
                            "message": issue.message,
                            "affected_memories": issue.affected_memory_ids,
                            "suggestions": issue.suggestions,
                        }
                        for issue in health.issues
                    ],
                    "recommendations": health.recommendations,
                }
            )

        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/health/stale")
async def detect_stale_memories(user_id: str, threshold_days: int = 90):
    """Detect stale memories that haven't been accessed recently."""
    try:
        client = MemoryClient()
        memories = client.get_memories(user_id=user_id)

        stale = health_monitor.detect_stale_memories(
            memories=memories, threshold_days=threshold_days
        )

        return {
            "stale_memories": [
                {
                    "id": sm.memory.id,
                    "text": sm.memory.text,
                    "staleness_score": sm.staleness_score,
                    "days_since_access": sm.days_since_access,
                    "should_archive": sm.should_archive,
                    "should_delete": sm.should_delete,
                }
                for sm in stale
            ],
            "count": len(stale),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/health/duplicates")
async def detect_duplicates(user_id: str, similarity_threshold: float = 0.90):
    """Detect duplicate or highly similar memories."""
    try:
        client = MemoryClient()
        memories = client.get_memories(user_id=user_id)

        clusters = health_monitor.detect_duplicate_clusters(
            memories=memories, cluster_type="soft"
        )

        return {
            "clusters": [
                {
                    "representative_id": cluster.representative_memory_id,
                    "representative_text": next(
                        (m.text for m in cluster.memories if m.id == cluster.representative_memory_id),
                        cluster.memories[0].text if cluster.memories else "",
                    ),
                    "duplicate_count": len(cluster.memories),
                    "duplicate_ids": [m.id for m in cluster.memories],
                    "average_similarity": (
                        sum(cluster.similarity_scores) / len(cluster.similarity_scores)
                        if cluster.similarity_scores
                        else 0.0
                    ),
                    "suggested_action": cluster.cluster_type.value,
                }
                for cluster in clusters
            ],
            "total_duplicates": sum(len(c.memories) for c in clusters),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/health/gaps")
async def detect_knowledge_gaps(user_id: str):
    """Detect gaps in knowledge coverage."""
    try:
        client = MemoryClient()
        memories = client.get_memories(user_id=user_id)

        coverage = health_monitor.analyze_topic_coverage(memories)

        gaps = [tc for tc in coverage if tc.coverage_level.value in ("missing", "minimal", "sparse")]

        return {
            "gaps": [
                {
                    "topic": gap.topic,
                    "coverage_level": gap.coverage_level.value,
                    "memory_count": gap.memory_count,
                    "quality_score": gap.quality_score,
                    "gaps": gap.gaps,
                }
                for gap in gaps
            ],
            "count": len(gaps),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# AUTO-HEALING OPERATIONS
# ============================================================================


@router.post("/cleanup")
async def auto_cleanup(request: CleanupRequest):
    """Run auto-cleanup to remove stale and low-quality memories."""
    try:
        client = MemoryClient()
        memories = client.get_memories(user_id=request.user_id)

        config = AutoHealingConfig(user_id=request.user_id, max_actions_per_run=request.max_actions)

        report = healing_engine.auto_cleanup(
            user_id=request.user_id, memories=memories, config=config, dry_run=request.dry_run
        )

        return {
            "success": True,
            "dry_run": request.dry_run,
            "actions_recommended": [
                {
                    "action_type": action.action_type.value,
                    "memory_ids": action.memory_ids,
                    "reason": action.reason,
                    "impact_score": action.impact_score,
                    "auto_applicable": action.auto_applicable,
                }
                for action in report.actions_recommended
            ],
            "actions_applied": [
                {
                    "action_type": action.action_type.value,
                    "memory_ids": action.memory_ids,
                    "reason": action.reason,
                }
                for action in report.actions_applied
            ],
            "health_before": report.health_before,
            "health_after": report.health_after,
            "summary": report.summary,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/deduplication")
async def auto_deduplication(request: DeduplicationRequest):
    """Run deduplication to merge or remove duplicate memories."""
    try:
        client = MemoryClient()
        memories = client.get_memories(user_id=request.user_id)

        config = AutoHealingConfig(
            user_id=request.user_id, dedup_similarity_threshold=request.similarity_threshold
        )

        report = healing_engine.auto_consolidate(
            user_id=request.user_id, memories=memories, config=config, dry_run=request.dry_run
        )

        return {
            "success": True,
            "dry_run": request.dry_run,
            "clusters_found": len(report.actions_recommended),
            "actions_recommended": [
                {
                    "action_type": action.action_type.value,
                    "memory_ids": action.memory_ids,
                    "reason": action.reason,
                    "merged_content": action.changes.get("merged_content")
                    if action.changes
                    else None,
                }
                for action in report.actions_recommended
            ],
            "duplicates_removed": sum(
                len(action.memory_ids) - 1
                for action in report.actions_applied
                if action.action_type == HealingActionType.MERGE
            ),
            "summary": report.summary,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/consolidate")
async def auto_consolidate(request: ConsolidationRequest):
    """Consolidate similar memories into higher-level insights."""
    try:
        client = MemoryClient()
        memories = client.get_memories(user_id=request.user_id)

        # Filter by tag if specified
        if request.tag:
            memories = [m for m in memories if request.tag in m.tags]

        config = AutoHealingConfig(user_id=request.user_id)

        report = healing_engine.auto_consolidate(
            user_id=request.user_id,
            memories=memories[: request.max_memories],
            config=config,
            dry_run=request.dry_run,
        )

        return {
            "success": True,
            "dry_run": request.dry_run,
            "consolidations": [
                {
                    "action_type": action.action_type.value,
                    "source_memory_ids": action.memory_ids,
                    "consolidated_text": action.changes.get("consolidated_text")
                    if action.changes
                    else None,
                    "reason": action.reason,
                }
                for action in report.actions_recommended
            ],
            "summary": report.summary,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tagging")
async def auto_tagging(request: TaggingRequest):
    """Automatically suggest or apply tags to untagged memories."""
    try:
        client = MemoryClient()
        memories = client.get_memories(user_id=request.user_id)

        tag_actions = healing_engine.auto_tag(memories=memories)

        # Apply if not dry run
        applied_actions = []
        if not request.dry_run:
            for action in tag_actions:
                if action.auto_applicable:
                    action.apply("auto_healing")
                    applied_actions.append(action)

        return {
            "success": True,
            "dry_run": request.dry_run,
            "tagging_suggestions": [
                {
                    "memory_id": action.memory_ids[0],
                    "suggested_tags": action.metadata.get("suggested_tags", []),
                    "reason": action.reason,
                }
                for action in tag_actions[: request.max_suggestions]
            ],
            "tags_applied": sum(
                len(action.metadata.get("suggested_tags", []))
                for action in applied_actions
            ),
            "summary": f"Generated {len(tag_actions)} tag suggestions, applied {len(applied_actions)}",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/importance")
async def auto_importance_adjustment(request: ImportanceAdjustmentRequest):
    """Automatically adjust importance scores based on usage patterns."""
    try:
        client = MemoryClient()
        memories = client.get_memories(user_id=request.user_id)

        importance_actions = healing_engine.auto_importance_adjustment(memories=memories)

        # Apply if not dry run
        applied_actions = []
        if not request.dry_run:
            for action in importance_actions:
                if action.auto_applicable:
                    action.apply("auto_healing")
                    applied_actions.append(action)

        return {
            "success": True,
            "dry_run": request.dry_run,
            "adjustments": [
                {
                    "memory_id": action.memory_ids[0],
                    "new_importance": action.metadata.get("new_importance"),
                    "reason": action.reason,
                }
                for action in importance_actions[: request.max_adjustments]
            ],
            "total_adjusted": len(applied_actions),
            "summary": f"Generated {len(importance_actions)} importance adjustments, applied {len(applied_actions)}",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/full-check")
async def run_full_health_check(user_id: str, dry_run: bool = True):
    """Run a comprehensive health check with all healing operations."""
    try:
        client = MemoryClient()
        memories = client.get_memories(user_id=user_id)

        config = AutoHealingConfig(
            user_id=user_id, auto_cleanup_enabled=True, auto_dedup_enabled=True
        )

        report = healing_engine.run_full_health_check(
            user_id=user_id, memories=memories, config=config, dry_run=dry_run
        )

        return {
            "success": True,
            "dry_run": dry_run,
            "health_before": report.health_before,
            "health_after": report.health_after,
            "health_improvement": report.health_after - report.health_before,
            "total_actions_recommended": len(report.actions_recommended),
            "total_actions_applied": len(report.actions_applied),
            "actions_by_type": {
                action_type.value: sum(
                    1 for action in report.actions_recommended if action.action_type == action_type
                )
                for action_type in HealingActionType
            },
            "summary": report.summary,
            "timestamp": report.started_at.isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# CONFIGURATION
# ============================================================================


@router.post("/config")
async def update_healing_config(request: HealingConfigRequest):
    """Update auto-healing configuration for a user."""
    try:
        config = AutoHealingConfig(
            user_id=request.user_id,
            enabled=request.enabled,
            auto_cleanup_enabled=request.auto_cleanup_enabled,
            auto_dedup_enabled=request.auto_dedup_enabled,
            cleanup_threshold_days=request.cleanup_threshold_days,
            dedup_similarity_threshold=request.dedup_similarity_threshold,
            require_user_approval=request.require_user_approval,
            max_actions_per_run=request.max_actions_per_run,
        )

        # In production, save config to database
        # For now, just return the config

        return {
            "success": True,
            "config": {
                "user_id": config.user_id,
                "enabled": config.enabled,
                "auto_cleanup_enabled": config.auto_cleanup_enabled,
                "auto_dedup_enabled": config.auto_dedup_enabled,
                "cleanup_threshold_days": config.cleanup_threshold_days,
                "dedup_similarity_threshold": config.dedup_similarity_threshold,
                "require_user_approval": config.require_user_approval,
                "max_actions_per_run": config.max_actions_per_run,
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/config/{user_id}")
async def get_healing_config(user_id: str):
    """Get auto-healing configuration for a user."""
    try:
        # In production, load from database
        # For now, return default config
        config = AutoHealingConfig(user_id=user_id)

        return {
            "config": {
                "user_id": config.user_id,
                "enabled": config.enabled,
                "auto_cleanup_enabled": config.auto_cleanup_enabled,
                "auto_dedup_enabled": config.auto_dedup_enabled,
                "cleanup_threshold_days": config.cleanup_threshold_days,
                "dedup_similarity_threshold": config.dedup_similarity_threshold,
                "require_user_approval": config.require_user_approval,
                "max_actions_per_run": config.max_actions_per_run,
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
