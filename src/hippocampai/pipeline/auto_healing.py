"""Auto-healing engine for memory health maintenance.

This module provides:
- Automatic memory cleanup
- Smart consolidation
- Scheduled maintenance
- Self-healing recommendations
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from hippocampai.embed.embedder import Embedder
from hippocampai.models.healing import (
    AutoHealingConfig,
    ConsolidationResult,
    ConsolidationStrategy,
    HealingAction,
    HealingActionType,
    HealingReport,
    HealthImprovement,
)
from hippocampai.models.memory import Memory
from hippocampai.monitoring.memory_health import MemoryHealthMonitor

logger = logging.getLogger(__name__)


class AutoHealingEngine:
    """
    Auto-healing engine for proactive memory maintenance.

    Features:
    - Automatic cleanup of stale memories
    - Smart duplicate consolidation
    - Auto-tagging and importance adjustment
    - Scheduled maintenance tasks
    """

    def __init__(
        self,
        health_monitor: MemoryHealthMonitor,
        embedder: Optional[Embedder] = None,
    ):
        """
        Initialize auto-healing engine.

        Args:
            health_monitor: Memory health monitoring system
            embedder: Embedder for similarity calculations
        """
        self.health_monitor = health_monitor
        self.embedder = embedder or health_monitor.embedder

    def auto_cleanup(
        self, user_id: str, memories: list[Memory], config: AutoHealingConfig, dry_run: bool = True
    ) -> HealingReport:
        """
        Automatically clean up stale and low-quality memories.

        Args:
            user_id: User ID
            memories: List of memories to analyze
            config: Auto-healing configuration
            dry_run: If True, only recommend actions without applying

        Returns:
            Healing report with results
        """
        started_at = datetime.now(timezone.utc)
        actions_recommended = []
        actions_applied = []

        # Get initial health score
        health_before = self.health_monitor.calculate_health_score(memories, detailed=False)

        # 1. Identify stale memories
        stale_memories = self.health_monitor.detect_stale_memories(
            memories, threshold_days=config.cleanup_threshold_days
        )

        for stale_mem in stale_memories:
            action = HealingAction(
                action_type=HealingActionType.ARCHIVE
                if stale_mem.should_archive
                else HealingActionType.DELETE,
                memory_ids=[stale_mem.memory.id],
                reason=f"Stale memory ({stale_mem.reason.value}): {stale_mem.staleness_score:.2f} staleness",
                impact=f"Will free up storage and improve health score",
                reversible=True,
                auto_applicable=not config.require_user_approval
                and stale_mem.staleness_score > 0.8,
            )
            actions_recommended.append(action)

            if not dry_run and action.auto_applicable:
                action.apply("auto_healing")
                actions_applied.append(action)

        # 2. Identify low confidence memories
        low_confidence_threshold = 0.3
        low_confidence_memories = [m for m in memories if m.confidence < low_confidence_threshold]

        for memory in low_confidence_memories:
            action = HealingAction(
                action_type=HealingActionType.UPDATE_CONFIDENCE,
                memory_ids=[memory.id],
                reason=f"Low confidence: {memory.confidence:.2f}",
                impact="Review and update or remove",
                reversible=True,
                auto_applicable=False,  # Requires review
            )
            actions_recommended.append(action)

        # Limit actions
        if len(actions_recommended) > config.max_actions_per_run:
            logger.info(
                f"Limiting actions from {len(actions_recommended)} to {config.max_actions_per_run}"
            )
            actions_recommended = actions_recommended[: config.max_actions_per_run]

        completed_at = datetime.now(timezone.utc)

        # Calculate health after (would need actual application)
        health_after = health_before.overall_score  # Same for dry run

        report = HealingReport(
            user_id=user_id,
            run_type="cleanup",
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            actions_recommended=actions_recommended,
            actions_applied=actions_applied,
            memories_analyzed=len(memories),
            memories_affected=len(actions_applied),
            health_before=health_before.overall_score,
            health_after=health_after,
            summary=f"Cleanup {'recommended' if dry_run else 'completed'}: {len(actions_recommended)} actions identified, {len(actions_applied)} applied",
        )

        logger.info(
            f"Auto-cleanup for user {user_id}: {len(actions_recommended)} actions, dry_run={dry_run}"
        )
        return report

    def auto_consolidate(
        self,
        user_id: str,
        memories: list[Memory],
        config: AutoHealingConfig,
        strategy: ConsolidationStrategy = ConsolidationStrategy.SEMANTIC_MERGE,
        dry_run: bool = True,
    ) -> HealingReport:
        """
        Automatically consolidate duplicate and similar memories.

        Args:
            user_id: User ID
            memories: List of memories
            config: Auto-healing configuration
            strategy: Consolidation strategy
            dry_run: If True, only recommend without applying

        Returns:
            Healing report
        """
        started_at = datetime.now(timezone.utc)
        actions_recommended = []
        actions_applied = []

        # Get initial health
        health_before = self.health_monitor.calculate_health_score(memories, detailed=False)

        # Detect duplicate clusters
        duplicate_clusters = self.health_monitor.detect_duplicate_clusters(
            memories, cluster_type="soft", min_cluster_size=2
        )

        for cluster in duplicate_clusters:
            # Only auto-consolidate if highly similar
            if cluster.confidence > config.dedup_similarity_threshold:
                action = HealingAction(
                    action_type=HealingActionType.MERGE,
                    memory_ids=[m.id for m in cluster.memories],
                    reason=f"Duplicate cluster ({cluster.cluster_type.value}): {cluster.confidence:.0%} similar",
                    impact=f"Merge {len(cluster.memories)} memories into one",
                    reversible=True,
                    auto_applicable=not config.require_user_approval and cluster.confidence > 0.95,
                    metadata={
                        "cluster_id": cluster.cluster_id,
                        "representative_id": cluster.representative_memory_id,
                        "strategy": strategy.value,
                    },
                )
                actions_recommended.append(action)

                if not dry_run and action.auto_applicable:
                    action.apply("auto_healing")
                    actions_applied.append(action)

        completed_at = datetime.now(timezone.utc)

        report = HealingReport(
            user_id=user_id,
            run_type="consolidate",
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            actions_recommended=actions_recommended,
            actions_applied=actions_applied,
            memories_analyzed=len(memories),
            memories_affected=len(actions_applied),
            health_before=health_before.overall_score,
            health_after=health_before.overall_score,  # Would improve with actual consolidation
            summary=f"Consolidation {'recommended' if dry_run else 'completed'}: {len(duplicate_clusters)} duplicate clusters found, {len(actions_applied)} merged",
        )

        logger.info(
            f"Auto-consolidate for user {user_id}: {len(duplicate_clusters)} clusters, dry_run={dry_run}"
        )
        return report

    def auto_tag(
        self, memories: list[Memory], llm_client: Optional[Any] = None
    ) -> list[HealingAction]:
        """
        Automatically suggest tags for memories.

        Args:
            memories: Memories to tag
            llm_client: Optional LLM client for intelligent tagging

        Returns:
            List of healing actions for tag updates
        """
        actions = []

        for memory in memories:
            if not memory.tags or len(memory.tags) < 2:
                # Simple rule-based tagging (can be enhanced with LLM)
                suggested_tags = self._suggest_tags(memory)

                if suggested_tags:
                    action = HealingAction(
                        action_type=HealingActionType.UPDATE_TAGS,
                        memory_ids=[memory.id],
                        reason=f"Memory has {len(memory.tags)} tags, suggesting {len(suggested_tags)} more",
                        impact=f"Add tags: {', '.join(suggested_tags)}",
                        reversible=True,
                        auto_applicable=True,
                        metadata={"suggested_tags": suggested_tags},
                    )
                    actions.append(action)

        logger.info(f"Auto-tag generated {len(actions)} tag suggestions")
        return actions

    def auto_importance_adjustment(
        self, memories: list[Memory], access_weight: float = 0.5
    ) -> list[HealingAction]:
        """
        Automatically adjust importance scores based on access patterns.

        Args:
            memories: Memories to analyze
            access_weight: Weight of access count in importance (0-1)

        Returns:
            List of healing actions for importance updates
        """
        actions = []

        # Calculate normalized access scores
        max_access = max((m.access_count for m in memories), default=1)

        for memory in memories:
            # Adjusted importance = original + (access_score * weight)
            access_score = (memory.access_count / max_access) * 10 if max_access > 0 else 0
            adjusted_importance = (
                memory.importance * (1 - access_weight) + access_score * access_weight
            )

            # If significantly different, suggest adjustment
            if abs(adjusted_importance - memory.importance) > 1.5:
                action = HealingAction(
                    action_type=HealingActionType.ADJUST_IMPORTANCE,
                    memory_ids=[memory.id],
                    reason=f"Access pattern suggests different importance (accessed {memory.access_count} times)",
                    impact=f"Adjust importance from {memory.importance:.1f} to {adjusted_importance:.1f}",
                    reversible=True,
                    auto_applicable=True,
                    metadata={"new_importance": adjusted_importance},
                )
                actions.append(action)

        logger.info(f"Auto-importance generated {len(actions)} adjustment suggestions")
        return actions

    def run_full_health_check(
        self, user_id: str, memories: list[Memory], config: AutoHealingConfig, dry_run: bool = True
    ) -> HealingReport:
        """
        Run comprehensive health check and healing.

        Args:
            user_id: User ID
            memories: List of memories
            config: Auto-healing configuration
            dry_run: If True, only recommend

        Returns:
            Comprehensive healing report
        """
        started_at = datetime.now(timezone.utc)
        all_actions_recommended = []
        all_actions_applied = []

        # Get initial health
        health_before = self.health_monitor.calculate_health_score(memories, detailed=True)

        # 1. Cleanup
        if config.auto_cleanup_enabled:
            cleanup_report = self.auto_cleanup(user_id, memories, config, dry_run)
            all_actions_recommended.extend(cleanup_report.actions_recommended)
            all_actions_applied.extend(cleanup_report.actions_applied)

        # 2. Deduplication
        if config.auto_dedup_enabled:
            dedup_report = self.auto_consolidate(user_id, memories, config, dry_run=dry_run)
            all_actions_recommended.extend(dedup_report.actions_recommended)
            all_actions_applied.extend(dedup_report.actions_applied)

        # 3. Auto-tagging
        if config.auto_tag_enabled:
            tag_actions = self.auto_tag(memories)
            all_actions_recommended.extend(tag_actions)
            if not dry_run:
                for action in tag_actions:
                    if action.auto_applicable:
                        action.apply("auto_healing")
                        all_actions_applied.append(action)

        # 4. Importance adjustment
        importance_actions = self.auto_importance_adjustment(memories)
        all_actions_recommended.extend(importance_actions)

        # Limit total actions
        if len(all_actions_recommended) > config.max_actions_per_run:
            all_actions_recommended = all_actions_recommended[: config.max_actions_per_run]

        completed_at = datetime.now(timezone.utc)

        # Get final health (would be different with actual application)
        health_after = health_before.overall_score

        # Calculate improvements
        improvements = {
            "stale_memories_reduced": 0,
            "duplicates_removed": len(
                [a for a in all_actions_applied if a.action_type == HealingActionType.MERGE]
            ),
            "tags_added": len(
                [a for a in all_actions_applied if a.action_type == HealingActionType.UPDATE_TAGS]
            ),
        }

        report = HealingReport(
            user_id=user_id,
            run_type="full_health_check",
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            actions_recommended=all_actions_recommended,
            actions_applied=all_actions_applied,
            memories_analyzed=len(memories),
            memories_affected=len(all_actions_applied),
            health_before=health_before.overall_score,
            health_after=health_after,
            improvements=improvements,
            summary=f"Full health check {'completed' if not dry_run else 'recommended'}: {len(all_actions_recommended)} actions, health: {health_before.overall_score:.1f}/100",
        )

        logger.info(
            f"Full health check for user {user_id}: {len(all_actions_recommended)} total actions"
        )
        return report

    def create_consolidation(
        self,
        user_id: str,
        memories: list[Memory],
        strategy: ConsolidationStrategy,
        llm_client: Optional[Any] = None,
    ) -> Optional[ConsolidationResult]:
        """
        Create a consolidated memory from multiple memories.

        Args:
            user_id: User ID
            memories: Memories to consolidate
            strategy: Consolidation strategy
            llm_client: Optional LLM for intelligent merging

        Returns:
            Consolidation result or None
        """
        if len(memories) < 2:
            return None

        # Extract common elements
        all_tags = set()
        all_texts = []
        total_importance = 0
        total_confidence = 0

        for memory in memories:
            all_tags.update(memory.tags)
            all_texts.append(memory.text)
            total_importance += memory.importance
            total_confidence += memory.confidence

        # Create consolidated text (simple concatenation or LLM-based summary)
        if strategy == ConsolidationStrategy.SEMANTIC_MERGE:
            # Would use LLM to create intelligent summary
            consolidated_text = " | ".join(all_texts[:3])  # Simple for now
        else:
            consolidated_text = " | ".join(all_texts)

        # Calculate preserved content (rough estimate)
        original_length = sum(len(t) for t in all_texts)
        preserved_length = len(consolidated_text)
        content_preserved = min(1.0, preserved_length / original_length)

        result = ConsolidationResult(
            user_id=user_id,
            strategy=strategy,
            original_memory_ids=[m.id for m in memories],
            consolidated_memory_id=None,  # Would be created
            summary=f"Consolidated {len(memories)} memories using {strategy.value}",
            content_preserved=content_preserved,
            metadata={
                "avg_importance": total_importance / len(memories),
                "avg_confidence": total_confidence / len(memories),
                "total_tags": len(all_tags),
            },
        )

        logger.info(f"Created consolidation for {len(memories)} memories using {strategy}")
        return result

    # === HELPER METHODS ===

    def _suggest_tags(self, memory: Memory) -> list[str]:
        """Suggest tags for a memory (simple rule-based)."""
        suggested = set()

        # Add memory type as tag
        suggested.add(memory.type.value)

        # Extract potential tags from text (simple keyword matching)
        keywords = {
            "work": ["work", "job", "office", "meeting", "project"],
            "personal": ["family", "friend", "home", "personal"],
            "health": ["health", "exercise", "diet", "sleep", "medical"],
            "finance": ["money", "finance", "budget", "investment", "payment"],
            "learning": ["learn", "study", "course", "book", "education"],
        }

        text_lower = memory.text.lower()
        for tag, words in keywords.items():
            if any(word in text_lower for word in words):
                suggested.add(tag)

        # Remove existing tags
        suggested = suggested - set(memory.tags)

        return list(suggested)[:3]  # Limit to 3 suggestions
