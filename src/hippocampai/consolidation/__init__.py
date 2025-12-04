"""Memory consolidation (Sleep Phase) module for HippocampAI."""

from hippocampai.consolidation.models import (
    ConsolidationDecision,
    ConsolidationRun,
    ConsolidationStats,
    ConsolidationStatus,
    MemoryCluster,
    MemoryConsolidationFields,
)
from hippocampai.consolidation.policy import (
    ConsolidationPolicy,
    ConsolidationPolicyEngine,
    apply_consolidation_decisions,
)
from hippocampai.consolidation.prompts import (
    CONSOLIDATION_SYSTEM_MESSAGE,
    build_cluster_theme_prompt,
    build_consolidation_prompt,
    build_simple_review_prompt,
    build_synthetic_memory_prompt,
)
from hippocampai.consolidation.tasks import (
    cluster_memories,
    collect_recent_memories,
    consolidate_user_memories,
    run_daily_consolidation,
)

__all__ = [
    # Models
    "ConsolidationRun",
    "ConsolidationStatus",
    "ConsolidationDecision",
    "ConsolidationStats",
    "MemoryCluster",
    "MemoryConsolidationFields",
    # Policy
    "ConsolidationPolicy",
    "ConsolidationPolicyEngine",
    "apply_consolidation_decisions",
    # Prompts
    "CONSOLIDATION_SYSTEM_MESSAGE",
    "build_consolidation_prompt",
    "build_simple_review_prompt",
    "build_cluster_theme_prompt",
    "build_synthetic_memory_prompt",
    # Tasks
    "run_daily_consolidation",
    "consolidate_user_memories",
    "collect_recent_memories",
    "cluster_memories",
]
