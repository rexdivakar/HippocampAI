"""Automation controller for SaaS and library integration.

Provides unified control over automated memory optimization tasks.
Library users can configure policies that the SaaS platform will execute.
"""

import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional, cast

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class PolicyType(str, Enum):
    """Types of automation policies."""

    THRESHOLD = "threshold"  # Trigger based on thresholds
    SCHEDULE = "schedule"  # Trigger on schedule
    CONTINUOUS = "continuous"  # Run continuously
    MANUAL = "manual"  # Only manual triggers


class AutomationSchedule(BaseModel):
    """Schedule configuration for automation."""

    enabled: bool = True
    cron_expression: Optional[str] = None  # e.g., "0 2 * * *" for daily 2 AM
    interval_hours: Optional[int] = None  # Alternative: run every N hours
    run_immediately: bool = False  # Run once immediately on enable


class AutomationPolicy(BaseModel):
    """Policy configuration for automated features."""

    # Core settings
    policy_id: str = Field(
        default_factory=lambda: f"policy_{int(datetime.now(timezone.utc).timestamp())}"
    )
    user_id: str
    policy_type: PolicyType = PolicyType.THRESHOLD
    enabled: bool = True

    # Feature toggles
    auto_summarization: bool = True
    auto_consolidation: bool = True
    auto_compression: bool = False
    importance_decay: bool = True
    health_monitoring: bool = True
    conflict_resolution: bool = True

    # Summarization settings
    summarization_threshold: int = 500  # Number of memories to trigger
    summarization_age_days: int = 30  # Only summarize memories older than N days
    summarization_schedule: Optional[AutomationSchedule] = None

    # Consolidation settings
    consolidation_threshold: int = 300  # Number of memories to trigger
    consolidation_similarity: float = 0.85
    consolidation_schedule: Optional[AutomationSchedule] = None

    # Compression settings
    compression_threshold: int = 1000  # Number of memories to trigger
    compression_target_reduction: float = 0.30  # 30% reduction
    compression_schedule: Optional[AutomationSchedule] = None

    # Decay settings
    decay_half_life_days: int = 90
    decay_schedule: Optional[AutomationSchedule] = None

    # Health monitoring settings
    health_check_schedule: Optional[AutomationSchedule] = None
    health_alert_threshold: float = 60.0  # Alert if health score below this

    # Conflict resolution settings
    conflict_strategy: str = "temporal"  # temporal, confidence, importance
    auto_resolve_conflicts: bool = True

    # Metadata
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AutomationController:
    """
    Central controller for automated memory optimization.

    This controller can be used by:
    1. Library users - Configure and trigger automation programmatically
    2. SaaS platform - Background workers read policies and execute tasks
    3. API endpoints - REST API for policy management
    """

    def __init__(
        self,
        memory_service: Any,
        llm: Any = None,
        embedder: Any = None,
        storage_backend: Any = None,
    ) -> None:
        """
        Initialize automation controller.

        Args:
            memory_service: Memory service instance
            llm: LLM provider (optional, uses memory_service's LLM if not provided)
            embedder: Embedder instance
            storage_backend: Where to store policies (Redis, DB, etc.)
        """
        self.memory_service = memory_service
        self.llm = llm or getattr(memory_service, "llm", None)
        self.embedder = embedder

        # Storage backend for policies (default to in-memory)
        self.storage_backend = storage_backend or {}
        self.policies: dict[str, Any] = {}  # In-memory policy store

        # Initialize feature modules (lazy loaded)
        self._summarizer: Any = None
        self._consolidator: Any = None
        self._compressor: Any = None
        self._decay: Any = None
        self._health_monitor: Any = None
        self._conflict_resolver: Any = None

        logger.info("AutomationController initialized")

    @property
    def summarizer(self) -> Any:
        """Lazy load auto-summarization."""
        if self._summarizer is None:
            from hippocampai.pipeline import AutoSummarizer

            self._summarizer = AutoSummarizer(
                llm=self.llm,
            )
        return self._summarizer

    @property
    def consolidator(self) -> Any:
        """Lazy load auto-consolidation."""
        if self._consolidator is None:
            from hippocampai.pipeline import AutoConsolidator

            self._consolidator = AutoConsolidator()
        return self._consolidator

    @property
    def compressor(self) -> Any:
        """Lazy load advanced compression."""
        if self._compressor is None:
            from hippocampai.pipeline import AdvancedCompressor

            self._compressor = AdvancedCompressor(
                llm=self.llm,
            )
        return self._compressor

    @property
    def decay(self) -> Any:
        """Lazy load importance decay."""
        if self._decay is None:
            from hippocampai.pipeline import ImportanceDecayEngine

            self._decay = ImportanceDecayEngine()
        return self._decay

    @property
    def health_monitor(self) -> Any:
        """Lazy load health monitor."""
        if self._health_monitor is None:
            from hippocampai.monitoring import MemoryHealthMonitor

            self._health_monitor = MemoryHealthMonitor(embedder=self.embedder)
        return self._health_monitor

    @property
    def conflict_resolver(self) -> Any:
        """Lazy load conflict resolver."""
        if self._conflict_resolver is None:
            from hippocampai.pipeline import MemoryConflictResolver

            self._conflict_resolver = MemoryConflictResolver(
                embedder=self.embedder,
                llm=self.llm,
            )
        return self._conflict_resolver

    def create_policy(self, policy: AutomationPolicy) -> AutomationPolicy:
        """
        Create or update automation policy.

        Args:
            policy: Policy configuration

        Returns:
            Created/updated policy
        """
        policy.updated_at = datetime.now(timezone.utc)
        self.policies[policy.user_id] = policy

        logger.info(f"Created/updated policy for user {policy.user_id}")
        return policy

    def get_policy(self, user_id: str) -> Optional[AutomationPolicy]:
        """Get automation policy for user."""
        return self.policies.get(user_id)

    def delete_policy(self, user_id: str) -> bool:
        """Delete automation policy for user."""
        if user_id in self.policies:
            del self.policies[user_id]
            logger.info(f"Deleted policy for user {user_id}")
            return True
        return False

    def should_run_summarization(self, user_id: str) -> bool:
        """Check if summarization should run for user."""
        policy = self.get_policy(user_id)
        if not policy or not policy.enabled or not policy.auto_summarization:
            return False

        # Check threshold
        if policy.policy_type == PolicyType.THRESHOLD:
            stats = self.memory_service.get_memory_statistics(user_id=user_id)
            memory_count = int(stats.get("total_memories", 0))
            return memory_count >= policy.summarization_threshold

        return True

    def should_run_consolidation(self, user_id: str) -> bool:
        """Check if consolidation should run for user."""
        policy = self.get_policy(user_id)
        if not policy or not policy.enabled or not policy.auto_consolidation:
            return False

        if policy.policy_type == PolicyType.THRESHOLD:
            stats = self.memory_service.get_memory_statistics(user_id=user_id)
            memory_count = int(stats.get("total_memories", 0))
            return memory_count >= policy.consolidation_threshold

        return True

    def should_run_compression(self, user_id: str) -> bool:
        """Check if compression should run for user."""
        policy = self.get_policy(user_id)
        if not policy or not policy.enabled or not policy.auto_compression:
            return False

        if policy.policy_type == PolicyType.THRESHOLD:
            stats = self.memory_service.get_memory_statistics(user_id=user_id)
            memory_count = int(stats.get("total_memories", 0))
            return memory_count >= policy.compression_threshold

        return True

    def run_summarization(self, user_id: str, force: bool = False) -> dict[str, Any]:
        """
        Run auto-summarization for user.

        Args:
            user_id: User identifier
            force: Force run even if policy says no

        Returns:
            Summarization results
        """
        if not force and not self.should_run_summarization(user_id):
            return {"status": "skipped", "reason": "policy check failed"}

        policy = self.get_policy(user_id)
        time_window_days = policy.summarization_age_days if policy else 30

        logger.info(f"Running summarization for user {user_id}")
        result = self.summarizer.summarize_memories(
            user_id=user_id, time_window_days=time_window_days
        )
        result["status"] = "success"
        return cast(dict[str, Any], result)

    def run_consolidation(self, user_id: str, force: bool = False) -> dict[str, Any]:
        """
        Run auto-consolidation for user.

        Args:
            user_id: User identifier
            force: Force run even if policy says no

        Returns:
            Consolidation results
        """
        if not force and not self.should_run_consolidation(user_id):
            return {"status": "skipped", "reason": "policy check failed"}

        policy = self.get_policy(user_id)
        similarity = policy.consolidation_similarity if policy else 0.85

        logger.info(f"Running consolidation for user {user_id}")
        result = self.consolidator.consolidate_memories(
            user_id=user_id, similarity_threshold=similarity
        )
        result["status"] = "success"
        return cast(dict[str, Any], result)

    def run_compression(self, user_id: str, force: bool = False) -> dict[str, Any]:
        """
        Run advanced compression for user.

        Args:
            user_id: User identifier
            force: Force run even if policy says no

        Returns:
            Compression results
        """
        if not force and not self.should_run_compression(user_id):
            return {"status": "skipped", "reason": "policy check failed"}

        policy = self.get_policy(user_id)
        target_reduction = policy.compression_target_reduction if policy else 0.30

        logger.info(f"Running compression for user {user_id}")
        result = self.compressor.compress_memories(
            user_id=user_id, target_reduction=target_reduction
        )
        result["status"] = "success"
        return cast(dict[str, Any], result)

    def run_decay(self, user_id: str, force: bool = False) -> dict[str, Any]:
        """
        Run importance decay for user.

        Args:
            user_id: User identifier
            force: Force run even if policy says no

        Returns:
            Decay results
        """
        policy = self.get_policy(user_id)
        if not force and (not policy or not policy.enabled or not policy.importance_decay):
            return {"status": "skipped", "reason": "policy check failed"}

        half_life = policy.decay_half_life_days if policy else 90

        logger.info(f"Running decay for user {user_id}")
        result = self.decay.apply_decay(user_id=user_id, half_life_days=half_life)
        result["status"] = "success"
        return cast(dict[str, Any], result)

    def run_health_check(self, user_id: str, force: bool = False) -> dict[str, Any]:
        """
        Run health check for user.

        Args:
            user_id: User identifier
            force: Force run even if policy says no

        Returns:
            Health check results
        """
        policy = self.get_policy(user_id)
        if not force and (not policy or not policy.enabled or not policy.health_monitoring):
            return {"status": "skipped", "reason": "policy check failed"}

        logger.info(f"Running health check for user {user_id}")

        # Get all memories
        memories = self.memory_service.get_memories(user_id=user_id, limit=1000)

        if not memories:
            return {"status": "skipped", "reason": "no memories"}

        # Generate report
        report = self.health_monitor.generate_quality_report(memories, user_id=user_id)

        # Check if alert needed
        alert_needed = False
        if policy and report.health_score.overall_score < policy.health_alert_threshold:
            alert_needed = True

        return {
            "status": "success",
            "health_score": report.health_score.overall_score,
            "health_status": report.health_score.status,
            "alert_needed": alert_needed,
            "recommendations": report.health_score.recommendations,
        }

    def run_all_optimizations(self, user_id: str, force: bool = False) -> dict[str, Any]:
        """
        Run all enabled optimizations for user.

        Args:
            user_id: User identifier
            force: Force run even if policies say no

        Returns:
            Combined results
        """
        logger.info(f"Running all optimizations for user {user_id}")

        results = {}

        # Run each optimization
        results["summarization"] = self.run_summarization(user_id, force)
        results["consolidation"] = self.run_consolidation(user_id, force)
        results["compression"] = self.run_compression(user_id, force)
        results["decay"] = self.run_decay(user_id, force)
        results["health_check"] = self.run_health_check(user_id, force)

        return results

    def get_user_statistics(self, user_id: str) -> dict[str, Any]:
        """
        Get comprehensive statistics for user.

        Args:
            user_id: User identifier

        Returns:
            Statistics dictionary
        """
        stats = self.memory_service.get_memory_statistics(user_id=user_id)

        # Add policy info
        policy = self.get_policy(user_id)
        if policy:
            stats["automation_enabled"] = policy.enabled
            stats["automation_features"] = {
                "summarization": policy.auto_summarization,
                "consolidation": policy.auto_consolidation,
                "compression": policy.auto_compression,
                "decay": policy.importance_decay,
                "health_monitoring": policy.health_monitoring,
            }

        return cast(dict[str, Any], stats)
