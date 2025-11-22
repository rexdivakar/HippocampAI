"""Advanced memory operations - clone, batch update, maintenance, archival, and garbage collection."""

import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ArchivalReason(str, Enum):
    """Reasons for archiving a memory."""

    STALE = "stale"
    LOW_IMPORTANCE = "low_importance"
    LOW_CONFIDENCE = "low_confidence"
    DUPLICATE = "duplicate"
    OBSOLETE = "obsolete"
    MANUAL = "manual"
    RETENTION_POLICY = "retention_policy"


class MaintenanceAction(str, Enum):
    """Types of maintenance actions."""

    REFRESH_EMBEDDINGS = "refresh_embeddings"
    UPDATE_METADATA = "update_metadata"
    RECALCULATE_IMPORTANCE = "recalculate_importance"
    CLEANUP_TAGS = "cleanup_tags"
    VALIDATE_INTEGRITY = "validate_integrity"
    OPTIMIZE_STORAGE = "optimize_storage"


class GarbageCollectionPolicy(str, Enum):
    """Garbage collection policies."""

    AGGRESSIVE = "aggressive"  # Remove low-value memories quickly
    MODERATE = "moderate"  # Balanced approach
    CONSERVATIVE = "conservative"  # Keep most memories
    CUSTOM = "custom"  # User-defined thresholds


class CloneOptions(BaseModel):
    """Options for cloning a memory."""

    preserve_id: bool = Field(default=False, description="Keep original ID")
    preserve_timestamps: bool = Field(
        default=False, description="Keep created/updated timestamps"
    )
    preserve_metadata: bool = Field(default=True, description="Copy metadata")
    preserve_tags: bool = Field(default=True, description="Copy tags")
    new_user_id: Optional[str] = Field(
        default=None, description="Assign to different user"
    )
    new_session_id: Optional[str] = Field(
        default=None, description="Assign to different session"
    )
    metadata_overrides: dict[str, Any] = Field(
        default_factory=dict, description="Override specific metadata fields"
    )
    tag_additions: list[str] = Field(
        default_factory=list, description="Add these tags to clone"
    )
    tag_removals: list[str] = Field(
        default_factory=list, description="Remove these tags from clone"
    )


class BatchUpdateFilter(BaseModel):
    """Filter criteria for batch updates."""

    user_ids: Optional[list[str]] = Field(default=None, description="Filter by users")
    session_ids: Optional[list[str]] = Field(
        default=None, description="Filter by sessions"
    )
    memory_types: Optional[list[str]] = Field(
        default=None, description="Filter by types"
    )
    tags_include: Optional[list[str]] = Field(
        default=None, description="Must have all these tags"
    )
    tags_exclude: Optional[list[str]] = Field(
        default=None, description="Must not have these tags"
    )
    importance_min: Optional[float] = Field(
        default=None, description="Minimum importance"
    )
    importance_max: Optional[float] = Field(
        default=None, description="Maximum importance"
    )
    confidence_min: Optional[float] = Field(
        default=None, description="Minimum confidence"
    )
    confidence_max: Optional[float] = Field(
        default=None, description="Maximum confidence"
    )
    created_before: Optional[datetime] = Field(
        default=None, description="Created before this date"
    )
    created_after: Optional[datetime] = Field(
        default=None, description="Created after this date"
    )
    updated_before: Optional[datetime] = Field(
        default=None, description="Updated before this date"
    )
    updated_after: Optional[datetime] = Field(
        default=None, description="Updated after this date"
    )
    metadata_filters: dict[str, Any] = Field(
        default_factory=dict, description="Metadata key-value filters"
    )


class BatchUpdateOperation(BaseModel):
    """Operation to perform in batch update."""

    set_importance: Optional[float] = Field(
        default=None, description="Set importance to this value"
    )
    adjust_importance: Optional[float] = Field(
        default=None, description="Add/subtract from importance"
    )
    set_confidence: Optional[float] = Field(
        default=None, description="Set confidence to this value"
    )
    adjust_confidence: Optional[float] = Field(
        default=None, description="Add/subtract from confidence"
    )
    add_tags: list[str] = Field(default_factory=list, description="Tags to add")
    remove_tags: list[str] = Field(default_factory=list, description="Tags to remove")
    set_metadata: dict[str, Any] = Field(
        default_factory=dict, description="Metadata to set/update"
    )
    remove_metadata_keys: list[str] = Field(
        default_factory=list, description="Metadata keys to remove"
    )
    archive: bool = Field(default=False, description="Archive matched memories")
    archive_reason: Optional[ArchivalReason] = Field(
        default=None, description="Reason for archival"
    )


class BatchUpdateResult(BaseModel):
    """Result of batch update operation."""

    total_matched: int = Field(description="Number of memories matched by filter")
    total_updated: int = Field(description="Number of memories actually updated")
    total_archived: int = Field(default=0, description="Number of memories archived")
    updated_memory_ids: list[str] = Field(
        default_factory=list, description="IDs of updated memories"
    )
    archived_memory_ids: list[str] = Field(
        default_factory=list, description="IDs of archived memories"
    )
    errors: list[str] = Field(default_factory=list, description="Errors encountered")
    execution_time_ms: float = Field(description="Execution time in milliseconds")


class MaintenanceSchedule(BaseModel):
    """Schedule for automatic memory maintenance."""

    schedule_id: str = Field(default_factory=lambda: f"schedule_{uuid4().hex[:8]}")
    action: MaintenanceAction = Field(description="Type of maintenance action")
    interval_hours: int = Field(description="Hours between executions")
    last_run: Optional[datetime] = Field(
        default=None, description="Last execution timestamp"
    )
    next_run: datetime = Field(description="Next scheduled execution")
    enabled: bool = Field(default=True, description="Whether schedule is active")
    filter_criteria: Optional[BatchUpdateFilter] = Field(
        default=None, description="Optional filter for targeted maintenance"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional configuration"
    )


class ArchivalPolicy(BaseModel):
    """Policy for automatic memory archival."""

    policy_id: str = Field(default_factory=lambda: f"policy_{uuid4().hex[:8]}")
    name: str = Field(description="Human-readable policy name")
    enabled: bool = Field(default=True, description="Whether policy is active")
    stale_threshold_days: Optional[int] = Field(
        default=None, description="Archive if not updated in N days"
    )
    low_importance_threshold: Optional[float] = Field(
        default=None, description="Archive if importance below this"
    )
    low_confidence_threshold: Optional[float] = Field(
        default=None, description="Archive if confidence below this"
    )
    access_count_threshold: Optional[int] = Field(
        default=None, description="Archive if accessed fewer than N times"
    )
    filter_criteria: Optional[BatchUpdateFilter] = Field(
        default=None, description="Additional filter criteria"
    )
    archive_reason: ArchivalReason = Field(
        default=ArchivalReason.RETENTION_POLICY, description="Reason for archival"
    )
    dry_run: bool = Field(
        default=False, description="Preview without actually archiving"
    )


class GarbageCollectionConfig(BaseModel):
    """Configuration for garbage collection."""

    policy: GarbageCollectionPolicy = Field(
        default=GarbageCollectionPolicy.MODERATE, description="GC policy"
    )
    min_age_days: int = Field(
        default=30, description="Minimum age before considering for GC"
    )
    importance_threshold: float = Field(
        default=3.0, description="Keep memories with importance above this"
    )
    confidence_threshold: float = Field(
        default=0.5, description="Keep memories with confidence above this"
    )
    access_count_threshold: int = Field(
        default=1, description="Keep memories accessed more than this"
    )
    preserve_types: list[str] = Field(
        default_factory=lambda: ["preference", "decision"],
        description="Never GC these types",
    )
    preserve_tags: list[str] = Field(
        default_factory=list, description="Never GC memories with these tags"
    )
    max_memories_per_run: int = Field(
        default=1000, description="Maximum memories to GC in one run"
    )
    dry_run: bool = Field(
        default=False, description="Preview without actually deleting"
    )


class GarbageCollectionResult(BaseModel):
    """Result of garbage collection operation."""

    total_evaluated: int = Field(description="Total memories evaluated")
    total_collected: int = Field(description="Total memories garbage collected")
    collected_memory_ids: list[str] = Field(
        default_factory=list, description="IDs of collected memories"
    )
    space_reclaimed_bytes: int = Field(
        default=0, description="Estimated space reclaimed"
    )
    execution_time_ms: float = Field(description="Execution time in milliseconds")
    dry_run: bool = Field(default=False, description="Was this a dry run")


class MemoryOperations:
    """Advanced memory operations manager."""

    def __init__(self) -> None:
        """Initialize memory operations manager."""
        self.maintenance_schedules: dict[str, MaintenanceSchedule] = {}
        self.archival_policies: dict[str, ArchivalPolicy] = {}

    def clone_memory(
        self, memory: dict[str, Any], options: Optional[CloneOptions] = None
    ) -> dict[str, Any]:
        """
        Clone a memory with optional modifications.

        Args:
            memory: Memory to clone
            options: Clone options

        Returns:
            Cloned memory dict
        """
        options = options or CloneOptions()

        # Create base clone
        cloned = memory.copy()

        # Handle ID
        if not options.preserve_id:
            cloned["id"] = f"mem_{uuid4().hex}"

        # Handle timestamps
        if not options.preserve_timestamps:
            now = datetime.now(timezone.utc)
            cloned["created_at"] = now
            cloned["updated_at"] = now
        else:
            # Ensure timestamps are datetime objects
            if isinstance(cloned.get("created_at"), str):
                cloned["created_at"] = datetime.fromisoformat(
                    cloned["created_at"].replace("Z", "+00:00")
                )
            if isinstance(cloned.get("updated_at"), str):
                cloned["updated_at"] = datetime.fromisoformat(
                    cloned["updated_at"].replace("Z", "+00:00")
                )

        # Handle user/session
        if options.new_user_id:
            cloned["user_id"] = options.new_user_id
        if options.new_session_id:
            cloned["session_id"] = options.new_session_id

        # Handle metadata
        if options.preserve_metadata:
            cloned["metadata"] = memory.get("metadata", {}).copy()
            # Apply overrides
            cloned["metadata"].update(options.metadata_overrides)
        else:
            cloned["metadata"] = options.metadata_overrides.copy()

        # Handle tags
        if options.preserve_tags:
            cloned["tags"] = list(memory.get("tags", []))
            # Add new tags
            for tag in options.tag_additions:
                if tag not in cloned["tags"]:
                    cloned["tags"].append(tag)
            # Remove tags
            cloned["tags"] = [
                tag for tag in cloned["tags"] if tag not in options.tag_removals
            ]
        else:
            cloned["tags"] = list(options.tag_additions)

        # Add clone metadata
        cloned["metadata"]["cloned_from"] = memory["id"]
        cloned["metadata"]["clone_timestamp"] = datetime.now(timezone.utc).isoformat()

        logger.info(f"Cloned memory {memory['id']} to {cloned['id']}")

        return cloned

    def create_template_memory(
        self,
        template_name: str,
        text_template: str,
        default_importance: float = 5.0,
        default_confidence: float = 0.7,
        default_type: str = "fact",
        default_tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Create a memory template that can be instantiated with variables.

        Args:
            template_name: Name of the template
            text_template: Text with placeholders like {variable}
            default_importance: Default importance score
            default_confidence: Default confidence score
            default_type: Default memory type
            default_tags: Default tags
            metadata: Template metadata

        Returns:
            Template memory dict
        """
        template = {
            "id": f"template_{template_name}_{uuid4().hex[:8]}",
            "template_name": template_name,
            "text_template": text_template,
            "is_template": True,
            "default_importance": default_importance,
            "default_confidence": default_confidence,
            "default_type": default_type,
            "default_tags": default_tags or [],
            "metadata": metadata or {},
            "created_at": datetime.now(timezone.utc),
        }

        logger.info(f"Created memory template: {template_name}")

        return template

    def instantiate_template(
        self,
        template: dict[str, Any],
        variables: dict[str, str],
        user_id: str,
        session_id: Optional[str] = None,
        overrides: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Instantiate a memory from a template.

        Args:
            template: Template memory
            variables: Variables to fill in template
            user_id: User ID for new memory
            session_id: Optional session ID
            overrides: Override default values

        Returns:
            Instantiated memory dict
        """
        overrides = overrides or {}

        # Fill in text template
        text = template["text_template"]
        for key, value in variables.items():
            text = text.replace(f"{{{key}}}", str(value))

        # Create memory from template
        memory = {
            "id": f"mem_{uuid4().hex}",
            "text": text,
            "user_id": user_id,
            "session_id": session_id,
            "type": overrides.get("type", template["default_type"]),
            "importance": overrides.get("importance", template["default_importance"]),
            "confidence": overrides.get("confidence", template["default_confidence"]),
            "tags": overrides.get("tags", template["default_tags"].copy()),
            "metadata": {
                "from_template": template["template_name"],
                "template_id": template["id"],
                **template.get("metadata", {}),
                **overrides.get("metadata", {}),
            },
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
        }

        logger.info(
            f"Instantiated memory from template {template['template_name']}: {memory['id']}"
        )

        return memory

    def matches_filter(
        self, memory: dict[str, Any], filter_criteria: BatchUpdateFilter
    ) -> bool:
        """
        Check if a memory matches filter criteria.

        Args:
            memory: Memory to check
            filter_criteria: Filter criteria

        Returns:
            True if memory matches all criteria
        """
        # User ID filter
        if filter_criteria.user_ids and memory.get("user_id") not in filter_criteria.user_ids:
            return False

        # Session ID filter
        if (
            filter_criteria.session_ids
            and memory.get("session_id") not in filter_criteria.session_ids
        ):
            return False

        # Memory type filter
        if filter_criteria.memory_types:
            mem_type = memory.get("type")
            if mem_type is not None and hasattr(mem_type, "value"):
                mem_type = mem_type.value
            if mem_type not in filter_criteria.memory_types:
                return False

        # Tags filter - include
        if filter_criteria.tags_include:
            memory_tags = set(memory.get("tags", []))
            if not all(tag in memory_tags for tag in filter_criteria.tags_include):
                return False

        # Tags filter - exclude
        if filter_criteria.tags_exclude:
            memory_tags = set(memory.get("tags", []))
            if any(tag in memory_tags for tag in filter_criteria.tags_exclude):
                return False

        # Importance filter
        importance = memory.get("importance", 0.0)
        if filter_criteria.importance_min is not None and importance < filter_criteria.importance_min:
            return False
        if filter_criteria.importance_max is not None and importance > filter_criteria.importance_max:
            return False

        # Confidence filter
        confidence = memory.get("confidence", 0.0)
        if filter_criteria.confidence_min is not None and confidence < filter_criteria.confidence_min:
            return False
        if filter_criteria.confidence_max is not None and confidence > filter_criteria.confidence_max:
            return False

        # Timestamp filters
        created_at = memory.get("created_at")
        if created_at:
            if isinstance(created_at, str):
                created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            if created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=timezone.utc)

            if filter_criteria.created_before and created_at > filter_criteria.created_before:
                return False
            if filter_criteria.created_after and created_at < filter_criteria.created_after:
                return False

        updated_at = memory.get("updated_at")
        if updated_at:
            if isinstance(updated_at, str):
                updated_at = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
            if updated_at.tzinfo is None:
                updated_at = updated_at.replace(tzinfo=timezone.utc)

            if filter_criteria.updated_before and updated_at > filter_criteria.updated_before:
                return False
            if filter_criteria.updated_after and updated_at < filter_criteria.updated_after:
                return False

        # Metadata filters
        memory_metadata = memory.get("metadata", {})
        for key, value in filter_criteria.metadata_filters.items():
            if memory_metadata.get(key) != value:
                return False

        return True

    def batch_update(
        self,
        memories: list[dict[str, Any]],
        filter_criteria: BatchUpdateFilter,
        operation: BatchUpdateOperation,
    ) -> BatchUpdateResult:
        """
        Perform batch update on memories matching filter.

        Args:
            memories: List of memories to process
            filter_criteria: Filter criteria
            operation: Update operation to perform

        Returns:
            Batch update result
        """
        start_time = datetime.now()

        result = BatchUpdateResult(
            total_matched=0, total_updated=0, execution_time_ms=0.0
        )

        for memory in memories:
            if not self.matches_filter(memory, filter_criteria):
                continue

            result.total_matched += 1

            try:
                # Apply importance operations
                if operation.set_importance is not None:
                    memory["importance"] = max(
                        0.0, min(10.0, operation.set_importance)
                    )
                elif operation.adjust_importance is not None:
                    memory["importance"] = max(
                        0.0,
                        min(
                            10.0,
                            memory.get("importance", 5.0) + operation.adjust_importance,
                        ),
                    )

                # Apply confidence operations
                if operation.set_confidence is not None:
                    memory["confidence"] = max(0.0, min(1.0, operation.set_confidence))
                elif operation.adjust_confidence is not None:
                    memory["confidence"] = max(
                        0.0,
                        min(
                            1.0,
                            memory.get("confidence", 0.5) + operation.adjust_confidence,
                        ),
                    )

                # Tag operations
                if operation.add_tags:
                    memory_tags = set(memory.get("tags", []))
                    memory_tags.update(operation.add_tags)
                    memory["tags"] = list(memory_tags)

                if operation.remove_tags:
                    memory["tags"] = [
                        tag
                        for tag in memory.get("tags", [])
                        if tag not in operation.remove_tags
                    ]

                # Metadata operations
                if operation.set_metadata:
                    if "metadata" not in memory:
                        memory["metadata"] = {}
                    memory["metadata"].update(operation.set_metadata)

                if operation.remove_metadata_keys:
                    for key in operation.remove_metadata_keys:
                        memory.get("metadata", {}).pop(key, None)

                # Archival
                if operation.archive:
                    memory["archived"] = True
                    memory["archived_at"] = datetime.now(timezone.utc)
                    memory["archive_reason"] = (
                        operation.archive_reason.value
                        if operation.archive_reason
                        else ArchivalReason.MANUAL.value
                    )
                    result.total_archived += 1
                    result.archived_memory_ids.append(memory["id"])

                # Update timestamp
                memory["updated_at"] = datetime.now(timezone.utc)

                result.total_updated += 1
                result.updated_memory_ids.append(memory["id"])

            except Exception as e:
                error_msg = f"Failed to update memory {memory.get('id')}: {e}"
                logger.error(error_msg)
                result.errors.append(error_msg)

        end_time = datetime.now()
        result.execution_time_ms = (end_time - start_time).total_seconds() * 1000

        logger.info(
            f"Batch update: {result.total_updated}/{result.total_matched} memories updated "
            f"in {result.execution_time_ms:.2f}ms"
        )

        return result

    def schedule_maintenance(
        self, schedule: MaintenanceSchedule
    ) -> MaintenanceSchedule:
        """
        Schedule automatic memory maintenance.

        Args:
            schedule: Maintenance schedule configuration

        Returns:
            Created schedule
        """
        self.maintenance_schedules[schedule.schedule_id] = schedule

        logger.info(
            f"Scheduled {schedule.action.value} maintenance "
            f"every {schedule.interval_hours} hours"
        )

        return schedule

    def check_maintenance_due(self) -> list[MaintenanceSchedule]:
        """
        Check which maintenance schedules are due for execution.

        Returns:
            List of schedules that should run
        """
        now = datetime.now(timezone.utc)
        due_schedules = []

        for schedule in self.maintenance_schedules.values():
            if not schedule.enabled:
                continue

            if schedule.next_run <= now:
                due_schedules.append(schedule)

        return due_schedules

    def create_archival_policy(self, policy: ArchivalPolicy) -> ArchivalPolicy:
        """
        Create automatic archival policy.

        Args:
            policy: Archival policy configuration

        Returns:
            Created policy
        """
        self.archival_policies[policy.policy_id] = policy

        logger.info(f"Created archival policy: {policy.name}")

        return policy

    def apply_archival_policies(
        self, memories: list[dict[str, Any]], policies: Optional[list[ArchivalPolicy]] = None
    ) -> BatchUpdateResult:
        """
        Apply archival policies to memories.

        Args:
            memories: List of memories to evaluate
            policies: Optional list of policies (uses all if not specified)

        Returns:
            Batch update result with archival statistics
        """
        policies = policies or list(self.archival_policies.values())
        now = datetime.now(timezone.utc)

        total_result = BatchUpdateResult(
            total_matched=0, total_updated=0, execution_time_ms=0.0
        )
        start_time = datetime.now()

        for policy in policies:
            if not policy.enabled:
                continue

            # Build filter for this policy
            filter_criteria = policy.filter_criteria or BatchUpdateFilter()

            for memory in memories:
                # Skip already archived
                if memory.get("archived"):
                    continue

                # Check policy conditions
                should_archive = False

                # Stale check
                if policy.stale_threshold_days is not None:
                    updated_at = memory.get("updated_at")
                    if isinstance(updated_at, str):
                        updated_at = datetime.fromisoformat(
                            updated_at.replace("Z", "+00:00")
                        )
                    if updated_at is not None and updated_at.tzinfo is None:
                        updated_at = updated_at.replace(tzinfo=timezone.utc)

                    if updated_at is not None:
                        days_stale = (now - updated_at).total_seconds() / 86400
                        if days_stale > policy.stale_threshold_days:
                            should_archive = True

                # Low importance check
                if (
                    policy.low_importance_threshold is not None
                    and memory.get("importance", 10.0)
                    < policy.low_importance_threshold
                ):
                    should_archive = True

                # Low confidence check
                if (
                    policy.low_confidence_threshold is not None
                    and memory.get("confidence", 1.0)
                    < policy.low_confidence_threshold
                ):
                    should_archive = True

                # Access count check
                if (
                    policy.access_count_threshold is not None
                    and memory.get("access_count", 0)
                    < policy.access_count_threshold
                ):
                    should_archive = True

                # Additional filter check
                if should_archive and not self.matches_filter(memory, filter_criteria):
                    should_archive = False

                if should_archive:
                    total_result.total_matched += 1

                    if not policy.dry_run:
                        memory["archived"] = True
                        memory["archived_at"] = now
                        memory["archive_reason"] = policy.archive_reason.value
                        memory["updated_at"] = now
                        total_result.total_updated += 1
                        total_result.total_archived += 1
                        total_result.archived_memory_ids.append(memory["id"])

        end_time = datetime.now()
        total_result.execution_time_ms = (
            end_time - start_time
        ).total_seconds() * 1000

        logger.info(
            f"Archival policies: {total_result.total_archived} memories archived "
            f"out of {total_result.total_matched} matched"
        )

        return total_result

    def garbage_collect(
        self, memories: list[dict[str, Any]], config: GarbageCollectionConfig
    ) -> GarbageCollectionResult:
        """
        Perform garbage collection on memories.

        Args:
            memories: List of memories to evaluate
            config: GC configuration

        Returns:
            Garbage collection result
        """
        start_time = datetime.now()
        now = datetime.now(timezone.utc)

        result = GarbageCollectionResult(
            total_evaluated=0, total_collected=0, execution_time_ms=0.0, dry_run=config.dry_run
        )

        # Adjust thresholds based on policy
        if config.policy == GarbageCollectionPolicy.AGGRESSIVE:
            importance_threshold = config.importance_threshold + 1.0
            confidence_threshold = config.confidence_threshold + 0.1
            access_threshold = config.access_count_threshold + 2
        elif config.policy == GarbageCollectionPolicy.CONSERVATIVE:
            importance_threshold = max(0.0, config.importance_threshold - 1.0)
            confidence_threshold = max(0.0, config.confidence_threshold - 0.1)
            access_threshold = max(0, config.access_count_threshold - 1)
        else:  # MODERATE or CUSTOM
            importance_threshold = config.importance_threshold
            confidence_threshold = config.confidence_threshold
            access_threshold = config.access_count_threshold

        collected_count = 0

        for memory in memories:
            if collected_count >= config.max_memories_per_run:
                break

            result.total_evaluated += 1

            # Skip if already archived or deleted
            if memory.get("archived") or memory.get("deleted"):
                continue

            # Skip preserved types
            mem_type = memory.get("type")
            if mem_type is not None and hasattr(mem_type, "value"):
                mem_type = mem_type.value
            if mem_type in config.preserve_types:
                continue

            # Skip preserved tags
            memory_tags = set(memory.get("tags", []))
            if any(tag in memory_tags for tag in config.preserve_tags):
                continue

            # Check age
            created_at = memory.get("created_at")
            if isinstance(created_at, str):
                created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            if created_at is not None and created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=timezone.utc)

            if created_at is not None:
                age_days = (now - created_at).total_seconds() / 86400
                if age_days < config.min_age_days:
                    continue

            # Evaluate for collection
            should_collect = False

            importance = memory.get("importance", 10.0)
            confidence = memory.get("confidence", 1.0)
            access_count = memory.get("access_count", 0)

            if (
                importance < importance_threshold
                and confidence < confidence_threshold
                and access_count <= access_threshold
            ):
                should_collect = True

            if should_collect:
                result.total_collected += 1
                result.collected_memory_ids.append(memory["id"])
                collected_count += 1

                # Estimate space reclaimed (text length)
                text_size = len(memory.get("text", "").encode("utf-8"))
                result.space_reclaimed_bytes += text_size

                if not config.dry_run:
                    # Mark for deletion
                    memory["deleted"] = True
                    memory["deleted_at"] = now
                    memory["deletion_reason"] = "garbage_collection"

        end_time = datetime.now()
        result.execution_time_ms = (end_time - start_time).total_seconds() * 1000

        mode = "DRY RUN: " if config.dry_run else ""
        logger.info(
            f"{mode}Garbage collection: {result.total_collected} memories collected "
            f"out of {result.total_evaluated} evaluated "
            f"({result.space_reclaimed_bytes} bytes reclaimed)"
        )

        return result
