"""Policy engine for memory consolidation decisions."""

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from hippocampai.models import Memory, MemoryType

logger = logging.getLogger(__name__)


@dataclass
class ConsolidationPolicy:
    """Configuration for consolidation policies."""

    # Deletion thresholds
    min_importance_to_keep: float = float(os.getenv("CONSOLIDATION_MIN_IMPORTANCE", "3.0"))
    min_age_days_for_deletion: int = int(os.getenv("CONSOLIDATION_MIN_AGE_DAYS", "7"))
    never_delete_if_accessed: bool = os.getenv("CONSOLIDATION_PROTECT_ACCESSED", "true").lower() == "true"

    # Archival thresholds
    archive_importance_threshold: float = float(os.getenv("CONSOLIDATION_ARCHIVE_THRESHOLD", "4.0"))
    archive_age_days: int = int(os.getenv("CONSOLIDATION_ARCHIVE_AGE_DAYS", "30"))

    # Promotion thresholds
    promotion_multiplier: float = float(os.getenv("CONSOLIDATION_PROMOTION_MULTIPLIER", "1.3"))
    max_importance_after_promotion: float = 10.0

    # Memory limits
    max_memories_per_user: int = int(os.getenv("CONSOLIDATION_MAX_MEMORIES_PER_USER", "10000"))
    max_memories_per_type: dict[str, int] | None = None

    # Type-specific rules
    never_delete_types: set[MemoryType] = None
    always_archive_types: set[MemoryType] = None

    def __post_init__(self):
        """Initialize default type-specific rules."""
        if self.never_delete_types is None:
            # Never auto-delete goals, preferences, or important facts
            self.never_delete_types = {
                MemoryType.GOAL,
                MemoryType.PREFERENCE,
                MemoryType.FACT,
            }

        if self.always_archive_types is None:
            # Always archive (don't delete) events and context
            self.always_archive_types = {
                MemoryType.EVENT,
                MemoryType.CONTEXT,
            }

        if self.max_memories_per_type is None:
            self.max_memories_per_type = {
                "event": 5000,
                "context": 3000,
                "fact": 2000,
                "preference": 1000,
                "goal": 500,
                "habit": 500,
            }


class ConsolidationPolicyEngine:
    """Applies consolidation policies to memories."""

    def __init__(self, policy: ConsolidationPolicy | None = None):
        """Initialize the policy engine."""
        self.policy = policy or ConsolidationPolicy()
        self.logger = logging.getLogger(f"{__name__}.PolicyEngine")

    def should_delete(self, memory: Memory) -> tuple[bool, str]:
        """
        Determine if a memory should be deleted.

        Returns:
            (should_delete: bool, reason: str)
        """
        # Never delete protected types
        if memory.type in self.policy.never_delete_types:
            return False, f"Protected type: {memory.type}"

        # Never delete if accessed (if policy says so)
        if self.policy.never_delete_if_accessed and memory.access_count > 0:
            return False, f"Memory accessed {memory.access_count} times"

        # Check age requirement
        age_days = (datetime.now(timezone.utc) - memory.created_at).days
        if age_days < self.policy.min_age_days_for_deletion:
            return False, f"Too recent: {age_days} days old"

        # Check importance
        if memory.importance < self.policy.min_importance_to_keep:
            return True, f"Low importance: {memory.importance:.1f} < {self.policy.min_importance_to_keep}"

        return False, "Does not meet deletion criteria"

    def should_archive(self, memory: Memory) -> tuple[bool, str]:
        """
        Determine if a memory should be archived (soft delete).

        Returns:
            (should_archive: bool, reason: str)
        """
        # Don't archive if already archived
        if hasattr(memory, "is_archived") and memory.is_archived:
            return False, "Already archived"

        # Always archive certain types if old enough
        age_days = (datetime.now(timezone.utc) - memory.created_at).days
        if memory.type in self.policy.always_archive_types and age_days > self.policy.archive_age_days:
            return True, f"Old {memory.type}: {age_days} days"

        # Archive low-importance, unaccessed memories
        if memory.importance < self.policy.archive_importance_threshold and memory.access_count == 0:
            return True, f"Low importance + never accessed"

        return False, "Does not meet archival criteria"

    def should_promote(self, memory: Memory, llm_decision: dict[str, Any] | None = None) -> tuple[bool, float, str]:
        """
        Determine if a memory should be promoted (importance increased).

        Returns:
            (should_promote: bool, new_importance: float, reason: str)
        """
        # LLM explicit promotion
        if llm_decision and llm_decision.get("new_importance"):
            new_importance = min(llm_decision["new_importance"], self.policy.max_importance_after_promotion)
            return True, new_importance, llm_decision.get("reason", "LLM decision")

        # Auto-promote frequently accessed memories
        if memory.access_count >= 3:
            new_importance = min(
                memory.importance * self.policy.promotion_multiplier,
                self.policy.max_importance_after_promotion,
            )
            return True, new_importance, f"Frequently accessed: {memory.access_count} times"

        # Auto-promote goals and preferences
        if memory.type in {MemoryType.GOAL, MemoryType.PREFERENCE} and memory.importance < 7.0:
            new_importance = min(7.0, self.policy.max_importance_after_promotion)
            return True, new_importance, f"Important type: {memory.type}"

        return False, memory.importance, "No promotion needed"

    def apply_decay(self, memory: Memory, days_elapsed: int = 1) -> float:
        """
        Apply importance decay to a memory.

        Args:
            memory: Memory to decay
            days_elapsed: Number of days since last consolidation

        Returns:
            New importance score after decay
        """
        # Get decay factor from memory or use default
        decay_factor = getattr(memory, "decay_factor", 0.95)

        # Apply exponential decay
        new_importance = memory.importance * (decay_factor**days_elapsed)

        # Floor at 0.0
        new_importance = max(0.0, new_importance)

        self.logger.debug(
            f"Applied decay to {memory.id}: {memory.importance:.2f} -> {new_importance:.2f} "
            f"(factor={decay_factor}, days={days_elapsed})"
        )

        return new_importance

    def enforce_memory_limit(
        self,
        memories: list[Memory],
        user_id: str,
    ) -> dict[str, list[str]]:
        """
        Enforce per-user memory limits by archiving oldest, least important memories.

        Args:
            memories: All memories for a user
            user_id: User ID

        Returns:
            Dictionary with 'to_archive' and 'to_delete' memory IDs
        """
        result = {"to_archive": [], "to_delete": []}

        # Check total limit
        if len(memories) <= self.policy.max_memories_per_user:
            return result

        # Sort by importance (ascending) and age (descending = oldest first)
        sorted_memories = sorted(
            memories,
            key=lambda m: (m.importance, -m.created_at.timestamp()),
        )

        # Calculate how many to remove
        excess_count = len(memories) - self.policy.max_memories_per_user

        # Archive the excess, prioritizing low-importance, old memories
        for memory in sorted_memories[:excess_count]:
            # Check if we should delete or archive
            should_del, reason = self.should_delete(memory)
            if should_del:
                result["to_delete"].append(memory.id)
                self.logger.info(f"Limit enforcement: deleting {memory.id} ({reason})")
            else:
                result["to_archive"].append(memory.id)
                self.logger.info(f"Limit enforcement: archiving {memory.id}")

        return result

    def consolidate_redundant(
        self,
        memories: list[Memory],
        similarity_threshold: float = 0.95,
    ) -> list[dict[str, Any]]:
        """
        Identify redundant/duplicate memories for consolidation.

        Args:
            memories: List of memories to check
            similarity_threshold: Cosine similarity threshold for duplicates

        Returns:
            List of merge operations: [{"keep_id": "...", "merge_ids": [...], "reason": "..."}]
        """
        merge_operations = []

        # Group by type for faster comparison
        by_type: dict[str, list[Memory]] = {}
        for memory in memories:
            by_type.setdefault(memory.type.value, []).append(memory)

        # Check for duplicates within each type
        for mem_type, type_memories in by_type.items():
            # Simple text similarity check (can be enhanced with embeddings)
            seen = {}
            for memory in type_memories:
                # Normalize text for comparison
                normalized = memory.text.lower().strip()

                # Check exact matches
                if normalized in seen:
                    existing_id = seen[normalized]
                    merge_operations.append(
                        {
                            "keep_id": existing_id,
                            "merge_ids": [memory.id],
                            "reason": f"Exact duplicate text in {mem_type}",
                        }
                    )
                else:
                    seen[normalized] = memory.id

        return merge_operations


def apply_consolidation_decisions(
    memories: list[Memory],
    llm_decisions: dict[str, Any],
    policy: ConsolidationPolicy | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """
    Apply LLM consolidation decisions with policy validation.

    Args:
        memories: All memories being consolidated
        llm_decisions: Decision dict from LLM
        policy: Consolidation policy to apply
        dry_run: If True, don't modify memories, just return actions

    Returns:
        Dictionary with actions taken and statistics
    """
    engine = ConsolidationPolicyEngine(policy)
    logger = logging.getLogger(f"{__name__}.apply_decisions")

    # Build memory lookup
    memory_map = {mem.id: mem for mem in memories}

    result = {
        "to_delete": [],
        "to_archive": [],
        "to_promote": [],
        "to_update": [],
        "to_create": [],
        "blocked": [],
        "stats": {
            "reviewed": len(memories),
            "deleted": 0,
            "archived": 0,
            "promoted": 0,
            "updated": 0,
            "synthesized": 0,
        },
    }

    # 1. Process deletions/archival (low-value memories)
    for mem_id in llm_decisions.get("low_value_memory_ids", []):
        if mem_id not in memory_map:
            logger.warning(f"LLM suggested deleting non-existent memory: {mem_id}")
            continue

        memory = memory_map[mem_id]

        # Check policy
        should_del, del_reason = engine.should_delete(memory)
        should_arch, arch_reason = engine.should_archive(memory)

        if should_del:
            result["to_delete"].append({"id": mem_id, "reason": del_reason})
            result["stats"]["deleted"] += 1
        elif should_arch:
            result["to_archive"].append({"id": mem_id, "reason": arch_reason})
            result["stats"]["archived"] += 1
        else:
            result["blocked"].append({"id": mem_id, "action": "delete", "reason": "Policy blocked deletion"})
            logger.warning(f"Policy blocked deletion of {mem_id}: not deletable")

    # 2. Process promotions
    for promotion in llm_decisions.get("promoted_facts", []):
        mem_id = promotion.get("id")
        if mem_id not in memory_map:
            logger.warning(f"LLM suggested promoting non-existent memory: {mem_id}")
            continue

        memory = memory_map[mem_id]
        should_prom, new_importance, reason = engine.should_promote(memory, promotion)

        if should_prom:
            result["to_promote"].append(
                {
                    "id": mem_id,
                    "old_importance": memory.importance,
                    "new_importance": new_importance,
                    "reason": reason,
                }
            )
            result["stats"]["promoted"] += 1

            # Update memory object if not dry run
            if not dry_run:
                memory.importance = new_importance
                if hasattr(memory, "promotion_count"):
                    memory.promotion_count += 1

    # 3. Process updates/merges
    for update in llm_decisions.get("updated_memories", []):
        mem_id = update.get("id")
        if mem_id not in memory_map:
            logger.warning(f"LLM suggested updating non-existent memory: {mem_id}")
            continue

        result["to_update"].append(update)
        result["stats"]["updated"] += 1

        # Update memory object if not dry run
        if not dry_run:
            memory = memory_map[mem_id]
            if "new_text" in update:
                memory.text = update["new_text"]
            if "new_importance" in update:
                memory.importance = min(update["new_importance"], policy.max_importance_after_promotion)

    # 4. Process synthetic memories
    for synthetic in llm_decisions.get("synthetic_memories", []):
        result["to_create"].append(synthetic)
        result["stats"]["synthesized"] += 1

    return result


if __name__ == "__main__":
    # Demo policy engine
    from hippocampai.models import Memory, MemoryType
    from datetime import datetime, timezone, timedelta

    policy = ConsolidationPolicy()
    engine = ConsolidationPolicyEngine(policy)

    # Test memory
    old_event = Memory(
        id="mem-old",
        text="Had coffee this morning",
        user_id="user-123",
        type=MemoryType.EVENT,
        importance=2.0,
        access_count=0,
        created_at=datetime.now(timezone.utc) - timedelta(days=10),
    )

    important_goal = Memory(
        id="mem-goal",
        text="Launch product by Q4",
        user_id="user-123",
        type=MemoryType.GOAL,
        importance=8.0,
        access_count=5,
        created_at=datetime.now(timezone.utc) - timedelta(days=2),
    )

    print("Policy Engine Test")
    print("=" * 60)

    print(f"\n1. Old event: {old_event.text}")
    should_del, reason = engine.should_delete(old_event)
    print(f"   Delete: {should_del} ({reason})")
    should_arch, reason = engine.should_archive(old_event)
    print(f"   Archive: {should_arch} ({reason})")

    print(f"\n2. Important goal: {important_goal.text}")
    should_del, reason = engine.should_delete(important_goal)
    print(f"   Delete: {should_del} ({reason})")
    should_prom, new_imp, reason = engine.should_promote(important_goal)
    print(f"   Promote: {should_prom} to {new_imp:.1f} ({reason})")

    print(f"\n3. Apply decay to old event")
    new_importance = engine.apply_decay(old_event, days_elapsed=5)
    print(f"   {old_event.importance:.2f} -> {new_importance:.2f}")
