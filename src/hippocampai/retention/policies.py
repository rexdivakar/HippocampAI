"""Retention policy manager for automatic memory cleanup."""

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from hippocampai.models.search import RetentionPolicy
from hippocampai.vector.qdrant_store import QdrantStore

logger = logging.getLogger(__name__)


class RetentionPolicyManager:
    """Manage retention policies and automatic memory cleanup."""

    def __init__(self, qdrant_store: QdrantStore):
        """
        Initialize retention policy manager.

        Args:
            qdrant_store: QdrantStore instance for memory operations
        """
        self.qdrant = qdrant_store
        self._policies: dict[str, RetentionPolicy] = {}  # policy_id -> RetentionPolicy
        self._user_policies: dict[str, list[str]] = {}  # user_id -> [policy_ids]

    def create_policy(
        self,
        name: str,
        retention_days: int,
        user_id: Optional[str] = None,
        memory_type: Optional[str] = None,
        min_importance: Optional[float] = None,
        min_access_count: Optional[int] = None,
        tags_to_preserve: Optional[list[str]] = None,
        enabled: bool = True,
    ) -> RetentionPolicy:
        """
        Create a new retention policy.

        Args:
            name: Policy name
            retention_days: Days to retain memories
            user_id: User ID (None = global policy)
            memory_type: Memory type to apply to (None = all types)
            min_importance: Preserve memories with importance >= threshold
            min_access_count: Preserve memories accessed >= N times
            tags_to_preserve: Tags that prevent deletion
            enabled: Whether policy is active

        Returns:
            RetentionPolicy object
        """
        policy = RetentionPolicy(
            name=name,
            user_id=user_id,
            memory_type=memory_type,
            retention_days=retention_days,
            min_importance=min_importance,
            min_access_count=min_access_count,
            tags_to_preserve=tags_to_preserve or [],
            enabled=enabled,
        )

        self._policies[policy.id] = policy

        if user_id:
            if user_id not in self._user_policies:
                self._user_policies[user_id] = []
            self._user_policies[user_id].append(policy.id)

        logger.info(f"Created retention policy '{name}' (retention: {retention_days} days)")
        return policy

    def get_policy(self, policy_id: str) -> Optional[RetentionPolicy]:
        """Get a policy by ID."""
        return self._policies.get(policy_id)

    def get_policies(
        self, user_id: Optional[str] = None, enabled_only: bool = True
    ) -> list[RetentionPolicy]:
        """
        Get all policies, optionally filtered by user.

        Args:
            user_id: User ID to filter by (None = all policies)
            enabled_only: Only return enabled policies

        Returns:
            List of RetentionPolicy objects
        """
        if user_id:
            policy_ids = self._user_policies.get(user_id, [])
            policies = [self._policies[pid] for pid in policy_ids if pid in self._policies]
        else:
            policies = list(self._policies.values())

        if enabled_only:
            policies = [p for p in policies if p.enabled]

        return policies

    def update_policy(
        self,
        policy_id: str,
        name: Optional[str] = None,
        retention_days: Optional[int] = None,
        min_importance: Optional[float] = None,
        min_access_count: Optional[int] = None,
        tags_to_preserve: Optional[list[str]] = None,
        enabled: Optional[bool] = None,
    ) -> Optional[RetentionPolicy]:
        """
        Update a retention policy.

        Returns:
            Updated policy or None if not found
        """
        policy = self._policies.get(policy_id)
        if not policy:
            return None

        if name is not None:
            policy.name = name
        if retention_days is not None:
            policy.retention_days = retention_days
        if min_importance is not None:
            policy.min_importance = min_importance
        if min_access_count is not None:
            policy.min_access_count = min_access_count
        if tags_to_preserve is not None:
            policy.tags_to_preserve = tags_to_preserve
        if enabled is not None:
            policy.enabled = enabled

        logger.info(f"Updated retention policy {policy_id}")
        return policy

    def delete_policy(self, policy_id: str) -> bool:
        """
        Delete a retention policy.

        Returns:
            True if deleted, False if not found
        """
        policy = self._policies.get(policy_id)
        if not policy:
            return False

        del self._policies[policy_id]

        # Remove from user index
        if policy.user_id and policy.user_id in self._user_policies:
            self._user_policies[policy.user_id] = [
                pid for pid in self._user_policies[policy.user_id] if pid != policy_id
            ]

        logger.info(f"Deleted retention policy {policy_id}")
        return True

    def apply_policies(
        self, user_id: Optional[str] = None, dry_run: bool = False
    ) -> dict[str, Any]:
        """
        Apply retention policies and delete expired memories.

        Args:
            user_id: Apply policies for specific user (None = all users)
            dry_run: If True, don't delete, just report what would be deleted

        Returns:
            Dictionary with deletion statistics
        """
        policies = self.get_policies(user_id=user_id, enabled_only=True)
        if not policies:
            logger.info("No enabled retention policies to apply")
            return {"deleted": 0, "policies_applied": 0}

        total_deleted = 0
        policies_applied = 0
        deleted_by_policy = {}

        for policy in policies:
            logger.info(f"Applying retention policy '{policy.name}'")

            # Determine which collection to scan
            collections = []
            if policy.memory_type:
                # Specific type
                if policy.memory_type in {"preference", "goal", "habit"}:
                    collections = [self.qdrant.collection_prefs]
                else:
                    collections = [self.qdrant.collection_facts]
            else:
                # All types
                collections = [self.qdrant.collection_facts, self.qdrant.collection_prefs]

            policy_deleted = 0

            for collection in collections:
                # Build filters
                filters = {}
                if policy.user_id:
                    filters["user_id"] = policy.user_id
                if policy.memory_type:
                    filters["type"] = policy.memory_type

                # Scroll through memories
                memories = self.qdrant.scroll(
                    collection_name=collection, filters=filters if filters else None, limit=10000
                )

                # Check each memory against policy
                to_delete = []
                for mem in memories:
                    payload = mem["payload"]
                    if policy.should_delete(payload):
                        to_delete.append(mem["id"])

                # Delete memories
                if to_delete:
                    if not dry_run:
                        self.qdrant.delete(collection_name=collection, ids=to_delete)
                        policy_deleted += len(to_delete)
                        total_deleted += len(to_delete)
                        logger.info(
                            f"Deleted {len(to_delete)} memories from {collection} (policy: {policy.name})"
                        )
                    else:
                        logger.info(
                            f"[DRY RUN] Would delete {len(to_delete)} memories from {collection} (policy: {policy.name})"
                        )
                        policy_deleted += len(to_delete)
                        total_deleted += len(to_delete)

            # Update policy statistics
            if not dry_run:
                policy.deleted_count += policy_deleted
                policy.last_run_at = datetime.now(timezone.utc)

            deleted_by_policy[policy.name] = policy_deleted
            policies_applied += 1

        logger.info(
            f"Retention policies applied: {policies_applied} policies, {total_deleted} memories deleted"
        )

        return {
            "deleted": total_deleted,
            "policies_applied": policies_applied,
            "deleted_by_policy": deleted_by_policy,
            "dry_run": dry_run,
        }

    def get_expiring_memories(
        self, user_id: Optional[str] = None, days_threshold: int = 7
    ) -> list[dict[str, Any]]:
        """
        Get memories that will expire within threshold days.

        Args:
            user_id: User ID to filter by
            days_threshold: Days until expiration

        Returns:
            List of memory data with policy information
        """
        policies = self.get_policies(user_id=user_id, enabled_only=True)
        if not policies:
            return []

        expiring = []

        for policy in policies:
            # Determine collections
            collections = []
            if policy.memory_type:
                if policy.memory_type in {"preference", "goal", "habit"}:
                    collections = [self.qdrant.collection_prefs]
                else:
                    collections = [self.qdrant.collection_facts]
            else:
                collections = [self.qdrant.collection_facts, self.qdrant.collection_prefs]

            for collection in collections:
                filters = {}
                if policy.user_id:
                    filters["user_id"] = policy.user_id
                if policy.memory_type:
                    filters["type"] = policy.memory_type

                memories = self.qdrant.scroll(
                    collection_name=collection, filters=filters if filters else None, limit=10000
                )

                # Check expiration
                for mem in memories:
                    payload = mem["payload"]
                    created_at = payload.get("created_at")
                    if not created_at:
                        continue

                    from hippocampai.utils.time import parse_iso_datetime

                    if isinstance(created_at, str):
                        created_at = parse_iso_datetime(created_at)

                    age_days = (datetime.now(timezone.utc) - created_at).days
                    days_until_expiration = policy.retention_days - age_days

                    if 0 < days_until_expiration <= days_threshold:
                        # Check if it would be deleted
                        if policy.should_delete(payload):
                            expiring.append(
                                {
                                    "memory_id": mem["id"],
                                    "text": payload.get("text", ""),
                                    "user_id": payload.get("user_id"),
                                    "type": payload.get("type"),
                                    "importance": payload.get("importance", 5.0),
                                    "days_until_expiration": days_until_expiration,
                                    "policy_name": policy.name,
                                }
                            )

        # Sort by days until expiration
        expiring.sort(key=lambda x: x["days_until_expiration"])
        return expiring

    def get_statistics(self) -> dict:
        """Get retention policy statistics."""
        total_policies = len(self._policies)
        enabled_policies = sum(1 for p in self._policies.values() if p.enabled)
        total_deleted = sum(p.deleted_count for p in self._policies.values())

        return {
            "total_policies": total_policies,
            "enabled_policies": enabled_policies,
            "disabled_policies": total_policies - enabled_policies,
            "total_memories_deleted": total_deleted,
        }
