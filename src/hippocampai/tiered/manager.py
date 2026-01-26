"""Tiered storage manager for memory lifecycle."""

import hashlib
import logging
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

from hippocampai.tiered.models import (
    CompressedMemory,
    MigrationStats,
    StorageTier,
    TierConfig,
    TierStats,
)

if TYPE_CHECKING:
    from hippocampai.client import MemoryClient
    from hippocampai.models.memory import Memory

logger = logging.getLogger(__name__)


class TieredStorageManager:
    """Manages tiered storage for memory optimization.

    Automatically migrates memories between tiers based on access
    patterns and age, applying compression to reduce storage costs.

    Example:
        >>> manager = TieredStorageManager(client)
        >>> stats = manager.migrate_tiers(user_id="alice")
        >>> print(f"Saved {stats.bytes_saved} bytes")
    """

    def __init__(
        self,
        client: "MemoryClient",
        config: Optional[TierConfig] = None,
    ):
        self.client = client
        self.config = config or TierConfig()
        self._compressed_store: dict[str, CompressedMemory] = {}

    def configure_tiers(
        self,
        hot_days: int = 30,
        warm_days: int = 90,
        cold_days: int = 365,
    ) -> None:
        """Configure tier transition thresholds."""
        self.config.hot_to_warm_days = hot_days
        self.config.warm_to_cold_days = warm_days
        self.config.cold_to_frozen_days = cold_days

    def get_tier(self, memory: "Memory") -> StorageTier:
        """Determine current tier for a memory."""
        # Check if already compressed
        tier_meta = memory.metadata.get("storage_tier")
        if tier_meta:
            return StorageTier(tier_meta)

        # Calculate based on last access
        last_access = memory.metadata.get("last_accessed")
        if last_access:
            if isinstance(last_access, str):
                last_access = datetime.fromisoformat(last_access.replace("Z", "+00:00"))
        else:
            last_access = memory.updated_at

        now = datetime.now(timezone.utc)
        days_since_access = (now - last_access).days

        if days_since_access < self.config.hot_to_warm_days:
            return StorageTier.HOT
        elif days_since_access < self.config.warm_to_cold_days:
            return StorageTier.WARM
        elif days_since_access < self.config.cold_to_frozen_days:
            return StorageTier.COLD
        else:
            return StorageTier.FROZEN

    def migrate_tiers(self, user_id: str) -> MigrationStats:
        """Migrate memories between tiers based on access patterns.

        Args:
            user_id: User ID to migrate

        Returns:
            Migration statistics
        """
        start_time = time.time()
        stats = MigrationStats()

        try:
            memories = self.client.get_memories(user_id, limit=100000)

            for memory in memories:
                current_tier = self.get_tier(memory)
                target_tier = self._calculate_target_tier(memory)

                if target_tier.value > current_tier.value:
                    # Demote to colder tier
                    try:
                        self._demote_memory(memory, target_tier)
                        if target_tier == StorageTier.WARM:
                            stats.hot_to_warm += 1
                        elif target_tier == StorageTier.COLD:
                            stats.warm_to_cold += 1
                        elif target_tier == StorageTier.FROZEN:
                            stats.cold_to_frozen += 1
                    except Exception as e:
                        stats.errors.append(f"Demote {memory.id}: {e}")

                elif target_tier.value < current_tier.value:
                    # Promote to hotter tier (due to recent access)
                    try:
                        self._promote_memory(memory, target_tier)
                        stats.promoted += 1
                    except Exception as e:
                        stats.errors.append(f"Promote {memory.id}: {e}")

        except Exception as e:
            stats.errors.append(f"Migration failed: {e}")
            logger.error(f"Tier migration failed: {e}")

        stats.duration_seconds = time.time() - start_time
        logger.info(
            f"Tier migration complete: {stats.hot_to_warm} to warm, "
            f"{stats.warm_to_cold} to cold, {stats.promoted} promoted"
        )
        return stats

    def _calculate_target_tier(self, memory: "Memory") -> StorageTier:
        """Calculate target tier based on access patterns."""
        # High importance memories stay hot longer
        importance_factor = memory.importance / 10.0

        # Frequently accessed memories stay hot
        access_count = memory.access_count or 0
        access_factor = min(1.0, access_count / 100)

        # Combine factors
        retention_bonus = int((importance_factor + access_factor) * 30)

        last_access = memory.metadata.get("last_accessed")
        if last_access:
            if isinstance(last_access, str):
                last_access = datetime.fromisoformat(last_access.replace("Z", "+00:00"))
        else:
            last_access = memory.updated_at

        now = datetime.now(timezone.utc)
        days_since_access = (now - last_access).days - retention_bonus

        if days_since_access < self.config.hot_to_warm_days:
            return StorageTier.HOT
        elif days_since_access < self.config.warm_to_cold_days:
            return StorageTier.WARM
        elif days_since_access < self.config.cold_to_frozen_days:
            return StorageTier.COLD
        else:
            return StorageTier.FROZEN

    def _demote_memory(self, memory: "Memory", target_tier: StorageTier) -> None:
        """Demote memory to colder tier with compression."""
        # Compress based on target tier
        if target_tier == StorageTier.WARM:
            compressed_text = self._compress_warm(memory)
        elif target_tier == StorageTier.COLD:
            compressed_text = self._compress_cold(memory)
        else:
            compressed_text = self._compress_frozen(memory)

        # Store compressed version
        compressed = CompressedMemory(
            id=f"compressed_{memory.id}",
            original_id=memory.id,
            user_id=memory.user_id,
            tier=target_tier,
            summary=compressed_text,
            original_length=len(memory.text),
            compressed_length=len(compressed_text),
            compression_ratio=len(compressed_text) / max(1, len(memory.text)),
            memory_type=memory.type.value if hasattr(memory.type, "value") else str(memory.type),
            importance=memory.importance,
            tags=memory.tags,
            created_at=memory.created_at.isoformat(),
            last_accessed=datetime.now(timezone.utc).isoformat(),
            access_count=memory.access_count or 0,
            original_hash=hashlib.sha256(memory.text.encode()).hexdigest(),
        )

        self._compressed_store[memory.id] = compressed

        # Update memory metadata
        self.client.update_memory(
            memory_id=memory.id,
            metadata={
                **memory.metadata,
                "storage_tier": target_tier.value,
                "compressed": True,
                "original_length": len(memory.text),
            },
        )

        # For cold/frozen, replace text with summary
        if target_tier in [StorageTier.COLD, StorageTier.FROZEN]:
            self.client.update_memory(
                memory_id=memory.id,
                text=compressed_text,
            )

    def _promote_memory(self, memory: "Memory", target_tier: StorageTier) -> None:
        """Promote memory to hotter tier, restoring if needed."""
        # Check if we have original (for future restoration support)
        _ = self._compressed_store.get(memory.id)

        # Update tier metadata
        self.client.update_memory(
            memory_id=memory.id,
            metadata={
                **memory.metadata,
                "storage_tier": target_tier.value,
                "promoted_at": datetime.now(timezone.utc).isoformat(),
            },
        )

    def _compress_warm(self, memory: "Memory") -> str:
        """Light compression for warm tier."""
        text: str = memory.text
        # Truncate very long memories
        if len(text) > 500:
            return text[:500] + "..."
        return text

    def _compress_cold(self, memory: "Memory") -> str:
        """Heavy compression for cold tier using LLM summarization."""
        if self.client.llm:
            try:
                prompt = f"Summarize this memory in 1-2 sentences:\n\n{memory.text[:1000]}"
                summary: str = self.client.llm.generate(prompt, max_tokens=100)
                return summary.strip()
            except Exception as e:
                logger.warning(f"LLM summarization failed: {e}")

        # Fallback: extract key sentences
        sentences = memory.text.split(". ")
        if len(sentences) > 2:
            return ". ".join(sentences[:2]) + "."
        result: str = memory.text[:200]
        return result

    def _compress_frozen(self, memory: "Memory") -> str:
        """Minimal compression for frozen tier."""
        # Just keep type and key metadata
        return f"[{memory.type}] {memory.text[:100]}..."

    def get_tier_stats(self, user_id: str) -> dict[StorageTier, TierStats]:
        """Get statistics for each tier."""
        stats = {tier: TierStats(tier=tier) for tier in StorageTier}

        memories = self.client.get_memories(user_id, limit=100000)
        for memory in memories:
            tier = self.get_tier(memory)
            stats[tier].memory_count += 1
            stats[tier].storage_bytes += len(memory.text.encode())

            if stats[tier].oldest_memory is None or memory.created_at < stats[tier].oldest_memory:
                stats[tier].oldest_memory = memory.created_at
            if stats[tier].newest_memory is None or memory.created_at > stats[tier].newest_memory:
                stats[tier].newest_memory = memory.created_at

        return stats

    def restore_memory(self, memory_id: str) -> Optional["Memory"]:
        """Restore a compressed memory to full fidelity.

        Note: Only possible if original was preserved.
        """
        compressed = self._compressed_store.get(memory_id)
        if not compressed or not compressed.can_restore:
            return None

        # Would need to restore from backup/archive
        logger.warning(f"Full restoration not available for {memory_id}")
        return None
