"""Background tasks for automatic memory maintenance."""

import asyncio
import logging
from typing import Any, Optional

from hippocampai.services.memory_service import MemoryManagementService

logger = logging.getLogger(__name__)


class BackgroundTaskManager:
    """
    Manager for background maintenance tasks.

    Handles:
    - Automatic deduplication
    - Automatic consolidation
    - Memory expiration
    - Cache cleanup
    """

    def __init__(
        self,
        service: MemoryManagementService,
        dedup_interval_hours: int = 24,
        consolidation_interval_hours: int = 168,  # 7 days
        expiration_interval_hours: int = 1,
        auto_dedup_enabled: bool = True,
        auto_consolidation_enabled: bool = False,  # Off by default
        dedup_threshold: float = 0.88,
        consolidation_threshold: float = 0.85,
    ):
        """
        Initialize background task manager.

        Args:
            service: Memory management service
            dedup_interval_hours: Hours between dedup runs
            consolidation_interval_hours: Hours between consolidation runs
            expiration_interval_hours: Hours between expiration runs
            auto_dedup_enabled: Whether to run automatic deduplication
            auto_consolidation_enabled: Whether to run automatic consolidation
            dedup_threshold: Similarity threshold for deduplication
            consolidation_threshold: Similarity threshold for consolidation
        """
        self.service = service
        self.dedup_interval_hours = dedup_interval_hours
        self.consolidation_interval_hours = consolidation_interval_hours
        self.expiration_interval_hours = expiration_interval_hours
        self.auto_dedup_enabled = auto_dedup_enabled
        self.auto_consolidation_enabled = auto_consolidation_enabled
        self.dedup_threshold = dedup_threshold
        self.consolidation_threshold = consolidation_threshold

        self._tasks: list[asyncio.Task] = []
        self._running = False

    async def start(self) -> None:
        """Start all background tasks."""
        if self._running:
            logger.warning("Background tasks already running")
            return

        self._running = True
        logger.info("Starting background tasks...")

        # Start expiration task (always enabled)
        self._tasks.append(asyncio.create_task(self._expiration_loop()))

        # Start deduplication task if enabled
        if self.auto_dedup_enabled:
            self._tasks.append(asyncio.create_task(self._deduplication_loop()))

        # Start consolidation task if enabled
        if self.auto_consolidation_enabled:
            self._tasks.append(asyncio.create_task(self._consolidation_loop()))

        logger.info(f"Started {len(self._tasks)} background tasks")

    async def stop(self) -> None:
        """Stop all background tasks."""
        if not self._running:
            return

        self._running = False
        logger.info("Stopping background tasks...")

        # Cancel all tasks
        for task in self._tasks:
            task.cancel()

        # Wait for cancellation
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

        logger.info("Background tasks stopped")

    async def _expiration_loop(self) -> None:
        """Periodic task to expire old memories."""
        logger.info(f"Expiration task started (interval: {self.expiration_interval_hours}h)")

        while self._running:
            try:
                # Wait for interval
                await asyncio.sleep(self.expiration_interval_hours * 3600)

                logger.info("Running memory expiration...")
                expired_count = await self.service.expire_memories()
                logger.info(f"Expired {expired_count} memories")

            except asyncio.CancelledError:
                logger.info("Expiration task cancelled")
                break
            except Exception as e:
                logger.error(f"Error in expiration task: {e}", exc_info=True)
                # Continue running despite errors
                await asyncio.sleep(60)  # Wait a minute before retrying

    async def _deduplication_loop(self) -> None:
        """Periodic task to deduplicate memories."""
        logger.info(f"Deduplication task started (interval: {self.dedup_interval_hours}h)")

        while self._running:
            try:
                # Wait for interval
                await asyncio.sleep(self.dedup_interval_hours * 3600)

                logger.info("Running automatic deduplication...")

                # Get all unique users (this is expensive, consider caching)
                # For now, we'll just log that dedup is available per-user
                logger.info(
                    "Automatic deduplication requires per-user execution. "
                    "Use the /v1/memories/deduplicate endpoint for specific users."
                )

                # If you have a user list, you could iterate:
                # for user_id in get_active_users():
                #     result = await self.service.deduplicate_user_memories(
                #         user_id=user_id, dry_run=False
                #     )
                #     logger.info(f"Deduplicated user {user_id}: {result}")

            except asyncio.CancelledError:
                logger.info("Deduplication task cancelled")
                break
            except Exception as e:
                logger.error(f"Error in deduplication task: {e}", exc_info=True)
                await asyncio.sleep(60)

    async def _consolidation_loop(self) -> None:
        """Periodic task to consolidate similar memories."""
        logger.info(f"Consolidation task started (interval: {self.consolidation_interval_hours}h)")

        while self._running:
            try:
                # Wait for interval
                await asyncio.sleep(self.consolidation_interval_hours * 3600)

                logger.info("Running automatic consolidation...")

                # Similar to deduplication, requires per-user execution
                logger.info(
                    "Automatic consolidation requires per-user execution. "
                    "Use the /v1/memories/consolidate endpoint for specific users."
                )

            except asyncio.CancelledError:
                logger.info("Consolidation task cancelled")
                break
            except Exception as e:
                logger.error(f"Error in consolidation task: {e}", exc_info=True)
                await asyncio.sleep(60)

    async def trigger_deduplication(self, user_id: str, dry_run: bool = False) -> dict[str, Any]:
        """
        Manually trigger deduplication for a user.

        Args:
            user_id: User to deduplicate
            dry_run: Whether to only analyze without making changes
        """
        logger.info(f"Triggering deduplication for user {user_id} (dry_run={dry_run})")
        result: dict[str, Any] = await self.service.deduplicate_user_memories(
            user_id=user_id, dry_run=dry_run
        )
        return result

    async def trigger_consolidation(
        self, user_id: str, dry_run: bool = False, threshold: Optional[float] = None
    ) -> dict[str, Any]:
        """
        Manually trigger consolidation for a user.

        Args:
            user_id: User to consolidate
            dry_run: Whether to only analyze without making changes
            threshold: Optional custom similarity threshold
        """
        threshold = threshold or self.consolidation_threshold
        logger.info(
            f"Triggering consolidation for user {user_id} (dry_run={dry_run}, threshold={threshold})"
        )
        result: dict[str, Any] = await self.service.consolidate_memories(
            user_id=user_id, dry_run=dry_run, similarity_threshold=threshold
        )
        return result

    def get_status(self) -> dict:
        """Get status of background tasks."""
        return {
            "running": self._running,
            "active_tasks": len([t for t in self._tasks if not t.done()]),
            "total_tasks": len(self._tasks),
            "config": {
                "dedup_interval_hours": self.dedup_interval_hours,
                "consolidation_interval_hours": self.consolidation_interval_hours,
                "expiration_interval_hours": self.expiration_interval_hours,
                "auto_dedup_enabled": self.auto_dedup_enabled,
                "auto_consolidation_enabled": self.auto_consolidation_enabled,
                "dedup_threshold": self.dedup_threshold,
                "consolidation_threshold": self.consolidation_threshold,
            },
        }
