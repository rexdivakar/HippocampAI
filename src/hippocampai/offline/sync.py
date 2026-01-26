"""Sync manager for offline operations."""

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from hippocampai.offline.queue import OfflineQueue, OperationType, QueuedOperation

if TYPE_CHECKING:
    from hippocampai.client import MemoryClient

logger = logging.getLogger(__name__)


@dataclass
class SyncStats:
    """Statistics from sync operation."""

    total_operations: int = 0
    successful: int = 0
    failed: int = 0
    retried: int = 0
    duration_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)


class SyncManager:
    """Manages syncing offline operations to backend.

    Example:
        >>> sync_manager = SyncManager(client, queue)
        >>> stats = sync_manager.sync()
        >>> print(f"Synced {stats.successful} operations")
    """

    def __init__(self, client: "MemoryClient", queue: OfflineQueue):
        self.client = client
        self.queue = queue

    def sync(self, batch_size: int = 50) -> SyncStats:
        """Sync pending operations to backend.

        Args:
            batch_size: Number of operations per batch

        Returns:
            Sync statistics
        """
        start_time = time.time()
        stats = SyncStats()

        pending = self.queue.get_pending(limit=batch_size)
        stats.total_operations = len(pending)

        for op in pending:
            try:
                self._execute_operation(op)
                self.queue.mark_completed(op.id)
                stats.successful += 1
            except Exception as e:
                error_msg = str(e)
                should_retry = self.queue.mark_failed(op.id, error_msg)
                if should_retry:
                    stats.retried += 1
                else:
                    stats.failed += 1
                    stats.errors.append(f"{op.id}: {error_msg}")
                logger.warning(f"Operation {op.id} failed: {e}")

        stats.duration_seconds = time.time() - start_time
        logger.info(
            f"Sync complete: {stats.successful} successful, "
            f"{stats.failed} failed, {stats.retried} retried"
        )
        return stats

    def _execute_operation(self, op: QueuedOperation) -> Any:
        """Execute a queued operation."""
        if op.operation == OperationType.REMEMBER:
            return self.client.remember(
                text=op.data["text"],
                user_id=op.user_id,
                session_id=op.session_id,
                type=op.data.get("type", "fact"),
                importance=op.data.get("importance"),
                tags=op.data.get("tags", []),
            )
        elif op.operation == OperationType.UPDATE:
            return self.client.update_memory(
                memory_id=op.data["memory_id"],
                text=op.data.get("text"),
                importance=op.data.get("importance"),
                tags=op.data.get("tags"),
            )
        elif op.operation == OperationType.DELETE:
            return self.client.delete_memory(
                op.data["memory_id"],
                op.user_id,
            )
        else:
            raise ValueError(f"Unknown operation type: {op.operation}")
