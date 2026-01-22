"""Offline-capable memory client wrapper."""

import logging
import threading
from typing import TYPE_CHECKING, Any, Optional

from hippocampai.offline.queue import OfflineQueue, OperationType
from hippocampai.offline.sync import SyncManager, SyncStats

if TYPE_CHECKING:
    from hippocampai.client import MemoryClient
    from hippocampai.models.memory import Memory, RetrievalResult

logger = logging.getLogger(__name__)


class OfflineClient:
    """Memory client wrapper with offline support.

    Wraps a MemoryClient to provide resilience when backend services
    are unavailable. Operations are queued locally and synced when
    connectivity is restored.

    Example:
        >>> from hippocampai import MemoryClient
        >>> from hippocampai.offline import OfflineClient
        >>>
        >>> client = MemoryClient()
        >>> offline = OfflineClient(client)
        >>>
        >>> # Works even if Qdrant is down
        >>> offline.remember("Important fact", user_id="alice")
        >>>
        >>> # Sync when ready
        >>> stats = offline.sync()
    """

    def __init__(
        self,
        client: "MemoryClient",
        queue_path: str = "~/.hippocampai/offline_queue.db",
        auto_sync: bool = True,
        sync_interval_seconds: int = 60,
    ):
        """Initialize offline client.

        Args:
            client: Underlying MemoryClient
            queue_path: Path to offline queue database
            auto_sync: Whether to auto-sync in background
            sync_interval_seconds: Sync interval for auto-sync
        """
        self.client = client
        self.queue = OfflineQueue(queue_path)
        self.sync_manager = SyncManager(client, self.queue)
        self._auto_sync = auto_sync
        self._sync_interval = sync_interval_seconds
        self._sync_thread: threading.Thread | None = None

        if auto_sync:
            self._start_auto_sync()

    def _start_auto_sync(self) -> None:
        """Start background sync thread."""

        def sync_loop() -> None:
            import time

            while self._auto_sync:
                try:
                    if self.is_online():
                        self.sync()
                except Exception as e:
                    logger.warning(f"Auto-sync failed: {e}")
                time.sleep(self._sync_interval)

        self._sync_thread = threading.Thread(target=sync_loop, daemon=True)
        self._sync_thread.start()

    def stop_auto_sync(self) -> None:
        """Stop background sync."""
        self._auto_sync = False

    def is_online(self) -> bool:
        """Check if backend is available."""
        try:
            # Quick health check
            self.client.qdrant.client.get_collections()
            return True
        except Exception:
            return False

    def remember(
        self,
        text: str,
        user_id: str,
        session_id: Optional[str] = None,
        type: str = "fact",
        importance: Optional[float] = None,
        tags: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> Optional["Memory"]:
        """Store a memory, queuing if offline.

        Args:
            text: Memory text
            user_id: User ID
            session_id: Optional session ID
            type: Memory type
            importance: Importance score
            tags: Tags
            **kwargs: Additional arguments

        Returns:
            Memory object if online, None if queued
        """
        try:
            if self.is_online():
                return self.client.remember(
                    text=text,
                    user_id=user_id,
                    session_id=session_id,
                    type=type,
                    importance=importance,
                    tags=tags,
                    **kwargs,
                )
        except Exception as e:
            logger.warning(f"Backend unavailable, queuing operation: {e}")

        # Queue for later
        self.queue.enqueue(
            operation=OperationType.REMEMBER,
            user_id=user_id,
            data={
                "text": text,
                "type": type,
                "importance": importance,
                "tags": tags or [],
                **kwargs,
            },
            session_id=session_id,
        )
        logger.info(f"Queued remember operation for user {user_id}")
        return None

    def recall(
        self,
        query: str,
        user_id: str,
        session_id: Optional[str] = None,
        k: int = 5,
        **kwargs: Any,
    ) -> list["RetrievalResult"]:
        """Recall memories, returning empty if offline.

        Note: Recall cannot be queued as it requires immediate results.
        Returns empty list if offline.
        """
        try:
            if self.is_online():
                result: list[RetrievalResult] = self.client.recall(
                    query=query,
                    user_id=user_id,
                    session_id=session_id,
                    k=k,
                    **kwargs,
                )
                return result
        except Exception as e:
            logger.warning(f"Recall failed (offline): {e}")

        return []

    def update_memory(
        self,
        memory_id: str,
        user_id: str,
        text: Optional[str] = None,
        importance: Optional[float] = None,
        tags: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> bool:
        """Update a memory, queuing if offline."""
        try:
            if self.is_online():
                self.client.update_memory(
                    memory_id=memory_id,
                    text=text,
                    importance=importance,
                    tags=tags,
                    **kwargs,
                )
                return True
        except Exception as e:
            logger.warning(f"Backend unavailable, queuing update: {e}")

        self.queue.enqueue(
            operation=OperationType.UPDATE,
            user_id=user_id,
            data={
                "memory_id": memory_id,
                "text": text,
                "importance": importance,
                "tags": tags,
                **kwargs,
            },
        )
        return False

    def delete_memory(self, memory_id: str, user_id: str) -> bool:
        """Delete a memory, queuing if offline."""
        try:
            if self.is_online():
                self.client.delete_memory(memory_id, user_id)
                return True
        except Exception as e:
            logger.warning(f"Backend unavailable, queuing delete: {e}")

        self.queue.enqueue(
            operation=OperationType.DELETE,
            user_id=user_id,
            data={"memory_id": memory_id},
        )
        return False

    def sync(self) -> SyncStats:
        """Sync queued operations to backend.

        Returns:
            Sync statistics
        """
        return self.sync_manager.sync()

    def get_queue_stats(self) -> dict[str, int]:
        """Get offline queue statistics."""
        stats: dict[str, int] = self.queue.get_stats()
        return stats

    def __getattr__(self, name: str) -> Any:
        """Proxy other methods to underlying client."""
        return getattr(self.client, name)
