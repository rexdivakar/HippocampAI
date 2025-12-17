"""Offline Mode for HippocampAI.

Queue operations when Qdrant/Redis are unavailable and sync later.
Provides resilience for intermittent connectivity scenarios.

Example:
    >>> from hippocampai.offline import OfflineQueue, OfflineClient
    >>>
    >>> # Wrap client with offline support
    >>> offline_client = OfflineClient(client, queue_path="~/.hippocampai/queue")
    >>>
    >>> # Operations are queued if backend is unavailable
    >>> offline_client.remember("Important fact", user_id="alice")
    >>>
    >>> # Sync when connection is restored
    >>> stats = offline_client.sync()
"""

from hippocampai.offline.client import OfflineClient
from hippocampai.offline.queue import OfflineQueue, OperationType, QueuedOperation
from hippocampai.offline.sync import SyncManager, SyncStats

__all__ = [
    "OfflineQueue",
    "QueuedOperation",
    "OperationType",
    "OfflineClient",
    "SyncManager",
    "SyncStats",
]
