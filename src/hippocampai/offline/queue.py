"""Offline operation queue for resilient memory operations."""

import json
import logging
import os
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


class OperationType(str, Enum):
    """Types of queued operations."""

    REMEMBER = "remember"
    UPDATE = "update"
    DELETE = "delete"
    RECALL = "recall"


@dataclass
class QueuedOperation:
    """A queued offline operation."""

    id: str
    operation: OperationType
    user_id: str
    data: dict[str, Any]
    created_at: str
    retry_count: int = 0
    max_retries: int = 3
    priority: int = 0  # Higher = more important
    session_id: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "operation": self.operation.value,
            "user_id": self.user_id,
            "data": self.data,
            "created_at": self.created_at,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "priority": self.priority,
            "session_id": self.session_id,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QueuedOperation":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            operation=OperationType(data["operation"]),
            user_id=data["user_id"],
            data=data["data"],
            created_at=data["created_at"],
            retry_count=data.get("retry_count", 0),
            max_retries=data.get("max_retries", 3),
            priority=data.get("priority", 0),
            session_id=data.get("session_id"),
            error=data.get("error"),
        )


class OfflineQueue:
    """SQLite-backed queue for offline operations.

    Persists operations to disk when backend is unavailable,
    allowing sync when connectivity is restored.

    Example:
        >>> queue = OfflineQueue("~/.hippocampai/queue.db")
        >>> queue.enqueue(OperationType.REMEMBER, "alice", {"text": "Hello"})
        >>> pending = queue.get_pending(limit=10)
        >>> queue.mark_completed(pending[0].id)
    """

    def __init__(self, db_path: str = "~/.hippocampai/offline_queue.db"):
        """Initialize offline queue.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = os.path.expanduser(db_path)
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS operations (
                    id TEXT PRIMARY KEY,
                    operation TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    session_id TEXT,
                    data TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    retry_count INTEGER DEFAULT 0,
                    max_retries INTEGER DEFAULT 3,
                    priority INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'pending',
                    error TEXT,
                    completed_at TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_status ON operations(status)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_user ON operations(user_id)
            """)
            conn.commit()

    def enqueue(
        self,
        operation: OperationType,
        user_id: str,
        data: dict[str, Any],
        session_id: Optional[str] = None,
        priority: int = 0,
    ) -> str:
        """Add operation to queue.

        Args:
            operation: Type of operation
            user_id: User ID
            data: Operation data
            session_id: Optional session ID
            priority: Priority (higher = more important)

        Returns:
            Operation ID
        """
        op_id = str(uuid4())
        created_at = datetime.now(timezone.utc).isoformat()

        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO operations 
                    (id, operation, user_id, session_id, data, created_at, priority)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        op_id,
                        operation.value,
                        user_id,
                        session_id,
                        json.dumps(data),
                        created_at,
                        priority,
                    ),
                )
                conn.commit()

        logger.debug(f"Queued {operation.value} operation: {op_id}")
        return op_id

    def get_pending(
        self,
        limit: int = 100,
        user_id: Optional[str] = None,
    ) -> list[QueuedOperation]:
        """Get pending operations.

        Args:
            limit: Maximum operations to return
            user_id: Optional filter by user

        Returns:
            List of pending operations
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            if user_id:
                cursor = conn.execute(
                    """
                    SELECT * FROM operations 
                    WHERE status = 'pending' AND user_id = ?
                    ORDER BY priority DESC, created_at ASC
                    LIMIT ?
                    """,
                    (user_id, limit),
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT * FROM operations 
                    WHERE status = 'pending'
                    ORDER BY priority DESC, created_at ASC
                    LIMIT ?
                    """,
                    (limit,),
                )

            operations = []
            for row in cursor:
                operations.append(
                    QueuedOperation(
                        id=row["id"],
                        operation=OperationType(row["operation"]),
                        user_id=row["user_id"],
                        data=json.loads(row["data"]),
                        created_at=row["created_at"],
                        retry_count=row["retry_count"],
                        max_retries=row["max_retries"],
                        priority=row["priority"],
                        session_id=row["session_id"],
                        error=row["error"],
                    )
                )
            return operations

    def mark_completed(self, op_id: str) -> None:
        """Mark operation as completed."""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    UPDATE operations 
                    SET status = 'completed', completed_at = ?
                    WHERE id = ?
                    """,
                    (datetime.now(timezone.utc).isoformat(), op_id),
                )
                conn.commit()

    def mark_failed(self, op_id: str, error: str) -> bool:
        """Mark operation as failed, increment retry count.

        Returns:
            True if operation should be retried
        """
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                # Get current retry count
                cursor = conn.execute(
                    "SELECT retry_count, max_retries FROM operations WHERE id = ?",
                    (op_id,),
                )
                row = cursor.fetchone()
                if not row:
                    return False

                retry_count, max_retries = row
                new_retry_count = retry_count + 1

                if new_retry_count >= max_retries:
                    # Max retries reached
                    conn.execute(
                        """
                        UPDATE operations 
                        SET status = 'failed', error = ?, retry_count = ?
                        WHERE id = ?
                        """,
                        (error, new_retry_count, op_id),
                    )
                    conn.commit()
                    return False
                else:
                    # Can retry
                    conn.execute(
                        """
                        UPDATE operations 
                        SET error = ?, retry_count = ?
                        WHERE id = ?
                        """,
                        (error, new_retry_count, op_id),
                    )
                    conn.commit()
                    return True

    def get_stats(self) -> dict[str, int]:
        """Get queue statistics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT status, COUNT(*) as count 
                FROM operations 
                GROUP BY status
            """)
            stats = {"pending": 0, "completed": 0, "failed": 0}
            for row in cursor:
                stats[row[0]] = row[1]
            return stats

    def clear_completed(self, older_than_days: int = 7) -> int:
        """Clear completed operations older than N days."""
        from datetime import timedelta

        cutoff = (datetime.now(timezone.utc) - timedelta(days=older_than_days)).isoformat()

        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    DELETE FROM operations 
                    WHERE status = 'completed' AND completed_at < ?
                    """,
                    (cutoff,),
                )
                conn.commit()
                return cursor.rowcount

    def clear_all(self) -> int:
        """Clear all operations (use with caution)."""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("DELETE FROM operations")
                conn.commit()
                return cursor.rowcount
