"""DuckDB-based storage for users and sessions."""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional
from uuid import uuid4

import duckdb

from hippocampai.storage.models import Session, SoftDeleteRecord, User

logger = logging.getLogger(__name__)

_store_instance: Optional["UserStore"] = None


def get_user_store() -> "UserStore":
    """Get singleton UserStore instance."""
    global _store_instance
    if _store_instance is None:
        db_path = os.getenv("USER_STORE_DB_PATH", "data/users.duckdb")
        _store_instance = UserStore(db_path)
    return _store_instance


class UserStore:
    """DuckDB-based storage for users, sessions, and soft-delete records."""

    def __init__(self, db_path: str = "data/users.duckdb"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = duckdb.connect(db_path)
        self._init_tables()
        logger.info(f"UserStore initialized at {db_path}")

    def _init_tables(self) -> None:
        """Initialize database tables."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id VARCHAR PRIMARY KEY,
                username VARCHAR,
                email VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT TRUE,
                is_deleted BOOLEAN DEFAULT FALSE,
                deleted_at TIMESTAMP,
                metadata JSON
            )
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id VARCHAR PRIMARY KEY,
                user_id VARCHAR NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_active_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT TRUE,
                is_deleted BOOLEAN DEFAULT FALSE,
                deleted_at TIMESTAMP,
                deleted_by VARCHAR,
                delete_reason VARCHAR,
                memory_count INTEGER DEFAULT 0,
                metadata JSON
            )
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS soft_deletes (
                id VARCHAR PRIMARY KEY,
                entity_type VARCHAR NOT NULL,
                entity_id VARCHAR NOT NULL,
                user_id VARCHAR NOT NULL,
                session_id VARCHAR,
                deleted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                deleted_by VARCHAR NOT NULL,
                reason VARCHAR,
                original_data JSON,
                is_restored BOOLEAN DEFAULT FALSE,
                restored_at TIMESTAMP,
                restored_by VARCHAR
            )
        """)

        # Create indexes
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_deleted ON sessions(is_deleted)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_soft_deletes_user ON soft_deletes(user_id)")

    # ========================================
    # USER OPERATIONS
    # ========================================

    def create_user(self, user: User) -> User:
        """Create a new user."""
        self.conn.execute("""
            INSERT INTO users (user_id, username, email, created_at, updated_at, is_active, is_deleted, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            user.user_id, user.username, user.email,
            user.created_at, user.updated_at,
            user.is_active, user.is_deleted,
            json.dumps(user.metadata)
        ])
        logger.info(f"Created user: {user.user_id}")
        return user

    def get_user(self, user_id: str, include_deleted: bool = False) -> Optional[User]:
        """Get user by ID."""
        query = "SELECT * FROM users WHERE user_id = ?"
        if not include_deleted:
            query += " AND is_deleted = FALSE"
        
        result = self.conn.execute(query, [user_id]).fetchone()
        if result:
            return self._row_to_user(result)
        return None

    def list_users(self, include_deleted: bool = False, limit: int = 100) -> list[User]:
        """List all users."""
        query = "SELECT * FROM users"
        if not include_deleted:
            query += " WHERE is_deleted = FALSE"
        query += f" ORDER BY created_at DESC LIMIT {limit}"
        
        results = self.conn.execute(query).fetchall()
        return [self._row_to_user(r) for r in results]

    def _row_to_user(self, row: tuple) -> User:
        """Convert database row to User model."""
        return User(
            user_id=row[0],
            username=row[1],
            email=row[2],
            created_at=row[3],
            updated_at=row[4],
            is_active=row[5],
            is_deleted=row[6],
            deleted_at=row[7],
            metadata=json.loads(row[8]) if row[8] else {}
        )

    # ========================================
    # SESSION OPERATIONS
    # ========================================

    def create_session(self, session: Session) -> Session:
        """Create a new session."""
        self.conn.execute("""
            INSERT INTO sessions (session_id, user_id, created_at, last_active_at, is_active, is_deleted, memory_count, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            session.session_id, session.user_id,
            session.created_at, session.last_active_at,
            session.is_active, session.is_deleted,
            session.memory_count, json.dumps(session.metadata)
        ])
        logger.info(f"Created session: {session.session_id} for user {session.user_id}")
        return session

    def get_session(self, session_id: str, include_deleted: bool = False) -> Optional[Session]:
        """Get session by ID."""
        query = "SELECT * FROM sessions WHERE session_id = ?"
        if not include_deleted:
            query += " AND is_deleted = FALSE"
        
        result = self.conn.execute(query, [session_id]).fetchone()
        if result:
            return self._row_to_session(result)
        return None

    def get_user_sessions(self, user_id: str, include_deleted: bool = False) -> list[Session]:
        """Get all sessions for a user."""
        query = "SELECT * FROM sessions WHERE user_id = ?"
        if not include_deleted:
            query += " AND is_deleted = FALSE"
        query += " ORDER BY created_at DESC"
        
        results = self.conn.execute(query, [user_id]).fetchall()
        return [self._row_to_session(r) for r in results]

    def update_session_activity(self, session_id: str) -> None:
        """Update session last activity timestamp."""
        self.conn.execute("""
            UPDATE sessions SET last_active_at = ? WHERE session_id = ?
        """, [datetime.utcnow(), session_id])

    def update_session_memory_count(self, session_id: str, count: int) -> None:
        """Update session memory count."""
        self.conn.execute("""
            UPDATE sessions SET memory_count = ? WHERE session_id = ?
        """, [count, session_id])

    def _row_to_session(self, row: tuple) -> Session:
        """Convert database row to Session model."""
        return Session(
            session_id=row[0],
            user_id=row[1],
            created_at=row[2],
            last_active_at=row[3],
            is_active=row[4],
            is_deleted=row[5],
            deleted_at=row[6],
            deleted_by=row[7],
            delete_reason=row[8],
            memory_count=row[9],
            metadata=json.loads(row[10]) if row[10] else {}
        )

    # ========================================
    # SOFT DELETE OPERATIONS
    # ========================================

    def soft_delete_session(
        self,
        session_id: str,
        deleted_by: str,
        reason: Optional[str] = None
    ) -> SoftDeleteRecord:
        """Soft delete a session and its memories."""
        session = self.get_session(session_id, include_deleted=True)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        # Create soft delete record
        record = SoftDeleteRecord(
            id=str(uuid4()),
            entity_type="session",
            entity_id=session_id,
            user_id=session.user_id,
            session_id=session_id,
            deleted_by=deleted_by,
            reason=reason,
            original_data=session.model_dump()
        )

        # Store soft delete record
        self.conn.execute("""
            INSERT INTO soft_deletes (id, entity_type, entity_id, user_id, session_id, deleted_at, deleted_by, reason, original_data)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            record.id, record.entity_type, record.entity_id,
            record.user_id, record.session_id, record.deleted_at,
            record.deleted_by, record.reason, json.dumps(record.original_data, default=str)
        ])

        # Mark session as deleted
        self.conn.execute("""
            UPDATE sessions 
            SET is_deleted = TRUE, is_active = FALSE, deleted_at = ?, deleted_by = ?, delete_reason = ?
            WHERE session_id = ?
        """, [datetime.utcnow(), deleted_by, reason, session_id])

        logger.info(f"Soft deleted session {session_id} by {deleted_by}")
        return record

    def soft_delete_user_data(
        self,
        user_id: str,
        deleted_by: str,
        reason: Optional[str] = None
    ) -> list[SoftDeleteRecord]:
        """Soft delete all data for a user."""
        records = []
        
        # Get all sessions for user
        sessions = self.get_user_sessions(user_id, include_deleted=False)
        
        for session in sessions:
            record = self.soft_delete_session(session.session_id, deleted_by, reason)
            records.append(record)

        # Mark user as deleted
        self.conn.execute("""
            UPDATE users 
            SET is_deleted = TRUE, is_active = FALSE, deleted_at = ?
            WHERE user_id = ?
        """, [datetime.utcnow(), user_id])

        logger.info(f"Soft deleted all data for user {user_id} ({len(records)} sessions)")
        return records

    def get_soft_deletes(
        self,
        user_id: Optional[str] = None,
        include_restored: bool = False,
        limit: int = 100
    ) -> list[SoftDeleteRecord]:
        """Get soft delete records (admin only)."""
        query = "SELECT * FROM soft_deletes WHERE 1=1"
        params = []
        
        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)
        
        if not include_restored:
            query += " AND is_restored = FALSE"
        
        query += f" ORDER BY deleted_at DESC LIMIT {limit}"
        
        results = self.conn.execute(query, params).fetchall()
        return [self._row_to_soft_delete(r) for r in results]

    def restore_session(self, record_id: str, restored_by: str) -> Session:
        """Restore a soft-deleted session."""
        result = self.conn.execute(
            "SELECT * FROM soft_deletes WHERE id = ?", [record_id]
        ).fetchone()
        
        if not result:
            raise ValueError(f"Soft delete record {record_id} not found")
        
        record = self._row_to_soft_delete(result)
        
        if record.is_restored:
            raise ValueError(f"Record {record_id} already restored")

        # Restore session
        self.conn.execute("""
            UPDATE sessions 
            SET is_deleted = FALSE, is_active = TRUE, deleted_at = NULL, deleted_by = NULL, delete_reason = NULL
            WHERE session_id = ?
        """, [record.session_id])

        # Mark soft delete as restored
        self.conn.execute("""
            UPDATE soft_deletes 
            SET is_restored = TRUE, restored_at = ?, restored_by = ?
            WHERE id = ?
        """, [datetime.utcnow(), restored_by, record_id])

        logger.info(f"Restored session {record.session_id} by {restored_by}")
        return self.get_session(record.session_id, include_deleted=True)

    def _row_to_soft_delete(self, row: tuple) -> SoftDeleteRecord:
        """Convert database row to SoftDeleteRecord model."""
        return SoftDeleteRecord(
            id=row[0],
            entity_type=row[1],
            entity_id=row[2],
            user_id=row[3],
            session_id=row[4],
            deleted_at=row[5],
            deleted_by=row[6],
            reason=row[7],
            original_data=json.loads(row[8]) if row[8] else {},
            is_restored=row[9],
            restored_at=row[10],
            restored_by=row[11]
        )

    # ========================================
    # STATS
    # ========================================

    def get_stats(self) -> dict:
        """Get storage statistics."""
        total_users = self.conn.execute("SELECT COUNT(*) FROM users WHERE is_deleted = FALSE").fetchone()[0]
        total_sessions = self.conn.execute("SELECT COUNT(*) FROM sessions WHERE is_deleted = FALSE").fetchone()[0]
        deleted_sessions = self.conn.execute("SELECT COUNT(*) FROM sessions WHERE is_deleted = TRUE").fetchone()[0]
        total_memories = self.conn.execute("SELECT SUM(memory_count) FROM sessions WHERE is_deleted = FALSE").fetchone()[0] or 0
        
        return {
            "total_users": total_users,
            "total_sessions": total_sessions,
            "deleted_sessions": deleted_sessions,
            "total_memories": total_memories
        }

    def close(self) -> None:
        """Close database connection."""
        self.conn.close()
