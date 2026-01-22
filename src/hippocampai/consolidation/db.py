"""Database abstraction layer for consolidation runs - supports SQLite and PostgreSQL."""

import json
import logging
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Generator, Optional

from hippocampai.consolidation.models import ConsolidationRun, ConsolidationStatus

logger = logging.getLogger(__name__)

# Database configuration
DB_TYPE = os.getenv("CONSOLIDATION_DB_TYPE", "sqlite")  # 'sqlite' or 'postgres'
SQLITE_DB_PATH = os.getenv("CONSOLIDATION_DB_PATH", "data/consolidation.db")
POSTGRES_URL = os.getenv("DATABASE_URL", "postgresql://localhost/hippocampai")


class ConsolidationDatabase:
    """Database abstraction for consolidation runs supporting SQLite and PostgreSQL."""

    def __init__(self, db_type: Optional[str] = None, db_path: Optional[str] = None):
        """
        Initialize database connection.

        Args:
            db_type: 'sqlite' or 'postgres' (defaults to env var CONSOLIDATION_DB_TYPE)
            db_path: Path to SQLite database or Postgres URL
        """
        self.db_type = db_type or DB_TYPE
        self.db_path = db_path or (SQLITE_DB_PATH if self.db_type == "sqlite" else POSTGRES_URL)

        if self.db_type == "sqlite":
            self._init_sqlite()
        elif self.db_type == "postgres":
            self._init_postgres()
        else:
            raise ValueError(f"Unsupported database type: {self.db_type}")

        logger.info(f"Initialized {self.db_type} database for consolidation runs")

    def _init_sqlite(self):
        """Initialize SQLite database and create tables."""
        # Ensure directory exists
        db_file = Path(self.db_path)
        db_file.parent.mkdir(parents=True, exist_ok=True)

        # Create tables
        with self.get_connection() as conn:
            schema_path = Path(__file__).parent / "schema_sqlite.sql"
            if schema_path.exists():
                with open(schema_path) as f:
                    conn.executescript(f.read())
                logger.info(f"SQLite database initialized at {self.db_path}")
            else:
                logger.warning(f"Schema file not found: {schema_path}")

    def _init_postgres(self):
        """Initialize PostgreSQL connection and create tables."""
        try:
            import psycopg2
            from psycopg2.extras import DictCursor

            self._psycopg2 = psycopg2
            self._DictCursor = DictCursor

            # Test connection and create tables
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    schema_path = Path(__file__).parent / "schema.sql"
                    if schema_path.exists():
                        with open(schema_path) as f:
                            cur.execute(f.read())
                        conn.commit()
                        logger.info("PostgreSQL database initialized")
                    else:
                        logger.warning(f"Schema file not found: {schema_path}")
        except ImportError:
            logger.error("psycopg2 not installed. Install with: pip install psycopg2-binary")
            raise

    @contextmanager
    def get_connection(self) -> Generator[Any, None, None]:
        """Get database connection context manager."""
        if self.db_type == "sqlite":
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row  # Enable dict-like access
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()
        elif self.db_type == "postgres":
            conn = self._psycopg2.connect(self.db_path)
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    def create_run(self, run: ConsolidationRun) -> str:
        """
        Create a new consolidation run.

        Args:
            run: ConsolidationRun object

        Returns:
            Run ID
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()

            if self.db_type == "sqlite":
                cursor.execute(
                    """
                    INSERT INTO consolidation_runs (
                        id, user_id, agent_id, status, mode, started_at,
                        lookback_hours, min_importance, dry_run, triggered_by
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run.id,
                        run.user_id,
                        run.agent_id,
                        run.status.value,
                        "preview" if run.dry_run else "live",
                        run.started_at.isoformat(),
                        run.lookback_hours,
                        run.min_importance,
                        1 if run.dry_run else 0,
                        "manual",
                    ),
                )
            else:  # postgres
                cursor.execute(
                    """
                    INSERT INTO consolidation_runs (
                        id, user_id, agent_id, status, mode, started_at,
                        lookback_hours, min_importance, dry_run, triggered_by
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        run.id,
                        run.user_id,
                        run.agent_id,
                        run.status.value,
                        "preview" if run.dry_run else "live",
                        run.started_at,
                        run.lookback_hours,
                        run.min_importance,
                        run.dry_run,
                        "manual",
                    ),
                )

        logger.info(f"Created consolidation run {run.id} for user {run.user_id}")
        return run.id

    def update_run(self, run_id: str, updates: dict[str, Any]) -> None:
        """
        Update a consolidation run.

        Args:
            run_id: Run ID
            updates: Dictionary of fields to update
        """
        if not updates:
            return

        # Build dynamic UPDATE query
        set_clauses = []
        values = []

        for key, value in updates.items():
            if key == "status" and isinstance(value, ConsolidationStatus):
                value = value.value
            elif key == "completed_at" and isinstance(value, datetime):
                value = value.isoformat() if self.db_type == "sqlite" else value

            set_clauses.append(f"{key} = ?")
            values.append(value)

        values.append(run_id)

        query = f"UPDATE consolidation_runs SET {', '.join(set_clauses)} WHERE id = ?"
        if self.db_type == "postgres":
            query = query.replace("?", "%s")

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, values)

        logger.debug(f"Updated consolidation run {run_id}: {list(updates.keys())}")

    def get_run(self, run_id: str) -> Optional[dict[str, Any]]:
        """
        Get a consolidation run by ID.

        Args:
            run_id: Run ID

        Returns:
            Run data as dict or None
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            param = "?" if self.db_type == "sqlite" else "%s"
            cursor.execute(f"SELECT * FROM consolidation_runs WHERE id = {param}", (run_id,))
            row = cursor.fetchone()

            if row:
                return dict(row)
            return None

    def get_user_runs(self, user_id: str, limit: int = 10, offset: int = 0) -> list[dict[str, Any]]:
        """
        Get consolidation runs for a user.

        Args:
            user_id: User ID
            limit: Maximum number of runs to return
            offset: Offset for pagination

        Returns:
            List of run data as dicts
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            param = "?" if self.db_type == "sqlite" else "%s"
            cursor.execute(
                f"""
                SELECT * FROM consolidation_runs
                WHERE user_id = {param}
                ORDER BY started_at DESC
                LIMIT {param} OFFSET {param}
                """,
                (user_id, limit, offset),
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_stats(self, user_id: str) -> dict[str, Any]:
        """
        Get aggregate statistics for a user.

        Args:
            user_id: User ID

        Returns:
            Statistics dictionary
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            param = "?" if self.db_type == "sqlite" else "%s"
            cursor.execute(
                f"""
                SELECT * FROM consolidation_statistics
                WHERE user_id = {param}
                """,
                (user_id,),
            )
            row = cursor.fetchone()

            if row:
                return dict(row)

            # Return empty stats if no runs yet
            return {
                "user_id": user_id,
                "total_runs": 0,
                "successful_runs": 0,
                "failed_runs": 0,
                "total_memories_promoted": 0,
                "total_memories_archived": 0,
                "total_memories_deleted": 0,
                "total_memories_synthesized": 0,
            }

    def get_latest_run(self, user_id: str) -> Optional[dict[str, Any]]:
        """
        Get the most recent run for a user.

        Args:
            user_id: User ID

        Returns:
            Run data as dict or None
        """
        runs = self.get_user_runs(user_id, limit=1)
        return runs[0] if runs else None

    def store_run_detail(
        self,
        run_id: str,
        memory_id: str,
        action: str,
        cluster_id: Optional[str] = None,
        reason: Optional[str] = None,
        old_importance: Optional[float] = None,
        new_importance: Optional[float] = None,
        old_text: Optional[str] = None,
        new_text: Optional[str] = None,
        source_memory_ids: Optional[list[str]] = None,
    ) -> None:
        """
        Store detail about a specific memory action.

        Args:
            run_id: Run ID
            memory_id: Memory ID
            action: Action taken (promoted, archived, deleted, etc.)
            cluster_id: Cluster ID (optional)
            reason: Reason for action (optional)
            old_importance: Old importance score (optional)
            new_importance: New importance score (optional)
            old_text: Old text (optional)
            new_text: New text (optional)
            source_memory_ids: Source memory IDs for synthesized memories (optional)
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Convert source_memory_ids to JSON string for SQLite
            source_ids_json = json.dumps(source_memory_ids) if source_memory_ids else None

            if self.db_type == "sqlite":
                cursor.execute(
                    """
                    INSERT INTO consolidation_run_details (
                        run_id, memory_id, cluster_id, action, reason,
                        old_importance, new_importance, old_text, new_text, source_memory_ids
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        memory_id,
                        cluster_id,
                        action,
                        reason,
                        old_importance,
                        new_importance,
                        old_text,
                        new_text,
                        source_ids_json,
                    ),
                )
            else:  # postgres
                cursor.execute(
                    """
                    INSERT INTO consolidation_run_details (
                        run_id, memory_id, cluster_id, action, reason,
                        old_importance, new_importance, old_text, new_text, source_memory_ids
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        run_id,
                        memory_id,
                        cluster_id,
                        action,
                        reason,
                        old_importance,
                        new_importance,
                        old_text,
                        new_text,
                        source_memory_ids,
                    ),
                )

    def get_run_details(self, run_id: str) -> list[dict[str, Any]]:
        """
        Get all details for a consolidation run.

        Args:
            run_id: Run ID

        Returns:
            List of detail records
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            param = "?" if self.db_type == "sqlite" else "%s"
            cursor.execute(
                f"""
                SELECT * FROM consolidation_run_details
                WHERE run_id = {param}
                ORDER BY created_at
                """,
                (run_id,),
            )
            details = [dict(row) for row in cursor.fetchall()]

            # Parse JSON source_memory_ids for SQLite
            if self.db_type == "sqlite":
                for detail in details:
                    if detail.get("source_memory_ids"):
                        try:
                            detail["source_memory_ids"] = json.loads(detail["source_memory_ids"])
                        except (json.JSONDecodeError, TypeError):
                            detail["source_memory_ids"] = []

            return details


# Global database instance
_db: Optional[ConsolidationDatabase] = None


def get_db() -> ConsolidationDatabase:
    """Get or create global database instance."""
    global _db
    if _db is None:
        _db = ConsolidationDatabase()
    return _db


def init_db(db_type: Optional[str] = None, db_path: Optional[str] = None) -> None:
    """
    Initialize or reinitialize database.

    Args:
        db_type: 'sqlite' or 'postgres'
        db_path: Database path or URL
    """
    global _db
    _db = ConsolidationDatabase(db_type=db_type, db_path=db_path)
