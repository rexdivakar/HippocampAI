"""Database abstraction layer for dual SQLite/PostgreSQL support.

Provides a thin wrapper so that aiosqlite connections expose the same
interface as asyncpg (acquire, fetchrow, fetch, execute, transaction).
Switch between backends via the DB_TYPE environment variable.
"""

import logging
import os
import re
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SQL translation helpers
# ---------------------------------------------------------------------------

# Match $1, $2, ... positional params (asyncpg style)
_PG_PARAM_RE = re.compile(r"\$(\d+)")

# Match PostgreSQL type casts like ::jsonb, ::text, ::integer
_PG_CAST_RE = re.compile(r"::(\w+)")

# Match NOW() — replace with SQLite equivalent
_NOW_RE = re.compile(r"\bNOW\(\)", re.IGNORECASE)


def _pg_to_sqlite_sql(sql: str) -> str:
    """Translate PostgreSQL SQL to SQLite-compatible SQL."""
    # Strip type casts (::jsonb, ::text, etc.)
    sql = _PG_CAST_RE.sub("", sql)

    # Replace NOW() with strftime
    sql = _NOW_RE.sub("strftime('%Y-%m-%dT%H:%M:%fZ', 'now')", sql)

    # Replace $N positional params with ?
    # We need to replace them in order, mapping $1 -> first ?, $2 -> second ?, etc.
    sql = _PG_PARAM_RE.sub("?", sql)

    return sql


def _generate_uuid() -> str:
    """Generate a UUID string for SQLite rows."""
    return str(uuid.uuid4())


def _sqlite_adapt(value: Any) -> Any:
    """Convert Python types to SQLite-compatible types for parameters.

    Handles: bool -> int, UUID -> str, datetime -> ISO string.
    """
    if isinstance(value, bool):
        return 1 if value else 0
    if hasattr(value, "hex") and hasattr(value, "int"):
        # UUID object — convert to string
        return str(value)
    return value


def _row_to_dict(cursor_description: Any, row: tuple) -> dict:
    """Convert a sqlite3 row tuple to a dict using cursor column names."""
    columns = [desc[0] for desc in cursor_description]
    d = dict(zip(columns, row))
    # Convert SQLite integer bools back to Python bools for known boolean columns
    bool_columns = {"is_active", "is_admin", "email_verified"}
    for col in bool_columns:
        if col in d and isinstance(d[col], int):
            d[col] = bool(d[col])
    return d


# ---------------------------------------------------------------------------
# SQLite wrapper — exposes asyncpg-compatible interface
# ---------------------------------------------------------------------------


class _SQLiteRecord(dict):
    """Dict subclass that supports attribute-style access like asyncpg Record."""

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, int):
            return list(self.values())[key]
        return super().__getitem__(key)


class _SQLiteConnection:
    """Wraps an aiosqlite connection to look like an asyncpg connection."""

    def __init__(self, conn: Any) -> None:
        self._conn = conn
        self._in_transaction = False

    async def fetchrow(self, query: str, *args: Any) -> Optional[_SQLiteRecord]:
        """Fetch a single row, returning a dict-like Record or None."""
        sql = _pg_to_sqlite_sql(query)
        safe_args = tuple(_sqlite_adapt(a) for a in args)

        # Handle RETURNING * by splitting into execute + SELECT
        returning_row = await self._handle_returning(sql, safe_args)
        if returning_row is not None:
            return returning_row

        cursor = await self._conn.execute(sql, safe_args)
        row = await cursor.fetchone()
        if row is None:
            return None
        return _SQLiteRecord(_row_to_dict(cursor.description, row))

    async def fetch(self, query: str, *args: Any) -> list[_SQLiteRecord]:
        """Fetch multiple rows."""
        sql = _pg_to_sqlite_sql(query)
        safe_args = tuple(_sqlite_adapt(a) for a in args)
        cursor = await self._conn.execute(sql, safe_args)
        rows = await cursor.fetchall()
        if not rows:
            return []
        return [_SQLiteRecord(_row_to_dict(cursor.description, r)) for r in rows]

    async def execute(self, query: str, *args: Any) -> str:
        """Execute a statement and return an asyncpg-style command tag string."""
        sql = _pg_to_sqlite_sql(query)
        safe_args = tuple(_sqlite_adapt(a) for a in args)

        # Strip RETURNING clause for plain execute (not fetchrow)
        sql_for_exec = re.sub(r"\s+RETURNING\s+\*", "", sql, flags=re.IGNORECASE)

        cursor = await self._conn.execute(sql_for_exec, safe_args)
        if not self._in_transaction:
            await self._conn.commit()

        rowcount = cursor.rowcount
        sql_upper = sql_for_exec.strip().upper()
        if sql_upper.startswith("INSERT"):
            return f"INSERT 0 {rowcount}"
        elif sql_upper.startswith("UPDATE"):
            return f"UPDATE {rowcount}"
        elif sql_upper.startswith("DELETE"):
            return f"DELETE {rowcount}"
        return f"OK {rowcount}"

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[None]:
        """Transaction context manager matching asyncpg's conn.transaction()."""
        self._in_transaction = True
        try:
            yield
            await self._conn.commit()
        except Exception:
            await self._conn.rollback()
            raise
        finally:
            self._in_transaction = False

    async def _handle_returning(
        self, sql: str, args: tuple
    ) -> Optional[_SQLiteRecord]:
        """Handle INSERT/UPDATE ... RETURNING * for SQLite.

        SQLite < 3.35 doesn't support RETURNING. We execute the mutation,
        then fetch the affected row by rowid or known columns.
        """
        returning_match = re.search(r"\s+RETURNING\s+\*", sql, re.IGNORECASE)
        if not returning_match:
            return None

        # Determine the table name
        sql_upper = sql.strip().upper()
        if sql_upper.startswith("INSERT"):
            table_match = re.search(r"INSERT\s+INTO\s+(\w+)", sql, re.IGNORECASE)
        elif sql_upper.startswith("UPDATE"):
            table_match = re.search(r"UPDATE\s+(\w+)", sql, re.IGNORECASE)
        else:
            return None

        if not table_match:
            return None

        table = table_match.group(1)

        # Strip RETURNING * from the SQL for execution
        exec_sql = sql[: returning_match.start()] + sql[returning_match.end() :]

        # For INSERT, we need to inject a UUID if no id is provided
        if sql_upper.startswith("INSERT"):
            exec_sql, args = self._inject_uuid_if_needed(exec_sql, args)

        cursor = await self._conn.execute(exec_sql, args)
        if not self._in_transaction:
            await self._conn.commit()

        # Fetch the row back
        if sql_upper.startswith("INSERT"):
            # For inserts, use last_insert_rowid to find the row
            fetch_cursor = await self._conn.execute(
                f"SELECT * FROM {table} WHERE rowid = ?", (cursor.lastrowid,)
            )
        else:
            # For updates, try to find by id from the WHERE clause
            # Extract the id value from the args — it's typically the last positional param
            # in UPDATE ... WHERE id = ?
            where_match = re.search(r"WHERE\s+id\s*=\s*\?", exec_sql, re.IGNORECASE)
            if where_match:
                # The ? for WHERE id is the last param
                fetch_cursor = await self._conn.execute(
                    f"SELECT * FROM {table} WHERE id = ?", (args[-1],)
                )
            else:
                # Fallback: return None if we can't identify the row
                return None

        row = await fetch_cursor.fetchone()
        if row is None:
            return None
        return _SQLiteRecord(_row_to_dict(fetch_cursor.description, row))

    def _inject_uuid_if_needed(
        self, sql: str, args: tuple
    ) -> tuple[str, tuple]:
        """If the INSERT doesn't include an id column, inject a generated UUID."""
        # Match the column list and VALUES keyword together
        cols_match = re.search(r"\(([^)]+)\)\s*VALUES\s*\(([^)]+)\)", sql, re.IGNORECASE)
        if cols_match:
            columns = [c.strip().lower() for c in cols_match.group(1).split(",")]
            if "id" not in columns:
                new_cols = "id, " + cols_match.group(1)
                new_vals = "?, " + cols_match.group(2)
                replacement = f"({new_cols}) VALUES ({new_vals})"
                sql = sql[: cols_match.start()] + replacement + sql[cols_match.end():]
                args = (_generate_uuid(),) + args
        return sql, args


class _SQLitePool:
    """Connection pool-like wrapper for aiosqlite matching asyncpg Pool interface."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._conn: Any = None  # aiosqlite connection (single connection, serialized)

    async def connect(self) -> None:
        """Open the database connection and initialize schema."""
        import aiosqlite

        # Ensure the directory exists
        db_dir = os.path.dirname(self._db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

        self._conn = await aiosqlite.connect(self._db_path)
        # Enable WAL mode for better concurrent read performance
        await self._conn.execute("PRAGMA journal_mode=WAL")
        # Enable foreign keys
        await self._conn.execute("PRAGMA foreign_keys=ON")
        await self._conn.commit()

        # Initialize schema
        schema_path = Path(__file__).parent / "schema_sqlite.sql"
        if schema_path.exists():
            schema_sql = schema_path.read_text()
            await self._conn.executescript(schema_sql)
            logger.info(f"SQLite schema initialized from {schema_path}")

        logger.info(f"SQLite database opened at {self._db_path}")

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[_SQLiteConnection]:
        """Yield a connection wrapper, mimicking asyncpg pool.acquire()."""
        if self._conn is None:
            raise RuntimeError("SQLite pool not connected. Call connect() first.")
        yield _SQLiteConnection(self._conn)

    async def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            await self._conn.close()
            self._conn = None
            logger.info("SQLite connection closed")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


async def create_db_pool(
    db_type: str = "sqlite",
    # PostgreSQL options
    pg_host: str = "localhost",
    pg_port: int = 5432,
    pg_database: str = "hippocampai",
    pg_user: str = "hippocampai",
    pg_password: str = "hippocampai_secret",
    pg_min_size: int = 2,
    pg_max_size: int = 10,
    # SQLite options
    sqlite_path: str = "data/hippocampai_auth.db",
) -> Any:
    """Create a database pool based on DB_TYPE.

    Returns either an asyncpg.Pool or a _SQLitePool, both of which expose:
      - acquire() -> async context manager yielding a connection
      - close() -> coroutine

    The connection objects expose:
      - fetchrow(query, *args) -> Optional[Record]
      - fetch(query, *args) -> list[Record]
      - execute(query, *args) -> str (command tag)
      - transaction() -> async context manager
    """
    if db_type == "postgres":
        import asyncpg

        pool = await asyncpg.create_pool(
            host=pg_host,
            port=pg_port,
            database=pg_database,
            user=pg_user,
            password=pg_password,
            min_size=pg_min_size,
            max_size=pg_max_size,
        )
        logger.info(f"PostgreSQL pool created ({pg_host}:{pg_port}/{pg_database})")
        return pool

    elif db_type == "sqlite":
        pool = _SQLitePool(sqlite_path)
        await pool.connect()
        return pool

    else:
        raise ValueError(f"Unsupported DB_TYPE: {db_type!r}. Use 'sqlite' or 'postgres'.")
