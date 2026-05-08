"""End-to-end tests for the dual database (SQLite / PostgreSQL) abstraction layer.

Tests the SQLite wrapper, SQL translation, and full AuthService CRUD through SQLite.
"""

import asyncio
import os
import tempfile

import pytest
import pytest_asyncio

# Ensure src is on the path
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hippocampai.auth.db import (
    _pg_to_sqlite_sql,
    _generate_uuid,
    _sqlite_adapt,
    create_db_pool,
    _SQLitePool,
)


# ---------------------------------------------------------------------------
# Unit tests: SQL translation
# ---------------------------------------------------------------------------

class TestSQLTranslation:
    """Test PostgreSQL -> SQLite SQL translation."""

    def test_positional_params(self):
        sql = "SELECT * FROM users WHERE id = $1 AND email = $2"
        result = _pg_to_sqlite_sql(sql)
        assert result == "SELECT * FROM users WHERE id = ? AND email = ?"

    def test_jsonb_cast_stripped(self):
        sql = "INSERT INTO api_keys (scopes) VALUES ($1::jsonb)"
        result = _pg_to_sqlite_sql(sql)
        assert "::jsonb" not in result
        assert "?" in result

    def test_text_cast_stripped(self):
        sql = "SELECT $1::text"
        result = _pg_to_sqlite_sql(sql)
        assert "::text" not in result

    def test_now_replaced(self):
        sql = "UPDATE users SET last_login_at = NOW() WHERE id = $1"
        result = _pg_to_sqlite_sql(sql)
        assert "NOW()" not in result
        assert "strftime" in result

    def test_combined_translation(self):
        sql = """
            INSERT INTO api_keys (
                user_id, key_prefix, key_hash, name, scopes,
                rate_limit_tier, expires_at
            )
            VALUES ($1, $2, $3, $4, $5::jsonb, $6, $7)
            RETURNING *
        """
        result = _pg_to_sqlite_sql(sql)
        assert "::jsonb" not in result
        assert "$" not in result
        assert result.count("?") == 7

    def test_multiple_now_calls(self):
        sql = "INSERT INTO t (a, b) VALUES (NOW(), NOW())"
        result = _pg_to_sqlite_sql(sql)
        assert result.count("strftime") == 2

    def test_no_params_passthrough(self):
        sql = "SELECT * FROM users"
        result = _pg_to_sqlite_sql(sql)
        assert result == "SELECT * FROM users"


class TestHelpers:
    """Test helper functions."""

    def test_generate_uuid(self):
        uid = _generate_uuid()
        assert isinstance(uid, str)
        assert len(uid) == 36  # UUID format: 8-4-4-4-12

    def test_generate_uuid_unique(self):
        uids = {_generate_uuid() for _ in range(100)}
        assert len(uids) == 100

    def test_sqlite_adapt_true(self):
        assert _sqlite_adapt(True) == 1

    def test_sqlite_adapt_false(self):
        assert _sqlite_adapt(False) == 0

    def test_sqlite_adapt_passthrough_int(self):
        assert _sqlite_adapt(42) == 42

    def test_sqlite_adapt_passthrough_str(self):
        assert _sqlite_adapt("hello") == "hello"

    def test_sqlite_adapt_passthrough_none(self):
        assert _sqlite_adapt(None) is None


# ---------------------------------------------------------------------------
# Integration tests: SQLite pool + connection
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def sqlite_pool():
    """Create a temporary SQLite pool for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_auth.db")
        pool = await create_db_pool(db_type="sqlite", sqlite_path=db_path)
        yield pool
        await pool.close()


@pytest.mark.asyncio
class TestSQLitePool:
    """Test the SQLite pool and connection wrapper."""

    async def test_pool_creation(self, sqlite_pool):
        """Pool should create and connect successfully."""
        assert sqlite_pool is not None

    async def test_acquire_connection(self, sqlite_pool):
        """Should be able to acquire a connection."""
        async with sqlite_pool.acquire() as conn:
            assert conn is not None

    async def test_schema_initialized(self, sqlite_pool):
        """Schema tables should exist after pool creation."""
        async with sqlite_pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
            table_names = {r["name"] for r in rows}
            assert "users" in table_names
            assert "api_keys" in table_names
            assert "api_key_usage" in table_names
            assert "organizations" in table_names
            assert "sessions" in table_names
            assert "audit_log" in table_names
            assert "rate_limit_buckets" in table_names

    async def test_insert_and_fetchrow(self, sqlite_pool):
        """INSERT with RETURNING * should work via the emulation layer."""
        async with sqlite_pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO users (email, hashed_password, full_name, tier, is_admin)
                VALUES ($1, $2, $3, $4, $5)
                RETURNING *
                """,
                "test@example.com",
                "$2b$12$fakehash",
                "Test User",
                "free",
                False,
            )
            assert row is not None
            assert row["email"] == "test@example.com"
            assert row["full_name"] == "Test User"
            assert row["tier"] == "free"
            assert row["is_admin"] is False
            assert row["is_active"] is True
            assert row["id"] is not None  # UUID should be auto-generated

    async def test_fetch_multiple_rows(self, sqlite_pool):
        """fetch() should return a list of records."""
        async with sqlite_pool.acquire() as conn:
            # Insert two users
            await conn.fetchrow(
                "INSERT INTO users (email, hashed_password, tier, is_admin) VALUES ($1, $2, $3, $4) RETURNING *",
                "user1@test.com", "$2b$12$hash1", "free", False,
            )
            await conn.fetchrow(
                "INSERT INTO users (email, hashed_password, tier, is_admin) VALUES ($1, $2, $3, $4) RETURNING *",
                "user2@test.com", "$2b$12$hash2", "pro", False,
            )

            rows = await conn.fetch(
                "SELECT * FROM users ORDER BY email LIMIT $1 OFFSET $2",
                10, 0,
            )
            assert len(rows) == 2
            assert rows[0]["email"] == "user1@test.com"
            assert rows[1]["email"] == "user2@test.com"

    async def test_fetchrow_not_found(self, sqlite_pool):
        """fetchrow should return None when no row matches."""
        async with sqlite_pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM users WHERE email = $1",
                "nonexistent@test.com",
            )
            assert row is None

    async def test_execute_delete(self, sqlite_pool):
        """execute() should return asyncpg-style command tags."""
        async with sqlite_pool.acquire() as conn:
            # Insert a user first
            row = await conn.fetchrow(
                "INSERT INTO users (email, hashed_password, tier, is_admin) VALUES ($1, $2, $3, $4) RETURNING *",
                "delete_me@test.com", "$2b$12$hash", "free", False,
            )
            user_id = row["id"]

            # Delete and check command tag
            result = await conn.execute(
                "DELETE FROM users WHERE id = $1", user_id
            )
            assert result.endswith("1")  # "DELETE 1"

    async def test_execute_update(self, sqlite_pool):
        """execute() UPDATE should return proper command tag."""
        async with sqlite_pool.acquire() as conn:
            row = await conn.fetchrow(
                "INSERT INTO users (email, hashed_password, tier, is_admin) VALUES ($1, $2, $3, $4) RETURNING *",
                "update_me@test.com", "$2b$12$hash", "free", False,
            )
            user_id = row["id"]

            result = await conn.execute(
                "UPDATE users SET last_login_at = NOW() WHERE id = $1",
                user_id,
            )
            assert result.endswith("1")  # "UPDATE 1"

    async def test_execute_delete_not_found(self, sqlite_pool):
        """execute() on non-existent row should return 0."""
        async with sqlite_pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM users WHERE id = $1",
                "00000000-0000-0000-0000-000000000000",
            )
            assert result.endswith("0")  # "DELETE 0"

    async def test_update_with_returning(self, sqlite_pool):
        """UPDATE ... RETURNING * should work."""
        async with sqlite_pool.acquire() as conn:
            row = await conn.fetchrow(
                "INSERT INTO users (email, hashed_password, full_name, tier, is_admin) VALUES ($1, $2, $3, $4, $5) RETURNING *",
                "update_ret@test.com", "$2b$12$hash", "Original Name", "free", False,
            )
            user_id = row["id"]

            updated = await conn.fetchrow(
                "UPDATE users SET full_name = $1 WHERE id = $2 RETURNING *",
                "Updated Name",
                user_id,
            )
            assert updated is not None
            assert updated["full_name"] == "Updated Name"
            assert updated["email"] == "update_ret@test.com"

    async def test_transaction_commit(self, sqlite_pool):
        """Transaction context manager should commit on success."""
        async with sqlite_pool.acquire() as conn:
            async with conn.transaction():
                await conn.fetchrow(
                    "INSERT INTO users (email, hashed_password, tier, is_admin) VALUES ($1, $2, $3, $4) RETURNING *",
                    "tx_commit@test.com", "$2b$12$hash", "free", False,
                )

            # Should be visible after transaction
            row = await conn.fetchrow(
                "SELECT * FROM users WHERE email = $1", "tx_commit@test.com"
            )
            assert row is not None

    async def test_transaction_rollback(self, sqlite_pool):
        """Transaction should rollback on exception."""
        async with sqlite_pool.acquire() as conn:
            try:
                async with conn.transaction():
                    await conn.fetchrow(
                        "INSERT INTO users (email, hashed_password, tier, is_admin) VALUES ($1, $2, $3, $4) RETURNING *",
                        "tx_rollback@test.com", "$2b$12$hash", "free", False,
                    )
                    raise RuntimeError("Force rollback")
            except RuntimeError:
                pass

            # Should NOT be visible after rollback
            row = await conn.fetchrow(
                "SELECT * FROM users WHERE email = $1", "tx_rollback@test.com"
            )
            assert row is None

    async def test_jsonb_insert(self, sqlite_pool):
        """Inserting JSON via ::jsonb cast should work (cast stripped)."""
        async with sqlite_pool.acquire() as conn:
            # First create a user
            user_row = await conn.fetchrow(
                "INSERT INTO users (email, hashed_password, tier, is_admin) VALUES ($1, $2, $3, $4) RETURNING *",
                "jsonb_test@test.com", "$2b$12$hash", "free", False,
            )
            user_id = user_row["id"]

            # Insert an API key with scopes as JSON via ::jsonb cast
            import json
            scopes_json = json.dumps(["memories:read", "memories:write"])
            key_row = await conn.fetchrow(
                """
                INSERT INTO api_keys (
                    user_id, key_prefix, key_hash, name, scopes, rate_limit_tier, expires_at
                )
                VALUES ($1, $2, $3, $4, $5::jsonb, $6, $7)
                RETURNING *
                """,
                user_id,
                "hc_live_abc",
                "$2b$12$keyhash",
                "Test Key",
                scopes_json,
                "free",
                None,
            )
            assert key_row is not None
            assert key_row["name"] == "Test Key"
            assert key_row["user_id"] == user_id

    async def test_join_query(self, sqlite_pool):
        """JOIN queries (used in validate_api_key) should work."""
        async with sqlite_pool.acquire() as conn:
            user_row = await conn.fetchrow(
                "INSERT INTO users (email, hashed_password, tier, is_admin) VALUES ($1, $2, $3, $4) RETURNING *",
                "join_test@test.com", "$2b$12$hash", "pro", False,
            )
            user_id = user_row["id"]

            import json
            await conn.fetchrow(
                """
                INSERT INTO api_keys (user_id, key_prefix, key_hash, name, scopes, rate_limit_tier, expires_at)
                VALUES ($1, $2, $3, $4, $5::jsonb, $6, $7) RETURNING *
                """,
                user_id, "hc_live_xyz", "$2b$12$keyhash", "Join Key",
                json.dumps(["memories:read"]), "pro", None,
            )

            # The validate_api_key JOIN query
            rows = await conn.fetch(
                """
                SELECT ak.*, u.tier as user_tier
                FROM api_keys ak
                JOIN users u ON ak.user_id = u.id
                WHERE ak.is_active = true
                    AND u.is_active = true
                    AND (ak.expires_at IS NULL OR ak.expires_at > NOW())
                """
            )
            assert len(rows) >= 1
            found = [r for r in rows if r["key_prefix"] == "hc_live_xyz"]
            assert len(found) == 1
            assert found[0]["user_tier"] == "pro"


# ---------------------------------------------------------------------------
# Integration tests: Full AuthService through SQLite
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def auth_service(sqlite_pool):
    """Create an AuthService backed by SQLite."""
    from hippocampai.auth.auth_service import AuthService
    return AuthService(sqlite_pool)


@pytest.mark.asyncio
class TestAuthServiceSQLite:
    """Test the full AuthService using SQLite backend."""

    async def test_create_user(self, auth_service):
        """Should create a user and return it."""
        from hippocampai.auth.models import UserCreate
        user = await auth_service.create_user(
            UserCreate(
                email="newuser@test.com",
                password="securepassword123",
                full_name="New User",
            )
        )
        assert user.email == "newuser@test.com"
        assert user.full_name == "New User"
        assert user.tier.value == "free"
        assert user.is_active is True
        assert user.is_admin is False
        assert user.id is not None

    async def test_create_user_duplicate_email(self, auth_service):
        """Should raise ValueError for duplicate email."""
        from hippocampai.auth.models import UserCreate
        await auth_service.create_user(
            UserCreate(email="dupe@test.com", password="password123")
        )
        with pytest.raises(ValueError, match="already exists"):
            await auth_service.create_user(
                UserCreate(email="dupe@test.com", password="password456")
            )

    async def test_authenticate_user(self, auth_service):
        """Should authenticate with correct credentials."""
        from hippocampai.auth.models import UserCreate, UserLogin
        await auth_service.create_user(
            UserCreate(email="auth@test.com", password="mypassword123")
        )
        user = await auth_service.authenticate_user(
            UserLogin(email="auth@test.com", password="mypassword123")
        )
        assert user is not None
        assert user.email == "auth@test.com"

    async def test_authenticate_wrong_password(self, auth_service):
        """Should return None for wrong password."""
        from hippocampai.auth.models import UserCreate, UserLogin
        await auth_service.create_user(
            UserCreate(email="authfail@test.com", password="correct_pass1")
        )
        user = await auth_service.authenticate_user(
            UserLogin(email="authfail@test.com", password="wrong_password")
        )
        assert user is None

    async def test_get_user(self, auth_service):
        """Should retrieve user by ID."""
        from hippocampai.auth.models import UserCreate
        created = await auth_service.create_user(
            UserCreate(email="getme@test.com", password="password123")
        )
        fetched = await auth_service.get_user(created.id)
        assert fetched is not None
        assert fetched.email == "getme@test.com"
        assert fetched.id == created.id

    async def test_get_user_by_email(self, auth_service):
        """Should retrieve user by email."""
        from hippocampai.auth.models import UserCreate
        await auth_service.create_user(
            UserCreate(email="byemail@test.com", password="password123")
        )
        fetched = await auth_service.get_user_by_email("byemail@test.com")
        assert fetched is not None
        assert fetched.email == "byemail@test.com"

    async def test_update_user(self, auth_service):
        """Should update user fields."""
        from hippocampai.auth.models import UserCreate, UserUpdate, UserTier
        created = await auth_service.create_user(
            UserCreate(email="updateme@test.com", password="password123", full_name="Old Name")
        )
        updated = await auth_service.update_user(
            created.id,
            UserUpdate(full_name="New Name", tier=UserTier.PRO),
        )
        assert updated is not None
        assert updated.full_name == "New Name"
        assert updated.tier == UserTier.PRO

    async def test_delete_user(self, auth_service):
        """Should delete user and return True."""
        from hippocampai.auth.models import UserCreate
        created = await auth_service.create_user(
            UserCreate(email="deleteme@test.com", password="password123")
        )
        result = await auth_service.delete_user(created.id)
        assert result is True

        # Verify gone
        fetched = await auth_service.get_user(created.id)
        assert fetched is None

    async def test_delete_user_not_found(self, auth_service):
        """Should return False for non-existent user."""
        from uuid import UUID
        result = await auth_service.delete_user(
            UUID("00000000-0000-0000-0000-000000000000")
        )
        assert result is False

    async def test_list_users(self, auth_service):
        """Should list users with pagination."""
        from hippocampai.auth.models import UserCreate
        for i in range(5):
            await auth_service.create_user(
                UserCreate(email=f"list{i}@test.com", password="password123")
            )
        users = await auth_service.list_users(limit=3, offset=0)
        assert len(users) == 3
        all_users = await auth_service.list_users(limit=100, offset=0)
        assert len(all_users) == 5

    async def test_create_api_key(self, auth_service):
        """Should create an API key and return the secret."""
        from hippocampai.auth.models import UserCreate, APIKeyCreate
        user = await auth_service.create_user(
            UserCreate(email="apikey@test.com", password="password123")
        )
        response = await auth_service.create_api_key(
            user.id,
            APIKeyCreate(name="My Key", scopes=["memories:read", "memories:write"]),
        )
        assert response.secret_key.startswith("hc_live_")
        assert response.api_key.name == "My Key"
        assert response.api_key.user_id == user.id
        assert response.api_key.is_active is True

    async def test_validate_api_key(self, auth_service):
        """Should validate a correct API key."""
        from hippocampai.auth.models import UserCreate, APIKeyCreate
        user = await auth_service.create_user(
            UserCreate(email="validate@test.com", password="password123")
        )
        response = await auth_service.create_api_key(
            user.id,
            APIKeyCreate(name="Validate Key"),
        )
        result = await auth_service.validate_api_key(response.secret_key)
        assert result is not None
        assert str(result["user_id"]) == str(user.id)

    async def test_validate_api_key_invalid(self, auth_service):
        """Should return None for invalid API key."""
        result = await auth_service.validate_api_key("hc_live_invalid_key_here")
        assert result is None

    async def test_list_user_api_keys(self, auth_service):
        """Should list all API keys for a user."""
        from hippocampai.auth.models import UserCreate, APIKeyCreate
        user = await auth_service.create_user(
            UserCreate(email="listkeys@test.com", password="password123")
        )
        for i in range(3):
            await auth_service.create_api_key(
                user.id, APIKeyCreate(name=f"Key {i}")
            )
        keys = await auth_service.list_user_api_keys(user.id)
        assert len(keys) == 3

    async def test_revoke_api_key(self, auth_service):
        """Should deactivate an API key."""
        from hippocampai.auth.models import UserCreate, APIKeyCreate
        user = await auth_service.create_user(
            UserCreate(email="revoke@test.com", password="password123")
        )
        response = await auth_service.create_api_key(
            user.id, APIKeyCreate(name="Revoke Me")
        )
        result = await auth_service.revoke_api_key(response.api_key.id)
        assert result is True

        # Verify it's deactivated
        key = await auth_service.get_api_key(response.api_key.id)
        assert key is not None
        assert key.is_active is False

    async def test_delete_api_key(self, auth_service):
        """Should permanently delete an API key."""
        from hippocampai.auth.models import UserCreate, APIKeyCreate
        user = await auth_service.create_user(
            UserCreate(email="delkey@test.com", password="password123")
        )
        response = await auth_service.create_api_key(
            user.id, APIKeyCreate(name="Delete Me")
        )
        result = await auth_service.delete_api_key(response.api_key.id)
        assert result is True

        key = await auth_service.get_api_key(response.api_key.id)
        assert key is None

    async def test_rotate_api_key(self, auth_service):
        """Should rotate an API key — new secret, same metadata."""
        from hippocampai.auth.models import UserCreate, APIKeyCreate
        user = await auth_service.create_user(
            UserCreate(email="rotate@test.com", password="password123")
        )
        original = await auth_service.create_api_key(
            user.id, APIKeyCreate(name="Rotate Me")
        )
        rotated = await auth_service.rotate_api_key(original.api_key.id)
        assert rotated is not None
        assert rotated.secret_key != original.secret_key
        assert rotated.api_key.name == "Rotate Me"
        assert rotated.api_key.id == original.api_key.id

        # Old key should no longer validate
        old_result = await auth_service.validate_api_key(original.secret_key)
        assert old_result is None

        # New key should validate
        new_result = await auth_service.validate_api_key(rotated.secret_key)
        assert new_result is not None

    async def test_log_api_usage(self, auth_service):
        """Should log API usage without errors."""
        from hippocampai.auth.models import UserCreate, APIKeyCreate
        user = await auth_service.create_user(
            UserCreate(email="usage@test.com", password="password123")
        )
        response = await auth_service.create_api_key(
            user.id, APIKeyCreate(name="Usage Key")
        )
        # Should not raise
        await auth_service.log_api_usage(
            api_key_id=response.api_key.id,
            endpoint="/v1/memories",
            method="POST",
            status_code=200,
            tokens_used=100,
            response_time_ms=45.2,
        )


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestConfig:
    """Test config.py dual DB fields."""

    def test_default_db_type(self):
        from hippocampai.config import Config
        # Clear any existing env vars for clean test
        env_backup = os.environ.get("DB_TYPE")
        os.environ.pop("DB_TYPE", None)
        try:
            config = Config()
            assert config.db_type == "sqlite"
        finally:
            if env_backup is not None:
                os.environ["DB_TYPE"] = env_backup

    def test_postgres_db_type(self):
        from hippocampai.config import Config
        os.environ["DB_TYPE"] = "postgres"
        try:
            config = Config()
            assert config.db_type == "postgres"
        finally:
            os.environ.pop("DB_TYPE", None)

    def test_sqlite_path_default(self):
        from hippocampai.config import Config
        os.environ.pop("DB_TYPE", None)
        os.environ.pop("SQLITE_PATH", None)
        try:
            config = Config()
            assert config.sqlite_path == "data/hippocampai_auth.db"
        finally:
            pass

    def test_sqlite_path_custom(self):
        from hippocampai.config import Config
        os.environ["SQLITE_PATH"] = "/tmp/custom.db"
        try:
            config = Config()
            assert config.sqlite_path == "/tmp/custom.db"
        finally:
            os.environ.pop("SQLITE_PATH", None)


# ---------------------------------------------------------------------------
# Factory tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestFactory:
    """Test the create_db_pool factory."""

    async def test_sqlite_factory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "factory_test.db")
            pool = await create_db_pool(db_type="sqlite", sqlite_path=db_path)
            assert pool is not None
            assert isinstance(pool, _SQLitePool)
            await pool.close()

    async def test_invalid_db_type(self):
        with pytest.raises(ValueError, match="Unsupported DB_TYPE"):
            await create_db_pool(db_type="mysql")
