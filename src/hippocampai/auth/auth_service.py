"""Authentication service for user and API key management."""

import json
import secrets
from datetime import datetime, timedelta
from typing import Optional
from uuid import UUID

import bcrypt
from asyncpg import Pool

from hippocampai.auth.models import (
    APIKey,
    APIKeyCreate,
    APIKeyResponse,
    User,
    UserCreate,
    UserLogin,
    UserUpdate,
)


class AuthService:
    """Service for managing users and API keys."""

    def __init__(self, db_pool: Pool):
        """Initialize with database connection pool.

        Args:
            db_pool: asyncpg connection pool
        """
        self.db_pool = db_pool

    @staticmethod
    def _parse_api_key_row(row: dict) -> dict:
        """Parse API key row from database, handling JSONB scopes.

        Args:
            row: Row dict from database

        Returns:
            Parsed row dict with scopes as list
        """
        row_dict = dict(row)
        if isinstance(row_dict.get('scopes'), str):
            row_dict['scopes'] = json.loads(row_dict['scopes'])
        return row_dict

    @staticmethod
    def hash_password(password: str) -> str:
        """Hash a password using bcrypt.

        Args:
            password: Plain text password

        Returns:
            Bcrypt hash of password
        """
        return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

    @staticmethod
    def verify_password(password: str, hashed: str) -> bool:
        """Verify password against bcrypt hash.

        Args:
            password: Plain text password
            hashed: Bcrypt hash to compare against

        Returns:
            True if password matches hash
        """
        return bcrypt.checkpw(password.encode(), hashed.encode())

    @staticmethod
    def generate_api_key() -> tuple[str, str]:
        """Generate a new API key.

        Returns:
            Tuple of (full_key, key_hash) where:
            - full_key is shown once to user (e.g., 'hc_live_abc123...')
            - key_hash is bcrypt hash stored in database
        """
        random_part = secrets.token_urlsafe(32)
        full_key = f"hc_live_{random_part}"
        key_hash = bcrypt.hashpw(full_key.encode(), bcrypt.gensalt()).decode()
        return full_key, key_hash

    async def create_user(self, user_data: UserCreate) -> User:
        """Create a new user.

        Args:
            user_data: User creation data

        Returns:
            Created user

        Raises:
            ValueError: If email already exists
        """
        hashed_password = self.hash_password(user_data.password)

        async with self.db_pool.acquire() as conn:
            try:
                row = await conn.fetchrow(
                    """
                    INSERT INTO users (email, hashed_password, full_name, tier, is_admin)
                    VALUES ($1, $2, $3, $4, $5)
                    RETURNING *
                    """,
                    user_data.email,
                    hashed_password,
                    user_data.full_name,
                    user_data.tier.value,
                    user_data.is_admin,
                )
                return User(**dict(row))
            except Exception as e:
                if "unique constraint" in str(e).lower():
                    raise ValueError(f"User with email {user_data.email} already exists")
                raise

    async def authenticate_user(self, login_data: UserLogin) -> Optional[User]:
        """Authenticate user with email and password.

        Args:
            login_data: User login credentials

        Returns:
            User if authentication successful, None otherwise
        """
        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM users WHERE email = $1 AND is_active = true",
                login_data.email,
            )

            if not row:
                return None

            if not self.verify_password(login_data.password, row["hashed_password"]):
                return None

            # Update last login time
            await conn.execute(
                "UPDATE users SET last_login_at = NOW() WHERE id = $1", row["id"]
            )

            return User(**dict(row))

    async def get_user(self, user_id: UUID) -> Optional[User]:
        """Get user by ID.

        Args:
            user_id: User ID

        Returns:
            User if found, None otherwise
        """
        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM users WHERE id = $1", user_id)
            return User(**dict(row)) if row else None

    async def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email.

        Args:
            email: User email

        Returns:
            User if found, None otherwise
        """
        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM users WHERE email = $1", email)
            return User(**dict(row)) if row else None

    async def update_user(self, user_id: UUID, user_data: UserUpdate) -> Optional[User]:
        """Update user information.

        Args:
            user_id: User ID
            user_data: Updated user data

        Returns:
            Updated user if found, None otherwise
        """
        updates = []
        values = []
        param_idx = 1

        if user_data.full_name is not None:
            updates.append(f"full_name = ${param_idx}")
            values.append(user_data.full_name)
            param_idx += 1

        if user_data.tier is not None:
            updates.append(f"tier = ${param_idx}")
            values.append(user_data.tier.value)
            param_idx += 1

        if user_data.is_active is not None:
            updates.append(f"is_active = ${param_idx}")
            values.append(user_data.is_active)
            param_idx += 1

        if user_data.is_admin is not None:
            updates.append(f"is_admin = ${param_idx}")
            values.append(user_data.is_admin)
            param_idx += 1

        if not updates:
            return await self.get_user(user_id)

        values.append(user_id)
        query = f"UPDATE users SET {', '.join(updates)} WHERE id = ${param_idx} RETURNING *"

        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow(query, *values)
            return User(**dict(row)) if row else None

    async def delete_user(self, user_id: UUID) -> bool:
        """Delete a user (and cascade to API keys).

        Args:
            user_id: User ID

        Returns:
            True if user was deleted, False if not found
        """
        async with self.db_pool.acquire() as conn:
            result = await conn.execute("DELETE FROM users WHERE id = $1", user_id)
            return result.endswith("1")

    async def list_users(
        self, limit: int = 100, offset: int = 0
    ) -> list[User]:
        """List all users with pagination.

        Args:
            limit: Maximum number of users to return
            offset: Number of users to skip

        Returns:
            List of users
        """
        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM users ORDER BY created_at DESC LIMIT $1 OFFSET $2",
                limit,
                offset,
            )
            return [User(**dict(row)) for row in rows]

    async def create_api_key(
        self, user_id: UUID, key_data: APIKeyCreate
    ) -> APIKeyResponse:
        """Create a new API key for a user.

        Args:
            user_id: User ID
            key_data: API key creation data

        Returns:
            API key response with secret key (shown only once)
        """
        full_key, key_hash = self.generate_api_key()
        key_prefix = full_key[:15]  # 'hc_live_' + first few chars

        expires_at = None
        if key_data.expires_in_days:
            expires_at = datetime.utcnow() + timedelta(days=key_data.expires_in_days)

        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO api_keys (
                    user_id, key_prefix, key_hash, name, scopes,
                    rate_limit_tier, expires_at
                )
                VALUES ($1, $2, $3, $4, $5::jsonb, $6, $7)
                RETURNING *
                """,
                user_id,
                key_prefix,
                key_hash,
                key_data.name,
                json.dumps(key_data.scopes),
                key_data.rate_limit_tier,
                expires_at,
            )

            # Convert row to dict and parse JSONB scopes field
            row_dict = self._parse_api_key_row(row)
            api_key = APIKey(**row_dict)
            return APIKeyResponse(api_key=api_key, secret_key=full_key)

    async def validate_api_key(self, api_key: str) -> Optional[dict]:
        """Validate an API key and return associated data.

        Args:
            api_key: Full API key to validate

        Returns:
            Dict with api_key_id, user_id, tier if valid, None otherwise
        """
        async with self.db_pool.acquire() as conn:
            # Get all active API keys
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

            # Check hash against submitted key
            for row in rows:
                if bcrypt.checkpw(api_key.encode(), row["key_hash"].encode()):
                    # Update last_used_at
                    await conn.execute(
                        "UPDATE api_keys SET last_used_at = NOW() WHERE id = $1",
                        row["id"],
                    )

                    return {
                        "api_key_id": row["id"],
                        "user_id": row["user_id"],
                        "tier": row["rate_limit_tier"],
                        "scopes": row["scopes"],
                    }

            return None

    async def get_api_key(self, key_id: UUID) -> Optional[APIKey]:
        """Get API key by ID.

        Args:
            key_id: API key ID

        Returns:
            API key if found, None otherwise
        """
        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM api_keys WHERE id = $1", key_id)
            return APIKey(**self._parse_api_key_row(row)) if row else None

    async def list_user_api_keys(self, user_id: UUID) -> list[APIKey]:
        """List all API keys for a user.

        Args:
            user_id: User ID

        Returns:
            List of API keys
        """
        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM api_keys WHERE user_id = $1 ORDER BY created_at DESC",
                user_id,
            )
            return [APIKey(**self._parse_api_key_row(row)) for row in rows]

    async def revoke_api_key(self, key_id: UUID) -> bool:
        """Revoke (deactivate) an API key.

        Args:
            key_id: API key ID

        Returns:
            True if key was revoked, False if not found
        """
        async with self.db_pool.acquire() as conn:
            result = await conn.execute(
                "UPDATE api_keys SET is_active = false WHERE id = $1", key_id
            )
            return result.endswith("1")

    async def delete_api_key(self, key_id: UUID) -> bool:
        """Delete an API key permanently.

        Args:
            key_id: API key ID

        Returns:
            True if key was deleted, False if not found
        """
        async with self.db_pool.acquire() as conn:
            result = await conn.execute("DELETE FROM api_keys WHERE id = $1", key_id)
            return result.endswith("1")

    async def log_api_usage(
        self,
        api_key_id: UUID,
        endpoint: str,
        method: str,
        status_code: int,
        tokens_used: int = 0,
        response_time_ms: float = 0.0,
    ) -> None:
        """Log API key usage.

        Args:
            api_key_id: API key ID
            endpoint: API endpoint called
            method: HTTP method
            status_code: Response status code
            tokens_used: Number of tokens used
            response_time_ms: Response time in milliseconds
        """
        async with self.db_pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO api_key_usage (
                    api_key_id, endpoint, method, status_code,
                    tokens_used, response_time_ms
                )
                VALUES ($1, $2, $3, $4, $5, $6)
                """,
                api_key_id,
                endpoint,
                method,
                status_code,
                tokens_used,
                response_time_ms,
            )
