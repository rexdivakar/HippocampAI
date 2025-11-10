"""Admin API routes for user and API key management."""

from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, status

from hippocampai.api.middleware import require_admin
from hippocampai.auth.auth_service import AuthService
from hippocampai.auth.models import (
    APIKey,
    APIKeyCreate,
    APIKeyResponse,
    APIKeyStatistics,
    User,
    UserCreate,
    UserLogin,
    UserStatistics,
    UserUpdate,
)

# Create router
router = APIRouter(prefix="/admin", tags=["admin"])


def get_auth_service(request: Request) -> AuthService:
    """Dependency to get auth service from app state.

    Args:
        request: FastAPI request

    Returns:
        AuthService instance
    """
    return request.app.state.auth_service


# Authentication Endpoints


@router.post("/login")
async def admin_login(
    credentials: UserLogin,
    request: Request,
    auth_service: AuthService = Depends(get_auth_service),
):
    """Admin login endpoint.

    Args:
        credentials: Email and password
        request: FastAPI request
        auth_service: Authentication service

    Returns:
        User data with session token

    Raises:
        HTTPException: If login fails
    """
    user = await auth_service.authenticate_user(credentials.email, credentials.password)
    if not user or not user.get("is_admin"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials or not an admin user",
        )

    # Create session
    import secrets
    session_token = secrets.token_urlsafe(32)

    # Get client info
    ip_address = request.client.host if request.client else None
    user_agent = request.headers.get("user-agent", "")

    # Store session in database
    async with auth_service.db_pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO sessions (user_id, session_token, ip_address, user_agent, expires_at)
            VALUES ($1, $2, $3, $4, NOW() + INTERVAL '7 days')
            """,
            user["id"],
            session_token,
            ip_address,
            user_agent,
        )

        # Update last login
        await conn.execute(
            """
            UPDATE users
            SET last_login_at = NOW(), last_login_ip = $2
            WHERE id = $1
            """,
            user["id"],
            ip_address,
        )

    return {
        "user": user,
        "session_token": session_token,
        "expires_in": 604800,  # 7 days in seconds
    }


@router.post("/logout")
async def admin_logout(
    request: Request,
    auth_service: AuthService = Depends(get_auth_service),
):
    """Admin logout endpoint.

    Args:
        request: FastAPI request
        auth_service: Authentication service

    Returns:
        Success message
    """
    session_token = request.headers.get("X-Session-Token")
    if session_token:
        async with auth_service.db_pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM sessions WHERE session_token = $1",
                session_token,
            )

    return {"message": "Logged out successfully"}


# User Management Endpoints


@router.get("/users", response_model=List[User], dependencies=[Depends(require_admin)])
async def list_users(
    limit: int = 100,
    offset: int = 0,
    auth_service: AuthService = Depends(get_auth_service),
):
    """List all users with pagination (admin only).

    Args:
        limit: Maximum number of users to return
        offset: Number of users to skip
        auth_service: Auth service dependency

    Returns:
        List of users
    """
    users = await auth_service.list_users(limit=limit, offset=offset)
    return users


@router.post(
    "/users", response_model=User, status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_admin)]
)
async def create_user(
    user_data: UserCreate,
    auth_service: AuthService = Depends(get_auth_service),
):
    """Create a new user (admin only).

    Args:
        user_data: User creation data
        auth_service: Auth service dependency

    Returns:
        Created user

    Raises:
        HTTPException: If email already exists
    """
    try:
        user = await auth_service.create_user(user_data)
        return user
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.get("/users/{user_id}", response_model=User, dependencies=[Depends(require_admin)])
async def get_user(
    user_id: UUID,
    auth_service: AuthService = Depends(get_auth_service),
):
    """Get user by ID (admin only).

    Args:
        user_id: User ID
        auth_service: Auth service dependency

    Returns:
        User

    Raises:
        HTTPException: If user not found
    """
    user = await auth_service.get_user(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User {user_id} not found",
        )
    return user


@router.patch("/users/{user_id}", response_model=User, dependencies=[Depends(require_admin)])
async def update_user(
    user_id: UUID,
    user_data: UserUpdate,
    auth_service: AuthService = Depends(get_auth_service),
):
    """Update user (admin only).

    Args:
        user_id: User ID
        user_data: Updated user data
        auth_service: Auth service dependency

    Returns:
        Updated user

    Raises:
        HTTPException: If user not found
    """
    user = await auth_service.update_user(user_id, user_data)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User {user_id} not found",
        )
    return user


@router.delete(
    "/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[Depends(require_admin)]
)
async def delete_user(
    user_id: UUID,
    auth_service: AuthService = Depends(get_auth_service),
):
    """Delete user (admin only).

    This will cascade delete all API keys.

    Args:
        user_id: User ID
        auth_service: Auth service dependency

    Raises:
        HTTPException: If user not found
    """
    deleted = await auth_service.delete_user(user_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User {user_id} not found",
        )


# API Key Management Endpoints


@router.get(
    "/users/{user_id}/api-keys", response_model=List[APIKey],
    dependencies=[Depends(require_admin)]
)
async def list_user_api_keys(
    user_id: UUID,
    auth_service: AuthService = Depends(get_auth_service),
):
    """List all API keys for a user (admin only).

    Args:
        user_id: User ID
        auth_service: Auth service dependency

    Returns:
        List of API keys
    """
    keys = await auth_service.list_user_api_keys(user_id)
    return keys


@router.post(
    "/users/{user_id}/api-keys",
    response_model=APIKeyResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_admin)],
)
async def create_api_key(
    user_id: UUID,
    key_data: APIKeyCreate,
    auth_service: AuthService = Depends(get_auth_service),
):
    """Create a new API key for a user (admin only).

    The secret_key in the response is shown ONLY ONCE.
    Make sure to save it securely.

    Args:
        user_id: User ID
        key_data: API key creation data
        auth_service: Auth service dependency

    Returns:
        API key response with secret key (shown only once)

    Raises:
        HTTPException: If user not found
    """
    # Verify user exists
    user = await auth_service.get_user(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User {user_id} not found",
        )

    key_response = await auth_service.create_api_key(user_id, key_data)
    return key_response


@router.get("/api-keys/{key_id}", response_model=APIKey, dependencies=[Depends(require_admin)])
async def get_api_key(
    key_id: UUID,
    auth_service: AuthService = Depends(get_auth_service),
):
    """Get API key by ID (admin only).

    Args:
        key_id: API key ID
        auth_service: Auth service dependency

    Returns:
        API key

    Raises:
        HTTPException: If key not found
    """
    key = await auth_service.get_api_key(key_id)
    if not key:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"API key {key_id} not found",
        )
    return key


@router.post(
    "/api-keys/{key_id}/revoke", status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[Depends(require_admin)]
)
async def revoke_api_key(
    key_id: UUID,
    auth_service: AuthService = Depends(get_auth_service),
):
    """Revoke (deactivate) an API key (admin only).

    Args:
        key_id: API key ID
        auth_service: Auth service dependency

    Raises:
        HTTPException: If key not found
    """
    revoked = await auth_service.revoke_api_key(key_id)
    if not revoked:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"API key {key_id} not found",
        )


@router.delete(
    "/api-keys/{key_id}", status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[Depends(require_admin)]
)
async def delete_api_key(
    key_id: UUID,
    auth_service: AuthService = Depends(get_auth_service),
):
    """Delete an API key permanently (admin only).

    Args:
        key_id: API key ID
        auth_service: Auth service dependency

    Raises:
        HTTPException: If key not found
    """
    deleted = await auth_service.delete_api_key(key_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"API key {key_id} not found",
        )


# Statistics Endpoints


@router.get(
    "/statistics/users", response_model=List[UserStatistics],
    dependencies=[Depends(require_admin)]
)
async def get_user_statistics(
    limit: int = 100,
    offset: int = 0,
    request: Request = None,
):
    """Get user statistics (admin only).

    Args:
        limit: Maximum number of entries
        offset: Number of entries to skip
        request: FastAPI request

    Returns:
        List of user statistics
    """
    db_pool = request.app.state.db_pool

    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT * FROM user_statistics
            ORDER BY created_at DESC
            LIMIT $1 OFFSET $2
            """,
            limit,
            offset,
        )
        return [UserStatistics(**dict(row)) for row in rows]


@router.get(
    "/statistics/api-keys", response_model=List[APIKeyStatistics],
    dependencies=[Depends(require_admin)]
)
async def get_api_key_statistics(
    user_id: Optional[UUID] = None,
    limit: int = 100,
    offset: int = 0,
    request: Request = None,
):
    """Get API key statistics (admin only).

    Args:
        user_id: Optional filter by user ID
        limit: Maximum number of entries
        offset: Number of entries to skip
        request: FastAPI request

    Returns:
        List of API key statistics
    """
    db_pool = request.app.state.db_pool

    async with db_pool.acquire() as conn:
        if user_id:
            rows = await conn.fetch(
                """
                SELECT * FROM api_key_statistics
                WHERE user_id = $1
                ORDER BY created_at DESC
                LIMIT $2 OFFSET $3
                """,
                user_id,
                limit,
                offset,
            )
        else:
            rows = await conn.fetch(
                """
                SELECT * FROM api_key_statistics
                ORDER BY created_at DESC
                LIMIT $1 OFFSET $2
                """,
                limit,
                offset,
            )

        return [APIKeyStatistics(**dict(row)) for row in rows]


# Health check endpoint (public)


@router.get("/health")
async def health_check():
    """Health check endpoint (public).

    Returns:
        Health status
    """
    return {"status": "healthy", "service": "hippocampai-admin"}
