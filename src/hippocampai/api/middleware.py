"""FastAPI middleware for authentication and rate limiting."""

import time
from typing import Callable, Optional

from fastapi import HTTPException, Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from hippocampai.auth.auth_service import AuthService
from hippocampai.auth.rate_limiter import RateLimiter


class AuthMiddleware(BaseHTTPMiddleware):
    """Authentication middleware for FastAPI.

    Handles:
    - Local mode bypass (X-User-Auth: false)
    - API key validation
    - Rate limiting
    - Request context injection
    """

    # Public endpoints that don't require authentication
    PUBLIC_PATHS = {
        "/health",
        "/docs",
        "/redoc",
        "/openapi.json",
    }

    # Admin endpoints that require admin role
    ADMIN_PATHS_PREFIX = [
        "/admin",
        "/api/admin",
    ]

    def __init__(
        self,
        app,
        auth_service: Optional[AuthService] = None,
        rate_limiter: Optional[RateLimiter] = None,
        user_auth_enabled: bool = True,
    ):
        """Initialize middleware.

        Args:
            app: FastAPI app
            auth_service: Authentication service (optional, will use app.state if not provided)
            rate_limiter: Rate limiter service (optional, will use app.state if not provided)
            user_auth_enabled: Whether to enforce authentication (False for local mode)
        """
        super().__init__(app)
        self._auth_service = auth_service
        self._rate_limiter = rate_limiter
        self._user_auth_enabled = user_auth_enabled

    def _get_auth_service(self, request: Request) -> Optional[AuthService]:
        """Get auth service from instance or app state."""
        if self._auth_service:
            return self._auth_service
        return getattr(request.app.state, "auth_service", None)

    def _get_rate_limiter(self, request: Request) -> Optional[RateLimiter]:
        """Get rate limiter from instance or app state."""
        if self._rate_limiter:
            return self._rate_limiter
        return getattr(request.app.state, "rate_limiter", None)

    def _get_user_auth_enabled(self, request: Request) -> bool:
        """Get user auth setting from instance or app state."""
        if hasattr(request.app.state, "user_auth_enabled"):
            return request.app.state.user_auth_enabled
        return self._user_auth_enabled

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request through authentication and rate limiting.

        Args:
            request: FastAPI request
            call_next: Next middleware/handler

        Returns:
            Response from handler or error response
        """
        # Skip authentication for public paths
        if request.url.path in self.PUBLIC_PATHS:
            return await call_next(request)

        # Get services from app state (lazy loading)
        auth_service = self._get_auth_service(request)
        rate_limiter = self._get_rate_limiter(request)
        user_auth_enabled = self._get_user_auth_enabled(request)

        # If auth service not available, skip auth
        if not auth_service:
            return await call_next(request)

        # Start timing
        start_time = time.time()

        # Check if user auth is enabled via header (overrides global setting)
        x_user_auth = request.headers.get("X-User-Auth", "").lower()
        user_auth_required = user_auth_enabled and x_user_auth != "false"

        # Local mode bypass
        if not user_auth_required:
            # Inject admin-level permissions for local mode
            request.state.user_id = None
            request.state.api_key_id = None
            request.state.tier = "admin"
            request.state.is_local_mode = True

            response = await call_next(request)
            return response

        # Extract Authorization header
        authorization = request.headers.get("Authorization")
        if not authorization:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={
                    "error": "Authentication required",
                    "message": "Missing Authorization header. Provide 'Authorization: Bearer <api_key>'",
                },
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Extract API key
        if not authorization.startswith("Bearer "):
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={
                    "error": "Invalid authorization format",
                    "message": "Authorization header must be 'Bearer <api_key>'",
                },
            )

        api_key = authorization.replace("Bearer ", "").strip()

        # Validate API key
        key_data = await auth_service.validate_api_key(api_key)
        if not key_data:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={
                    "error": "Invalid API key",
                    "message": "The provided API key is invalid, expired, or has been revoked",
                },
            )

        # Inject authentication context
        request.state.user_id = key_data["user_id"]
        request.state.api_key_id = key_data["api_key_id"]
        request.state.tier = key_data["tier"]
        request.state.scopes = key_data["scopes"]
        request.state.is_local_mode = False

        # Check if admin endpoint
        is_admin_endpoint = any(
            request.url.path.startswith(prefix) for prefix in self.ADMIN_PATHS_PREFIX
        )

        if is_admin_endpoint:
            # Verify admin access
            user = await auth_service.get_user(key_data["user_id"])
            if not user or not user.is_admin:
                return JSONResponse(
                    status_code=status.HTTP_403_FORBIDDEN,
                    content={
                        "error": "Admin access required",
                        "message": "This endpoint requires administrator privileges",
                    },
                )

        # Check rate limits (skip for admin tier)
        if key_data["tier"] != "admin" and rate_limiter:
            allowed, rate_info = await rate_limiter.check_rate_limit(
                key_data["api_key_id"], key_data["tier"]
            )

            if not allowed:
                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={
                        "error": "Rate limit exceeded",
                        "message": f"Rate limit of {rate_info.limit} requests per {rate_info.window} exceeded",
                        "limit": rate_info.limit,
                        "remaining": rate_info.remaining,
                        "reset_at": rate_info.reset_at,
                        "window": rate_info.window,
                    },
                    headers={
                        "X-RateLimit-Limit": str(rate_info.limit),
                        "X-RateLimit-Remaining": str(rate_info.remaining),
                        "X-RateLimit-Reset": str(rate_info.reset_at),
                        "Retry-After": str(
                            max(1, rate_info.reset_at - int(time.time()))
                        ),
                    },
                )

            # Add rate limit headers to request state for response
            request.state.rate_limit_info = rate_info

        # Call next handler
        try:
            response = await call_next(request)

            # Add rate limit headers to response
            if hasattr(request.state, "rate_limit_info"):
                rate_info = request.state.rate_limit_info
                response.headers["X-RateLimit-Limit"] = str(rate_info.limit)
                response.headers["X-RateLimit-Remaining"] = str(rate_info.remaining)
                response.headers["X-RateLimit-Reset"] = str(rate_info.reset_at)

            # Log API usage
            response_time_ms = (time.time() - start_time) * 1000
            if hasattr(request.state, "api_key_id") and auth_service:
                await auth_service.log_api_usage(
                    api_key_id=request.state.api_key_id,
                    endpoint=request.url.path,
                    method=request.method,
                    status_code=response.status_code,
                    tokens_used=0,  # TODO: Extract from response if available
                    response_time_ms=response_time_ms,
                )

            return response

        except HTTPException as e:
            # Log failed requests too
            if hasattr(request.state, "api_key_id") and auth_service:
                await auth_service.log_api_usage(
                    api_key_id=request.state.api_key_id,
                    endpoint=request.url.path,
                    method=request.method,
                    status_code=e.status_code,
                    tokens_used=0,
                    response_time_ms=(time.time() - start_time) * 1000,
                )
            raise


async def require_admin(request: Request) -> None:
    """Dependency to require admin access.

    Args:
        request: FastAPI request with state

    Raises:
        HTTPException: If user is not admin
    """
    # Check for X-User-Auth header bypass (when middleware not installed)
    x_user_auth = request.headers.get("X-User-Auth", "").lower()
    if x_user_auth == "false":
        # Set local mode state for consistency
        request.state.is_local_mode = True
        request.state.tier = "admin"
        return

    # Local mode always has admin access (when middleware is installed)
    if hasattr(request.state, "is_local_mode") and request.state.is_local_mode:
        return

    # Check if user_id is present (authenticated)
    if not hasattr(request.state, "user_id") or request.state.user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )

    # This should be verified by middleware, but double-check
    # The middleware already checks admin status, but we can add extra validation here if needed
    pass


async def require_scope(request: Request, required_scope: str) -> None:
    """Dependency to require specific scope.

    Args:
        request: FastAPI request with state
        required_scope: Required scope (e.g., 'memories:write')

    Raises:
        HTTPException: If user doesn't have required scope
    """
    # Local mode has all scopes
    if hasattr(request.state, "is_local_mode") and request.state.is_local_mode:
        return

    # Check authentication
    if not hasattr(request.state, "scopes"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )

    # Check scope
    if required_scope not in request.state.scopes:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Missing required scope: {required_scope}",
        )
