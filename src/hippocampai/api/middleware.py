"""FastAPI middleware for authentication and rate limiting."""

import time
from typing import Any, Callable, Optional, cast
from uuid import UUID

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
        app: Any,
        auth_service: Optional[AuthService] = None,
        rate_limiter: Optional[RateLimiter] = None,
        user_auth_enabled: bool = True,
    ) -> None:
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
            return cast(bool, request.app.state.user_auth_enabled)
        return self._user_auth_enabled

    def _extract_api_key(self, request: Request) -> tuple[Optional[str], Optional[JSONResponse]]:
        """Extract API key from Authorization header. Returns (api_key, error_response)."""
        authorization = request.headers.get("Authorization")
        if not authorization:
            return None, JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"error": "Authentication required",
                         "message": "Missing Authorization header. Provide 'Authorization: Bearer <api_key>'"},
                headers={"WWW-Authenticate": "Bearer"},
            )
        if not authorization.startswith("Bearer "):
            return None, JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"error": "Invalid authorization format",
                         "message": "Authorization header must be 'Bearer <api_key>'"},
            )
        return authorization.replace("Bearer ", "").strip(), None

    async def _check_admin_access(
        self, request: Request, auth_service: AuthService, user_id: str
    ) -> Optional[JSONResponse]:
        """Check admin access for admin endpoints. Returns error response if denied."""
        is_admin_endpoint = any(
            request.url.path.startswith(prefix) for prefix in self.ADMIN_PATHS_PREFIX
        )
        if not is_admin_endpoint:
            return None
        user = await auth_service.get_user(UUID(user_id))
        if not user or not user.is_admin:
            return JSONResponse(
                status_code=status.HTTP_403_FORBIDDEN,
                content={"error": "Admin access required",
                         "message": "This endpoint requires administrator privileges"},
            )
        return None

    async def _check_rate_limit(
        self, request: Request, rate_limiter: Optional[RateLimiter], key_data: dict
    ) -> Optional[JSONResponse]:
        """Check rate limits. Returns error response if exceeded."""
        if key_data["tier"] == "admin" or not rate_limiter:
            return None
        allowed, rate_info = await rate_limiter.check_rate_limit(key_data["api_key_id"], key_data["tier"])
        if allowed:
            request.state.rate_limit_info = rate_info
            return None
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={
                "error": "Rate limit exceeded",
                "message": f"Rate limit of {rate_info.limit} requests per {rate_info.window} exceeded",
                "limit": rate_info.limit, "remaining": rate_info.remaining,
                "reset_at": rate_info.reset_at, "window": rate_info.window,
            },
            headers={
                "X-RateLimit-Limit": str(rate_info.limit),
                "X-RateLimit-Remaining": str(rate_info.remaining),
                "X-RateLimit-Reset": str(rate_info.reset_at),
                "Retry-After": str(max(1, rate_info.reset_at - int(time.time()))),
            },
        )

    async def _log_api_usage(
        self, request: Request, auth_service: Optional[AuthService],
        status_code: int, start_time: float, response: Optional[Response] = None
    ) -> None:
        """Log API usage."""
        if not hasattr(request.state, "api_key_id") or not auth_service:
            return
        tokens_used = getattr(request.state, "tokens_used", 0)
        if response and "X-Tokens-Used" in response.headers:
            try:
                tokens_used = int(response.headers["X-Tokens-Used"])
            except (ValueError, TypeError):
                pass
        await auth_service.log_api_usage(
            api_key_id=request.state.api_key_id, endpoint=request.url.path,
            method=request.method, status_code=status_code,
            tokens_used=tokens_used, response_time_ms=(time.time() - start_time) * 1000,
        )

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request through authentication and rate limiting."""
        # Skip authentication for public paths
        if request.url.path in self.PUBLIC_PATHS:
            return cast(Response, await call_next(request))

        auth_service = self._get_auth_service(request)
        rate_limiter = self._get_rate_limiter(request)
        user_auth_enabled = self._get_user_auth_enabled(request)

        if not auth_service:
            return cast(Response, await call_next(request))

        start_time = time.time()
        x_user_auth = request.headers.get("X-User-Auth", "").lower()
        user_auth_required = user_auth_enabled and x_user_auth != "false"

        # Local mode bypass
        if not user_auth_required:
            request.state.user_id = None
            request.state.api_key_id = None
            request.state.tier = "admin"
            request.state.is_local_mode = True
            return cast(Response, await call_next(request))

        # Extract and validate API key
        api_key, error_response = self._extract_api_key(request)
        if error_response or api_key is None:
            return error_response or JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"error": "Authentication required",
                         "message": "API key is required"},
            )

        key_data = await auth_service.validate_api_key(api_key)
        if not key_data:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"error": "Invalid API key",
                         "message": "The provided API key is invalid, expired, or has been revoked"},
            )

        # Inject authentication context
        request.state.user_id = key_data["user_id"]
        request.state.api_key_id = key_data["api_key_id"]
        request.state.tier = key_data["tier"]
        request.state.scopes = key_data["scopes"]
        request.state.is_local_mode = False

        # Check admin access
        admin_error = await self._check_admin_access(request, auth_service, key_data["user_id"])
        if admin_error:
            return admin_error

        # Check rate limits
        rate_error = await self._check_rate_limit(request, rate_limiter, key_data)
        if rate_error:
            return rate_error

        # Call next handler
        try:
            response = await call_next(request)

            # Add rate limit headers to response
            if hasattr(request.state, "rate_limit_info"):
                rate_info = request.state.rate_limit_info
                response.headers["X-RateLimit-Limit"] = str(rate_info.limit)
                response.headers["X-RateLimit-Remaining"] = str(rate_info.remaining)
                response.headers["X-RateLimit-Reset"] = str(rate_info.reset_at)

            await self._log_api_usage(request, auth_service, response.status_code, start_time, response)
            return cast(Response, response)

        except HTTPException as e:
            await self._log_api_usage(request, auth_service, e.status_code, start_time)
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
