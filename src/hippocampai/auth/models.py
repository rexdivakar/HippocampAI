"""Authentication models for users, API keys, and sessions."""

from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, EmailStr, Field


class UserTier(str, Enum):
    """User subscription tiers."""

    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"
    ADMIN = "admin"


class User(BaseModel):
    """User model."""

    id: UUID
    email: EmailStr
    full_name: Optional[str] = None
    organization_id: Optional[UUID] = None
    tier: UserTier = UserTier.FREE
    is_active: bool = True
    is_admin: bool = False
    email_verified: bool = False
    created_at: datetime
    updated_at: datetime
    last_login_at: Optional[datetime] = None
    # Metadata fields
    country: Optional[str] = None
    region: Optional[str] = None
    city: Optional[str] = None
    timezone: Optional[str] = None
    signup_ip: Optional[str] = None
    last_login_ip: Optional[str] = None
    user_agent: Optional[str] = None
    referrer: Optional[str] = None
    metadata: Optional[dict] = None

    class Config:
        from_attributes = True


class UserCreate(BaseModel):
    """Schema for creating a new user."""

    email: EmailStr
    password: str = Field(..., min_length=8)
    full_name: Optional[str] = None
    tier: UserTier = UserTier.FREE
    is_admin: bool = False


class UserUpdate(BaseModel):
    """Schema for updating a user."""

    full_name: Optional[str] = None
    tier: Optional[UserTier] = None
    is_active: Optional[bool] = None
    is_admin: Optional[bool] = None


class UserLogin(BaseModel):
    """Schema for user login."""

    email: EmailStr
    password: str


class APIKey(BaseModel):
    """API Key model."""

    id: UUID
    user_id: UUID
    key_prefix: str
    name: Optional[str] = None
    scopes: list[str] = Field(default_factory=lambda: ["memories:read", "memories:write"])
    rate_limit_tier: str = "free"
    last_used_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    is_active: bool = True
    created_at: datetime

    class Config:
        from_attributes = True


class APIKeyCreate(BaseModel):
    """Schema for creating an API key."""

    name: Optional[str] = None
    scopes: list[str] = Field(default_factory=lambda: ["memories:read", "memories:write"])
    rate_limit_tier: str = "free"
    expires_in_days: Optional[int] = None  # None = no expiration


class APIKeyResponse(BaseModel):
    """Response when creating an API key (includes the actual key)."""

    api_key: APIKey
    secret_key: str  # Only shown once!


class Session(BaseModel):
    """User session model."""

    id: UUID
    user_id: UUID
    session_token: str
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    expires_at: datetime
    created_at: datetime

    class Config:
        from_attributes = True


class AuditLog(BaseModel):
    """Audit log entry."""

    id: UUID
    user_id: Optional[UUID] = None
    action: str
    resource_type: Optional[str] = None
    resource_id: Optional[UUID] = None
    details: Optional[dict] = None
    ip_address: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


class UserStatistics(BaseModel):
    """User usage statistics."""

    id: UUID
    email: EmailStr
    tier: UserTier
    is_active: bool
    api_key_count: int = 0
    total_requests: int = 0
    total_tokens_used: Optional[int] = 0
    last_api_usage: Optional[datetime] = None
    created_at: datetime


class APIKeyStatistics(BaseModel):
    """API key usage statistics."""

    id: UUID
    user_id: UUID
    name: Optional[str]
    key_prefix: str
    rate_limit_tier: str
    is_active: bool
    total_requests: int = 0
    total_tokens_used: Optional[int] = 0
    avg_response_time: Optional[float] = None
    last_request_at: Optional[datetime] = None
    created_at: datetime


class RateLimitInfo(BaseModel):
    """Rate limit information."""

    limit: int
    remaining: int
    reset_at: int  # Unix timestamp
    window: str  # 'minute', 'hour', 'day'
