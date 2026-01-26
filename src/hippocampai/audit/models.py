"""Audit logging models for compliance tracking."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class AuditAction(str, Enum):
    """Types of auditable actions."""

    # Authentication
    LOGIN = "login"
    LOGOUT = "logout"
    LOGIN_FAILED = "login_failed"
    PASSWORD_CHANGE = "password_change"

    # API Keys
    API_KEY_CREATE = "api_key_create"
    API_KEY_REVOKE = "api_key_revoke"
    API_KEY_ROTATE = "api_key_rotate"
    API_KEY_DELETE = "api_key_delete"

    # User Management
    USER_CREATE = "user_create"
    USER_UPDATE = "user_update"
    USER_DELETE = "user_delete"
    USER_TIER_CHANGE = "user_tier_change"

    # Memory Operations
    MEMORY_CREATE = "memory_create"
    MEMORY_UPDATE = "memory_update"
    MEMORY_DELETE = "memory_delete"
    MEMORY_BULK_DELETE = "memory_bulk_delete"
    MEMORY_EXPORT = "memory_export"
    MEMORY_IMPORT = "memory_import"

    # Admin Actions
    ADMIN_ACCESS = "admin_access"
    SETTINGS_CHANGE = "settings_change"
    RATE_LIMIT_OVERRIDE = "rate_limit_override"

    # System Events
    SYSTEM_ERROR = "system_error"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    UNAUTHORIZED_ACCESS = "unauthorized_access"


class AuditSeverity(str, Enum):
    """Severity levels for audit events."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AuditEvent(BaseModel):
    """Individual audit event."""

    id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Who
    user_id: Optional[UUID] = None
    user_email: Optional[str] = None
    api_key_id: Optional[UUID] = None
    tenant_id: Optional[str] = None

    # What
    action: AuditAction
    severity: AuditSeverity = AuditSeverity.INFO
    resource_type: Optional[str] = None  # e.g., "memory", "user", "api_key"
    resource_id: Optional[str] = None

    # Where
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    endpoint: Optional[str] = None
    method: Optional[str] = None

    # Details
    description: str
    old_value: Optional[dict[str, Any]] = None
    new_value: Optional[dict[str, Any]] = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Status
    success: bool = True
    error_message: Optional[str] = None


class AuditLog(BaseModel):
    """Collection of audit events with metadata."""

    events: list[AuditEvent]
    total_count: int
    page: int = 1
    page_size: int = 100
    has_more: bool = False


class AuditQuery(BaseModel):
    """Query parameters for searching audit logs."""

    user_id: Optional[UUID] = None
    tenant_id: Optional[str] = None
    action: Optional[AuditAction] = None
    actions: Optional[list[AuditAction]] = None
    severity: Optional[AuditSeverity] = None
    min_severity: Optional[AuditSeverity] = None
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None
    ip_address: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    success: Optional[bool] = None
    search_text: Optional[str] = None
    page: int = 1
    page_size: int = 100


class AuditStats(BaseModel):
    """Statistics about audit logs."""

    total_events: int = 0
    events_by_action: dict[str, int] = Field(default_factory=dict)
    events_by_severity: dict[str, int] = Field(default_factory=dict)
    events_by_user: dict[str, int] = Field(default_factory=dict)
    failed_events: int = 0
    success_rate: float = 100.0
    time_range_start: Optional[datetime] = None
    time_range_end: Optional[datetime] = None
