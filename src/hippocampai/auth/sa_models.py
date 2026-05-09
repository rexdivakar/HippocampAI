"""SQLAlchemy declarative models for HippocampAI auth and audit tables.

These models mirror the SQL schemas defined in:
  - src/hippocampai/auth/schema.sql (PostgreSQL)
  - src/hippocampai/auth/schema_sqlite.sql (SQLite)
  - src/hippocampai/audit/schema.sql (audit_logs, audit_retention_policies)

Type notes:
  - UUIDs are stored as UUID on Postgres and as Text on SQLite via with_variant.
  - JSONB / JSON are stored as JSON (SQLAlchemy's JSON type, maps to JSONB on Postgres).
  - INET (Postgres-only) is stored as String(45) for cross-dialect compatibility.
  - Timestamps use DateTime(timezone=True) for both dialects.
  - SQLite boolean columns are INTEGER 0/1 on-disk but Python bool at the ORM layer.

The autogenerate target_metadata is exposed as `metadata` at module level.
"""

import uuid
from datetime import datetime, date as date_type
from typing import Any, Optional, Union

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    MetaData,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.engine.interfaces import Dialect
from sqlalchemy.orm import DeclarativeBase, relationship
from sqlalchemy.types import TypeDecorator, CHAR, TypeEngine


# ---------------------------------------------------------------------------
# Cross-dialect UUID type
# ---------------------------------------------------------------------------


class UUIDType(TypeDecorator[uuid.UUID]):
    """Platform-independent UUID type.

    Stores as PostgreSQL native UUID on Postgres, and as CHAR(36) on SQLite.
    Accepts ``uuid.UUID`` objects and string representations.
    """

    impl = CHAR
    cache_ok = True

    def load_dialect_impl(self, dialect: Dialect) -> TypeEngine[Any]:
        if dialect.name == "postgresql":
            return dialect.type_descriptor(PG_UUID(as_uuid=True))
        return dialect.type_descriptor(CHAR(36))

    def process_bind_param(
        self, value: Optional[Union[uuid.UUID, str]], dialect: Dialect
    ) -> Optional[Union[uuid.UUID, str]]:
        if value is None:
            return None
        if dialect.name == "postgresql":
            return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))
        return str(value) if isinstance(value, uuid.UUID) else str(value)

    def process_result_value(
        self, value: Optional[Union[uuid.UUID, str]], dialect: Dialect
    ) -> Optional[uuid.UUID]:
        if value is None:
            return None
        return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))


# ---------------------------------------------------------------------------
# Shared metadata — imported by env.py for autogenerate support
# ---------------------------------------------------------------------------

metadata = MetaData()


class Base(DeclarativeBase):
    metadata = metadata


# ---------------------------------------------------------------------------
# Auth tables
# ---------------------------------------------------------------------------


class Organization(Base):
    """Multi-tenant organizations for enterprise accounts."""

    __tablename__ = "organizations"

    id: uuid.UUID = Column(UUIDType, primary_key=True, default=uuid.uuid4)
    name: str = Column(String(255), nullable=False)
    tier: str = Column(String(50), default="enterprise")
    max_users: int = Column(Integer, default=10)
    max_api_keys: int = Column(Integer, default=50)
    created_at: datetime = Column(DateTime(timezone=True), nullable=False)
    updated_at: datetime = Column(DateTime(timezone=True), nullable=False)

    users: list = relationship("User", back_populates="organization")


class User(Base):
    """Application users — both human admins and API consumers."""

    __tablename__ = "users"
    __table_args__ = (
        CheckConstraint("tier IN ('free', 'pro', 'enterprise', 'admin')", name="ck_users_tier"),
    )

    id: uuid.UUID = Column(UUIDType, primary_key=True, default=uuid.uuid4)
    email: str = Column(String(255), unique=True, nullable=False)
    hashed_password: str = Column(String(255), nullable=False)
    full_name: Optional[str] = Column(String(255))
    organization_id: Optional[uuid.UUID] = Column(
        UUIDType,
        ForeignKey("organizations.id", ondelete="SET NULL"),
        nullable=True,
    )
    tier: str = Column(String(50), default="free")
    is_active: bool = Column(Boolean, default=True)
    is_admin: bool = Column(Boolean, default=False)
    email_verified: bool = Column(Boolean, default=False)
    created_at: datetime = Column(DateTime(timezone=True), nullable=False)
    updated_at: datetime = Column(DateTime(timezone=True), nullable=False)
    last_login_at: Optional[datetime] = Column(DateTime(timezone=True))

    organization: Optional[Organization] = relationship(
        "Organization", back_populates="users"
    )
    api_keys: list = relationship("APIKey", back_populates="user")
    sessions: list = relationship("Session", back_populates="user")
    audit_entries: list = relationship("AuditLog", back_populates="user")


Index("idx_users_email", User.email)
Index("idx_users_tier", User.tier)
Index("idx_users_is_admin", User.is_admin)


class APIKey(Base):
    """API keys issued to users for programmatic access."""

    __tablename__ = "api_keys"

    id: uuid.UUID = Column(UUIDType, primary_key=True, default=uuid.uuid4)
    user_id: uuid.UUID = Column(
        UUIDType, ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    key_prefix: str = Column(String(20), nullable=False)
    key_hash: str = Column(String(255), nullable=False)
    name: Optional[str] = Column(String(255))
    # JSON maps to JSONB on Postgres; JSON text on SQLite.
    scopes: list = Column(JSON, default=lambda: ["memories:read", "memories:write"])
    rate_limit_tier: str = Column(String(50), default="free")
    last_used_at: Optional[datetime] = Column(DateTime(timezone=True))
    expires_at: Optional[datetime] = Column(DateTime(timezone=True))
    is_active: bool = Column(Boolean, default=True)
    created_at: datetime = Column(DateTime(timezone=True), nullable=False)

    user: User = relationship("User", back_populates="api_keys")
    usage: list = relationship("APIKeyUsage", back_populates="api_key")
    rate_limit_buckets: list = relationship("RateLimitBucket", back_populates="api_key")


Index("idx_api_keys_user_id", APIKey.user_id)
Index("idx_api_keys_key_prefix", APIKey.key_prefix)
Index("idx_api_keys_is_active", APIKey.is_active)


class APIKeyUsage(Base):
    """Per-request usage records for API keys.

    On Postgres the underlying table is partitioned by ``date`` (RANGE). The
    model omits partitioning — it is applied as raw DDL in the migration.
    The composite PK ``(id, date)`` is required by Postgres partitioning.
    """

    __tablename__ = "api_key_usage"

    id: uuid.UUID = Column(UUIDType, primary_key=True, default=uuid.uuid4)
    api_key_id: uuid.UUID = Column(
        UUIDType, ForeignKey("api_keys.id", ondelete="CASCADE"), nullable=False
    )
    endpoint: str = Column(String(255), nullable=False)
    method: str = Column(String(10), nullable=False)
    status_code: Optional[int] = Column(Integer)
    request_count: int = Column(Integer, default=1)
    tokens_used: int = Column(Integer, default=0)
    response_time_ms: Optional[float] = Column(Float)
    date: date_type = Column(Date, nullable=False)
    created_at: datetime = Column(DateTime(timezone=True), nullable=False)

    api_key: APIKey = relationship("APIKey", back_populates="usage")


Index("idx_api_key_usage_key_id", APIKeyUsage.api_key_id)
Index("idx_api_key_usage_date", APIKeyUsage.date)
Index("idx_api_key_usage_endpoint", APIKeyUsage.endpoint)


class RateLimitBucket(Base):
    """Token-bucket state for per-key rate limiting."""

    __tablename__ = "rate_limit_buckets"
    __table_args__ = (UniqueConstraint("api_key_id", "time_window", name="uq_rate_limit_bucket"),)

    id: uuid.UUID = Column(UUIDType, primary_key=True, default=uuid.uuid4)
    api_key_id: uuid.UUID = Column(
        UUIDType, ForeignKey("api_keys.id", ondelete="CASCADE"), nullable=False
    )
    time_window: str = Column(String(20), nullable=False)
    tokens_remaining: int = Column(Integer, nullable=False)
    tokens_capacity: int = Column(Integer, nullable=False)
    last_refill_at: datetime = Column(DateTime(timezone=True), nullable=False)
    updated_at: datetime = Column(DateTime(timezone=True), nullable=False)

    api_key: APIKey = relationship("APIKey", back_populates="rate_limit_buckets")


class Session(Base):
    """Admin UI login sessions."""

    __tablename__ = "sessions"

    id: uuid.UUID = Column(UUIDType, primary_key=True, default=uuid.uuid4)
    user_id: uuid.UUID = Column(
        UUIDType, ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    session_token: str = Column(String(255), unique=True, nullable=False)
    # INET on Postgres; String(45) covers IPv6 on both dialects.
    ip_address: Optional[str] = Column(String(45))
    user_agent: Optional[str] = Column(Text)
    expires_at: datetime = Column(DateTime(timezone=True), nullable=False)
    created_at: datetime = Column(DateTime(timezone=True), nullable=False)

    user: User = relationship("User", back_populates="sessions")


Index("idx_sessions_token", Session.session_token)
Index("idx_sessions_user_id", Session.user_id)
Index("idx_sessions_expires_at", Session.expires_at)


class AuditLog(Base):
    """Simple auth-scoped audit log (auth/schema.sql — table: audit_log, singular)."""

    __tablename__ = "audit_log"

    id: uuid.UUID = Column(UUIDType, primary_key=True, default=uuid.uuid4)
    user_id: Optional[uuid.UUID] = Column(
        UUIDType, ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    action: str = Column(String(100), nullable=False)
    resource_type: Optional[str] = Column(String(50))
    resource_id: Optional[uuid.UUID] = Column(UUIDType)
    details: Optional[dict] = Column(JSON)
    ip_address: Optional[str] = Column(String(45))
    created_at: datetime = Column(DateTime(timezone=True), nullable=False)

    user: Optional[User] = relationship("User", back_populates="audit_entries")


Index("idx_audit_log_user_id", AuditLog.user_id)
Index("idx_audit_log_action", AuditLog.action)
Index("idx_audit_log_created_at", AuditLog.created_at)


# ---------------------------------------------------------------------------
# Compliance audit tables (audit/schema.sql)
# ---------------------------------------------------------------------------


class AuditLogs(Base):
    """Full compliance audit trail (audit/schema.sql — table: audit_logs, plural).

    Separate from ``AuditLog`` (auth/schema.sql). The plural table records
    all system events for compliance; the singular table is a lightweight
    auth-action log.
    """

    __tablename__ = "audit_logs"

    id: uuid.UUID = Column(UUIDType, primary_key=True, default=uuid.uuid4)
    timestamp: datetime = Column(DateTime(timezone=True), nullable=False)
    user_id: Optional[uuid.UUID] = Column(
        UUIDType, ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    user_email: Optional[str] = Column(String(255))
    api_key_id: Optional[uuid.UUID] = Column(
        UUIDType, ForeignKey("api_keys.id", ondelete="SET NULL"), nullable=True
    )
    tenant_id: Optional[str] = Column(String(100))
    action: str = Column(String(50), nullable=False)
    severity: str = Column(String(20), nullable=False, default="info")
    resource_type: Optional[str] = Column(String(50))
    resource_id: Optional[str] = Column(String(255))
    ip_address: Optional[str] = Column(String(45))
    user_agent: Optional[str] = Column(Text)
    endpoint: Optional[str] = Column(String(255))
    method: Optional[str] = Column(String(10))
    description: str = Column(Text, nullable=False)
    old_value: Optional[dict] = Column(JSON)
    new_value: Optional[dict] = Column(JSON)
    extra_metadata: Optional[dict] = Column(
        JSON, default=dict, name="metadata"
    )
    success: bool = Column(Boolean, nullable=False, default=True)
    error_message: Optional[str] = Column(Text)


Index("idx_audit_logs_timestamp", AuditLogs.timestamp)
Index("idx_audit_logs_user_id", AuditLogs.user_id)
Index("idx_audit_logs_tenant_id", AuditLogs.tenant_id)
Index("idx_audit_logs_action", AuditLogs.action)
Index("idx_audit_logs_severity", AuditLogs.severity)
Index("idx_audit_logs_success", AuditLogs.success)
Index("idx_audit_logs_tenant_time", AuditLogs.tenant_id, AuditLogs.timestamp)
Index("idx_audit_logs_user_time", AuditLogs.user_id, AuditLogs.timestamp)


class AuditRetentionPolicy(Base):
    """Per-tenant (or global) audit log retention configuration."""

    __tablename__ = "audit_retention_policies"
    __table_args__ = (UniqueConstraint("tenant_id", name="uq_audit_retention_tenant"),)

    id: uuid.UUID = Column(UUIDType, primary_key=True, default=uuid.uuid4)
    # NULL tenant_id means the global default policy.
    tenant_id: Optional[str] = Column(String(100), unique=True, nullable=True)
    debug_retention: str = Column(String(20), nullable=False, default="30_days")
    info_retention: str = Column(String(20), nullable=False, default="90_days")
    warning_retention: str = Column(String(20), nullable=False, default="1_year")
    error_retention: str = Column(String(20), nullable=False, default="3_years")
    critical_retention: str = Column(String(20), nullable=False, default="7_years")
    auth_events_retention: str = Column(String(20), nullable=False, default="3_years")
    admin_events_retention: str = Column(String(20), nullable=False, default="7_years")
    data_export_retention: str = Column(String(20), nullable=False, default="7_years")
    enabled: bool = Column(Boolean, nullable=False, default=True)
    archive_before_delete: bool = Column(Boolean, nullable=False, default=True)
    archive_location: Optional[str] = Column(Text)
    created_at: datetime = Column(DateTime(timezone=True), nullable=False)
    updated_at: datetime = Column(DateTime(timezone=True), nullable=False)
