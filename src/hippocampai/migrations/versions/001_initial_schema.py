"""Initial schema — auth and audit tables.

Revision ID: a1b2c3d4e5f6
Revises:
Create Date: 2026-05-08

Captures the full baseline schema from:
  - src/hippocampai/auth/schema.sql       (PostgreSQL)
  - src/hippocampai/auth/schema_sqlite.sql (SQLite)
  - src/hippocampai/audit/schema.sql       (compliance audit)

On PostgreSQL, the api_key_usage table is partitioned by RANGE(date) with
annual child partitions (2024-2028) plus a default fallback partition. This
DDL is emitted as dialect-specific raw SQL since SQLAlchemy's DDL layer does
not model Postgres partitioning.

On SQLite, a plain unpartitioned table is created instead. SQLite does not
support RANGE partitioning, INET types, or JSONB — the migration branches
on dialect to handle these differences.

If a fresh install already ran schema.sql via the Docker entrypoint, this
migration is safe to stamp without re-applying: use
    alembic stamp a1b2c3d4e5f6
to record it as applied without executing upgrade().
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "a1b2c3d4e5f6"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _is_postgresql() -> bool:
    bind = op.get_bind()
    return bind.dialect.name == "postgresql"


def upgrade() -> None:
    is_pg = _is_postgresql()

    # ------------------------------------------------------------------
    # PostgreSQL-only: uuid-ossp extension
    # ------------------------------------------------------------------
    if is_pg:
        op.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp";')

    # ------------------------------------------------------------------
    # organizations
    # ------------------------------------------------------------------
    op.create_table(
        "organizations",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True) if is_pg else sa.String(36),
            primary_key=True,
        ),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("tier", sa.String(50), server_default="enterprise"),
        sa.Column("max_users", sa.Integer, server_default="10"),
        sa.Column("max_api_keys", sa.Integer, server_default="50"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()") if is_pg else sa.text("(strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()") if is_pg else sa.text("(strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))"),
            nullable=False,
        ),
    )

    # ------------------------------------------------------------------
    # users
    # ------------------------------------------------------------------
    op.create_table(
        "users",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True) if is_pg else sa.String(36),
            primary_key=True,
        ),
        sa.Column("email", sa.String(255), unique=True, nullable=False),
        sa.Column("hashed_password", sa.String(255), nullable=False),
        sa.Column("full_name", sa.String(255), nullable=True),
        sa.Column(
            "organization_id",
            postgresql.UUID(as_uuid=True) if is_pg else sa.String(36),
            sa.ForeignKey("organizations.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("tier", sa.String(50), server_default="free"),
        sa.Column("is_active", sa.Boolean, server_default="true" if is_pg else "1"),
        sa.Column("is_admin", sa.Boolean, server_default="false" if is_pg else "0"),
        sa.Column("email_verified", sa.Boolean, server_default="false" if is_pg else "0"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()") if is_pg else sa.text("(strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()") if is_pg else sa.text("(strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))"),
            nullable=False,
        ),
        sa.Column("last_login_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "tier IN ('free', 'pro', 'enterprise', 'admin')",
            name="ck_users_tier",
        ),
    )

    op.create_index("idx_users_email", "users", ["email"])
    op.create_index("idx_users_tier", "users", ["tier"])
    op.create_index("idx_users_is_admin", "users", ["is_admin"])

    # ------------------------------------------------------------------
    # api_keys
    # ------------------------------------------------------------------
    op.create_table(
        "api_keys",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True) if is_pg else sa.String(36),
            primary_key=True,
        ),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True) if is_pg else sa.String(36),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("key_prefix", sa.String(20), nullable=False),
        sa.Column("key_hash", sa.String(255), nullable=False),
        sa.Column("name", sa.String(255), nullable=True),
        # JSONB on Postgres; JSON text on SQLite.
        sa.Column(
            "scopes",
            postgresql.JSONB() if is_pg else sa.JSON(),
            server_default=sa.text("""'["memories:read", "memories:write"]'::jsonb""") if is_pg else sa.text("""'["memories:read", "memories:write"]'"""),
        ),
        sa.Column("rate_limit_tier", sa.String(50), server_default="free"),
        sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("is_active", sa.Boolean, server_default="true" if is_pg else "1"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()") if is_pg else sa.text("(strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))"),
            nullable=False,
        ),
    )

    op.create_index("idx_api_keys_user_id", "api_keys", ["user_id"])
    op.create_index("idx_api_keys_key_prefix", "api_keys", ["key_prefix"])
    op.create_index("idx_api_keys_is_active", "api_keys", ["is_active"])

    # ------------------------------------------------------------------
    # api_key_usage — partitioned on Postgres, plain table on SQLite
    # ------------------------------------------------------------------
    if is_pg:
        # Raw DDL is needed for PARTITION BY RANGE which SQLAlchemy does not
        # model natively. The composite PK (id, date) is a Postgres requirement
        # for partitioned tables.
        op.execute("""
            CREATE TABLE IF NOT EXISTS api_key_usage (
                id UUID DEFAULT uuid_generate_v4(),
                api_key_id UUID NOT NULL REFERENCES api_keys(id) ON DELETE CASCADE,
                endpoint VARCHAR(255) NOT NULL,
                method VARCHAR(10) NOT NULL,
                status_code INTEGER,
                request_count INTEGER DEFAULT 1,
                tokens_used INTEGER DEFAULT 0,
                response_time_ms FLOAT,
                date DATE NOT NULL DEFAULT CURRENT_DATE,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                PRIMARY KEY (id, date)
            ) PARTITION BY RANGE (date);
        """)
        for year in range(2024, 2029):
            op.execute(f"""
                CREATE TABLE IF NOT EXISTS api_key_usage_{year}
                    PARTITION OF api_key_usage
                    FOR VALUES FROM ('{year}-01-01') TO ('{year + 1}-01-01');
            """)
        op.execute("""
            CREATE TABLE IF NOT EXISTS api_key_usage_default
                PARTITION OF api_key_usage DEFAULT;
        """)
    else:
        op.create_table(
            "api_key_usage",
            sa.Column("id", sa.String(36), primary_key=True),
            sa.Column(
                "api_key_id",
                sa.String(36),
                sa.ForeignKey("api_keys.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column("endpoint", sa.String(255), nullable=False),
            sa.Column("method", sa.String(10), nullable=False),
            sa.Column("status_code", sa.Integer, nullable=True),
            sa.Column("request_count", sa.Integer, server_default="1"),
            sa.Column("tokens_used", sa.Integer, server_default="0"),
            sa.Column("response_time_ms", sa.Float, nullable=True),
            sa.Column(
                "date",
                sa.Date,
                server_default=sa.text("(date('now'))"),
                nullable=False,
            ),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                server_default=sa.text("(strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))"),
                nullable=False,
            ),
        )

    op.create_index("idx_api_key_usage_key_id", "api_key_usage", ["api_key_id"])
    op.create_index("idx_api_key_usage_date", "api_key_usage", ["date"])
    op.create_index("idx_api_key_usage_endpoint", "api_key_usage", ["endpoint"])

    # ------------------------------------------------------------------
    # rate_limit_buckets
    # ------------------------------------------------------------------
    op.create_table(
        "rate_limit_buckets",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True) if is_pg else sa.String(36),
            primary_key=True,
        ),
        sa.Column(
            "api_key_id",
            postgresql.UUID(as_uuid=True) if is_pg else sa.String(36),
            sa.ForeignKey("api_keys.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("time_window", sa.String(20), nullable=False),
        sa.Column("tokens_remaining", sa.Integer, nullable=False),
        sa.Column("tokens_capacity", sa.Integer, nullable=False),
        sa.Column(
            "last_refill_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()") if is_pg else sa.text("(strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()") if is_pg else sa.text("(strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))"),
            nullable=False,
        ),
        sa.UniqueConstraint("api_key_id", "time_window", name="uq_rate_limit_bucket"),
    )

    # ------------------------------------------------------------------
    # sessions
    # ------------------------------------------------------------------
    op.create_table(
        "sessions",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True) if is_pg else sa.String(36),
            primary_key=True,
        ),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True) if is_pg else sa.String(36),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("session_token", sa.String(255), unique=True, nullable=False),
        # INET on Postgres; String(45) covers IPv6 on SQLite.
        sa.Column(
            "ip_address",
            postgresql.INET() if is_pg else sa.String(45),
            nullable=True,
        ),
        sa.Column("user_agent", sa.Text, nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()") if is_pg else sa.text("(strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))"),
            nullable=False,
        ),
    )

    op.create_index("idx_sessions_token", "sessions", ["session_token"])
    op.create_index("idx_sessions_user_id", "sessions", ["user_id"])
    op.create_index("idx_sessions_expires_at", "sessions", ["expires_at"])

    # ------------------------------------------------------------------
    # audit_log (auth-scoped, singular)
    # ------------------------------------------------------------------
    op.create_table(
        "audit_log",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True) if is_pg else sa.String(36),
            primary_key=True,
        ),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True) if is_pg else sa.String(36),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("action", sa.String(100), nullable=False),
        sa.Column("resource_type", sa.String(50), nullable=True),
        sa.Column(
            "resource_id",
            postgresql.UUID(as_uuid=True) if is_pg else sa.String(36),
            nullable=True,
        ),
        sa.Column(
            "details",
            postgresql.JSONB() if is_pg else sa.JSON(),
            nullable=True,
        ),
        sa.Column(
            "ip_address",
            postgresql.INET() if is_pg else sa.String(45),
            nullable=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()") if is_pg else sa.text("(strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))"),
            nullable=False,
        ),
    )

    op.create_index("idx_audit_log_user_id", "audit_log", ["user_id"])
    op.create_index("idx_audit_log_action", "audit_log", ["action"])
    op.create_index("idx_audit_log_created_at", "audit_log", ["created_at"])

    # ------------------------------------------------------------------
    # PostgreSQL-only: updated_at trigger function + triggers
    # ------------------------------------------------------------------
    if is_pg:
        op.execute("""
            CREATE OR REPLACE FUNCTION update_updated_at_column()
            RETURNS TRIGGER AS $$
            BEGIN
                NEW.updated_at = NOW();
                RETURN NEW;
            END;
            $$ language 'plpgsql';
        """)
        op.execute("""
            CREATE TRIGGER update_users_updated_at
                BEFORE UPDATE ON users
                FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
        """)
        op.execute("""
            CREATE TRIGGER update_organizations_updated_at
                BEFORE UPDATE ON organizations
                FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
        """)

    # ------------------------------------------------------------------
    # audit_logs (compliance audit, plural — audit/schema.sql)
    # ------------------------------------------------------------------
    op.create_table(
        "audit_logs",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True) if is_pg else sa.String(36),
            primary_key=True,
        ),
        sa.Column(
            "timestamp",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()") if is_pg else sa.text("(strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))"),
            nullable=False,
        ),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True) if is_pg else sa.String(36),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("user_email", sa.String(255), nullable=True),
        sa.Column(
            "api_key_id",
            postgresql.UUID(as_uuid=True) if is_pg else sa.String(36),
            sa.ForeignKey("api_keys.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("tenant_id", sa.String(100), nullable=True),
        sa.Column("action", sa.String(50), nullable=False),
        sa.Column("severity", sa.String(20), nullable=False, server_default="info"),
        sa.Column("resource_type", sa.String(50), nullable=True),
        sa.Column("resource_id", sa.String(255), nullable=True),
        sa.Column(
            "ip_address",
            postgresql.INET() if is_pg else sa.String(45),
            nullable=True,
        ),
        sa.Column("user_agent", sa.Text, nullable=True),
        sa.Column("endpoint", sa.String(255), nullable=True),
        sa.Column("method", sa.String(10), nullable=True),
        sa.Column("description", sa.Text, nullable=False),
        sa.Column(
            "old_value",
            postgresql.JSONB() if is_pg else sa.JSON(),
            nullable=True,
        ),
        sa.Column(
            "new_value",
            postgresql.JSONB() if is_pg else sa.JSON(),
            nullable=True,
        ),
        sa.Column(
            "metadata",
            postgresql.JSONB() if is_pg else sa.JSON(),
            server_default=sa.text("'{}'::jsonb") if is_pg else sa.text("'{}'"),
        ),
        sa.Column("success", sa.Boolean, nullable=False, server_default="true" if is_pg else "1"),
        sa.Column("error_message", sa.Text, nullable=True),
    )

    op.create_index("idx_audit_logs_timestamp", "audit_logs", ["timestamp"])
    op.create_index("idx_audit_logs_user_id", "audit_logs", ["user_id"])
    op.create_index("idx_audit_logs_tenant_id", "audit_logs", ["tenant_id"])
    op.create_index("idx_audit_logs_action", "audit_logs", ["action"])
    op.create_index("idx_audit_logs_severity", "audit_logs", ["severity"])
    op.create_index("idx_audit_logs_success", "audit_logs", ["success"])
    op.create_index("idx_audit_logs_tenant_time", "audit_logs", ["tenant_id", "timestamp"])
    op.create_index("idx_audit_logs_user_time", "audit_logs", ["user_id", "timestamp"])

    # Postgres-only: full-text search index on description
    if is_pg:
        op.execute("""
            CREATE INDEX IF NOT EXISTS idx_audit_logs_description_search
                ON audit_logs USING gin(to_tsvector('english', description));
        """)

    # ------------------------------------------------------------------
    # audit_retention_policies
    # ------------------------------------------------------------------
    op.create_table(
        "audit_retention_policies",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True) if is_pg else sa.String(36),
            primary_key=True,
        ),
        sa.Column("tenant_id", sa.String(100), unique=True, nullable=True),
        sa.Column("debug_retention", sa.String(20), nullable=False, server_default="30_days"),
        sa.Column("info_retention", sa.String(20), nullable=False, server_default="90_days"),
        sa.Column("warning_retention", sa.String(20), nullable=False, server_default="1_year"),
        sa.Column("error_retention", sa.String(20), nullable=False, server_default="3_years"),
        sa.Column("critical_retention", sa.String(20), nullable=False, server_default="7_years"),
        sa.Column("auth_events_retention", sa.String(20), nullable=False, server_default="3_years"),
        sa.Column("admin_events_retention", sa.String(20), nullable=False, server_default="7_years"),
        sa.Column("data_export_retention", sa.String(20), nullable=False, server_default="7_years"),
        sa.Column("enabled", sa.Boolean, nullable=False, server_default="true" if is_pg else "1"),
        sa.Column(
            "archive_before_delete",
            sa.Boolean,
            nullable=False,
            server_default="true" if is_pg else "1",
        ),
        sa.Column("archive_location", sa.Text, nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()") if is_pg else sa.text("(strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()") if is_pg else sa.text("(strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))"),
            nullable=False,
        ),
        sa.UniqueConstraint("tenant_id", name="uq_audit_retention_tenant"),
    )

    # PostgreSQL-only: updated_at trigger for audit_retention_policies
    if is_pg:
        op.execute("""
            CREATE OR REPLACE FUNCTION update_audit_retention_updated_at()
            RETURNS TRIGGER AS $$
            BEGIN
                NEW.updated_at = NOW();
                RETURN NEW;
            END;
            $$ LANGUAGE plpgsql;
        """)
        op.execute("""
            CREATE TRIGGER audit_retention_updated_at
                BEFORE UPDATE ON audit_retention_policies
                FOR EACH ROW
                EXECUTE FUNCTION update_audit_retention_updated_at();
        """)

    # Seed the global default retention policy (NULL tenant = global)
    if is_pg:
        op.execute("""
            INSERT INTO audit_retention_policies (id, tenant_id)
            VALUES (gen_random_uuid(), NULL)
            ON CONFLICT (tenant_id) DO NOTHING;
        """)
    else:
        op.execute("""
            INSERT OR IGNORE INTO audit_retention_policies
                (id, tenant_id,
                 debug_retention, info_retention, warning_retention,
                 error_retention, critical_retention,
                 auth_events_retention, admin_events_retention, data_export_retention,
                 enabled, archive_before_delete,
                 created_at, updated_at)
            VALUES
                (lower(hex(randomblob(4))) || '-' || lower(hex(randomblob(2))) || '-4' ||
                 substr(lower(hex(randomblob(2))),2) || '-' ||
                 substr('89ab', abs(random()) % 4 + 1, 1) ||
                 substr(lower(hex(randomblob(2))),2) || '-' || lower(hex(randomblob(6))),
                 NULL,
                 '30_days', '90_days', '1_year', '3_years', '7_years',
                 '3_years', '7_years', '7_years',
                 1, 1,
                 strftime('%Y-%m-%dT%H:%M:%fZ', 'now'),
                 strftime('%Y-%m-%dT%H:%M:%fZ', 'now'));
        """)


def downgrade() -> None:
    is_pg = _is_postgresql()

    # Drop in reverse dependency order
    op.execute("DROP TABLE IF EXISTS audit_retention_policies CASCADE;" if is_pg else "DROP TABLE IF EXISTS audit_retention_policies;")
    op.execute("DROP TABLE IF EXISTS audit_logs CASCADE;" if is_pg else "DROP TABLE IF EXISTS audit_logs;")
    op.execute("DROP TABLE IF EXISTS audit_log CASCADE;" if is_pg else "DROP TABLE IF EXISTS audit_log;")
    op.execute("DROP TABLE IF EXISTS sessions CASCADE;" if is_pg else "DROP TABLE IF EXISTS sessions;")
    op.execute("DROP TABLE IF EXISTS rate_limit_buckets CASCADE;" if is_pg else "DROP TABLE IF EXISTS rate_limit_buckets;")

    if is_pg:
        # Partitioned table: drop parent cascades all children
        op.execute("DROP TABLE IF EXISTS api_key_usage CASCADE;")
    else:
        op.execute("DROP TABLE IF EXISTS api_key_usage;")

    op.execute("DROP TABLE IF EXISTS api_keys CASCADE;" if is_pg else "DROP TABLE IF EXISTS api_keys;")
    op.execute("DROP TABLE IF EXISTS users CASCADE;" if is_pg else "DROP TABLE IF EXISTS users;")
    op.execute("DROP TABLE IF EXISTS organizations CASCADE;" if is_pg else "DROP TABLE IF EXISTS organizations;")

    if is_pg:
        op.execute("DROP FUNCTION IF EXISTS update_updated_at_column() CASCADE;")
        op.execute("DROP FUNCTION IF EXISTS update_audit_retention_updated_at() CASCADE;")
