"""Initial schema — baseline from src/hippocampai/auth/schema.sql.

Revision ID: 0001
Revises:
Create Date: 2026-04-12

This migration captures the state of the database as created by the Docker
entrypoint init script (schema.sql). It exists so that Alembic's revision
table is populated and future incremental migrations can be applied cleanly.

On a fresh install the Docker entrypoint already runs schema.sql, so this
migration performs no destructive work — it uses CREATE TABLE IF NOT EXISTS
and CREATE INDEX IF NOT EXISTS throughout.
"""

from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\";")

    op.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            email VARCHAR(255) UNIQUE NOT NULL,
            hashed_password VARCHAR(255) NOT NULL,
            full_name VARCHAR(255),
            organization_id UUID,
            tier VARCHAR(50) DEFAULT 'free' CHECK (tier IN ('free', 'pro', 'enterprise', 'admin')),
            is_active BOOLEAN DEFAULT true,
            is_admin BOOLEAN DEFAULT false,
            email_verified BOOLEAN DEFAULT false,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            last_login_at TIMESTAMP WITH TIME ZONE
        );
    """)

    op.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);")
    op.execute("CREATE INDEX IF NOT EXISTS idx_users_tier ON users(tier);")
    op.execute("CREATE INDEX IF NOT EXISTS idx_users_is_admin ON users(is_admin);")

    op.execute("""
        CREATE TABLE IF NOT EXISTS api_keys (
            id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            key_prefix VARCHAR(20) NOT NULL,
            key_hash VARCHAR(255) NOT NULL,
            name VARCHAR(255),
            scopes JSONB DEFAULT '["memories:read", "memories:write"]'::jsonb,
            rate_limit_tier VARCHAR(50) DEFAULT 'free',
            last_used_at TIMESTAMP WITH TIME ZONE,
            expires_at TIMESTAMP WITH TIME ZONE,
            is_active BOOLEAN DEFAULT true,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );
    """)

    op.execute("CREATE INDEX IF NOT EXISTS idx_api_keys_user_id ON api_keys(user_id);")
    op.execute("CREATE INDEX IF NOT EXISTS idx_api_keys_key_prefix ON api_keys(key_prefix);")
    op.execute("CREATE INDEX IF NOT EXISTS idx_api_keys_is_active ON api_keys(is_active);")

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

    op.execute("CREATE INDEX IF NOT EXISTS idx_api_key_usage_key_id ON api_key_usage(api_key_id);")
    op.execute("CREATE INDEX IF NOT EXISTS idx_api_key_usage_date ON api_key_usage(date);")
    op.execute("CREATE INDEX IF NOT EXISTS idx_api_key_usage_endpoint ON api_key_usage(endpoint);")

    op.execute("""
        CREATE TABLE IF NOT EXISTS organizations (
            id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            name VARCHAR(255) NOT NULL,
            tier VARCHAR(50) DEFAULT 'enterprise',
            max_users INTEGER DEFAULT 10,
            max_api_keys INTEGER DEFAULT 50,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );
    """)

    op.execute("""
        ALTER TABLE users ADD CONSTRAINT IF NOT EXISTS fk_users_organization
            FOREIGN KEY (organization_id) REFERENCES organizations(id) ON DELETE SET NULL;
    """)

    op.execute("""
        CREATE TABLE IF NOT EXISTS rate_limit_buckets (
            id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            api_key_id UUID NOT NULL REFERENCES api_keys(id) ON DELETE CASCADE,
            time_window VARCHAR(20) NOT NULL,
            tokens_remaining INTEGER NOT NULL,
            tokens_capacity INTEGER NOT NULL,
            last_refill_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            UNIQUE(api_key_id, time_window)
        );
    """)

    op.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            session_token VARCHAR(255) UNIQUE NOT NULL,
            ip_address INET,
            user_agent TEXT,
            expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );
    """)

    op.execute("CREATE INDEX IF NOT EXISTS idx_sessions_token ON sessions(session_token);")
    op.execute("CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id);")
    op.execute("CREATE INDEX IF NOT EXISTS idx_sessions_expires_at ON sessions(expires_at);")

    op.execute("""
        CREATE TABLE IF NOT EXISTS audit_log (
            id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            user_id UUID REFERENCES users(id) ON DELETE SET NULL,
            action VARCHAR(100) NOT NULL,
            resource_type VARCHAR(50),
            resource_id UUID,
            details JSONB,
            ip_address INET,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );
    """)

    op.execute("CREATE INDEX IF NOT EXISTS idx_audit_log_user_id ON audit_log(user_id);")
    op.execute("CREATE INDEX IF NOT EXISTS idx_audit_log_action ON audit_log(action);")
    op.execute("CREATE INDEX IF NOT EXISTS idx_audit_log_created_at ON audit_log(created_at);")


def downgrade() -> None:
    # Drop in reverse dependency order
    op.execute("DROP TABLE IF EXISTS audit_log CASCADE;")
    op.execute("DROP TABLE IF EXISTS sessions CASCADE;")
    op.execute("DROP TABLE IF EXISTS rate_limit_buckets CASCADE;")
    op.execute("DROP TABLE IF EXISTS api_key_usage CASCADE;")
    op.execute("DROP TABLE IF EXISTS api_keys CASCADE;")
    op.execute("ALTER TABLE users DROP CONSTRAINT IF EXISTS fk_users_organization;")
    op.execute("DROP TABLE IF EXISTS users CASCADE;")
    op.execute("DROP TABLE IF EXISTS organizations CASCADE;")
