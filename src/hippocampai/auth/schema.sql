-- HippocampAI Authentication Schema
-- PostgreSQL database schema for users, API keys, and usage tracking

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Users table
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

-- Create index on email for faster lookups
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_tier ON users(tier);
CREATE INDEX IF NOT EXISTS idx_users_is_admin ON users(is_admin);

-- API Keys table
CREATE TABLE IF NOT EXISTS api_keys (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    key_prefix VARCHAR(20) NOT NULL,  -- e.g., 'hc_live', 'hc_test'
    key_hash VARCHAR(255) NOT NULL,  -- bcrypt hash of the full key
    name VARCHAR(255),  -- User-friendly name
    scopes JSONB DEFAULT '["memories:read", "memories:write"]'::jsonb,
    rate_limit_tier VARCHAR(50) DEFAULT 'free',
    last_used_at TIMESTAMP WITH TIME ZONE,
    expires_at TIMESTAMP WITH TIME ZONE,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for API key lookups
CREATE INDEX IF NOT EXISTS idx_api_keys_user_id ON api_keys(user_id);
CREATE INDEX IF NOT EXISTS idx_api_keys_key_prefix ON api_keys(key_prefix);
CREATE INDEX IF NOT EXISTS idx_api_keys_is_active ON api_keys(is_active);

-- API Key Usage tracking (partitioned by date)
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

-- Annual partitions — add a new one each year before Jan 1.
-- A DEFAULT partition catches any date not covered (prevents constraint violations).
CREATE TABLE IF NOT EXISTS api_key_usage_2024 PARTITION OF api_key_usage
    FOR VALUES FROM ('2024-01-01') TO ('2025-01-01');

CREATE TABLE IF NOT EXISTS api_key_usage_2025 PARTITION OF api_key_usage
    FOR VALUES FROM ('2025-01-01') TO ('2026-01-01');

CREATE TABLE IF NOT EXISTS api_key_usage_2026 PARTITION OF api_key_usage
    FOR VALUES FROM ('2026-01-01') TO ('2027-01-01');

CREATE TABLE IF NOT EXISTS api_key_usage_2027 PARTITION OF api_key_usage
    FOR VALUES FROM ('2027-01-01') TO ('2028-01-01');

CREATE TABLE IF NOT EXISTS api_key_usage_2028 PARTITION OF api_key_usage
    FOR VALUES FROM ('2028-01-01') TO ('2029-01-01');

-- Default partition: catches any date not covered by the annual partitions above.
-- This prevents INSERT failures when a new year starts before the annual partition
-- is manually created. Records here should be migrated to the correct partition
-- by running scripts/create_usage_partition.py at the start of each new year.
CREATE TABLE IF NOT EXISTS api_key_usage_default PARTITION OF api_key_usage DEFAULT;

-- Create indexes on usage table
CREATE INDEX IF NOT EXISTS idx_api_key_usage_key_id ON api_key_usage(api_key_id);
CREATE INDEX IF NOT EXISTS idx_api_key_usage_date ON api_key_usage(date);
CREATE INDEX IF NOT EXISTS idx_api_key_usage_endpoint ON api_key_usage(endpoint);

-- Organizations table (for enterprise multi-tenancy)
CREATE TABLE IF NOT EXISTS organizations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    tier VARCHAR(50) DEFAULT 'enterprise',
    max_users INTEGER DEFAULT 10,
    max_api_keys INTEGER DEFAULT 50,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Add foreign key for organization
ALTER TABLE users ADD CONSTRAINT fk_users_organization
    FOREIGN KEY (organization_id) REFERENCES organizations(id) ON DELETE SET NULL;

-- Rate limit buckets (for token bucket algorithm)
CREATE TABLE IF NOT EXISTS rate_limit_buckets (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    api_key_id UUID NOT NULL REFERENCES api_keys(id) ON DELETE CASCADE,
    time_window VARCHAR(20) NOT NULL,  -- 'minute', 'hour', 'day' (renamed from 'window' - reserved keyword)
    tokens_remaining INTEGER NOT NULL,
    tokens_capacity INTEGER NOT NULL,
    last_refill_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(api_key_id, time_window)
);

-- Sessions table (for admin UI login)
CREATE TABLE IF NOT EXISTS sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    session_token VARCHAR(255) UNIQUE NOT NULL,
    ip_address INET,
    user_agent TEXT,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_sessions_token ON sessions(session_token);
CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_expires_at ON sessions(expires_at);

-- Audit log for admin actions
CREATE TABLE IF NOT EXISTS audit_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    action VARCHAR(100) NOT NULL,  -- 'create_user', 'delete_api_key', etc.
    resource_type VARCHAR(50),  -- 'user', 'api_key', etc.
    resource_id UUID,
    details JSONB,
    ip_address INET,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_audit_log_user_id ON audit_log(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_log_action ON audit_log(action);
CREATE INDEX IF NOT EXISTS idx_audit_log_created_at ON audit_log(created_at);

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers for updated_at
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_organizations_updated_at BEFORE UPDATE ON organizations
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Function to create the initial admin user.
-- NEVER call this directly from SQL with a hardcoded password.
-- Use scripts/init_admin.py which reads ADMIN_PASSWORD from the environment,
-- generates a bcrypt hash, and calls: SELECT create_default_admin('<hash>');
CREATE OR REPLACE FUNCTION create_default_admin(p_hashed_password TEXT)
RETURNS VOID AS $$
BEGIN
    IF p_hashed_password IS NULL OR length(trim(p_hashed_password)) = 0 THEN
        RAISE EXCEPTION 'p_hashed_password must not be empty. Generate a bcrypt hash via scripts/init_admin.py.';
    END IF;
    INSERT INTO users (email, hashed_password, full_name, tier, is_admin, is_active, email_verified)
    VALUES (
        'admin@hippocampai.com',
        p_hashed_password,
        'System Administrator',
        'admin',
        true,
        true,
        true
    )
    ON CONFLICT (email) DO NOTHING;
END;
$$ LANGUAGE plpgsql;

-- Admin is NOT seeded here. Run scripts/init_admin.py with ADMIN_PASSWORD set
-- in the environment to create the initial admin account.

-- View for user statistics
CREATE OR REPLACE VIEW user_statistics AS
SELECT
    u.id,
    u.email,
    u.tier,
    u.is_active,
    COUNT(DISTINCT ak.id) as api_key_count,
    COUNT(DISTINCT aku.id) as total_requests,
    SUM(aku.tokens_used) as total_tokens_used,
    MAX(ak.last_used_at) as last_api_usage,
    u.created_at
FROM users u
LEFT JOIN api_keys ak ON u.id = ak.user_id
LEFT JOIN api_key_usage aku ON ak.id = aku.api_key_id
GROUP BY u.id, u.email, u.tier, u.is_active, u.created_at;

-- View for API key statistics
CREATE OR REPLACE VIEW api_key_statistics AS
SELECT
    ak.id,
    ak.user_id,
    ak.name,
    ak.key_prefix,
    ak.rate_limit_tier,
    ak.is_active,
    COUNT(aku.id) as total_requests,
    SUM(aku.tokens_used) as total_tokens_used,
    AVG(aku.response_time_ms) as avg_response_time,
    MAX(aku.created_at) as last_request_at,
    ak.created_at
FROM api_keys ak
LEFT JOIN api_key_usage aku ON ak.id = aku.api_key_id
GROUP BY ak.id, ak.user_id, ak.name, ak.key_prefix, ak.rate_limit_tier, ak.is_active, ak.created_at;

-- Grant permissions (adjust as needed)
-- GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO hippocampai_app;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO hippocampai_app;
