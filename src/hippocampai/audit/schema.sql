-- Audit logs table for compliance tracking
-- Run this migration to add audit logging support

CREATE TABLE IF NOT EXISTS audit_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Who
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    user_email VARCHAR(255),
    api_key_id UUID REFERENCES api_keys(id) ON DELETE SET NULL,
    tenant_id VARCHAR(100),
    
    -- What
    action VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL DEFAULT 'info',
    resource_type VARCHAR(50),
    resource_id VARCHAR(255),
    
    -- Where
    ip_address INET,
    user_agent TEXT,
    endpoint VARCHAR(255),
    method VARCHAR(10),
    
    -- Details
    description TEXT NOT NULL,
    old_value JSONB,
    new_value JSONB,
    metadata JSONB DEFAULT '{}',
    
    -- Status
    success BOOLEAN NOT NULL DEFAULT true,
    error_message TEXT
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_audit_logs_timestamp ON audit_logs(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_audit_logs_user_id ON audit_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_logs_tenant_id ON audit_logs(tenant_id);
CREATE INDEX IF NOT EXISTS idx_audit_logs_action ON audit_logs(action);
CREATE INDEX IF NOT EXISTS idx_audit_logs_severity ON audit_logs(severity);
CREATE INDEX IF NOT EXISTS idx_audit_logs_resource ON audit_logs(resource_type, resource_id);
CREATE INDEX IF NOT EXISTS idx_audit_logs_success ON audit_logs(success);

-- Composite index for common filter combinations
CREATE INDEX IF NOT EXISTS idx_audit_logs_tenant_time 
    ON audit_logs(tenant_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_audit_logs_user_time 
    ON audit_logs(user_id, timestamp DESC);

-- Full text search on description
CREATE INDEX IF NOT EXISTS idx_audit_logs_description_search 
    ON audit_logs USING gin(to_tsvector('english', description));

-- Retention policy table
CREATE TABLE IF NOT EXISTS audit_retention_policies (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(100) UNIQUE,  -- NULL for global policy
    
    debug_retention VARCHAR(20) NOT NULL DEFAULT '30_days',
    info_retention VARCHAR(20) NOT NULL DEFAULT '90_days',
    warning_retention VARCHAR(20) NOT NULL DEFAULT '1_year',
    error_retention VARCHAR(20) NOT NULL DEFAULT '3_years',
    critical_retention VARCHAR(20) NOT NULL DEFAULT '7_years',
    
    auth_events_retention VARCHAR(20) NOT NULL DEFAULT '3_years',
    admin_events_retention VARCHAR(20) NOT NULL DEFAULT '7_years',
    data_export_retention VARCHAR(20) NOT NULL DEFAULT '7_years',
    
    enabled BOOLEAN NOT NULL DEFAULT true,
    archive_before_delete BOOLEAN NOT NULL DEFAULT true,
    archive_location TEXT,
    
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Function to auto-update updated_at
CREATE OR REPLACE FUNCTION update_audit_retention_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger for auto-updating updated_at
DROP TRIGGER IF EXISTS audit_retention_updated_at ON audit_retention_policies;
CREATE TRIGGER audit_retention_updated_at
    BEFORE UPDATE ON audit_retention_policies
    FOR EACH ROW
    EXECUTE FUNCTION update_audit_retention_updated_at();

-- Insert default global policy
INSERT INTO audit_retention_policies (tenant_id)
VALUES (NULL)
ON CONFLICT (tenant_id) DO NOTHING;

-- Comments for documentation
COMMENT ON TABLE audit_logs IS 'Audit trail for all system events - compliance requirement';
COMMENT ON COLUMN audit_logs.action IS 'Type of action: login, api_key_create, memory_delete, etc.';
COMMENT ON COLUMN audit_logs.severity IS 'Event severity: debug, info, warning, error, critical';
COMMENT ON COLUMN audit_logs.old_value IS 'Previous value for update operations (JSONB)';
COMMENT ON COLUMN audit_logs.new_value IS 'New value for create/update operations (JSONB)';
