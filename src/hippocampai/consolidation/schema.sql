-- HippocampAI Sleep Phase / Consolidation Schema
-- PostgreSQL database schema for consolidation runs and memory optimization tracking

-- Enable UUID extension (if not already enabled)
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Consolidation runs table - tracks each sleep phase execution
CREATE TABLE IF NOT EXISTS consolidation_runs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(255) NOT NULL,
    agent_id VARCHAR(255),

    -- Execution metadata
    status VARCHAR(20) NOT NULL CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled')),
    mode VARCHAR(10) NOT NULL DEFAULT 'live' CHECK (mode IN ('live', 'preview')),
    started_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,
    duration_seconds FLOAT,

    -- Configuration
    lookback_hours INTEGER NOT NULL DEFAULT 24,
    min_importance FLOAT DEFAULT 3.0,
    dry_run BOOLEAN DEFAULT false,

    -- Statistics
    memories_reviewed INTEGER DEFAULT 0,
    memories_deleted INTEGER DEFAULT 0,
    memories_archived INTEGER DEFAULT 0,
    memories_promoted INTEGER DEFAULT 0,
    memories_updated INTEGER DEFAULT 0,
    memories_synthesized INTEGER DEFAULT 0,
    clusters_created INTEGER DEFAULT 0,
    llm_calls_made INTEGER DEFAULT 0,

    -- Result
    dream_report TEXT,  -- Human-readable summary
    error_message TEXT,
    error_stacktrace TEXT,

    -- Metadata
    celery_task_id VARCHAR(255),
    triggered_by VARCHAR(50) DEFAULT 'scheduled',  -- 'scheduled', 'manual', 'api'
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for efficient queries
CREATE INDEX IF NOT EXISTS idx_consolidation_runs_user_id ON consolidation_runs(user_id);
CREATE INDEX IF NOT EXISTS idx_consolidation_runs_status ON consolidation_runs(status);
CREATE INDEX IF NOT EXISTS idx_consolidation_runs_started_at ON consolidation_runs(started_at DESC);
CREATE INDEX IF NOT EXISTS idx_consolidation_runs_completed_at ON consolidation_runs(completed_at DESC NULLS LAST);
CREATE INDEX IF NOT EXISTS idx_consolidation_runs_mode ON consolidation_runs(mode);

-- Composite index for common queries (user's recent runs)
CREATE INDEX IF NOT EXISTS idx_consolidation_runs_user_started ON consolidation_runs(user_id, started_at DESC);

-- Consolidation run details - stores detailed information about what happened to each memory
CREATE TABLE IF NOT EXISTS consolidation_run_details (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    run_id UUID NOT NULL REFERENCES consolidation_runs(id) ON DELETE CASCADE,

    -- Memory identification
    memory_id VARCHAR(255) NOT NULL,
    cluster_id VARCHAR(255),

    -- Action taken
    action VARCHAR(20) NOT NULL CHECK (action IN ('promoted', 'archived', 'deleted', 'updated', 'synthesized', 'unchanged')),

    -- Details
    reason TEXT,
    old_importance FLOAT,
    new_importance FLOAT,
    old_text TEXT,
    new_text TEXT,
    source_memory_ids TEXT[],  -- For synthesized memories

    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for run details
CREATE INDEX IF NOT EXISTS idx_consolidation_run_details_run_id ON consolidation_run_details(run_id);
CREATE INDEX IF NOT EXISTS idx_consolidation_run_details_memory_id ON consolidation_run_details(memory_id);
CREATE INDEX IF NOT EXISTS idx_consolidation_run_details_action ON consolidation_run_details(action);

-- Memory clusters - stores the grouping information
CREATE TABLE IF NOT EXISTS consolidation_clusters (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    run_id UUID NOT NULL REFERENCES consolidation_runs(id) ON DELETE CASCADE,

    -- Cluster identification
    cluster_id VARCHAR(255) NOT NULL,
    theme VARCHAR(500),

    -- Cluster metadata
    memory_ids TEXT[] NOT NULL,
    time_window_start TIMESTAMP WITH TIME ZONE,
    time_window_end TIMESTAMP WITH TIME ZONE,
    avg_importance FLOAT,

    -- LLM decision
    llm_reasoning TEXT,

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    UNIQUE(run_id, cluster_id)
);

-- Index for efficient cluster lookups
CREATE INDEX IF NOT EXISTS idx_consolidation_clusters_run_id ON consolidation_clusters(run_id);

-- View for consolidation statistics
CREATE OR REPLACE VIEW consolidation_statistics AS
SELECT
    user_id,
    COUNT(*) as total_runs,
    COUNT(*) FILTER (WHERE status = 'completed') as successful_runs,
    COUNT(*) FILTER (WHERE status = 'failed') as failed_runs,
    COUNT(*) FILTER (WHERE mode = 'live') as live_runs,
    COUNT(*) FILTER (WHERE mode = 'preview') as preview_runs,
    SUM(memories_reviewed) as total_memories_reviewed,
    SUM(memories_promoted) as total_memories_promoted,
    SUM(memories_archived) as total_memories_archived,
    SUM(memories_deleted) as total_memories_deleted,
    SUM(memories_synthesized) as total_memories_synthesized,
    SUM(memories_updated) as total_memories_updated,
    AVG(duration_seconds) as avg_duration_seconds,
    MAX(started_at) as last_run_at,
    MIN(started_at) as first_run_at
FROM consolidation_runs
WHERE status IN ('completed', 'failed')  -- Only count finished runs
GROUP BY user_id;

-- View for recent consolidation activity
CREATE OR REPLACE VIEW recent_consolidation_activity AS
SELECT
    cr.id,
    cr.user_id,
    cr.status,
    cr.mode,
    cr.started_at,
    cr.completed_at,
    cr.duration_seconds,
    cr.memories_reviewed,
    cr.memories_promoted,
    cr.memories_archived,
    cr.memories_deleted,
    cr.memories_synthesized,
    cr.dream_report,
    cr.triggered_by,
    COUNT(crd.id) FILTER (WHERE crd.action = 'promoted') as promoted_count,
    COUNT(crd.id) FILTER (WHERE crd.action = 'archived') as archived_count,
    COUNT(crd.id) FILTER (WHERE crd.action = 'deleted') as deleted_count
FROM consolidation_runs cr
LEFT JOIN consolidation_run_details crd ON cr.id = crd.run_id
WHERE cr.started_at >= NOW() - INTERVAL '30 days'
GROUP BY cr.id, cr.user_id, cr.status, cr.mode, cr.started_at, cr.completed_at,
         cr.duration_seconds, cr.memories_reviewed, cr.memories_promoted,
         cr.memories_archived, cr.memories_deleted, cr.memories_synthesized,
         cr.dream_report, cr.triggered_by
ORDER BY cr.started_at DESC;

-- Function to clean up old consolidation runs (optional maintenance)
CREATE OR REPLACE FUNCTION cleanup_old_consolidation_runs(days_to_keep INTEGER DEFAULT 90)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM consolidation_runs
    WHERE started_at < NOW() - (days_to_keep || ' days')::INTERVAL
    AND status IN ('completed', 'failed', 'cancelled');

    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Grant permissions (adjust as needed for your setup)
-- GRANT SELECT, INSERT, UPDATE, DELETE ON consolidation_runs TO hippocampai_app;
-- GRANT SELECT, INSERT, UPDATE, DELETE ON consolidation_run_details TO hippocampai_app;
-- GRANT SELECT, INSERT, UPDATE, DELETE ON consolidation_clusters TO hippocampai_app;
-- GRANT SELECT ON consolidation_statistics TO hippocampai_app;
-- GRANT SELECT ON recent_consolidation_activity TO hippocampai_app;
