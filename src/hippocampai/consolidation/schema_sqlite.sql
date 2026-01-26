-- HippocampAI Sleep Phase / Consolidation Schema (SQLite)
-- SQLite database schema for consolidation runs and memory optimization tracking

-- Consolidation runs table - tracks each sleep phase execution
CREATE TABLE IF NOT EXISTS consolidation_runs (
    id TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
    user_id TEXT NOT NULL,
    agent_id TEXT,

    -- Execution metadata
    status TEXT NOT NULL CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled')),
    mode TEXT NOT NULL DEFAULT 'live' CHECK (mode IN ('live', 'preview')),
    started_at TEXT NOT NULL DEFAULT (datetime('now')),
    completed_at TEXT,
    duration_seconds REAL,

    -- Configuration
    lookback_hours INTEGER NOT NULL DEFAULT 24,
    min_importance REAL DEFAULT 3.0,
    dry_run INTEGER DEFAULT 0,  -- SQLite uses INTEGER for boolean

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
    dream_report TEXT,
    error_message TEXT,
    error_stacktrace TEXT,

    -- Metadata
    celery_task_id TEXT,
    triggered_by TEXT DEFAULT 'scheduled',  -- 'scheduled', 'manual', 'api'
    created_at TEXT DEFAULT (datetime('now'))
);

-- Indexes for efficient queries
CREATE INDEX IF NOT EXISTS idx_consolidation_runs_user_id ON consolidation_runs(user_id);
CREATE INDEX IF NOT EXISTS idx_consolidation_runs_status ON consolidation_runs(status);
CREATE INDEX IF NOT EXISTS idx_consolidation_runs_started_at ON consolidation_runs(started_at DESC);
CREATE INDEX IF NOT EXISTS idx_consolidation_runs_completed_at ON consolidation_runs(completed_at DESC);
CREATE INDEX IF NOT EXISTS idx_consolidation_runs_mode ON consolidation_runs(mode);
CREATE INDEX IF NOT EXISTS idx_consolidation_runs_user_started ON consolidation_runs(user_id, started_at DESC);

-- Consolidation run details - stores detailed information about what happened to each memory
CREATE TABLE IF NOT EXISTS consolidation_run_details (
    id TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
    run_id TEXT NOT NULL REFERENCES consolidation_runs(id) ON DELETE CASCADE,

    -- Memory identification
    memory_id TEXT NOT NULL,
    cluster_id TEXT,

    -- Action taken
    action TEXT NOT NULL CHECK (action IN ('promoted', 'archived', 'deleted', 'updated', 'synthesized', 'unchanged')),

    -- Details
    reason TEXT,
    old_importance REAL,
    new_importance REAL,
    old_text TEXT,
    new_text TEXT,
    source_memory_ids TEXT,  -- JSON array as TEXT in SQLite

    -- Metadata
    created_at TEXT DEFAULT (datetime('now'))
);

-- Indexes for run details
CREATE INDEX IF NOT EXISTS idx_consolidation_run_details_run_id ON consolidation_run_details(run_id);
CREATE INDEX IF NOT EXISTS idx_consolidation_run_details_memory_id ON consolidation_run_details(memory_id);
CREATE INDEX IF NOT EXISTS idx_consolidation_run_details_action ON consolidation_run_details(action);

-- Memory clusters - stores the grouping information
CREATE TABLE IF NOT EXISTS consolidation_clusters (
    id TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
    run_id TEXT NOT NULL REFERENCES consolidation_runs(id) ON DELETE CASCADE,

    -- Cluster identification
    cluster_id TEXT NOT NULL,
    theme TEXT,

    -- Cluster metadata
    memory_ids TEXT NOT NULL,  -- JSON array as TEXT in SQLite
    time_window_start TEXT,
    time_window_end TEXT,
    avg_importance REAL,

    -- LLM decision
    llm_reasoning TEXT,

    created_at TEXT DEFAULT (datetime('now')),

    UNIQUE(run_id, cluster_id)
);

-- Index for efficient cluster lookups
CREATE INDEX IF NOT EXISTS idx_consolidation_clusters_run_id ON consolidation_clusters(run_id);

-- View for consolidation statistics
CREATE VIEW IF NOT EXISTS consolidation_statistics AS
SELECT
    user_id,
    COUNT(*) as total_runs,
    SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as successful_runs,
    SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed_runs,
    SUM(CASE WHEN mode = 'live' THEN 1 ELSE 0 END) as live_runs,
    SUM(CASE WHEN mode = 'preview' THEN 1 ELSE 0 END) as preview_runs,
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
WHERE status IN ('completed', 'failed')
GROUP BY user_id;

-- View for recent consolidation activity
CREATE VIEW IF NOT EXISTS recent_consolidation_activity AS
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
    SUM(CASE WHEN crd.action = 'promoted' THEN 1 ELSE 0 END) as promoted_count,
    SUM(CASE WHEN crd.action = 'archived' THEN 1 ELSE 0 END) as archived_count,
    SUM(CASE WHEN crd.action = 'deleted' THEN 1 ELSE 0 END) as deleted_count
FROM consolidation_runs cr
LEFT JOIN consolidation_run_details crd ON cr.id = crd.run_id
WHERE datetime(cr.started_at) >= datetime('now', '-30 days')
GROUP BY cr.id, cr.user_id, cr.status, cr.mode, cr.started_at, cr.completed_at,
         cr.duration_seconds, cr.memories_reviewed, cr.memories_promoted,
         cr.memories_archived, cr.memories_deleted, cr.memories_synthesized,
         cr.dream_report, cr.triggered_by
ORDER BY cr.started_at DESC;
