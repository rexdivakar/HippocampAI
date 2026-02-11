# Sleep Phase / Active Consolidation Architecture

## Overview

The Sleep Phase mimics the brain's hippocampal replay during sleep, consolidating short-term memories into long-term knowledge through automated nightly processing.

## System Components

```
┌─────────────────────────────────────────────────────────────────┐
│                     CELERY BEAT SCHEDULER                       │
│                   (Triggers at 3:00 AM daily)                   │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              run_daily_consolidation() [Main Task]              │
│  • Reads configuration (ACTIVE_CONSOLIDATION_ENABLED)           │
│  • Gets list of active users/agents                             │
│  • Chains subtasks for each user                                │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│         collect_recent_memories(user_id, lookback_hours)        │
│  • Query Postgres for memories created/updated in last 24h      │
│  • Filter by type (event, context, transient)                   │
│  • Return list of Memory objects                                │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              cluster_memories(memories) [Optional]              │
│  • Group related memories by:                                   │
│    - Session ID                                                 │
│    - Temporal proximity (same day)                              │
│    - Semantic similarity (embeddings)                           │
│  • Return dict of {cluster_id: [Memory]}                        │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│           llm_review_cluster(cluster, user_context)             │
│  • Prepare prompt with cluster memories                         │
│  • Call LLM (abstracted via UnifiedClient)                      │
│  • Parse JSON response with consolidation decisions             │
│  • Return: {promoted_facts, low_value_ids, updated, synthetic}  │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│         apply_consolidation_decisions(review_result)            │
│  • Apply policy rules (delete/archive/promote)                  │
│  • Update importance scores                                     │
│  • Create consolidated memories                                 │
│  • Update metadata (source_memory_ids, consolidation_run_id)    │
│  • Persist changes to Postgres + Qdrant                         │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                log_consolidation_run(stats, duration)           │
│  • Create ConsolidationRun record                               │
│  • Log metrics (memories reviewed/archived/promoted)            │
│  • Emit Prometheus metrics                                      │
│  • Generate dream report for observability                      │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow

### Step 1: Collection (Postgres Query)
```
INPUT: user_id="user-123", lookback_hours=24

QUERY:
  SELECT * FROM memories
  WHERE user_id = 'user-123'
  AND (created_at > NOW() - INTERVAL '24 hours'
       OR updated_at > NOW() - INTERVAL '24 hours')
  AND type IN ('event', 'context')
  ORDER BY created_at DESC

OUTPUT: [Memory(id=1, text="Had coffee at 9am"), Memory(id=2, text="Meeting at 10am"), ...]
```

### Step 2: Clustering (Semantic Grouping)
```
INPUT: 50 memories from last 24h

PROCESS:
  1. Group by session_id (if available)
  2. For ungrouped, compute embedding similarity
  3. Use DBSCAN or simple time-window clustering

OUTPUT: {
  "morning_routine": [Memory(id=1), Memory(id=2), Memory(id=3)],
  "work_meeting": [Memory(id=4), Memory(id=5)],
  "evening": [Memory(id=6), Memory(id=7), Memory(id=8)]
}
```

### Step 3: LLM Review (Per Cluster)
```
INPUT: cluster=["Had coffee at 9am", "Meeting at 10am", "Discussed Q4 roadmap"]

PROMPT:
  """
  You are reviewing a day's memories for consolidation. Analyze these memories:

  1. Had coffee at 9am (type: event, importance: 3.0)
  2. Meeting at 10am (type: event, importance: 5.0)
  3. Discussed Q4 roadmap (type: event, importance: 7.0)

  Return JSON with:
  - promoted_facts: Key facts to make long-term
  - low_value_memory_ids: IDs to delete/archive
  - updated_memories: Changes to existing memories
  - synthetic_memories: New summary/insight memories
  """

OUTPUT (LLM Response):
{
  "promoted_facts": [
    {"id": "3", "reason": "Important strategic discussion"}
  ],
  "low_value_memory_ids": ["1"],
  "updated_memories": [
    {"id": "3", "new_importance": 8.5, "new_text": "Q4 roadmap meeting: focus on AI features"}
  ],
  "synthetic_memories": [
    {
      "text": "Morning: productive work session covering Q4 planning",
      "type": "context",
      "importance": 7.0,
      "tags": ["work", "planning", "Q4"],
      "source_ids": ["2", "3"]
    }
  ]
}
```

### Step 4: Apply Decisions (Policy Engine)
```
INPUT: review_result (from LLM)

PROCESS:
  FOR EACH low_value_memory_id:
    IF importance < 3.0 AND access_count == 0:
      DELETE from Postgres + Qdrant
    ELSE:
      SET is_archived = TRUE

  FOR EACH promoted_fact:
    SET importance = MIN(importance * 1.5, 10.0)
    SET metadata.promoted_at = NOW()

  FOR EACH updated_memory:
    UPDATE text, importance in Postgres + Qdrant
    SET updated_at = NOW()

  FOR EACH synthetic_memory:
    CREATE new Memory with source_memory_ids = [...]
    INSERT into Postgres + Qdrant

OUTPUT: {
  "deleted": 1,
  "archived": 0,
  "promoted": 1,
  "updated": 1,
  "synthesized": 1
}
```

### Step 5: Logging & Observability
```
INPUT: stats={deleted: 1, promoted: 1, ...}, duration=12.5s

PROCESS:
  1. Create ConsolidationRun(
       user_id="user-123",
       memories_reviewed=50,
       memories_deleted=1,
       memories_promoted=1,
       duration=12.5,
       status="completed"
     )

  2. Log to console/file:
     [2026-02-11 03:00:15] Consolidation completed for user-123:
       - Reviewed: 50 memories
       - Deleted: 1
       - Promoted: 1
       - Synthesized: 1
       - Duration: 12.5s

  3. Emit Prometheus metrics:
     consolidation_runs_total{status="completed"} +1
     consolidation_memories_reviewed{user_id="user-123"} 50
     consolidation_duration_seconds{user_id="user-123"} 12.5
```

## Database Schema Changes

### Extended Memory Model
```python
class Memory(BaseModel):
    # ... existing fields ...

    # Consolidation fields
    is_archived: bool = False
    source_memory_ids: Optional[list[str]] = None  # For synthetic memories
    consolidation_run_id: Optional[str] = None
    last_consolidated_at: Optional[datetime] = None
    decay_factor: float = 0.95  # Daily decay multiplier
    promotion_count: int = 0  # Times promoted by consolidation
```

### ConsolidationRun Model
```python
class ConsolidationRun(BaseModel):
    id: str
    user_id: str
    agent_id: Optional[str]
    started_at: datetime
    completed_at: Optional[datetime]
    status: str  # "running", "completed", "failed"
    memories_reviewed: int
    memories_deleted: int
    memories_archived: int
    memories_promoted: int
    memories_updated: int
    memories_synthesized: int
    duration_seconds: float
    error_message: Optional[str]
    metadata: dict
```

## Celery Task Chain

```python
# Nightly run at 3:00 AM
@celery_app.task(name="hippocampai.consolidation.run_daily_consolidation")
def run_daily_consolidation():
    """Main orchestration task."""
    users = get_active_users()  # Query users with recent activity

    for user_id in users:
        chain(
            collect_recent_memories.s(user_id, lookback_hours=24),
            cluster_memories.s(),
            process_clusters.s(user_id),
            finalize_consolidation.s(user_id)
        ).apply_async()

# Chain: collect → cluster → process → finalize
```

## Configuration

```python
# Environment variables
ACTIVE_CONSOLIDATION_ENABLED=true
CONSOLIDATION_SCHEDULE_HOUR=3  # 3 AM UTC
CONSOLIDATION_LOOKBACK_HOURS=24
CONSOLIDATION_DRY_RUN=false
CONSOLIDATION_MIN_IMPORTANCE=3.0
CONSOLIDATION_MAX_MEMORIES_PER_USER=10000
CONSOLIDATION_LLM_MODEL=gpt-4
CONSOLIDATION_LLM_TEMPERATURE=0.3
```

## Safety & Observability

### Dry Run Mode
```python
if os.getenv("CONSOLIDATION_DRY_RUN", "false") == "true":
    logger.info(f"[DRY RUN] Would delete {len(to_delete)} memories")
    logger.info(f"[DRY RUN] Would promote {len(to_promote)} memories")
    return {"dry_run": True, "stats": stats}
```

### Metrics (Prometheus)
```python
from prometheus_client import Counter, Histogram

consolidation_runs = Counter(
    'consolidation_runs_total',
    'Total consolidation runs',
    ['user_id', 'status']
)

consolidation_duration = Histogram(
    'consolidation_duration_seconds',
    'Consolidation run duration',
    ['user_id']
)
```

### Logging
```python
import logging

logger = logging.getLogger("hippocampai.consolidation")
logger.setLevel(logging.INFO)

# Log every decision
logger.info(f"Consolidation run {run_id}: reviewed {count} memories for {user_id}")
logger.warning(f"Deleting {len(to_delete)} low-value memories")
logger.error(f"Consolidation failed for {user_id}: {error}")
```

## Performance Considerations

### Batching
- Process users in batches of 10 to avoid overwhelming the system
- Use Celery chord to parallelize user processing
- Rate limit LLM calls (max 5 concurrent)

### Caching
- Cache user context (preferences, recent topics) for LLM prompts
- Use Redis to track ongoing consolidation runs
- Avoid reprocessing memories already consolidated today

### Optimization
- Use vector search to find similar memories (Qdrant)
- Batch database operations (bulk updates/deletes)
- Stream LLM responses for large clusters
- Archive old ConsolidationRun records after 90 days

## Error Handling

```python
try:
    result = llm_review_cluster(cluster)
except LLMTimeoutError:
    logger.error(f"LLM timeout for cluster {cluster_id}")
    # Fallback: use rule-based consolidation
    result = rule_based_consolidation(cluster)
except LLMParseError:
    logger.error(f"Failed to parse LLM response")
    # Skip this cluster, continue with others
    continue
except Exception as e:
    logger.exception(f"Unexpected error: {e}")
    # Mark run as failed, send alert
    mark_run_failed(run_id, str(e))
```

## Future Enhancements

1. **Adaptive Scheduling**: Run consolidation when user is idle (detect from activity patterns)
2. **Multi-Level Consolidation**: Daily → Weekly → Monthly → Yearly
3. **Cross-User Insights**: Find common patterns across users (privacy-preserving)
4. **Interactive Consolidation**: Allow users to review/approve deletions
5. **Predictive Consolidation**: Use ML to predict which memories will be important
6. **Federated Consolidation**: Distribute processing across multiple workers
