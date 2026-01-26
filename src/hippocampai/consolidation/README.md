# Sleep Phase / Active Consolidation

**Automated nightly memory consolidation inspired by the brain's hippocampal replay during sleep.**

## 📋 Overview

The Sleep Phase system mimics the brain's natural memory consolidation process. Every night, it:

1. **Reviews** recent memories (last 24 hours)
2. **Prunes** low-value, transient memories
3. **Promotes** important facts to long-term storage
4. **Synthesizes** higher-level insights from related events
5. **Logs** all changes for observability

## 🏗️ Architecture

```
Celery Beat (3 AM daily)
  ↓
run_daily_consolidation
  ↓
For each active user:
  ↓
  collect_recent_memories (Postgres query)
  ↓
  cluster_memories (group by session/time)
  ↓
  llm_review_cluster (LLM decision-making)
  ↓
  apply_consolidation_decisions (policy validation)
  ↓
  persist_consolidation_changes (DB/Qdrant updates)
  ↓
  generate_dream_report (observability)
```

For detailed architecture, see: [docs/SLEEP_PHASE_ARCHITECTURE.md](../../../../docs/SLEEP_PHASE_ARCHITECTURE.md)

## 📁 File Structure

```
src/hippocampai/consolidation/
├── __init__.py              # Module exports
├── models.py                # Data models (ConsolidationRun, etc.)
├── prompts.py               # LLM prompt templates
├── policy.py                # Consolidation policy engine
├── tasks.py                 # Celery tasks (main implementation)
├── config.example.env       # Configuration examples
└── README.md                # This file

docs/
└── SLEEP_PHASE_ARCHITECTURE.md  # Detailed architecture

examples/
└── consolidation_demo.py    # End-to-end demonstration
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Prometheus metrics (optional)
pip install prometheus-client

# Ensure your LLM client is configured
# (OpenAI, Anthropic, or your UnifiedClient)
```

### 2. Configure Environment

Copy the example config and customize:

```bash
cp src/hippocampai/consolidation/config.example.env .env
```

Key settings:

```bash
# Enable the feature
ACTIVE_CONSOLIDATION_ENABLED=true

# Schedule (3 AM UTC)
CONSOLIDATION_SCHEDULE_HOUR=3

# LLM model
CONSOLIDATION_LLM_MODEL=gpt-4-turbo-preview
CONSOLIDATION_LLM_TEMPERATURE=0.3

# Policy thresholds
CONSOLIDATION_MIN_IMPORTANCE=3.0
CONSOLIDATION_MIN_AGE_DAYS=7
CONSOLIDATION_MAX_MEMORIES_PER_USER=10000

# Safety
CONSOLIDATION_DRY_RUN=false  # Set to true for testing
```

### 3. Run the Demo

Test the system without making real changes:

```bash
cd examples
PYTHONPATH=/path/to/HippocampAI/src python consolidation_demo.py
```

This demonstrates the complete pipeline with sample data.

### 4. Enable in Production

Start Celery Beat with the scheduler:

```bash
# Start Celery worker
celery -A hippocampai.celery_app worker --loglevel=info --queues=scheduled

# Start Celery Beat (scheduler)
celery -A hippocampai.celery_app beat --loglevel=info
```

The consolidation will run automatically at the configured time (default: 3 AM UTC).

## 🔧 Integration Steps

### Step 1: Update Memory Model (Optional)

Add consolidation fields to your Memory model:

```python
from hippocampai.consolidation.models import MemoryConsolidationFields

class Memory(BaseModel):
    # ... existing fields ...

    # Consolidation fields
    is_archived: bool = False
    archived_at: Optional[datetime] = None
    source_memory_ids: Optional[list[str]] = None
    consolidation_run_id: Optional[str] = None
    last_consolidated_at: Optional[datetime] = None
    decay_factor: float = 0.95
    promotion_count: int = 0
    consolidation_metadata: dict[str, Any] = Field(default_factory=dict)
```

### Step 2: Implement LLM Integration

Replace the placeholder LLM call in `tasks.py`:

```python
def call_llm_for_consolidation(prompt: str, model: str, temperature: float) -> str:
    """Call your LLM API."""
    import openai  # or anthropic, or your UnifiedClient

    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": CONSOLIDATION_SYSTEM_MESSAGE},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        response_format={"type": "json_object"}  # For GPT-4 Turbo
    )

    return response.choices[0].message.content
```

### Step 3: Implement Database Queries

Update placeholder functions in `tasks.py`:

```python
def get_active_users(lookback_hours: int = 24) -> list[str]:
    """Get users with recent activity."""
    from hippocampai.backends.postgres import get_db

    query = """
        SELECT DISTINCT user_id FROM memories
        WHERE created_at > NOW() - INTERVAL '%s hours'
           OR updated_at > NOW() - INTERVAL '%s hours'
    """

    with get_db() as db:
        result = db.execute(query, (lookback_hours, lookback_hours))
        return [row[0] for row in result.fetchall()]
```

### Step 4: Add Prometheus Metrics (Optional)

If using Prometheus:

```python
# In your main app
from prometheus_client import start_http_server

# Start metrics server
start_http_server(9090)  # Metrics at http://localhost:9090/metrics
```

Metrics exposed:
- `consolidation_runs_total{user_id, status}`
- `consolidation_duration_seconds{user_id}`
- `consolidation_memories_processed_total{user_id, action}`

## 📊 Example Consolidation Flow

**Input:** 8 memories from the last 24 hours

```
1. [EVENT] Had morning coffee at 9am (importance: 2.5, accessed: 0x)
2. [EVENT] Q4 roadmap meeting... (importance: 7.5, accessed: 2x)
3. [FACT] Decision: Prioritize AI features (importance: 8.5, accessed: 3x)
4. [EVENT] Lunch break at 12:30pm (importance: 1.0, accessed: 0x)
5. [GOAL] Hire 2 ML engineers (importance: 7.5, accessed: 1x)
6. [EVENT] Daily standup (importance: 4.0, accessed: 0x)
7. [PREF] Team wants better async tools (importance: 6.5, accessed: 1x)
8. [EVENT] Coffee break at 3pm (importance: 1.5, accessed: 0x)
```

**LLM Analysis:**

```json
{
  "promoted_facts": [
    {"id": "mem-3", "reason": "Strategic decision", "new_importance": 9.0},
    {"id": "mem-5", "reason": "Critical hiring goal", "new_importance": 8.0}
  ],
  "low_value_memory_ids": ["mem-1", "mem-4", "mem-8"],
  "updated_memories": [
    {
      "id": "mem-2",
      "new_text": "Q4 roadmap: Prioritize AI features...",
      "new_importance": 8.0
    }
  ],
  "synthetic_memories": [
    {
      "text": "Productive planning day: Finalized Q4 AI roadmap, identified hiring needs...",
      "importance": 7.5,
      "source_ids": ["mem-2", "mem-3", "mem-5", "mem-7"]
    }
  ]
}
```

**Output:**
- ❌ Deleted: 3 low-value memories (coffee breaks, lunch)
- ⬆️ Promoted: 2 strategic memories (9.0, 8.0 importance)
- ✏️ Updated: 1 memory (merged details)
- 🔄 Synthesized: 1 high-level summary

**Result:** 6 memories → 5 memories (more valuable, better organized)

## 🎯 Policy Configuration

The policy engine validates LLM decisions based on configurable rules:

### When to Delete

- Importance < `CONSOLIDATION_MIN_IMPORTANCE` (default: 3.0)
- Age > `CONSOLIDATION_MIN_AGE_DAYS` (default: 7 days)
- Never accessed (`access_count == 0`)
- **Protected types**: Never delete GOAL, PREFERENCE, FACT

### When to Archive

- Importance < `CONSOLIDATION_ARCHIVE_THRESHOLD` (default: 4.0)
- Age > `CONSOLIDATION_ARCHIVE_AGE_DAYS` (default: 30 days)
- Never accessed

### When to Promote

- Frequently accessed (access_count >= 3)
- Strategic importance (GOAL, PREFERENCE types)
- LLM explicit recommendation
- **Multiplier**: `CONSOLIDATION_PROMOTION_MULTIPLIER` (default: 1.3x)

### Memory Limits

- Max per user: `CONSOLIDATION_MAX_MEMORIES_PER_USER` (default: 10,000)
- Oldest, lowest-importance memories archived when exceeded

## 🔍 Observability

### Logging

```python
import logging

logger = logging.getLogger("hippocampai.consolidation")
logger.setLevel(logging.INFO)

# Outputs:
# [2025-12-03 03:00:15] Consolidation completed for user-123:
#   - Reviewed: 50 memories
#   - Deleted: 5
#   - Promoted: 10
#   - Synthesized: 3
#   - Duration: 12.5s
```

### Metrics (Prometheus)

```promql
# Total runs
consolidation_runs_total{status="completed"}

# Average duration
rate(consolidation_duration_seconds_sum[1h]) / rate(consolidation_duration_seconds_count[1h])

# Memories processed
consolidation_memories_processed_total{action="deleted"}
consolidation_memories_processed_total{action="promoted"}
```

### Dry Run Mode

Test without making changes:

```bash
export CONSOLIDATION_DRY_RUN=true
celery -A hippocampai.celery_app call hippocampai.consolidation.consolidate_user_memories \
  --args='["user-123", 24, true]'
```

## 🛡️ Safety Features

1. **Dry Run Mode**: Test without making changes
2. **Protected Types**: Never auto-delete GOALs, PREFERENCEs, FACTs
3. **Access Protection**: Never delete accessed memories (configurable)
4. **Age Protection**: Minimum age before deletion (default: 7 days)
5. **Policy Validation**: LLM decisions validated by policy engine
6. **Audit Trail**: All changes logged with reasoning
7. **Rollback**: ConsolidationRun records enable audit/recovery

## ⚙️ Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `ACTIVE_CONSOLIDATION_ENABLED` | `false` | Enable/disable consolidation |
| `CONSOLIDATION_SCHEDULE_HOUR` | `3` | Hour to run (0-23, UTC) |
| `CONSOLIDATION_LOOKBACK_HOURS` | `24` | Hours of memory to review |
| `CONSOLIDATION_DRY_RUN` | `false` | Test mode (no changes) |
| `CONSOLIDATION_LLM_MODEL` | `gpt-4-turbo-preview` | LLM model to use |
| `CONSOLIDATION_LLM_TEMPERATURE` | `0.3` | LLM temperature (0.0-1.0) |
| `CONSOLIDATION_MIN_IMPORTANCE` | `3.0` | Delete threshold |
| `CONSOLIDATION_MIN_AGE_DAYS` | `7` | Min age for deletion |
| `CONSOLIDATION_PROTECT_ACCESSED` | `true` | Protect accessed memories |
| `CONSOLIDATION_MAX_MEMORIES_PER_USER` | `10000` | Per-user limit |
| `CONSOLIDATION_PROMOTION_MULTIPLIER` | `1.3` | Importance boost |

See `config.example.env` for complete reference.

## 🧪 Testing

### Unit Tests

```python
from hippocampai.consolidation.policy import ConsolidationPolicyEngine, ConsolidationPolicy

def test_should_delete():
    policy = ConsolidationPolicy(min_importance_to_keep=3.0)
    engine = ConsolidationPolicyEngine(policy)

    memory = Memory(
        id="test-1",
        text="Low value event",
        user_id="user-1",
        type=MemoryType.EVENT,
        importance=2.0,
        access_count=0
    )

    should_del, reason = engine.should_delete(memory)
    assert should_del == True
```

### Integration Test

```bash
# Run the demo
PYTHONPATH=src python examples/consolidation_demo.py
```

### Manual Test

```bash
# Trigger consolidation manually for a specific user
celery -A hippocampai.celery_app call \
  hippocampai.consolidation.consolidate_user_memories \
  --args='["user-123"]' \
  --kwargs='{"dry_run": true}'
```

## 📈 Performance

### Scalability

- **Users**: Processes users in parallel (Celery chord)
- **Memories**: Handles 10,000+ memories per user via clustering
- **LLM**: Batches memories (default: 50 per call)
- **Database**: Uses indexed queries on created_at/updated_at

### Optimization Tips

1. **Clustering**: Groups related memories to reduce LLM calls
2. **Caching**: Cache user context between runs
3. **Batching**: Process users in batches of 10
4. **Rate Limiting**: Limit concurrent LLM calls (configurable)
5. **Async**: Use Celery groups for parallel processing

### Expected Performance

| Users | Memories/user | LLM Calls | Duration |
|-------|---------------|-----------|----------|
| 10    | 100           | 20        | ~2 min   |
| 100   | 100           | 200       | ~10 min  |
| 1000  | 100           | 2000      | ~60 min  |

## 🔮 Future Enhancements

See [docs/SLEEP_PHASE_ARCHITECTURE.md](../../../../docs/SLEEP_PHASE_ARCHITECTURE.md#future-enhancements) for detailed roadmap:

- Adaptive scheduling (run when user is idle)
- Multi-level consolidation (daily → weekly → monthly)
- Cross-user insights (privacy-preserving)
- Interactive consolidation (user approval)
- Predictive consolidation (ML-based importance)

## 📞 Support

- **Documentation**: See `/docs/SLEEP_PHASE_ARCHITECTURE.md`
- **Examples**: Run `examples/consolidation_demo.py`
- **Issues**: Check Celery logs at `/app/logs/celery.log`
- **Metrics**: View Prometheus at `http://localhost:9090/metrics`

## 📄 License

Part of HippocampAI - see main project LICENSE.

---

**Built with ❤️ using Celery, FastAPI, and LLM intelligence**
