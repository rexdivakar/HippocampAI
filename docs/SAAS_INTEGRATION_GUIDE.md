# 🚀 SaaS Integration Guide - Unified Library & SaaS Control

This guide shows how HippocampAI provides **unified control** where library users can programmatically configure and control SaaS automation features.

## 🎯 Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Library User                             │
│                 (Python Application)                         │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       │ Configure Policies
                       ▼
         ┌─────────────────────────────┐
         │   AutomationController      │
         │   (Unified Control Plane)   │
         └──────────────┬──────────────┘
                        │
            ┌───────────┴───────────┐
            │                       │
            ▼                       ▼
    ┌──────────────┐        ┌──────────────┐
    │   Library    │        │    SaaS      │
    │   Direct     │        │  Background  │
    │  Execution   │        │   Workers    │
    └──────────────┘        └──────────────┘
```

### Key Principles:

1. **Single Control Plane**: `AutomationController` manages both library and SaaS execution
2. **Policy-Based**: Users configure `AutomationPolicy` objects
3. **Flexible Execution**: Same code runs in library (immediate) or SaaS (background)
4. **Task Management**: `TaskManager` handles scheduling and execution

---

## 📦 Installation

```bash
# Basic installation
pip install -e .

# With SaaS features (Celery)
pip install -e ".[saas]"

# Or manually
pip install celery redis
```

---

## 🔧 Quick Start - Library User Control

### 1. Basic Setup

```python
from hippocampai import MemoryClient, AutomationController, AutomationPolicy, PolicyType
from hippocampai.adapters import GroqLLM
from hippocampai.embed.embedder import Embedder

# Initialize your memory client
llm = GroqLLM(api_key="your-api-key")
client = MemoryClient(llm_provider=llm)
embedder = Embedder()

# Initialize automation controller
automation = AutomationController(
    memory_service=client.memory_service,
    llm=llm,
    embedder=embedder
)

print("✅ Automation controller ready!")
```

### 2. Create Automation Policy

```python
# Create a policy for automatic optimization
policy = AutomationPolicy(
    user_id="user123",
    policy_type=PolicyType.THRESHOLD,  # Trigger based on thresholds

    # Enable features
    auto_summarization=True,
    auto_consolidation=True,
    auto_compression=False,  # Disabled
    importance_decay=True,
    health_monitoring=True,
    conflict_resolution=True,

    # Configure thresholds
    summarization_threshold=500,  # Run when >500 memories
    consolidation_threshold=300,

    # Configure settings
    summarization_age_days=30,  # Only summarize old memories
    consolidation_similarity=0.85,
    decay_half_life_days=90,

    # Conflict resolution
    conflict_strategy="temporal",  # Latest wins
    auto_resolve_conflicts=True,
)

# Register the policy
automation.create_policy(policy)

print(f"✅ Created policy for {policy.user_id}")
print(f"   Features enabled: {policy.auto_summarization}, {policy.auto_consolidation}")
```

### 3. Run Optimizations (Library Mode)

```python
# Check if optimization should run
if automation.should_run_summarization("user123"):
    print("Running summarization...")
    result = automation.run_summarization("user123")
    print(f"  Processed: {result['memories_processed']}")
    print(f"  Summaries created: {result['summaries_created']}")

# Or run all optimizations
results = automation.run_all_optimizations("user123")
print(f"✅ All optimizations completed!")
print(f"   Summarization: {results['summarization']['status']}")
print(f"   Consolidation: {results['consolidation']['status']}")
print(f"   Health check: {results['health_check']['status']}")
```

### 4. Manual Triggers (Bypass Policy)

```python
# Force run regardless of policy
result = automation.run_summarization("user123", force=True)

# Run specific optimization
result = automation.run_consolidation("user123", force=True)
result = automation.run_compression("user123", force=True)
result = automation.run_health_check("user123", force=True)
```

---

## 🌐 SaaS Deployment - Background Workers

### Option 1: Task Manager (Abstracted)

The `TaskManager` provides abstraction over different backends (Celery, RQ, or inline).

```python
from hippocampai.saas import TaskManager, TaskPriority

# Initialize task manager
task_manager = TaskManager(
    automation_controller=automation,
    backend="celery"  # or "rq", "inline", None
)

# Submit tasks
task = task_manager.submit_task(
    user_id="user123",
    task_type="summarization",
    priority=TaskPriority.NORMAL,
    async_execution=True  # Run in background
)

print(f"✅ Task submitted: {task.task_id}")
print(f"   Status: {task.status}")

# Check task status later
task = task_manager.get_task(task.task_id)
print(f"Task status: {task.status}")
if task.result:
    print(f"Result: {task.result}")
```

### Option 2: Celery (Direct Integration)

#### A. Create `celeryconfig.py`:

```python
from celery.schedules import crontab

broker_url = 'redis://localhost:6379/0'
result_backend = 'redis://localhost:6379/0'

task_serializer = 'json'
result_serializer = 'json'
accept_content = ['json']
timezone = 'UTC'
enable_utc = True

# Periodic tasks
beat_schedule = {
    'run-summarization-daily': {
        'task': 'hippocampai_tasks.run_scheduled_tasks',
        'schedule': crontab(hour=2, minute=0),  # 2 AM daily
    },
    'run-decay-daily': {
        'task': 'hippocampai_tasks.apply_decay_all_users',
        'schedule': crontab(hour=1, minute=0),  # 1 AM daily
    },
    'health-checks-daily': {
        'task': 'hippocampai_tasks.health_checks_all_users',
        'schedule': crontab(hour=4, minute=0),  # 4 AM daily
    },
}
```

#### B. Create `celery_tasks.py`:

```python
from celery import Celery
from hippocampai import MemoryClient, AutomationController
from hippocampai.adapters import GroqLLM
from hippocampai.embed.embedder import Embedder

# Initialize Celery app
app = Celery('hippocampai_tasks')
app.config_from_object('celeryconfig')

# Initialize components (shared across workers)
llm = GroqLLM(api_key="your-api-key")
client = MemoryClient(llm_provider=llm)
embedder = Embedder()

automation = AutomationController(
    memory_service=client.memory_service,
    llm=llm,
    embedder=embedder
)

@app.task
def run_scheduled_tasks():
    """Run scheduled tasks for all users based on their policies."""
    # Get all users with policies
    user_ids = list(automation.policies.keys())

    for user_id in user_ids:
        policy = automation.get_policy(user_id)

        if not policy or not policy.enabled:
            continue

        # Run optimizations based on policy
        if policy.auto_summarization and automation.should_run_summarization(user_id):
            run_summarization.delay(user_id)

        if policy.auto_consolidation and automation.should_run_consolidation(user_id):
            run_consolidation.delay(user_id)

        if policy.auto_compression and automation.should_run_compression(user_id):
            run_compression.delay(user_id)

@app.task
def run_summarization(user_id: str):
    """Run summarization for a user."""
    result = automation.run_summarization(user_id, force=True)
    return result

@app.task
def run_consolidation(user_id: str):
    """Run consolidation for a user."""
    result = automation.run_consolidation(user_id, force=True)
    return result

@app.task
def run_compression(user_id: str):
    """Run compression for a user."""
    result = automation.run_compression(user_id, force=True)
    return result

@app.task
def apply_decay_all_users():
    """Apply importance decay for all users."""
    user_ids = list(automation.policies.keys())

    for user_id in user_ids:
        policy = automation.get_policy(user_id)
        if policy and policy.enabled and policy.importance_decay:
            automation.run_decay(user_id, force=True)

@app.task
def health_checks_all_users():
    """Run health checks for all users."""
    user_ids = list(automation.policies.keys())

    for user_id in user_ids:
        policy = automation.get_policy(user_id)
        if policy and policy.enabled and policy.health_monitoring:
            result = automation.run_health_check(user_id, force=True)

            # Send alert if health is low
            if result.get('alert_needed'):
                send_health_alert.delay(user_id, result)

@app.task
def send_health_alert(user_id: str, health_result: dict):
    """Send alert when memory health is low."""
    # Implement your alerting logic (email, webhook, etc.)
    print(f"⚠️ ALERT: User {user_id} memory health is low!")
    print(f"   Score: {health_result['health_score']}/100")
    print(f"   Recommendations: {health_result['recommendations']}")
```

#### C. Start Celery Workers:

```bash
# Start worker
celery -A celery_tasks worker --loglevel=info

# Start beat scheduler (in separate terminal)
celery -A celery_tasks beat --loglevel=info

# Or combined
celery -A celery_tasks worker --beat --loglevel=info
```

---

## 🔌 REST API Integration (For SaaS)

Create a FastAPI app that library users can call:

```python
from fastapi import FastAPI, HTTPException
from hippocampai import AutomationController, AutomationPolicy, PolicyType
from hippocampai.saas import TaskManager, TaskPriority

app = FastAPI(title="HippocampAI SaaS API")

# Initialize (singleton)
automation = AutomationController(...)  # Your setup
task_manager = TaskManager(automation, backend="celery")

@app.post("/api/v1/users/{user_id}/policy")
async def create_policy(user_id: str, policy: AutomationPolicy):
    """Create or update automation policy for user."""
    policy.user_id = user_id
    automation.create_policy(policy)
    return {"status": "success", "policy_id": policy.policy_id}

@app.get("/api/v1/users/{user_id}/policy")
async def get_policy(user_id: str):
    """Get user's automation policy."""
    policy = automation.get_policy(user_id)
    if not policy:
        raise HTTPException(status_code=404, detail="Policy not found")
    return policy

@app.post("/api/v1/users/{user_id}/optimize/summarization")
async def trigger_summarization(user_id: str, force: bool = False):
    """Trigger summarization for user."""
    task = task_manager.submit_task(
        user_id=user_id,
        task_type="summarization",
        priority=TaskPriority.NORMAL
    )
    return {"task_id": task.task_id, "status": task.status}

@app.post("/api/v1/users/{user_id}/optimize/all")
async def trigger_all_optimizations(user_id: str):
    """Trigger all optimizations for user."""
    task = task_manager.submit_task(
        user_id=user_id,
        task_type="all_optimizations",
        priority=TaskPriority.HIGH
    )
    return {"task_id": task.task_id, "status": task.status}

@app.get("/api/v1/tasks/{task_id}")
async def get_task_status(task_id: str):
    """Get task status."""
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task

@app.get("/api/v1/users/{user_id}/statistics")
async def get_statistics(user_id: str):
    """Get comprehensive statistics."""
    return automation.get_user_statistics(user_id)

@app.get("/api/v1/users/{user_id}/health")
async def get_health(user_id: str):
    """Get memory health report."""
    result = automation.run_health_check(user_id, force=True)
    return result
```

---

## 📱 Client Library Usage (Unified Control)

Library users can control SaaS features directly:

```python
import requests
from hippocampai import AutomationPolicy, PolicyType

# Configure your policy
policy = AutomationPolicy(
    user_id="user123",
    policy_type=PolicyType.THRESHOLD,
    auto_summarization=True,
    summarization_threshold=500,
)

# Send to SaaS API
response = requests.post(
    "https://your-saas.com/api/v1/users/user123/policy",
    json=policy.dict()
)

print(f"✅ Policy configured on SaaS: {response.json()}")

# Trigger optimization manually
response = requests.post(
    "https://your-saas.com/api/v1/users/user123/optimize/summarization"
)

task_id = response.json()["task_id"]
print(f"✅ Task submitted: {task_id}")

# Check task status
response = requests.get(
    f"https://your-saas.com/api/v1/tasks/{task_id}"
)

print(f"Task status: {response.json()['status']}")
```

Or use the library directly in your app:

```python
from hippocampai import AutomationController, AutomationPolicy

# Library user configures locally
automation = AutomationController(...)
policy = AutomationPolicy(user_id="user123", ...)
automation.create_policy(policy)

# Run immediately (library mode)
result = automation.run_summarization("user123")

# OR submit to SaaS (if connected)
task = task_manager.submit_task(
    user_id="user123",
    task_type="summarization",
    async_execution=True  # Will use Celery if available
)
```

---

## 🎮 Complete Example - Library User Controlling SaaS

```python
from hippocampai import (
    MemoryClient,
    AutomationController,
    AutomationPolicy,
    PolicyType,
    AutomationSchedule,
)
from hippocampai.adapters import GroqLLM
from hippocampai.embed.embedder import Embedder

# 1. Initialize
llm = GroqLLM(api_key="your-key")
client = MemoryClient(llm_provider=llm)
embedder = Embedder()

automation = AutomationController(
    memory_service=client.memory_service,
    llm=llm,
    embedder=embedder
)

# 2. Create sophisticated policy
policy = AutomationPolicy(
    user_id="user123",
    policy_type=PolicyType.THRESHOLD,

    # Features
    auto_summarization=True,
    auto_consolidation=True,
    auto_compression=True,
    importance_decay=True,
    health_monitoring=True,
    conflict_resolution=True,

    # Thresholds
    summarization_threshold=500,
    consolidation_threshold=300,
    compression_threshold=1000,

    # Schedules (optional - for SaaS cron jobs)
    summarization_schedule=AutomationSchedule(
        enabled=True,
        cron_expression="0 2 * * *",  # 2 AM daily
    ),
    consolidation_schedule=AutomationSchedule(
        enabled=True,
        cron_expression="0 3 * * 0",  # 3 AM Sundays
    ),
    health_check_schedule=AutomationSchedule(
        enabled=True,
        interval_hours=24,  # Every 24 hours
    ),

    # Settings
    summarization_age_days=30,
    consolidation_similarity=0.85,
    compression_target_reduction=0.30,
    decay_half_life_days=90,
    health_alert_threshold=60.0,
    conflict_strategy="temporal",
)

# 3. Register policy (works for both library and SaaS)
automation.create_policy(policy)
print("✅ Policy registered!")

# 4. Use in your application
# Add some memories
for i in range(600):  # Exceeds threshold
    client.add_memory(
        text=f"Memory {i}",
        user_id="user123",
        memory_type="fact"
    )

# Check if optimization should run
if automation.should_run_summarization("user123"):
    print("📊 Memory count exceeded threshold, running summarization...")
    result = automation.run_summarization("user123")
    print(f"✅ Summarization complete:")
    print(f"   Processed: {result['memories_processed']}")
    print(f"   Summaries: {result['summaries_created']}")
    print(f"   Space saved: {result['space_saved']:.1%}")

# Run health check
health = automation.run_health_check("user123", force=True)
print(f"\n🏥 Health Score: {health['health_score']}/100")
print(f"   Status: {health['health_status']}")

if health['alert_needed']:
    print(f"   ⚠️ Alert: Health below threshold!")
    print(f"   Recommendations:")
    for rec in health['recommendations']:
        print(f"     • {rec}")

# Get comprehensive statistics
stats = automation.get_user_statistics("user123")
print(f"\n📊 Statistics:")
print(f"   Total memories: {stats['total_memories']}")
print(f"   Automation enabled: {stats['automation_enabled']}")
print(f"   Features: {stats['automation_features']}")
```

---

## 🔄 Workflow Summary

### Library User Workflow:
1. Create `AutomationPolicy` with desired settings
2. Register policy with `AutomationController`
3. Run optimizations immediately or let SaaS handle them
4. Monitor results and adjust policy as needed

### SaaS Platform Workflow:
1. Background workers read policies from `AutomationController`
2. Celery Beat triggers scheduled tasks
3. Workers execute optimizations based on policies
4. Results stored and alerts sent as configured
5. Library users can query status via API or directly

### Key Benefits:
✅ **Unified Control**: Same API for library and SaaS
✅ **Policy-Based**: Configure once, works everywhere
✅ **Flexible Execution**: Immediate or background
✅ **No Vendor Lock-in**: Works with any task backend
✅ **Full Observability**: Monitor everything

---

## 🚀 Deployment Checklist

### For Library Users:
- [ ] Install HippocampAI: `pip install -e .`
- [ ] Create `AutomationController`
- [ ] Configure `AutomationPolicy`
- [ ] Choose execution mode (immediate or background)

### For SaaS Deployment:
- [ ] Install with Celery: `pip install -e ".[saas]"`
- [ ] Set up Redis
- [ ] Configure `celeryconfig.py`
- [ ] Create `celery_tasks.py`
- [ ] Start workers and beat scheduler
- [ ] Set up monitoring (Flower, Prometheus)
- [ ] Configure alerts
- [ ] Deploy FastAPI endpoints (optional)

---

## 📚 Additional Resources

- **Full Feature Guide**: See `NEW_FEATURES_GUIDE.md`
- **Advanced Chat Example**: Run `python chat_advanced.py`
- **Test Suite**: Run `python -m pytest tests/ -v`
- **Celery Docs**: https://docs.celeryq.dev/
- **FastAPI Docs**: https://fastapi.tiangolo.com/

---

**🎉 That's it! Library users now have full control over SaaS automation features!**
