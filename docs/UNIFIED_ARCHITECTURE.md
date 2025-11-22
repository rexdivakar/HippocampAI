# 🏗️ Unified Architecture - Library & SaaS Integration

## Overview

HippocampAI now provides a **unified architecture** where:
- **Library users** control all features programmatically
- **SaaS platforms** execute features automatically in background
- **Both share the same control plane** (AutomationController)
- **Zero vendor lock-in** - works with any task backend

## 📐 Architecture Diagram

```
┌───────────────────────────────────────────────────────────────────┐
│                     LIBRARY USER                                   │
│                  (Python Application)                              │
│                                                                     │
│  from hippocampai import (                                         │
│      AutomationController,                                         │
│      AutomationPolicy,                                             │
│      TaskManager                                                   │
│  )                                                                  │
│                                                                     │
│  # Create policy                                                   │
│  policy = AutomationPolicy(                                        │
│      user_id="user123",                                            │
│      auto_summarization=True,                                      │
│      summarization_threshold=500                                   │
│  )                                                                  │
│                                                                     │
│  # Register & control                                              │
│  automation.create_policy(policy)                                  │
│  automation.run_summarization("user123")                           │
│                                                                     │
└───────────────────────────┬───────────────────────────────────────┘
                            │
                            │ Configure
                            ▼
        ┌───────────────────────────────────────┐
        │     AUTOMATION CONTROLLER              │
        │    (Unified Control Plane)             │
        │                                        │
        │  • Policy Management                   │
        │  • Feature Toggle                      │
        │  • Threshold Checking                  │
        │  • Execution Routing                   │
        │                                        │
        └───────────┬───────────────────┬────────┘
                    │                   │
        ┌───────────▼────────┐  ┌──────▼─────────────┐
        │   LIBRARY MODE     │  │    SAAS MODE       │
        │  (Immediate)       │  │   (Background)     │
        │                    │  │                    │
        │  automation.run_   │  │  Celery Workers    │
        │  summarization()   │  │  read policies     │
        │                    │  │  and execute       │
        │  Result: {...}     │  │                    │
        │                    │  │  Task Queue        │
        └────────────────────┘  └────────────────────┘
```

## 🎯 Key Features

### 1. Unified Control Plane

**Single API** for both library and SaaS:

```python
# Same code works in library AND SaaS
automation = AutomationController(memory_service, llm, embedder)

# Create policy (works everywhere)
policy = AutomationPolicy(user_id="user123", auto_summarization=True)
automation.create_policy(policy)

# Run immediately (library mode)
result = automation.run_summarization("user123")

# OR submit to background (SaaS mode)
task = task_manager.submit_task("user123", "summarization")
```

### 2. Policy-Based Configuration

Configure once, works everywhere:

```python
policy = AutomationPolicy(
    user_id="user123",
    policy_type=PolicyType.THRESHOLD,

    # Feature toggles
    auto_summarization=True,
    auto_consolidation=True,
    auto_compression=False,
    importance_decay=True,
    health_monitoring=True,

    # Thresholds (for automatic triggers)
    summarization_threshold=500,
    consolidation_threshold=300,

    # Schedules (for SaaS cron jobs)
    summarization_schedule=AutomationSchedule(
        cron_expression="0 2 * * *"  # 2 AM daily
    ),

    # Settings
    summarization_age_days=30,
    conflict_strategy="temporal"
)

automation.create_policy(policy)
```

### 3. Flexible Execution

Choose your execution model:

```python
# Option 1: Immediate (Library)
result = automation.run_summarization("user123")

# Option 2: Background (SaaS with Celery)
task = task_manager.submit_task(
    user_id="user123",
    task_type="summarization",
    priority=TaskPriority.NORMAL,
    async_execution=True  # Uses Celery
)

# Option 3: Inline (Synchronous testing)
task = task_manager.submit_task(
    user_id="user123",
    task_type="summarization",
    async_execution=False  # Runs immediately
)
```

### 4. Backend Agnostic

Works with any task backend:

```python
# Celery backend
task_manager = TaskManager(automation, backend="celery")

# Redis Queue backend
task_manager = TaskManager(automation, backend="rq")

# Inline (no queue)
task_manager = TaskManager(automation, backend="inline")

# Auto-detect
task_manager = TaskManager(automation)  # Uses best available
```

## 🔧 Components

### AutomationController

**Purpose**: Central control plane for all automation

**Responsibilities**:
- Policy management (CRUD)
- Feature enablement/disablement
- Threshold checking
- Execution routing (library vs SaaS)
- Statistics aggregation

**Key Methods**:
```python
# Policy management
create_policy(policy) -> AutomationPolicy
get_policy(user_id) -> Optional[AutomationPolicy]
delete_policy(user_id) -> bool

# Threshold checks
should_run_summarization(user_id) -> bool
should_run_consolidation(user_id) -> bool
should_run_compression(user_id) -> bool

# Execution
run_summarization(user_id, force=False) -> dict
run_consolidation(user_id, force=False) -> dict
run_compression(user_id, force=False) -> dict
run_decay(user_id, force=False) -> dict
run_health_check(user_id, force=False) -> dict
run_all_optimizations(user_id, force=False) -> dict

# Statistics
get_user_statistics(user_id) -> dict
```

### AutomationPolicy

**Purpose**: Configuration object for user preferences

**Fields**:
```python
# Core
user_id: str
policy_type: PolicyType  # threshold, schedule, continuous, manual
enabled: bool

# Features
auto_summarization: bool
auto_consolidation: bool
auto_compression: bool
importance_decay: bool
health_monitoring: bool
conflict_resolution: bool

# Thresholds
summarization_threshold: int  # Number of memories
consolidation_threshold: int
compression_threshold: int

# Schedules (for SaaS)
summarization_schedule: Optional[AutomationSchedule]
consolidation_schedule: Optional[AutomationSchedule]
health_check_schedule: Optional[AutomationSchedule]

# Settings
summarization_age_days: int
consolidation_similarity: float
compression_target_reduction: float
decay_half_life_days: int
health_alert_threshold: float
conflict_strategy: str
```

### TaskManager

**Purpose**: Abstract task queue management

**Responsibilities**:
- Task creation and submission
- Backend routing (Celery/RQ/inline)
- Task status tracking
- Retry logic
- Scheduled task execution

**Key Methods**:
```python
# Task submission
submit_task(user_id, task_type, priority, async_execution) -> BackgroundTask
execute_task(task) -> BackgroundTask

# Task queries
get_task(task_id) -> Optional[BackgroundTask]
get_user_tasks(user_id, status, limit) -> list[BackgroundTask]

# Control
cancel_task(task_id) -> bool

# Scheduled execution
run_scheduled_tasks(user_ids) -> None
```

## 📚 Usage Patterns

### Pattern 1: Library User (Immediate Execution)

```python
from hippocampai import MemoryClient, AutomationController, AutomationPolicy

# Setup
client = MemoryClient(llm_provider=llm)
automation = AutomationController(client.memory_service, llm, embedder)

# Configure
policy = AutomationPolicy(user_id="user123", auto_summarization=True)
automation.create_policy(policy)

# Use in application
def add_memory(text, user_id):
    client.add_memory(text=text, user_id=user_id)

    # Check if optimization needed
    if automation.should_run_summarization(user_id):
        result = automation.run_summarization(user_id)
        print(f"Auto-summarized: {result['summaries_created']} summaries")
```

### Pattern 2: SaaS Platform (Background Workers)

```python
# In your Celery tasks file
from celery import Celery
from hippocampai import AutomationController

app = Celery('tasks')
automation = AutomationController(...)  # Shared instance

@app.task
def run_scheduled_optimizations():
    """Periodic task (cron) that runs optimizations."""
    user_ids = automation.policies.keys()

    for user_id in user_ids:
        policy = automation.get_policy(user_id)

        if not policy or not policy.enabled:
            continue

        # Check and run based on policy
        if policy.auto_summarization and automation.should_run_summarization(user_id):
            run_summarization.delay(user_id)

@app.task
def run_summarization(user_id: str):
    """Worker task for summarization."""
    result = automation.run_summarization(user_id, force=True)
    return result

# Start workers
# celery -A tasks worker --beat --loglevel=info
```

### Pattern 3: Hybrid (REST API + Library)

```python
# FastAPI server
from fastapi import FastAPI
from hippocampai import AutomationController, AutomationPolicy

app = FastAPI()
automation = AutomationController(...)  # Shared

@app.post("/api/v1/users/{user_id}/policy")
async def create_policy(user_id: str, policy: AutomationPolicy):
    """Library users call this to configure their policy."""
    policy.user_id = user_id
    automation.create_policy(policy)
    return {"status": "success"}

@app.post("/api/v1/users/{user_id}/optimize/{feature}")
async def trigger_optimization(user_id: str, feature: str):
    """Library users call this to trigger optimization."""
    if feature == "summarization":
        result = automation.run_summarization(user_id, force=True)
    elif feature == "consolidation":
        result = automation.run_consolidation(user_id, force=True)
    return result

@app.get("/api/v1/users/{user_id}/statistics")
async def get_stats(user_id: str):
    """Library users query their automation status."""
    return automation.get_user_statistics(user_id)

# Library user code
import requests

# Configure policy from library
policy = AutomationPolicy(user_id="user123", ...)
requests.post("https://yourapi.com/api/v1/users/user123/policy", json=policy.dict())

# Trigger optimization
requests.post("https://yourapi.com/api/v1/users/user123/optimize/summarization")

# Check status
stats = requests.get("https://yourapi.com/api/v1/users/user123/statistics").json()
```

## 🚀 Deployment Options

### Option 1: Library Only (No SaaS)

```python
# Just use AutomationController directly
automation = AutomationController(memory_service, llm, embedder)

# Run optimizations in your application
result = automation.run_summarization("user123")
```

**Pros**: Simple, no infrastructure needed
**Cons**: No background processing, runs in your app

### Option 2: SaaS with Celery

```python
# Set up Celery workers that read policies
# Workers run 24/7 in background
# Library users configure policies

# celery -A tasks worker --beat
```

**Pros**: True background processing, scalable
**Cons**: Requires Redis/RabbitMQ, more infrastructure

### Option 3: Hybrid (Best of Both)

```python
# Library users run some things immediately
result = automation.run_health_check("user123")

# SaaS handles heavy operations in background
task_manager.submit_task("user123", "summarization", async_execution=True)
```

**Pros**: Flexible, best performance
**Cons**: More complex setup

## 📊 Benefits Summary

### For Library Users:
✅ **Full programmatic control** over all features
✅ **Configure once**, works in library and SaaS
✅ **No vendor lock-in** - works offline or with any backend
✅ **Flexible execution** - immediate or background
✅ **Policy-based** - set it and forget it

### For SaaS Platforms:
✅ **Unified control plane** - same API everywhere
✅ **Backend agnostic** - Celery, RQ, or custom
✅ **Scalable** - distribute across workers
✅ **Observable** - full task tracking
✅ **User-controllable** - users configure their own policies

### For Everyone:
✅ **Same code** works in library and SaaS
✅ **No duplication** - single source of truth
✅ **Easy testing** - inline mode for tests
✅ **Gradual adoption** - start simple, scale up
✅ **Zero breaking changes** - opt-in features

## 🎯 Quick Start

```bash
# 1. Install
pip install -e .

# 2. Run example
python example_saas_control.py

# 3. Read guides
cat SAAS_INTEGRATION_GUIDE.md
cat NEW_FEATURES_GUIDE.md

# 4. Try advanced chat
python chat_advanced.py
```

## 📖 Documentation

- **`SAAS_INTEGRATION_GUIDE.md`** - Complete integration guide
- **`NEW_FEATURES_GUIDE.md`** - All features documentation
- **`example_saas_control.py`** - Runnable example
- **`chat_advanced.py`** - Interactive demo

## 🎉 Conclusion

HippocampAI now provides **true unified control** where:
- Library users configure automation **once**
- Same configuration works **everywhere**
- Choose execution model: **immediate** or **background**
- **Zero vendor lock-in** - works with any infrastructure

**One API. Two modes. Infinite flexibility.**
