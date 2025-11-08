# 🚀 HippocampAI - SaaS + Library Unified Control

> **A unified architecture where library users control ALL features programmatically, and SaaS platforms execute based on those settings.**

## Quick Links

- **📖 [Complete Implementation Summary](./IMPLEMENTATION_COMPLETE.md)** - What was built
- **🏗️ [Architecture Overview](./UNIFIED_ARCHITECTURE.md)** - How it works
- **📚 [Integration Guide](./SAAS_INTEGRATION_GUIDE.md)** - How to use it
- **🎯 [Features Guide](./NEW_FEATURES_GUIDE.md)** - All features explained

---

## 🎯 What Is This?

HippocampAI now provides **unified SaaS + Library control** where:

```python
from hippocampai import AutomationController, AutomationPolicy

# Library users create policies programmatically
policy = AutomationPolicy(
    user_id="user123",
    auto_summarization=True,
    summarization_threshold=500,  # Auto-run when >500 memories
    health_monitoring=True,
)

# Register policy (works for library AND SaaS)
automation.create_policy(policy)

# Run immediately (library mode)
result = automation.run_summarization("user123")

# OR let SaaS workers handle it automatically (background mode)
# Workers read the policy and execute when threshold is met
```

**Result**: One configuration, works everywhere. No vendor lock-in.

---

## ✨ Key Features

### For Library Users:
- ✅ **Full programmatic control** over all automation
- ✅ **Configure once**, works in library and SaaS
- ✅ **Choose execution mode**: immediate or background
- ✅ **Monitor everything**: tasks, health, metrics
- ✅ **Zero lock-in**: works offline or with any backend

### For SaaS Platforms:
- ✅ **Unified control plane** for all users
- ✅ **Scalable**: distribute across workers
- ✅ **User-controlled**: respect user preferences
- ✅ **Observable**: full task tracking
- ✅ **Backend agnostic**: Celery, RQ, or custom

---

## 🚀 Quick Start

### 1. Install

```bash
# Basic (library only)
pip install -e .

# With SaaS support (Celery)
pip install -e ".[saas]"
```

### 2. Run Examples

```bash
# Interactive demo with all features
python chat_advanced.py

# Runnable SaaS control example
python example_saas_control.py

# Original chat (basic features)
python chat.py
```

### 3. Use in Your Code

```python
from hippocampai import (
    MemoryClient,
    AutomationController,
    AutomationPolicy,
    PolicyType,
)
from hippocampai.saas import TaskManager

# Initialize
client = MemoryClient(llm_provider=llm)
automation = AutomationController(
    memory_service=client.memory_service,
    llm=llm,
    embedder=embedder
)

# Create policy
policy = AutomationPolicy(
    user_id="user123",
    policy_type=PolicyType.THRESHOLD,

    # Enable features
    auto_summarization=True,
    auto_consolidation=True,
    health_monitoring=True,

    # Set thresholds
    summarization_threshold=500,
    consolidation_threshold=300,
)

automation.create_policy(policy)

# Use it!
result = automation.run_summarization("user123")
print(f"Summaries created: {result['summaries_created']}")
```

---

## 📦 What's Included

### New Modules

**`src/hippocampai/saas/`**:
- `automation.py` - AutomationController, Policies, Schedules
- `tasks.py` - TaskManager, Background tasks

### New Features

1. **AutomationController** - Central control plane
2. **AutomationPolicy** - User configuration
3. **TaskManager** - Backend-agnostic task queue
4. **Policy-based triggers** - Automatic optimization
5. **Flexible execution** - Immediate or background
6. **Full observability** - Metrics, traces, health reports

### Updated Features

1. **chat_advanced.py** - Shows ALL features
2. **NEW_FEATURES_GUIDE.md** - Section 10 (SaaS deployment)
3. **Package exports** - All automation classes

---

## 🎮 Interactive Demo

### Advanced Chat with ALL Features

```bash
python chat_advanced.py
```

**Available Commands**:

**Basic**:
- `/help` - Show commands
- `/stats` - Memory statistics
- `/memories` - Recent memories
- `/patterns` - Behavioral patterns
- `/search <query>` - Search memories

**Advanced** (NEW):
- `/health` - Generate comprehensive health report
- `/summarize` - Run auto-summarization
- `/consolidate` - Run memory consolidation
- `/compress` - Run advanced compression
- `/metrics` - Show metrics and tracing stats
- `/traces` - Show recent operation traces

---

## 📚 Documentation

### Start Here:
1. **[UNIFIED_ARCHITECTURE.md](./UNIFIED_ARCHITECTURE.md)** - Understand the design
2. **[SAAS_INTEGRATION_GUIDE.md](./SAAS_INTEGRATION_GUIDE.md)** - Learn to integrate
3. **[example_saas_control.py](./example_saas_control.py)** - See it in action

### Deep Dive:
- **[NEW_FEATURES_GUIDE.md](./NEW_FEATURES_GUIDE.md)** - All features explained
- **[IMPLEMENTATION_COMPLETE.md](./IMPLEMENTATION_COMPLETE.md)** - What was built
- **Source Code**: `src/hippocampai/saas/` - Implementation details

---

## 🔧 How It Works

### Architecture

```
┌─────────────────────────────────────────┐
│         LIBRARY USER                     │
│      (Python Application)                │
│                                          │
│  automation.create_policy(policy)        │
│  automation.run_summarization()          │
└──────────────┬───────────────────────────┘
               │
               │ Configure
               ▼
    ┌──────────────────────────┐
    │  AutomationController    │
    │  (Unified Control Plane) │
    └──────┬────────────┬──────┘
           │            │
     ┌─────▼────┐  ┌───▼────────┐
     │ Library  │  │   SaaS     │
     │  Mode    │  │  Workers   │
     └──────────┘  └────────────┘
```

### Execution Modes

**1. Library Mode (Immediate)**:
```python
result = automation.run_summarization("user123")
# Executes now, returns result
```

**2. SaaS Mode (Background)**:
```python
task = task_manager.submit_task("user123", "summarization")
# Queued for background execution
```

**3. Hybrid (Both)**:
```python
# Quick ops: immediate
health = automation.run_health_check("user123")

# Heavy ops: background
task_manager.submit_task("user123", "summarization")
```

---

## 🧪 Testing

### Run All Tests

```bash
# All tests (110+)
python -m pytest tests/ -v

# Specific modules
python -m pytest tests/test_monitoring_tags_storage.py -v  # ✅ 14/14
```

### Verify Installation

```bash
# Check linting
ruff check .  # Should pass

# Check formatting
ruff format .  # All formatted

# Run examples
python example_saas_control.py
python chat_advanced.py
```

---

## 🌐 SaaS Deployment

### Option 1: Celery (Recommended)

**1. Create `celeryconfig.py`**:
```python
broker_url = 'redis://localhost:6379/0'
result_backend = 'redis://localhost:6379/0'

beat_schedule = {
    'run-optimizations': {
        'task': 'tasks.run_scheduled_tasks',
        'schedule': crontab(hour=2, minute=0),  # 2 AM daily
    },
}
```

**2. Create `celery_tasks.py`**:
```python
from celery import Celery
from hippocampai import AutomationController

app = Celery('tasks')
app.config_from_object('celeryconfig')

automation = AutomationController(...)

@app.task
def run_scheduled_tasks():
    """Run for all users with policies."""
    for user_id in automation.policies.keys():
        policy = automation.get_policy(user_id)
        if policy.enabled and automation.should_run_summarization(user_id):
            automation.run_summarization(user_id, force=True)
```

**3. Start Workers**:
```bash
celery -A celery_tasks worker --beat --loglevel=info
```

See **[SAAS_INTEGRATION_GUIDE.md](./SAAS_INTEGRATION_GUIDE.md)** for complete setup.

### Option 2: REST API

```python
from fastapi import FastAPI
from hippocampai import AutomationController, AutomationPolicy

app = FastAPI()
automation = AutomationController(...)

@app.post("/api/v1/users/{user_id}/policy")
async def create_policy(user_id: str, policy: AutomationPolicy):
    automation.create_policy(policy)
    return {"status": "success"}

@app.get("/api/v1/users/{user_id}/statistics")
async def get_stats(user_id: str):
    return automation.get_user_statistics(user_id)
```

**Library users call**:
```python
import requests

# Configure from library
policy = AutomationPolicy(user_id="user123", ...)
requests.post("https://api.com/users/user123/policy", json=policy.dict())

# Check status
stats = requests.get("https://api.com/users/user123/statistics").json()
```

---

## 🎯 Use Cases

### Use Case 1: Startup (Library Only)

**Scenario**: Small app, no infrastructure
**Solution**: Use library mode, run optimizations directly

```python
automation = AutomationController(...)
result = automation.run_summarization("user123")
```

**Pros**: Simple, no infra needed
**Cons**: Runs in your app process

### Use Case 2: SaaS Platform (Background Workers)

**Scenario**: Production SaaS with many users
**Solution**: Deploy Celery workers, users configure policies

```python
# Users configure via API or library
policy = AutomationPolicy(user_id="user123", ...)

# Workers execute automatically based on policies
# celery -A tasks worker --beat
```

**Pros**: Scalable, background processing
**Cons**: More infrastructure (Redis, workers)

### Use Case 3: Hybrid (Best of Both)

**Scenario**: Need both immediate and background
**Solution**: Mix execution modes

```python
# Quick ops: immediate
health = automation.run_health_check("user123")

# Heavy ops: background
task_manager.submit_task("user123", "summarization", async_execution=True)
```

**Pros**: Flexible, optimal performance
**Cons**: More complex logic

---

## 💡 Examples

### Example 1: Threshold-Based Auto-Optimization

```python
policy = AutomationPolicy(
    user_id="user123",
    policy_type=PolicyType.THRESHOLD,
    auto_summarization=True,
    summarization_threshold=500,  # Auto-run at 500 memories
)

automation.create_policy(policy)

# Later in your app...
def add_memory(text, user_id):
    client.add_memory(text=text, user_id=user_id)

    # Check if optimization should run
    if automation.should_run_summarization(user_id):
        print("Threshold reached! Running summarization...")
        result = automation.run_summarization(user_id)
        print(f"Created {result['summaries_created']} summaries")
```

### Example 2: Scheduled Background Tasks

```python
# Policy with schedules (for SaaS cron)
policy = AutomationPolicy(
    user_id="user123",
    auto_summarization=True,
    summarization_schedule=AutomationSchedule(
        enabled=True,
        cron_expression="0 2 * * *",  # 2 AM daily
    ),
)

automation.create_policy(policy)

# Celery beat will read this and schedule the task
# No code needed - workers handle it automatically!
```

### Example 3: REST API Control

```python
# Library user configures remotely
import requests

policy = AutomationPolicy(
    user_id="user123",
    auto_summarization=True,
    summarization_threshold=500,
)

# Send to SaaS API
response = requests.post(
    "https://yourapi.com/api/v1/users/user123/policy",
    json=policy.dict()
)

# Later: check statistics
stats = requests.get(
    "https://yourapi.com/api/v1/users/user123/statistics"
).json()

print(f"Total memories: {stats['total_memories']}")
print(f"Automation enabled: {stats['automation_enabled']}")
```

---

## 🎓 Learn More

### Recommended Reading Order:

1. **This README** - Overview and quick start
2. **[UNIFIED_ARCHITECTURE.md](./UNIFIED_ARCHITECTURE.md)** - Architecture deep dive
3. **[example_saas_control.py](./example_saas_control.py)** - Runnable code
4. **[SAAS_INTEGRATION_GUIDE.md](./SAAS_INTEGRATION_GUIDE.md)** - Production deployment
5. **[NEW_FEATURES_GUIDE.md](./NEW_FEATURES_GUIDE.md)** - All features explained

### Try It Yourself:

```bash
# 1. Run the demo
python example_saas_control.py

# 2. Try interactive chat
python chat_advanced.py

# 3. Read source code
cat src/hippocampai/saas/automation.py
cat src/hippocampai/saas/tasks.py
```

---

## ✅ Status

- ✅ **Implementation**: Complete
- ✅ **Documentation**: Comprehensive (4 guides, 2000+ lines)
- ✅ **Testing**: All tests pass (110+)
- ✅ **Code Quality**: All linting checks pass
- ✅ **Examples**: Runnable demos included
- ✅ **Backward Compatible**: No breaking changes

---

## 🎉 Conclusion

HippocampAI now provides **true unified control** where:

- ✅ Library users configure everything programmatically
- ✅ Same configuration works in library and SaaS
- ✅ No vendor lock-in - works offline or any backend
- ✅ Flexible execution - immediate or background
- ✅ Full observability - metrics, traces, health reports

**One API. Two modes. Infinite flexibility.**

---

## 📞 Support

- **Documentation**: See guides above
- **Examples**: Run `python example_saas_control.py`
- **Interactive**: Try `python chat_advanced.py`
- **Source Code**: Check `src/hippocampai/saas/`

---

**🚀 Ready to use! Start with `python example_saas_control.py`**
