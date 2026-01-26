# HippocampAI Package Structure

This document explains the organization of HippocampAI into separate components for flexibility and ease of use.

## Overview

HippocampAI is organized into two main packages:

```
hippocampai/
├── core/           # Core memory engine (library)
├── platform/       # SaaS platform components
├── models/         # Data models (Memory, Session, Agent)
├── pipeline/       # Processing pipelines
├── retrieval/      # Search and retrieval
├── adapters/       # LLM provider adapters
└── ...             # Other modules
```

## Package Components

### `hippocampai.core` — Core Library

The core library contains everything needed to use HippocampAI as a memory engine in your application. It has minimal dependencies and doesn't require any SaaS infrastructure.

**Requirements:**
- Qdrant (vector database)
- An LLM provider (Ollama, OpenAI, Groq, or Anthropic)

**Key exports:**
```python
from hippocampai.core import (
    # Main clients
    MemoryClient,
    AsyncMemoryClient,
    UnifiedMemoryClient,
    
    # Configuration
    Config,
    get_config,
    
    # Models
    Memory,
    MemoryType,
    Session,
    Agent,
    
    # Features
    SessionManager,
    MultiAgentManager,
    MemoryGraph,
    MemoryVersionControl,
)
```

**Use cases:**
- Embedding memory in your own application
- Building custom AI agents with persistent memory
- Local development and testing
- Lightweight deployments

### `hippocampai.platform` — SaaS Platform

The platform package contains everything needed to run HippocampAI as a SaaS service with full production infrastructure.

**Additional requirements:**
- Redis (caching, Celery broker)
- PostgreSQL (authentication, user management)
- Celery workers (background tasks)

**Key exports:**
```python
from hippocampai.platform import (
    # API Server
    create_app,
    run_api_server,
    
    # Authentication
    AuthService,
    RateLimiter,
    
    # Background Tasks
    AutomationController,
    TaskManager,
    celery_app,
    
    # Configuration
    PlatformConfig,
    get_platform_config,
)
```

**Use cases:**
- Self-hosted SaaS deployment
- Multi-tenant memory service
- Production deployments with monitoring
- API-first architectures

## Import Patterns

### Basic Usage (Recommended)

```python
# Most users should use the main package
from hippocampai import MemoryClient

client = MemoryClient()
memory = client.remember("I love coffee", user_id="alice")
```

### Core Library Only

```python
# For minimal dependencies and embedded use
from hippocampai.core import MemoryClient, Config

config = Config(
    qdrant_url="http://localhost:6333",
    llm_provider="ollama",
    llm_model="llama3"
)
client = MemoryClient(config=config)
```

### SaaS Platform

```python
# For running as a service
from hippocampai.platform import run_api_server, PlatformConfig

config = PlatformConfig()
run_api_server(host="0.0.0.0", port=8000)
```

### Mixed Usage

```python
# Use core for memory operations, platform for automation
from hippocampai.core import MemoryClient
from hippocampai.platform import AutomationController

client = MemoryClient()
automation = AutomationController(client)
automation.start()
```

## Configuration

### Core Configuration

The core library uses `Config` for memory engine settings:

```python
from hippocampai.core import Config, get_config

# Get default config (reads from .env)
config = get_config()

# Or create custom config
config = Config(
    qdrant_url="http://localhost:6333",
    llm_provider="groq",
    llm_model="llama-3.1-8b-instant",
    embed_model="BAAI/bge-small-en-v1.5",
)
```

### Platform Configuration

The platform uses `PlatformConfig` for SaaS-specific settings:

```python
from hippocampai.platform import PlatformConfig, get_platform_config

config = get_platform_config()

# Access platform-specific settings
print(config.redis_url)
print(config.celery_broker_url)
print(config.user_auth_enabled)
```

## Backward Compatibility

The main `hippocampai` package maintains full backward compatibility. All existing imports continue to work:

```python
# These all work as before
from hippocampai import MemoryClient
from hippocampai import Memory, MemoryType
from hippocampai import AutomationController
from hippocampai import Config, get_config
```

## Dependencies

### Core Dependencies (always installed)

```
pydantic>=2.6
pydantic-settings>=2.0
python-dotenv>=1.0
httpx>=0.25
cachetools>=5.3
qdrant-client>=1.7
rank-bm25>=0.2
sentence-transformers>=2.2
tenacity>=8.2
python-json-logger>=2.0
```

### SaaS Dependencies (optional, `pip install hippocampai[saas]`)

```
fastapi>=0.110
uvicorn>=0.24
celery[redis]>=5.3
redis>=5.0
asyncpg>=0.29
bcrypt>=4.0
apscheduler>=3.10
flower>=2.0
```

## Migration Guide

If you're upgrading from an older version:

1. **No changes required** for basic usage — all imports work as before
2. **For cleaner imports**, consider using the subpackages:
   - `from hippocampai.core import ...` for library features
   - `from hippocampai.platform import ...` for SaaS features
3. **For new projects**, we recommend using the subpackages for clarity

## Best Practices

1. **Use `hippocampai.core`** when:
   - Building a library or SDK
   - Embedding memory in an existing application
   - Minimizing dependencies
   - Running in serverless environments

2. **Use `hippocampai.platform`** when:
   - Deploying as a standalone service
   - Need authentication and rate limiting
   - Running background tasks
   - Building a multi-tenant SaaS

3. **Use the main `hippocampai` package** when:
   - Prototyping or experimenting
   - Need access to everything
   - Backward compatibility is important
