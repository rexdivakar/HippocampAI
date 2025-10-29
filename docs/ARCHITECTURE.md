# HippocampAI Architecture Guide

## Overview

HippocampAI is a **unified codebase** with a **unified library interface** that supports two connection modes:

1. **Local Mode** - Direct connection to Qdrant/Redis/Ollama
2. **Remote Mode** - HTTP connection to SaaS API server

**Key Innovation**: The same Python library (`UnifiedMemoryClient`) works with both modes. You simply choose your backend connection at initialization!

```python
# Local mode - direct connection
from hippocampai import UnifiedMemoryClient
client = UnifiedMemoryClient(mode="local")

# Remote mode - via API
from hippocampai import UnifiedMemoryClient
client = UnifiedMemoryClient(mode="remote", api_url="http://localhost:8000")

# Either way, same API!
memory = client.remember("text", user_id="user123")
results = client.recall("query", user_id="user123")
```

---

## The Unified Approach

### Traditional Architecture (What We DON'T Do)

```
❌ Separate codebases requiring different learning curves:

Library Users              API Users
├── Learn MemoryClient     ├── Learn HTTP endpoints
├── Python-only            ├── Any language
└── Different API          └── Different API
```

### Our Architecture (What We DO)

```
✅ Single library, multiple backends:

All Users
├── Learn UnifiedMemoryClient (same API!)
├── Choose backend: local or remote
└── Switch between modes with ONE parameter

Local Mode                 Remote Mode
├── mode="local"           ├── mode="remote"
├── Direct connection      ├── HTTP connection
├── 5-15ms latency         ├── 20-50ms latency
└── Python only            └── Python + any language via API
```

### Benefits of This Approach

1. **Consistent API**: Same `remember()`, `recall()`, etc. regardless of mode
2. **Easy Migration**: Switch from local to remote without code changes
3. **Flexible Deployment**: Start local, scale to remote when needed
4. **Best of Both Worlds**: Use local for internal services, remote for external clients
5. **No Vendor Lock-in**: Your code works with any backend

---

## Architecture Layers

```
┌─────────────────────────────────────────────────────────────────┐
│                  Application Layer (Your Code)                   │
│                                                                   │
│              from hippocampai import UnifiedMemoryClient         │
│                                                                   │
│     client = UnifiedMemoryClient(mode="local" | "remote")       │
│                                                                   │
│          memory = client.remember(...)  # Same API!              │
│          results = client.recall(...)   # Same API!              │
└───────────────────────────┬───────────────────────────────────────┘
                            │
                ┌───────────┴───────────┐
                │                       │
                ▼                       ▼
┌──────────────────────────┐  ┌──────────────────────────┐
│    LOCAL MODE BACKEND    │  │   REMOTE MODE BACKEND    │
│                          │  │                          │
│  UnifiedMemoryClient     │  │  UnifiedMemoryClient     │
│  ↓                       │  │  ↓                       │
│  LocalBackend            │  │  RemoteBackend           │
│  (client.py)             │  │  (HTTP client)           │
│  ↓                       │  │  ↓                       │
│  Direct calls to:        │  │  HTTP POST/GET to:       │
│  • Qdrant               │  │  • localhost:8000/v1/*   │
│  • Redis                │  │     ↓                    │
│  • Ollama               │  │     FastAPI Server       │
│  • MemoryService        │  │     ↓                    │
│                          │  │     MemoryService        │
│                          │  │     ↓                    │
│                          │  │  • Qdrant                │
│                          │  │  • Redis                 │
│                          │  │  • Ollama                │
└──────────┬───────────────┘  └──────────┬───────────────┘
           │                             │
           └─────────────┬───────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Service Layer                               │
│                 (Shared by Both Modes)                           │
│  ┌──────────────────────────────────────────────────────┐       │
│  │  MemoryManagementService                             │       │
│  │  (services/memory_service.py)                        │       │
│  │                                                       │       │
│  │  • create_memory()                                   │       │
│  │  • recall_memories()                                 │       │
│  │  • update_memory()                                   │       │
│  │  • deduplicate_memories()                            │       │
│  │  • consolidate_memories()                            │       │
│  └───────────────────────┬──────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Pipeline    │    │  Retrieval   │    │   Storage    │
│  Layer       │    │  Layer       │    │   Layer      │
├──────────────┤    ├──────────────┤    ├──────────────┤
│ • Fact       │    │ • Hybrid     │    │ • Qdrant     │
│   Extraction │    │   Search     │    │   Store      │
│ • Entity     │    │ • Reranking  │    │ • Redis      │
│   Recognition│    │ • BM25       │    │   Cache      │
│ • Relations  │    │ • RRF        │    │ • Vector     │
│ • Clustering │    │ • Scoring    │    │   Index      │
└──────────────┘    └──────────────┘    └──────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Qdrant     │    │    Redis     │    │     LLM      │
│  Vector DB   │    │    Cache     │    │   Provider   │
└──────────────┘    └──────────────┘    └──────────────┘
```

---

## Mode 1: Python Library

### Installation

```bash
# From source (development)
cd HippocampAI
pip install -e .

# From PyPI (when published)
pip install hippocampai
```

### Project Structure

```
your-project/
├── main.py
├── requirements.txt  # hippocampai==1.0.0
└── .env              # QDRANT_URL, REDIS_URL, etc.
```

### Usage Example

```python
# your-project/main.py
from hippocampai import MemoryClient

# Initialize client (connects to Qdrant, Redis directly)
client = MemoryClient()

# Use the library
memory = client.remember(
    text="User prefers dark mode",
    user_id="user123"
)

results = client.recall(
    query="UI preferences",
    user_id="user123"
)

print(results[0].memory.text)
# Output: User prefers dark mode
```

### How It Works

```
Your Application Process
├── main.py
│   ├── import hippocampai
│   └── client = MemoryClient()
│       └── Directly accesses:
│           ├── Qdrant (localhost:6333)
│           ├── Redis (localhost:6379)
│           └── Ollama (localhost:11434)
```

**Advantages:**

- ✅ No HTTP overhead
- ✅ Maximum performance
- ✅ Simple deployment (single process)
- ✅ Full control over configuration
- ✅ Can customize any component

**Disadvantages:**

- ❌ Python-only
- ❌ Runs in your application process
- ❌ No multi-language support

---

## Mode 2: REST API / SaaS

### Deployment

```bash
# Start the FastAPI server
uvicorn hippocampai.api.async_app:app --host 0.0.0.0 --port 8000

# Or with Docker
docker-compose up -d
```

### Server Architecture

```
FastAPI Server Process (Port 8000)
├── hippocampai.api.async_app
│   ├── POST /v1/memories
│   ├── POST /v1/memories/recall
│   ├── GET /v1/memories
│   └── Routes call →
│       └── MemoryManagementService
│           └── Same service layer as Mode 1
│               ├── Qdrant (localhost:6333)
│               ├── Redis (localhost:6379)
│               └── Ollama (localhost:11434)
```

### Usage from Any Language

**Python:**

```python
import requests

response = requests.post(
    "http://localhost:8000/v1/memories",
    json={
        "text": "User prefers dark mode",
        "user_id": "user123"
    }
)
memory = response.json()
```

**JavaScript:**

```javascript
const response = await fetch('http://localhost:8000/v1/memories', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
        text: 'User prefers dark mode',
        user_id: 'user123'
    })
});
const memory = await response.json();
```

**cURL:**

```bash
curl -X POST http://localhost:8000/v1/memories \
  -H "Content-Type: application/json" \
  -d '{"text": "User prefers dark mode", "user_id": "user123"}'
```

**Go:**

```go
import (
    "bytes"
    "encoding/json"
    "net/http"
)

payload := map[string]interface{}{
    "text": "User prefers dark mode",
    "user_id": "user123",
}
body, _ := json.Marshal(payload)
resp, _ := http.Post(
    "http://localhost:8000/v1/memories",
    "application/json",
    bytes.NewBuffer(body),
)
```

**Advantages:**

- ✅ Multi-language support
- ✅ Microservices architecture
- ✅ Centralized deployment
- ✅ Easy horizontal scaling
- ✅ True SaaS multi-tenancy

**Disadvantages:**

- ❌ HTTP overhead
- ❌ Network latency
- ❌ Additional deployment complexity

---

## Hybrid Mode: Best of Both Worlds

You can use **both modes** in the same deployment:

```
┌─────────────────────────────────────┐
│  Internal Python Services           │
│  ├── Use Mode 1 (Library)           │
│  │   └── Direct import              │
│  └── Max performance                │
└──────────────┬──────────────────────┘
               │
               ├── Shared Qdrant/Redis
               │
┌──────────────▼──────────────────────┐
│  External Clients                   │
│  ├── Use Mode 2 (REST API)          │
│  │   └── HTTP requests              │
│  └── Multi-language support         │
└─────────────────────────────────────┘
```

**Example:**

```python
# Internal Python service (high performance)
from hippocampai import MemoryClient
client = MemoryClient()
client.remember(...)  # Direct library call

# External web app (JavaScript)
fetch('http://api.hippocampai.com/v1/memories', {...})  # HTTP API
```

---

## Code Organization

### Core Library (src/hippocampai/)

```
src/hippocampai/
├── __init__.py              # Exports MemoryClient
├── client.py                # Mode 1: Python library interface
├── api/
│   ├── async_app.py         # Mode 2: FastAPI application
│   ├── deps.py              # Dependency injection
│   └── intelligence_routes.py
├── services/
│   └── memory_service.py    # Shared business logic
├── models/
│   ├── memory.py            # Data models
│   └── retrieval.py
├── pipeline/
│   ├── fact_extraction.py   # Intelligence features
│   ├── entity_recognition.py
│   └── consolidate.py
├── storage/
│   ├── redis_store.py       # Storage adapters
│   └── qdrant_store.py
└── adapters/
    ├── provider_ollama.py   # LLM adapters
    ├── provider_openai.py
    └── provider_groq.py
```

### Key Point: Shared Services

Both modes use the **same** `MemoryManagementService`:

```python
# Mode 1 (Library)
from hippocampai import MemoryClient
client = MemoryClient()
# Internally creates: MemoryManagementService(...)

# Mode 2 (API)
# FastAPI route handler
@app.post("/v1/memories")
async def create_memory(request: CreateMemoryRequest):
    service = get_memory_service()  # Same MemoryManagementService
    return await service.create_memory(...)
```

---

## Deployment Patterns

### Pattern 1: Library-Only (Embedded)

```
┌─────────────────────────────┐
│   Your Application          │
│   ├── app.py                │
│   │   └── import hippocampai│
│   └── Dependencies:         │
│       ├── Qdrant (Docker)   │
│       ├── Redis (Docker)    │
│       └── Ollama (Docker)   │
└─────────────────────────────┘
```

**Use case:** Single Python application, maximum performance

### Pattern 2: API-Only (Microservice)

```
┌─────────────────────────────┐
│   HippocampAI Server        │
│   ├── FastAPI (Port 8000)   │
│   └── Dependencies:         │
│       ├── Qdrant            │
│       ├── Redis             │
│       └── Ollama            │
└──────────┬──────────────────┘
           │
    ┌──────┴──────┬──────────┐
    ▼             ▼          ▼
┌────────┐  ┌────────┐  ┌────────┐
│Web App │  │Mobile  │  │ Go API │
│(JS)    │  │(Swift) │  │ Server │
└────────┘  └────────┘  └────────┘
```

**Use case:** Multi-language clients, centralized SaaS

### Pattern 3: Hybrid (Best Practice)

```
┌─────────────────────────────┐
│   Internal Services         │
│   ├── Python Service 1      │
│   │   └── Direct import     │
│   └── Python Service 2      │
│       └── Direct import     │
└──────────┬──────────────────┘
           │ Same DB
┌──────────▼──────────────────┐
│   HippocampAI API Server    │
│   └── For external clients  │
└──────────┬──────────────────┘
           │
    ┌──────┴──────┐
    ▼             ▼
┌────────┐  ┌────────┐
│Web App │  │Partner │
│        │  │API     │
└────────┘  └────────┘
```

**Use case:** Mix of internal Python services + external clients

---

## Configuration

Both modes share the same configuration:

```bash
# .env file (used by both modes)
QDRANT_URL=http://localhost:6333
REDIS_URL=redis://localhost:6379
LLM_PROVIDER=ollama
LLM_BASE_URL=http://localhost:11434
```

**Mode 1 (Library):**

```python
from hippocampai import MemoryClient

# Uses .env automatically
client = MemoryClient()

# Or explicit configuration
client = MemoryClient(
    qdrant_url="http://custom-qdrant:6333",
    redis_url="redis://custom-redis:6379"
)
```

**Mode 2 (API):**

```bash
# Server reads from .env on startup
uvicorn hippocampai.api.async_app:app

# Or via environment variables
QDRANT_URL=http://prod-qdrant:6333 \
REDIS_URL=redis://prod-redis:6379 \
uvicorn hippocampai.api.async_app:app
```

---

## Installation Paths

### Development (Editable Install)

```bash
git clone https://github.com/rexdivakar/HippocampAI.git
cd HippocampAI
pip install -e .
```

- Code lives in `src/hippocampai/`
- Changes immediately reflected
- Can use both Mode 1 and Mode 2

### Production (PyPI Install - Future)

```bash
pip install hippocampai
```

- Package installed in `site-packages/`
- Fixed version
- Can use Mode 1 (library)
- For Mode 2, need to run server separately

---

## Performance Comparison

| Aspect | Mode 1 (Library) | Mode 2 (API) |
|--------|------------------|--------------|
| Latency | ~5-15ms | ~20-50ms (+ network) |
| Throughput | ~100-200 ops/sec | ~50-100 ops/sec |
| Language Support | Python only | Any language |
| Deployment | Single process | Separate server |
| Scaling | Vertical only | Horizontal + Vertical |
| Multi-tenancy | Application-level | Built-in |

---

## When to Use Which Mode?

### Use Mode 1 (Library) When

- ✅ You're building a Python application
- ✅ Performance is critical
- ✅ You want maximum control
- ✅ Single-tenant deployment
- ✅ Embedding in existing app

### Use Mode 2 (API) When

- ✅ Multi-language clients
- ✅ Microservices architecture
- ✅ True SaaS deployment
- ✅ Multi-tenant requirements
- ✅ Centralized memory service

### Use Hybrid When

- ✅ Internal + external clients
- ✅ Need both performance and flexibility
- ✅ Gradual migration
- ✅ Enterprise deployment

---

## Summary

**Key Takeaway:** HippocampAI is **ONE codebase** with **TWO interfaces**:

1. **MemoryClient** (Python library) - Direct import
2. **FastAPI App** (REST API) - HTTP interface

Both use the **same underlying services** - there is no external dependency. You choose the consumption mode based on your needs.

```
src/hippocampai/  ← This IS the product
├── client.py     ← Interface for Mode 1
├── api/          ← Interface for Mode 2
└── services/     ← Shared by both modes
```

**Not like this:**

```
❌ hippocampai-server (depends on) → hippocampai-library
```

**But like this:**

```
✅ hippocampai (single codebase)
   ├── Can be used as library
   └── Can be used as API server
```
