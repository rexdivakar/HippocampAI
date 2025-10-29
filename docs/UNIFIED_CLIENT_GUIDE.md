# UnifiedMemoryClient - The Unified Approach

## Overview

HippocampAI introduces `UnifiedMemoryClient` - a **single library interface** that works with **two different backends**:

- **Local Mode**: Direct connection to Qdrant/Redis/Ollama
- **Remote Mode**: HTTP connection to SaaS API

**The Innovation**: Your application code stays the same. You only change ONE parameter to switch between modes!

---

## Why This Matters

### The Problem (Traditional Approach)

Most systems force you to choose:

```
Option A: Use a library        Option B: Use an API
├── Learn library API          ├── Learn REST API
├── Tied to one language       ├── Works with any language
├── Rewrite code to switch     ├── Rewrite code to switch
└── Different deployment       └── Different deployment
```

### The Solution (Our Approach)

```
UnifiedMemoryClient - One Interface, Multiple Backends
├── Learn once, use everywhere
├── Same API for both modes
├── Switch with ONE parameter
└── Flexible deployment options
```

---

## Quick Comparison

```python
# Same import
from hippocampai import UnifiedMemoryClient

# LOCAL MODE - Direct connection (5-15ms latency)
client = UnifiedMemoryClient(mode="local")

# REMOTE MODE - HTTP API (20-50ms latency, multi-language)
client = UnifiedMemoryClient(mode="remote", api_url="http://localhost:8000")

# EITHER WAY - Same API!
memory = client.remember("User prefers dark mode", user_id="user123")
results = client.recall("UI preferences", user_id="user123")
```

---

## Detailed Comparison

| Aspect | Local Mode | Remote Mode |
|--------|------------|-------------|
| **Import** | `from hippocampai import UnifiedMemoryClient` | `from hippocampai import UnifiedMemoryClient` |
| **Init** | `mode="local"` | `mode="remote", api_url="..."` |
| **API** | `client.remember()`, `client.recall()` | `client.remember()`, `client.recall()` |
| **Backend** | Direct Qdrant/Redis/Ollama | HTTP to SaaS API |
| **Latency** | 5-15ms | 20-50ms (+ network) |
| **Language** | Python only | Python + any language via HTTP |
| **Deployment** | Embedded in app | Separate API server |
| **Best For** | Single app, max performance | Microservices, multi-language |

---

## Installation & Setup

### Prerequisites

Both modes require the same services running:

```bash
# Start Qdrant, Redis, Ollama
docker-compose up -d

# Pull Ollama model
docker exec -it $(docker ps -q -f name=ollama) ollama pull llama3.2:3b
```

### Install HippocampAI

```bash
git clone https://github.com/rexdivakar/HippocampAI.git
cd HippocampAI
pip install -e .
```

---

## Using Local Mode

### When to Use

- Building a Python application
- Performance is critical (5-15ms latency)
- Want maximum control
- Single-tenant deployment
- Embedding in existing Python app

### Setup

```python
from hippocampai import UnifiedMemoryClient

# Initialize in local mode (default)
client = UnifiedMemoryClient(mode="local")

# Or with custom configuration
client = UnifiedMemoryClient(
    mode="local",
    qdrant_url="http://localhost:6333",
    llm_provider="ollama",
    llm_model="llama3.2:3b"
)
```

### Example

```python
from hippocampai import UnifiedMemoryClient

# Initialize
client = UnifiedMemoryClient(mode="local")

# Store
memory = client.remember(
    text="User completed Python course",
    user_id="user123",
    tags=["education", "programming"]
)

# Retrieve
results = client.recall(
    query="What courses did the user complete?",
    user_id="user123"
)

for result in results:
    print(f"{result.memory.text} (score: {result.score:.3f})")
```

### Connecting to Local Resources

Local mode connects directly to three services:

**1. Qdrant** (Vector Database)

```python
client = UnifiedMemoryClient(
    mode="local",
    qdrant_url="http://localhost:6333",  # Default
    collection_facts="hippocampai_facts",  # Default
    collection_prefs="hippocampai_prefs"  # Default
)
```

**2. Redis** (Cache)

```bash
# Configure via environment
export REDIS_URL=redis://localhost:6379  # Default
```

**3. Ollama** (LLM & Embeddings)

```python
client = UnifiedMemoryClient(
    mode="local",
    llm_provider="ollama",  # Default
    llm_model="llama3.2:3b",  # Default
)

# Configure base URL via environment
# export LLM_BASE_URL=http://localhost:11434
```

### Configuration Methods

**Method 1: .env File (Recommended)**

```bash
# .env
QDRANT_URL=http://localhost:6333
REDIS_URL=redis://localhost:6379
LLM_PROVIDER=ollama
LLM_BASE_URL=http://localhost:11434
LLM_MODEL=llama3.2:3b
```

**Method 2: Direct Parameters**

```python
client = UnifiedMemoryClient(
    mode="local",
    qdrant_url="http://custom-qdrant:6333",
    llm_provider="ollama",
    llm_model="llama3.2:3b"
)
```

**Method 3: Environment Variables**

```bash
export QDRANT_URL=http://localhost:6333
export REDIS_URL=redis://localhost:6379
python your_script.py
```

### Error Handling

The client provides helpful errors if services aren't running:

```python
try:
    client = UnifiedMemoryClient(mode="local")
except ConnectionError as e:
    # Shows which service failed and how to start it
    print(e)
    # Example output:
    # "Qdrant not running: docker run -p 6333:6333 qdrant/qdrant"
```

---

## Using Remote Mode

### When to Use

- Multi-language clients (JavaScript, Go, etc.)
- Microservices architecture
- True SaaS multi-tenant deployment
- Centralized memory service
- Need to scale horizontally

### Setup

```bash
# Start the API server
uvicorn hippocampai.api.async_app:app --host 0.0.0.0 --port 8000
```

```python
from hippocampai import UnifiedMemoryClient

# Initialize in remote mode
client = UnifiedMemoryClient(
    mode="remote",
    api_url="http://localhost:8000"
)

# Or with authentication
client = UnifiedMemoryClient(
    mode="remote",
    api_url="https://api.hippocampai.com",
    api_key="your-api-key",
    timeout=30
)
```

### Example (Python)

```python
from hippocampai import UnifiedMemoryClient

# Initialize
client = UnifiedMemoryClient(
    mode="remote",
    api_url="http://localhost:8000"
)

# Same API as local mode!
memory = client.remember(
    text="User completed Python course",
    user_id="user123",
    tags=["education", "programming"]
)

results = client.recall(
    query="What courses did the user complete?",
    user_id="user123"
)

for result in results:
    print(f"{result.memory.text} (score: {result.score:.3f})")
```

### Example (JavaScript/Node.js)

Remote mode enables multi-language support:

```javascript
const axios = require('axios');

const client = {
    baseUrl: 'http://localhost:8000',

    async remember(text, userId, options = {}) {
        const response = await axios.post(`${this.baseUrl}/v1/memories`, {
            text, user_id: userId, ...options
        });
        return response.data;
    },

    async recall(query, userId, limit = 10) {
        const response = await axios.post(`${this.baseUrl}/v1/memories/recall`, {
            query, user_id: userId, limit
        });
        return response.data;
    }
};

// Use it
const memory = await client.remember('User completed Python course', 'user123');
const results = await client.recall('What courses did the user complete?', 'user123');
```

---

## Complete Feature Parity

All features work in both modes:

| Feature | Local Mode | Remote Mode |
|---------|------------|-------------|
| Basic CRUD | ✅ | ✅ |
| Semantic Search | ✅ | ✅ |
| Filtering & Tags | ✅ | ✅ |
| Fact Extraction | ✅ | ✅ |
| Entity Recognition | ✅ | ✅ |
| Memory Consolidation | ✅ | ✅ |
| Temporal Queries | ✅ | ✅ |
| Memory Expiration | ✅ | ✅ |
| Batch Operations | ✅ | ✅ |
| Analytics | ✅ | ✅ |
| Duplicate Detection | ✅ | ✅ |

---

## Switching Between Modes

### Environment-Based Configuration

```python
import os
from hippocampai import UnifiedMemoryClient

# Read from environment
mode = os.getenv("HIPPOCAMP_MODE", "local")
api_url = os.getenv("HIPPOCAMP_API_URL", "http://localhost:8000")

if mode == "remote":
    client = UnifiedMemoryClient(mode="remote", api_url=api_url)
else:
    client = UnifiedMemoryClient(mode="local")

# Use same API regardless of mode
memory = client.remember("text", user_id="user123")
```

### Development vs Production

```python
import os
from hippocampai import UnifiedMemoryClient

env = os.getenv("ENVIRONMENT", "development")

if env == "production":
    # Production: Use remote SaaS
    client = UnifiedMemoryClient(
        mode="remote",
        api_url="https://api.hippocampai.com",
        api_key=os.getenv("HIPPOCAMP_API_KEY")
    )
else:
    # Development: Use local
    client = UnifiedMemoryClient(mode="local")
```

### Fallback Strategy

```python
from hippocampai import UnifiedMemoryClient

try:
    # Try remote first
    client = UnifiedMemoryClient(
        mode="remote",
        api_url="http://localhost:8000",
        timeout=5
    )
    client.health_check()  # Verify connection
    print("Using remote mode")
except Exception:
    # Fallback to local
    client = UnifiedMemoryClient(mode="local")
    print("Using local mode (fallback)")
```

---

## Hybrid Deployment

You can use BOTH modes in the same deployment!

```
┌─────────────────────────────┐
│  Internal Python Services   │
│  ├── Service A (local)      │  ← Fast, direct connection
│  └── Service B (local)      │
└──────────┬──────────────────┘
           │
           │ Share same Qdrant/Redis
           │
┌──────────▼──────────────────┐
│  HippocampAI API Server     │
│  └── For external clients   │
└──────────┬──────────────────┘
           │
    ┌──────┴──────┐
    ▼             ▼
┌────────┐  ┌────────┐
│Web App │  │Partner │  ← Multi-language, HTTP
│(JS)    │  │API     │
└────────┘  └────────┘
```

**Example:**

```python
# Internal Python Service A - Use local mode for speed
from hippocampai import UnifiedMemoryClient
client = UnifiedMemoryClient(mode="local")
memory = client.remember(...)  # 5ms latency

# External JavaScript App - Use remote mode via HTTP
fetch('http://api.example.com/v1/memories', {
    method: 'POST',
    body: JSON.stringify({...})
})  // 30ms latency but works from browser
```

---

## Migration Guide

### From Old MemoryClient

If you were using the old `MemoryClient`:

```python
# OLD CODE (still works, backward compatible)
from hippocampai import MemoryClient
client = MemoryClient()

# NEW CODE (recommended)
from hippocampai import UnifiedMemoryClient
client = UnifiedMemoryClient(mode="local")
```

The old `MemoryClient` still works, but `UnifiedMemoryClient` is recommended as it provides mode switching capability.

### From Direct HTTP API

If you were using HTTP API directly:

```python
# OLD CODE (direct HTTP)
import requests
response = requests.post('http://localhost:8000/v1/memories', json={...})

# NEW CODE (use UnifiedMemoryClient)
from hippocampai import UnifiedMemoryClient
client = UnifiedMemoryClient(mode="remote", api_url="http://localhost:8000")
memory = client.remember(...)  # Same result, cleaner API
```

---

## Best Practices

### 1. Use Environment Variables

```python
import os
from hippocampai import UnifiedMemoryClient

mode = os.getenv("HIPPOCAMP_MODE", "local")
client = UnifiedMemoryClient(
    mode=mode,
    api_url=os.getenv("HIPPOCAMP_API_URL") if mode == "remote" else None
)
```

### 2. Implement Health Checks

```python
from hippocampai import UnifiedMemoryClient

client = UnifiedMemoryClient(mode="remote", api_url="http://localhost:8000")

try:
    health = client.health_check()
    print(f"Server status: {health['status']}")
except Exception as e:
    print(f"Health check failed: {e}")
```

### 3. Use Type Hints

```python
from hippocampai import UnifiedMemoryClient, Memory
from typing import List

def store_memories(client: UnifiedMemoryClient, texts: List[str]) -> List[Memory]:
    return [client.remember(text, user_id="user123") for text in texts]
```

### 4. Handle Errors Gracefully

```python
from hippocampai import UnifiedMemoryClient

client = UnifiedMemoryClient(mode="remote", api_url="http://localhost:8000")

try:
    memory = client.remember("text", user_id="user123")
except Exception as e:
    print(f"Failed to store memory: {e}")
    # Implement retry logic or fallback
```

---

## Performance Considerations

### Local Mode

**Pros:**

- Lowest latency (5-15ms)
- No network overhead
- Direct access to storage

**Cons:**

- Python only
- Runs in app process
- No horizontal scaling

**Best for:**

- Real-time applications
- High-throughput scenarios
- Single-server deployments

### Remote Mode

**Pros:**

- Multi-language support
- Horizontal scaling
- Centralized management

**Cons:**

- Network latency (20-50ms)
- HTTP overhead
- Additional server to manage

**Best for:**

- Microservices
- Multi-language teams
- SaaS deployments

---

## Troubleshooting

### "Connection refused" in Local Mode

```bash
# Check if Qdrant is running
curl http://localhost:6333/healthz

# Check if Redis is running
redis-cli ping

# Restart services
docker-compose restart
```

### "Connection refused" in Remote Mode

```bash
# Check if API server is running
curl http://localhost:8000/health

# Start API server
uvicorn hippocampai.api.async_app:app --port 8000
```

### "Import Error: UnifiedMemoryClient"

```bash
# Reinstall
pip install -e .

# Verify
python -c "from hippocampai import UnifiedMemoryClient; print('OK')"
```

---

## Examples

See the `examples/` directory for complete examples:

- `unified_client_local_mode.py` - Local mode usage
- `unified_client_remote_mode.py` - Remote mode usage
- `unified_client_mode_switching.py` - Switching between modes
- `unified_client_configuration.py` - All configuration options

---

## Summary

**UnifiedMemoryClient** provides:

✅ **One API** - Same methods for both modes
✅ **Easy Switching** - Change ONE parameter to switch backends
✅ **Full Features** - All features work in both modes
✅ **Flexible Deployment** - Choose the best mode for your needs
✅ **No Lock-in** - Your code works with any backend

**Get Started:**

```python
from hippocampai import UnifiedMemoryClient

# Local mode (fastest)
client = UnifiedMemoryClient(mode="local")

# Remote mode (multi-language)
client = UnifiedMemoryClient(mode="remote", api_url="http://localhost:8000")

# Same API either way!
memory = client.remember("text", user_id="user123")
results = client.recall("query", user_id="user123")
```

For more details, see:

- [GETTING_STARTED.md](GETTING_STARTED.md) - Complete setup guide
- [ARCHITECTURE.md](ARCHITECTURE.md) - Technical architecture
- [examples/](examples/) - Code examples
