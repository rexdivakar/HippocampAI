# UnifiedMemoryClient - Complete Usage Guide

## Overview

The `UnifiedMemoryClient` is a single Python interface that works with both local and remote backends. This guide covers everything you need to know to use it effectively.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/rexdivakar/HippocampAI.git
cd HippocampAI

# Install with all dependencies
pip install -e .
```

---

## Prerequisites

### For Local Mode

Start the required services:

```bash
# Using docker-compose
docker-compose up -d

# Pull Ollama model
docker exec -it $(docker ps -q -f name=ollama) ollama pull llama3.2:3b

# Verify services
curl http://localhost:6333/healthz  # Qdrant
redis-cli ping                      # Redis
curl http://localhost:11434/api/tags # Ollama
```

### For Remote Mode

Start the API server:

```bash
# Start the FastAPI server
uvicorn hippocampai.api.async_app:app --host 0.0.0.0 --port 8000

# Verify server
curl http://localhost:8000/health
```

---

## Quick Start

### Local Mode (Direct Connection)

```python
from hippocampai import UnifiedMemoryClient

# Initialize in local mode
client = UnifiedMemoryClient(mode="local")

# Store a memory
memory = client.remember(
    text="User prefers dark mode",
    user_id="user123"
)

# Retrieve memories
results = client.recall(
    query="user preferences",
    user_id="user123"
)

print(f"Found: {results[0].memory.text}")
```

### Remote Mode (API Connection)

```python
from hippocampai import UnifiedMemoryClient

# Initialize in remote mode
client = UnifiedMemoryClient(
    mode="remote",
    api_url="http://localhost:8000"
)

# Same API as local mode!
memory = client.remember(
    text="User prefers dark mode",
    user_id="user123"
)

results = client.recall(
    query="user preferences",
    user_id="user123"
)

print(f"Found: {results[0].memory.text}")
```

---

## Connecting to Local Resources (Local Mode)

### Overview

When using `mode="local"`, UnifiedMemoryClient connects directly to your local services:

- **Qdrant** - Vector database for semantic search
- **Redis** - Cache for fast lookups
- **Ollama** - Local LLM for embeddings and intelligence features

### Method 1: Using .env File (Recommended)

Create a `.env` file in your project root:

```bash
# .env
QDRANT_URL=http://localhost:6333
REDIS_URL=redis://localhost:6379
LLM_PROVIDER=ollama
LLM_BASE_URL=http://localhost:11434
LLM_MODEL=llama3.2:3b
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

Then initialize with defaults:

```python
from hippocampai import UnifiedMemoryClient

# Automatically reads from .env
client = UnifiedMemoryClient(mode="local")

# Now connected to all local services!
memory = client.remember("test", user_id="user123")
```

### Method 2: Pass Configuration Directly

```python
from hippocampai import UnifiedMemoryClient

client = UnifiedMemoryClient(
    mode="local",

    # Qdrant configuration
    qdrant_url="http://localhost:6333",
    collection_facts="my_facts",
    collection_prefs="my_prefs",

    # Embedding model
    embed_model="all-MiniLM-L6-v2",

    # LLM configuration
    llm_provider="ollama",
    llm_model="llama3.2:3b",

    # HNSW optimization
    hnsw_M=48,
    ef_construction=256,
    ef_search=128
)
```

**Note:** Redis configuration comes from the `REDIS_URL` environment variable.

### Method 3: Environment Variables

```bash
# Set before running script
export QDRANT_URL=http://localhost:6333
export REDIS_URL=redis://localhost:6379
export LLM_PROVIDER=ollama
export LLM_BASE_URL=http://localhost:11434

# Run your script
python your_script.py
```

### Connecting to Services on Different Hosts

```python
# Services on different servers
client = UnifiedMemoryClient(
    mode="local",
    qdrant_url="http://qdrant-server:6333",  # Remote Qdrant
)

# Set Redis via environment
# REDIS_URL=redis://redis-server:6379
```

### Verifying Local Connections

```python
from hippocampai import UnifiedMemoryClient
import httpx

def verify_connections():
    """Verify all services are accessible."""
    # Check Qdrant
    try:
        response = httpx.get("http://localhost:6333/healthz", timeout=5)
        print(f"✓ Qdrant: {response.json()}")
    except Exception as e:
        print(f"✗ Qdrant not running: {e}")
        print("  Start: docker run -p 6333:6333 qdrant/qdrant")

    # Check Redis
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379)
        r.ping()
        print("✓ Redis: Connected")
    except Exception as e:
        print(f"✗ Redis not running: {e}")
        print("  Start: docker run -p 6379:6379 redis:alpine")

    # Check Ollama
    try:
        response = httpx.get("http://localhost:11434/api/tags", timeout=5)
        print(f"✓ Ollama: Connected")
    except Exception as e:
        print(f"✗ Ollama not running: {e}")
        print("  Start: docker run -p 11434:11434 ollama/ollama")

# Run verification
verify_connections()

# Initialize client (will show helpful errors if connections fail)
try:
    client = UnifiedMemoryClient(mode="local")
    print("✓ UnifiedMemoryClient initialized successfully!")
except ConnectionError as e:
    print(f"Connection failed: {e}")
```

### Error Handling

The client will automatically detect connection issues and provide helpful error messages:

```python
from hippocampai import UnifiedMemoryClient

try:
    client = UnifiedMemoryClient(mode="local")
except ConnectionError as e:
    # Will show helpful messages like:
    # "Qdrant not running: docker run -p 6333:6333 qdrant/qdrant"
    # "Redis not running: docker run -p 6379:6379 redis:alpine"
    # "Missing .env file: Create .env with QDRANT_URL, etc."
    print(f"Failed to connect: {e}")
except ImportError as e:
    # Dependencies not installed
    print(f"Installation error: {e}")
```

---

## Complete API Reference

### Initialization

#### Local Mode

```python
from hippocampai import UnifiedMemoryClient

# Basic (uses defaults from .env)
client = UnifiedMemoryClient(mode="local")

# Custom configuration
client = UnifiedMemoryClient(
    mode="local",
    qdrant_url="http://localhost:6333",
    collection_facts="my_facts",
    collection_prefs="my_prefs",
    embed_model="all-MiniLM-L6-v2",
    llm_provider="ollama",
    llm_model="llama3.2:3b",
    hnsw_M=48,
    ef_construction=256,
    ef_search=128
)
```

#### Remote Mode

```python
from hippocampai import UnifiedMemoryClient

# Basic
client = UnifiedMemoryClient(
    mode="remote",
    api_url="http://localhost:8000"
)

# With authentication
client = UnifiedMemoryClient(
    mode="remote",
    api_url="https://api.hippocampai.com",
    api_key="your-api-key-here",
    timeout=30  # seconds
)
```

---

### Core Methods

#### remember() - Store a Memory

```python
from datetime import datetime, timedelta, timezone

memory = client.remember(
    text="User completed Python course on 2024-01-15",
    user_id="user123",
    session_id="session_001",  # Optional
    metadata={"course": "python", "level": "advanced"},  # Optional
    tags=["education", "programming"],  # Optional
    importance=0.8,  # 0.0-1.0, optional
    expires_at=datetime.now(timezone.utc) + timedelta(days=30),  # Optional
    extract_entities=True,  # Extract people, places, etc.
    extract_facts=True,  # Extract factual statements
    extract_relationships=True  # Extract entity relationships
)

print(f"Created memory: {memory.id}")
print(f"Text: {memory.text}")
print(f"Entities: {memory.entities}")
print(f"Facts: {memory.facts}")
```

#### recall() - Semantic Search

```python
results = client.recall(
    query="What courses did the user complete?",
    user_id="user123",
    session_id="session_001",  # Optional filter
    limit=10,  # Number of results
    filters={"tags": ["education"]},  # Optional filters
    min_score=0.5  # Minimum relevance score
)

for result in results:
    print(f"Score: {result.score:.3f}")
    print(f"Memory: {result.memory.text}")
    print(f"Tags: {result.memory.tags}")
    print(f"Created: {result.memory.created_at}")
    print("---")
```

#### get_memory() - Get by ID

```python
memory = client.get_memory(memory_id="mem_123")

if memory:
    print(f"Found: {memory.text}")
else:
    print("Memory not found")
```

#### get_memories() - List All Memories

```python
from datetime import datetime, timedelta, timezone

# Get all memories for a user
memories = client.get_memories(user_id="user123")

# With filters
memories = client.get_memories(
    user_id="user123",
    session_id="session_001",  # Optional
    limit=100,  # Max results
    filters={"tags": ["work"]},  # Filter by tags
    min_importance=0.7,  # Filter by importance
    after=datetime.now(timezone.utc) - timedelta(days=7),  # Created after
    before=datetime.now(timezone.utc)  # Created before
)

print(f"Found {len(memories)} memories")
for memory in memories:
    print(f"- {memory.text}")
```

#### update_memory() - Update Existing Memory

```python
updated = client.update_memory(
    memory_id="mem_123",
    text="Updated text",  # Optional
    metadata={"updated": True},  # Optional
    tags=["new", "tags"],  # Optional
    importance=0.9,  # Optional
    expires_at=None  # Optional (None = no expiration)
)

if updated:
    print(f"Updated: {updated.text}")
else:
    print("Memory not found")
```

#### delete_memory() - Delete a Memory

```python
success = client.delete_memory(memory_id="mem_123")

if success:
    print("Memory deleted")
else:
    print("Memory not found")
```

---

### Batch Operations

#### batch_remember() - Store Multiple Memories

```python
memories_data = [
    {
        "text": "User likes Python",
        "user_id": "user123",
        "tags": ["preferences"]
    },
    {
        "text": "User completed React course",
        "user_id": "user123",
        "tags": ["education"]
    },
    {
        "text": "User works at Google",
        "user_id": "user123",
        "tags": ["work"]
    }
]

memories = client.batch_remember(memories_data)
print(f"Created {len(memories)} memories")
```

#### batch_get_memories() - Get Multiple by IDs

```python
memory_ids = ["mem_1", "mem_2", "mem_3"]
memories = client.batch_get_memories(memory_ids)

print(f"Retrieved {len(memories)} memories")
```

#### batch_delete_memories() - Delete Multiple

```python
memory_ids = ["mem_1", "mem_2", "mem_3"]
success = client.batch_delete_memories(memory_ids)

if success:
    print("All memories deleted")
```

---

### Advanced Features

#### consolidate_memories() - Merge Similar Memories

```python
# Store related memories
client.remember("User likes coffee", user_id="user123")
client.remember("User drinks coffee daily", user_id="user123")
client.remember("User prefers espresso", user_id="user123")

# Consolidate them
consolidated = client.consolidate_memories(
    user_id="user123",
    session_id="session_001"  # Optional
)

for group in consolidated:
    print(f"Summary: {group['summary']}")
    print(f"Consolidated {len(group['memory_ids'])} memories")
    print("---")
```

#### cleanup_expired_memories() - Remove Expired

```python
deleted_count = client.cleanup_expired_memories()
print(f"Deleted {deleted_count} expired memories")
```

#### get_memory_analytics() - Get Statistics

```python
analytics = client.get_memory_analytics(user_id="user123")

print(f"Total memories: {analytics['total_memories']}")
print(f"Average importance: {analytics['avg_importance']}")
print(f"Top tags: {analytics['top_tags']}")
print(f"Top entities: {analytics['top_entities']}")
print(f"Memory types: {analytics['memory_types']}")
```

---

## Complete Examples

### Example 1: Personal Assistant

```python
from hippocampai import UnifiedMemoryClient
from datetime import datetime, timezone

# Initialize
client = UnifiedMemoryClient(mode="local")
user_id = "user_alice"

# Store user information
client.remember(
    "Alice prefers morning meetings",
    user_id=user_id,
    tags=["preferences", "schedule"]
)

client.remember(
    "Alice is allergic to peanuts",
    user_id=user_id,
    tags=["health", "allergies"],
    importance=0.9
)

client.remember(
    "Alice working on ML project deadline March 15",
    user_id=user_id,
    tags=["work", "projects"],
    extract_entities=True
)

# Search for relevant memories
results = client.recall(
    "What are Alice's food restrictions?",
    user_id=user_id,
    limit=5
)

for result in results:
    print(f"{result.memory.text} (confidence: {result.score:.2f})")
```

### Example 2: Customer Support Bot

```python
from hippocampai import UnifiedMemoryClient

# Initialize
client = UnifiedMemoryClient(
    mode="remote",
    api_url="http://localhost:8000"
)

def handle_customer_interaction(customer_id: str, message: str):
    # Store interaction
    client.remember(
        text=f"Customer said: {message}",
        user_id=customer_id,
        tags=["interaction"],
        extract_entities=True,
        extract_facts=True
    )

    # Get relevant context
    context = client.recall(
        query=message,
        user_id=customer_id,
        limit=5
    )

    # Use context to inform response
    print("Relevant context:")
    for result in context:
        print(f"- {result.memory.text}")

    return context

# Usage
handle_customer_interaction(
    "customer_123",
    "I'm having issues with my last order"
)
```

### Example 3: Learning Management System

```python
from hippocampai import UnifiedMemoryClient
from datetime import datetime, timezone

client = UnifiedMemoryClient(mode="local")
student_id = "student_123"

# Track learning progress
courses = [
    ("Python Basics", 0.9, "completed"),
    ("Advanced Python", 0.7, "in_progress"),
    ("Web Development", 0.5, "started")
]

for course, score, status in courses:
    client.remember(
        f"Student {status} {course} with score {score}",
        user_id=student_id,
        tags=["education", "progress", status],
        importance=score,
        metadata={"course": course, "score": score, "status": status}
    )

# Get learning analytics
analytics = client.get_memory_analytics(user_id=student_id)
print(f"Total courses: {analytics['total_memories']}")
print(f"Average score: {analytics['avg_importance']:.2f}")

# Find areas needing improvement
weak_areas = client.get_memories(
    user_id=student_id,
    filters={"tags": ["in_progress"]},
    min_importance=0.0
)

print("\nAreas needing focus:")
for memory in weak_areas:
    if memory.importance < 0.7:
        print(f"- {memory.text}")
```

### Example 4: Configuration-Based Mode Switching

```python
import os
from hippocampai import UnifiedMemoryClient

# Configuration
MODE = os.getenv("HIPPOCAMP_MODE", "local")
API_URL = os.getenv("HIPPOCAMP_API_URL", "http://localhost:8000")
API_KEY = os.getenv("HIPPOCAMP_API_KEY")

# Initialize based on environment
if MODE == "remote":
    client = UnifiedMemoryClient(
        mode="remote",
        api_url=API_URL,
        api_key=API_KEY
    )
    print(f"Connected to API: {API_URL}")
else:
    client = UnifiedMemoryClient(mode="local")
    print("Using local mode")

# Same code works regardless of mode!
memory = client.remember(
    "Test memory",
    user_id="test_user"
)

print(f"Created: {memory.id}")
```

### Example 5: Hybrid Deployment

```python
from hippocampai import UnifiedMemoryClient

# Internal microservice (uses local mode for speed)
class InternalService:
    def __init__(self):
        self.client = UnifiedMemoryClient(mode="local")

    def process_data(self, data: str, user_id: str):
        # Fast, direct connection
        return self.client.remember(data, user_id=user_id)

# External API (uses remote mode for flexibility)
class ExternalAPI:
    def __init__(self, api_url: str, api_key: str):
        self.client = UnifiedMemoryClient(
            mode="remote",
            api_url=api_url,
            api_key=api_key
        )

    def store_user_data(self, data: str, user_id: str):
        # Remote connection via API
        return self.client.remember(data, user_id=user_id)
```

---

## Environment Variables

Create a `.env` file in your project root:

```bash
# Mode selection
HIPPOCAMP_MODE=local  # or "remote"

# For remote mode
HIPPOCAMP_API_URL=http://localhost:8000
HIPPOCAMP_API_KEY=your-api-key

# For local mode
QDRANT_URL=http://localhost:6333
REDIS_URL=redis://localhost:6379
LLM_PROVIDER=ollama
LLM_BASE_URL=http://localhost:11434
LLM_MODEL=llama3.2:3b
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

---

## Error Handling

```python
from hippocampai import UnifiedMemoryClient
import httpx

client = UnifiedMemoryClient(mode="remote", api_url="http://localhost:8000")

try:
    # Check server health (remote mode only)
    health = client.health_check()
    print(f"Server status: {health['status']}")

    # Store memory
    memory = client.remember("test", user_id="user123")

except httpx.ConnectError:
    print("Cannot connect to server. Is it running?")
    print("Start with: uvicorn hippocampai.api.async_app:app --port 8000")

except httpx.HTTPStatusError as e:
    if e.response.status_code == 404:
        print("Memory not found")
    elif e.response.status_code == 500:
        print("Server error")
    else:
        print(f"HTTP error: {e.response.status_code}")

except Exception as e:
    print(f"Unexpected error: {e}")
```

---

## Performance Tips

### For Local Mode

```python
# Local mode is fastest - use for high-throughput scenarios
client = UnifiedMemoryClient(mode="local")

# Optimize HNSW parameters for your use case
client = UnifiedMemoryClient(
    mode="local",
    hnsw_M=48,  # Higher = better recall, slower indexing
    ef_construction=256,  # Higher = better quality, slower indexing
    ef_search=128  # Higher = better recall, slower search
)
```

### For Remote Mode

```python
# Use batch operations when possible
memories_data = [...]  # List of dictionaries
memories = client.batch_remember(memories_data)  # 3-5x faster than individual

# Adjust timeout for slow networks
client = UnifiedMemoryClient(
    mode="remote",
    api_url="http://remote-server:8000",
    timeout=60  # Increase for slow connections
)
```

---

## Troubleshooting

### Issue: "Connection refused" (Local Mode)

```bash
# Check if services are running
docker-compose ps

# Restart services
docker-compose restart

# View logs
docker-compose logs -f
```

### Issue: "Connection refused" (Remote Mode)

```bash
# Check if API server is running
curl http://localhost:8000/health

# Start API server
uvicorn hippocampai.api.async_app:app --port 8000

# Check server logs
# Look for any startup errors
```

### Issue: "Model not found" (Local Mode)

```bash
# Pull Ollama model
docker exec -it $(docker ps -q -f name=ollama) ollama pull llama3.2:3b

# Verify models
docker exec -it $(docker ps -q -f name=ollama) ollama list
```

### Issue: "Import Error: UnifiedMemoryClient"

```bash
# Reinstall package
pip install -e .

# Verify installation
python -c "from hippocampai import UnifiedMemoryClient; print('OK')"
```

---

## Next Steps

1. ✅ Read the [Architecture Guide](ARCHITECTURE.md) for technical details
2. ✅ Check [Getting Started](GETTING_STARTED.md) for step-by-step setup
3. ✅ Explore [examples/](examples/) for more code samples
4. ✅ See [Unified Client Guide](UNIFIED_CLIENT_GUIDE.md) for concepts

---

## Support

- **GitHub Issues**: <https://github.com/rexdivakar/HippocampAI/issues>
- **Documentation**: <https://docs.hippocampai.com>
- **Discord**: <https://discord.gg/hippocampai>
