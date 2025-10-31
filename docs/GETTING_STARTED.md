# HippocampAI - Complete Getting Started Guide

This guide walks you through setting up and using HippocampAI with the **UnifiedMemoryClient** - a single library interface that works in both Local and Remote modes.

## The Unified Approach

**Key Concept**: Always use the library (`UnifiedMemoryClient`), just choose your backend:

- **Local Mode**: Direct connection to Qdrant/Redis/Ollama (5-15ms latency)
- **Remote Mode**: HTTP connection to SaaS API (20-50ms latency, multi-language support)

**Same code, different connection mode!**

```python
# Local mode (direct connection)
from hippocampai import UnifiedMemoryClient
client = UnifiedMemoryClient(mode="local")

# Remote mode (via SaaS API)
from hippocampai import UnifiedMemoryClient
client = UnifiedMemoryClient(mode="remote", api_url="http://localhost:8000")

# Either way, same API:
memory = client.remember("User prefers dark mode", user_id="user123")
results = client.recall("UI preferences", user_id="user123")
```

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start](#quick-start)
3. [Local Mode Setup](#local-mode-setup)
4. [Remote Mode Setup](#remote-mode-setup)
5. [Feature Examples](#feature-examples)
6. [Mode Comparison](#mode-comparison)
7. [Troubleshooting](#troubleshooting)

---

## Prerequisites

Before using HippocampAI in either mode, you need these services running:

### 1. Install Docker Desktop

```bash
# macOS
brew install --cask docker

# Linux
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Windows
# Download from https://www.docker.com/products/docker-desktop
```

### 2. Start Required Services

Create a `docker-compose.yml` file:

```yaml
version: '3.8'

services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant_data:/qdrant/storage

  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama

volumes:
  qdrant_data:
  redis_data:
  ollama_data:
```

Start the services:

```bash
docker-compose up -d
```

### 3. Pull Ollama Model

```bash
docker exec -it $(docker ps -q -f name=ollama) ollama pull llama3.2:3b
```

### 4. Verify Services

```bash
# Check Qdrant
curl http://localhost:6333/healthz

# Check Redis
redis-cli ping

# Check Ollama
curl http://localhost:11434/api/tags
```

---

## Quick Start

Get up and running in 5 minutes with either mode!

### Option A: Local Mode (Fastest Setup)

```bash
# 1. Clone and install
git clone https://github.com/rexdivakar/HippocampAI.git
cd HippocampAI
pip install -e .

# 2. Start services (if not already running)
docker-compose up -d

# 3. Create a Python script
cat > quickstart.py <<EOF
from hippocampai import UnifiedMemoryClient

# Initialize in local mode
client = UnifiedMemoryClient(mode="local")

# Store and retrieve
memory = client.remember("User prefers dark mode", user_id="user123")
results = client.recall("UI preferences", user_id="user123")
print(f"Found: {results[0].memory.text}")
EOF

# 4. Run it!
python quickstart.py
```

### Option B: Remote Mode (API Setup)

```bash
# 1. Clone and install
git clone https://github.com/rexdivakar/HippocampAI.git
cd HippocampAI
pip install -e .

# 2. Start API server
uvicorn hippocampai.api.async_app:app --port 8000 &

# 3. Create a Python script
cat > quickstart.py <<EOF
from hippocampai import UnifiedMemoryClient

# Initialize in remote mode
client = UnifiedMemoryClient(mode="remote", api_url="http://localhost:8000")

# Store and retrieve (same API!)
memory = client.remember("User prefers dark mode", user_id="user123")
results = client.recall("UI preferences", user_id="user123")
print(f"Found: {results[0].memory.text}")
EOF

# 4. Run it!
python quickstart.py
```

**That's it!** Same code, different backend. Now let's dive deeper...

---

## Local Mode Setup

Direct connection to Qdrant/Redis/Ollama for maximum performance.

### Library Setup

#### Step 1: Clone and Install

```bash
# Clone the repository
git clone https://github.com/rexdivakar/HippocampAI.git
cd HippocampAI

# Install in development mode
pip install -e .
```

#### Step 2: Configure Environment

Create a `.env` file in your project:

```bash
# .env
QDRANT_URL=http://localhost:6333
REDIS_URL=redis://localhost:6379
LLM_PROVIDER=ollama
LLM_BASE_URL=http://localhost:11434
LLM_MODEL=llama3.2:3b
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

#### Step 3: Verify Installation

```bash
python -c "from hippocampai import UnifiedMemoryClient; print('Installation successful!')"
```

### Local Mode Basic Usage

Create a file `quickstart_local.py`:

```python
from hippocampai import UnifiedMemoryClient

# Initialize client in LOCAL mode
client = UnifiedMemoryClient(mode="local")

# Store a memory
memory = client.remember(
    text="User prefers dark mode and large fonts",
    user_id="user123"
)
print(f"Stored memory: {memory.id}")

# Retrieve memories
results = client.recall(
    query="What are the user's UI preferences?",
    user_id="user123"
)

for result in results:
    print(f"Score: {result.score:.3f}")
    print(f"Text: {result.memory.text}")
```

Run it:

```bash
python quickstart_local.py
```

### Local Mode - All Features

All examples below use `UnifiedMemoryClient(mode="local")`. The same code works with `mode="remote"` - just change the initialization!

#### Feature 1: Basic Memory Operations

```python
from hippocampai import UnifiedMemoryClient
from datetime import datetime, timedelta, timezone

client = UnifiedMemoryClient(mode="local")  # or mode="remote"

# 1. Store a memory
memory = client.remember(
    text="User completed Python course on 2024-01-15",
    user_id="user123",
    metadata={"course": "python", "level": "beginner"},
    importance=0.8
)
print(f"Created: {memory.id}")

# 2. Retrieve by ID
retrieved = client.get_memory(memory.id)
print(f"Retrieved: {retrieved.text}")

# 3. Update memory
updated = client.update_memory(
    memory_id=memory.id,
    text="User completed Python advanced course on 2024-01-15",
    importance=0.9
)
print(f"Updated: {updated.text}")

# 4. Delete memory
success = client.delete_memory(memory.id)
print(f"Deleted: {success}")
```

#### Feature 2: Semantic Search

```python
from hippocampai import MemoryClient

client = MemoryClient()

# Store multiple memories
memories_data = [
    "User loves Italian food, especially pizza",
    "User is allergic to peanuts",
    "User prefers outdoor activities like hiking",
    "User works as a software engineer",
    "User speaks English and Spanish"
]

for text in memories_data:
    client.remember(text=text, user_id="user123")

# Semantic search
results = client.recall(
    query="What does the user like to eat?",
    user_id="user123",
    limit=3
)

print("Food preferences:")
for result in results:
    print(f"  - {result.memory.text} (score: {result.score:.3f})")
```

#### Feature 3: Filtering and Tags

```python
from hippocampai import MemoryClient

client = MemoryClient()

# Store memories with tags
client.remember(
    text="Completed project Alpha",
    user_id="user123",
    tags=["work", "project", "completed"]
)

client.remember(
    text="Started learning React",
    user_id="user123",
    tags=["learning", "frontend"]
)

client.remember(
    text="Meeting with Sarah at 3pm",
    user_id="user123",
    tags=["meeting", "schedule"]
)

# Filter by tags
work_memories = client.get_memories(
    user_id="user123",
    filters={"tags": ["work"]}
)
print(f"Work memories: {len(work_memories)}")

# Filter by importance
important_memories = client.get_memories(
    user_id="user123",
    min_importance=0.7
)
print(f"Important memories: {len(important_memories)}")
```

#### Feature 4: Fact Extraction

```python
from hippocampai import MemoryClient

client = MemoryClient()

# Store a complex memory
text = """
John Smith works at Google as a Senior Engineer.
He lives in San Francisco and has been programming for 15 years.
His favorite language is Python and he contributes to open source.
"""

memory = client.remember(
    text=text,
    user_id="user123",
    extract_entities=True
)

# Facts are automatically extracted
print("Extracted facts:")
for fact in memory.facts:
    print(f"  - {fact}")
```

#### Feature 5: Entity Recognition

```python
from hippocampai import MemoryClient

client = MemoryClient()

# Store text with entities
text = "Met with Dr. Sarah Johnson from Microsoft in Seattle about the Azure project."

memory = client.remember(
    text=text,
    user_id="user123",
    extract_entities=True
)

# Entities are automatically recognized
print("Recognized entities:")
print(f"  People: {memory.entities.get('persons', [])}")
print(f"  Organizations: {memory.entities.get('organizations', [])}")
print(f"  Locations: {memory.entities.get('locations', [])}")
print(f"  Projects: {memory.entities.get('projects', [])}")
```

#### Feature 6: Memory Consolidation

```python
from hippocampai import MemoryClient

client = MemoryClient()

# Store related memories over time
client.remember("User likes coffee", user_id="user123")
client.remember("User drinks coffee every morning", user_id="user123")
client.remember("User prefers espresso", user_id="user123")
client.remember("User bought a new espresso machine", user_id="user123")

# Consolidate related memories
consolidated = client.consolidate_memories(user_id="user123")

print(f"Consolidated {len(consolidated)} memory groups:")
for group in consolidated[:3]:
    print(f"\nSummary: {group['summary']}")
    print(f"Original memories: {len(group['memory_ids'])}")
```

#### Feature 7: Temporal Queries

```python
from hippocampai import MemoryClient
from datetime import datetime, timedelta, timezone

client = MemoryClient()

# Store memories with timestamps
now = datetime.now(timezone.utc)
client.remember("Completed task A", user_id="user123")

# Query recent memories
recent = client.get_memories(
    user_id="user123",
    after=now - timedelta(hours=1)
)
print(f"Memories in last hour: {len(recent)}")

# Query older memories
older = client.get_memories(
    user_id="user123",
    before=now - timedelta(days=7)
)
print(f"Memories older than 7 days: {len(older)}")
```

#### Feature 8: Memory Expiration

```python
from hippocampai import MemoryClient
from datetime import datetime, timedelta, timezone

client = MemoryClient()

# Store temporary memory
expires_at = datetime.now(timezone.utc) + timedelta(hours=24)
temp_memory = client.remember(
    text="Temporary note: Review document by EOD",
    user_id="user123",
    expires_at=expires_at
)

print(f"Memory expires at: {temp_memory.expires_at}")

# Cleanup expired memories
deleted_count = client.cleanup_expired_memories()
print(f"Cleaned up {deleted_count} expired memories")
```

#### Feature 9: Batch Operations

```python
from hippocampai import MemoryClient

client = MemoryClient()

# Batch create memories
memories_data = [
    {"text": "User likes reading", "user_id": "user123"},
    {"text": "User enjoys hiking", "user_id": "user123"},
    {"text": "User loves cooking", "user_id": "user123"},
]

created = client.batch_remember(memories_data)
print(f"Created {len(created)} memories in batch")

# Batch retrieve
memory_ids = [m.id for m in created]
retrieved = client.batch_get_memories(memory_ids)
print(f"Retrieved {len(retrieved)} memories")

# Batch delete
success = client.batch_delete_memories(memory_ids)
print(f"Batch delete: {success}")
```

#### Feature 10: Advanced Intelligence

```python
from hippocampai import MemoryClient

client = MemoryClient()

# Store memories with full intelligence pipeline
memory = client.remember(
    text="Working on machine learning project with TensorFlow and PyTorch at NVIDIA",
    user_id="user123",
    extract_entities=True,
    extract_facts=True,
    extract_relationships=True
)

# View extracted intelligence
print("Facts:", memory.facts)
print("Entities:", memory.entities)
print("Relationships:", memory.relationships)

# Get analytics
analytics = client.get_memory_analytics(user_id="user123")
print(f"\nMemory Analytics:")
print(f"  Total memories: {analytics['total_memories']}")
print(f"  Average importance: {analytics['avg_importance']:.2f}")
print(f"  Most common tags: {analytics['top_tags']}")
print(f"  Top entities: {analytics['top_entities']}")
```

#### Feature 11: Duplicate Detection

```python
from hippocampai import MemoryClient

client = MemoryClient()

# Store similar memories
memory1 = client.remember("User likes pizza", user_id="user123")
print(f"First memory: {memory1.id}")

# Try to store duplicate (will be detected)
memory2 = client.remember(
    "User likes pizza",
    user_id="user123",
    duplicate_action="skip"  # Options: skip, update, create
)

if memory2.id == memory1.id:
    print("Duplicate detected and skipped!")
```

#### Feature 12: Memory Search with Context

```python
from hippocampai import MemoryClient

client = MemoryClient()

# Store memories
client.remember("User is learning Python", user_id="user123", tags=["learning"])
client.remember("User completed JavaScript course", user_id="user123", tags=["learning"])
client.remember("User working on React project", user_id="user123", tags=["work"])

# Search with context
results = client.recall(
    query="programming languages",
    user_id="user123",
    filters={"tags": ["learning"]},
    limit=5
)

print("Learning-related programming memories:")
for result in results:
    print(f"  - {result.memory.text}")
```

---

## Mode 2: SaaS Mode

Use HippocampAI as a REST API server that any language can connect to.

### SaaS Setup

#### Step 1: Clone Repository

```bash
git clone https://github.com/rexdivakar/HippocampAI.git
cd HippocampAI
```

#### Step 2: Install Dependencies

```bash
pip install -e .
```

#### Step 3: Configure Environment

Create `.env` file:

```bash
# .env
QDRANT_URL=http://localhost:6333
REDIS_URL=redis://localhost:6379
LLM_PROVIDER=ollama
LLM_BASE_URL=http://localhost:11434
LLM_MODEL=llama3.2:3b
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

#### Step 4: Start the Server

```bash
uvicorn hippocampai.api.async_app:app --host 0.0.0.0 --port 8000 --reload
```

You should see:

```text
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

#### Step 5: Verify Server

```bash
curl http://localhost:8000/health
```

Response:

```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### SaaS Basic Usage

#### Using Python

Create `quickstart_api.py`:

```python
import requests

BASE_URL = "http://localhost:8000"

# Store a memory
response = requests.post(
    f"{BASE_URL}/v1/memories",
    json={
        "text": "User prefers dark mode and large fonts",
        "user_id": "user123"
    }
)
memory = response.json()
print(f"Stored memory: {memory['id']}")

# Retrieve memories
response = requests.post(
    f"{BASE_URL}/v1/memories/recall",
    json={
        "query": "What are the user's UI preferences?",
        "user_id": "user123"
    }
)
results = response.json()

for result in results:
    print(f"Score: {result['score']:.3f}")
    print(f"Text: {result['memory']['text']}")
```

Run it:

```bash
python quickstart_api.py
```

#### Using JavaScript/Node.js

Create `quickstart_api.js`:

```javascript
const axios = require('axios');

const BASE_URL = 'http://localhost:8000';

async function main() {
    // Store a memory
    const createResponse = await axios.post(`${BASE_URL}/v1/memories`, {
        text: 'User prefers dark mode and large fonts',
        user_id: 'user123'
    });
    console.log(`Stored memory: ${createResponse.data.id}`);

    // Retrieve memories
    const recallResponse = await axios.post(`${BASE_URL}/v1/memories/recall`, {
        query: "What are the user's UI preferences?",
        user_id: 'user123'
    });

    for (const result of recallResponse.data) {
        console.log(`Score: ${result.score.toFixed(3)}`);
        console.log(`Text: ${result.memory.text}`);
    }
}

main();
```

#### Using cURL

```bash
# Store a memory
curl -X POST http://localhost:8000/v1/memories \
  -H "Content-Type: application/json" \
  -d '{
    "text": "User prefers dark mode",
    "user_id": "user123"
  }'

# Retrieve memories
curl -X POST http://localhost:8000/v1/memories/recall \
  -H "Content-Type: application/json" \
  -d '{
    "query": "UI preferences",
    "user_id": "user123"
  }'
```

### SaaS All Features

#### Feature 1: Basic Memory Operations (REST API)

```python
import requests

BASE_URL = "http://localhost:8000"

# 1. Create a memory
response = requests.post(
    f"{BASE_URL}/v1/memories",
    json={
        "text": "User completed Python course on 2024-01-15",
        "user_id": "user123",
        "metadata": {"course": "python", "level": "beginner"},
        "importance": 0.8
    }
)
memory = response.json()
memory_id = memory["id"]
print(f"Created: {memory_id}")

# 2. Get memory by ID
response = requests.get(f"{BASE_URL}/v1/memories/{memory_id}")
retrieved = response.json()
print(f"Retrieved: {retrieved['text']}")

# 3. Update memory
response = requests.put(
    f"{BASE_URL}/v1/memories/{memory_id}",
    json={
        "text": "User completed Python advanced course on 2024-01-15",
        "importance": 0.9
    }
)
updated = response.json()
print(f"Updated: {updated['text']}")

# 4. Delete memory
response = requests.delete(f"{BASE_URL}/v1/memories/{memory_id}")
print(f"Deleted: {response.status_code == 204}")
```

#### Feature 2: Semantic Search (REST API)

```python
import requests

BASE_URL = "http://localhost:8000"

# Store multiple memories
memories_data = [
    "User loves Italian food, especially pizza",
    "User is allergic to peanuts",
    "User prefers outdoor activities like hiking",
    "User works as a software engineer",
    "User speaks English and Spanish"
]

for text in memories_data:
    requests.post(
        f"{BASE_URL}/v1/memories",
        json={"text": text, "user_id": "user123"}
    )

# Semantic search
response = requests.post(
    f"{BASE_URL}/v1/memories/recall",
    json={
        "query": "What does the user like to eat?",
        "user_id": "user123",
        "limit": 3
    }
)
results = response.json()

print("Food preferences:")
for result in results:
    print(f"  - {result['memory']['text']} (score: {result['score']:.3f})")
```

#### Feature 3: Filtering and Tags (REST API)

```python
import requests

BASE_URL = "http://localhost:8000"

# Store memories with tags
requests.post(
    f"{BASE_URL}/v1/memories",
    json={
        "text": "Completed project Alpha",
        "user_id": "user123",
        "tags": ["work", "project", "completed"]
    }
)

requests.post(
    f"{BASE_URL}/v1/memories",
    json={
        "text": "Started learning React",
        "user_id": "user123",
        "tags": ["learning", "frontend"]
    }
)

# Filter by tags
response = requests.get(
    f"{BASE_URL}/v1/memories",
    params={
        "user_id": "user123",
        "tags": "work"
    }
)
work_memories = response.json()
print(f"Work memories: {len(work_memories)}")

# Filter by importance
response = requests.get(
    f"{BASE_URL}/v1/memories",
    params={
        "user_id": "user123",
        "min_importance": 0.7
    }
)
important_memories = response.json()
print(f"Important memories: {len(important_memories)}")
```

#### Feature 4: Fact Extraction (REST API)

```python
import requests

BASE_URL = "http://localhost:8000"

# Store complex memory with fact extraction
text = """
John Smith works at Google as a Senior Engineer.
He lives in San Francisco and has been programming for 15 years.
His favorite language is Python and he contributes to open source.
"""

response = requests.post(
    f"{BASE_URL}/v1/memories",
    json={
        "text": text,
        "user_id": "user123",
        "extract_entities": True
    }
)
memory = response.json()

print("Extracted facts:")
for fact in memory.get("facts", []):
    print(f"  - {fact}")
```

#### Feature 5: Entity Recognition (REST API)

```python
import requests

BASE_URL = "http://localhost:8000"

# Store text with entities
response = requests.post(
    f"{BASE_URL}/v1/memories",
    json={
        "text": "Met with Dr. Sarah Johnson from Microsoft in Seattle about the Azure project.",
        "user_id": "user123",
        "extract_entities": True
    }
)
memory = response.json()

entities = memory.get("entities", {})
print("Recognized entities:")
print(f"  People: {entities.get('persons', [])}")
print(f"  Organizations: {entities.get('organizations', [])}")
print(f"  Locations: {entities.get('locations', [])}")
print(f"  Projects: {entities.get('projects', [])}")
```

#### Feature 6: Memory Consolidation (REST API)

```python
import requests

BASE_URL = "http://localhost:8000"

# Store related memories
memories = [
    "User likes coffee",
    "User drinks coffee every morning",
    "User prefers espresso",
    "User bought a new espresso machine"
]

for text in memories:
    requests.post(
        f"{BASE_URL}/v1/memories",
        json={"text": text, "user_id": "user123"}
    )

# Consolidate memories
response = requests.post(
    f"{BASE_URL}/v1/memories/consolidate",
    json={"user_id": "user123"}
)
consolidated = response.json()

print(f"Consolidated {len(consolidated)} memory groups:")
for group in consolidated[:3]:
    print(f"\nSummary: {group['summary']}")
    print(f"Original memories: {len(group['memory_ids'])}")
```

#### Feature 7: Temporal Queries (REST API)

```python
import requests
from datetime import datetime, timedelta, timezone

BASE_URL = "http://localhost:8000"

# Store memory
requests.post(
    f"{BASE_URL}/v1/memories",
    json={"text": "Completed task A", "user_id": "user123"}
)

# Query recent memories
now = datetime.now(timezone.utc)
after = (now - timedelta(hours=1)).isoformat()

response = requests.get(
    f"{BASE_URL}/v1/memories",
    params={
        "user_id": "user123",
        "after": after
    }
)
recent = response.json()
print(f"Memories in last hour: {len(recent)}")
```

#### Feature 8: Memory Expiration (REST API)

```python
import requests
from datetime import datetime, timedelta, timezone

BASE_URL = "http://localhost:8000"

# Store temporary memory
expires_at = datetime.now(timezone.utc) + timedelta(hours=24)

response = requests.post(
    f"{BASE_URL}/v1/memories",
    json={
        "text": "Temporary note: Review document by EOD",
        "user_id": "user123",
        "expires_at": expires_at.isoformat()
    }
)
memory = response.json()
print(f"Memory expires at: {memory['expires_at']}")

# Cleanup expired memories
response = requests.post(f"{BASE_URL}/v1/memories/cleanup")
result = response.json()
print(f"Cleaned up {result['deleted_count']} expired memories")
```

#### Feature 9: Batch Operations (REST API)

```python
import requests

BASE_URL = "http://localhost:8000"

# Batch create memories
response = requests.post(
    f"{BASE_URL}/v1/memories/batch",
    json={
        "memories": [
            {"text": "User likes reading", "user_id": "user123"},
            {"text": "User enjoys hiking", "user_id": "user123"},
            {"text": "User loves cooking", "user_id": "user123"}
        ]
    }
)
created = response.json()
print(f"Created {len(created)} memories in batch")

# Batch retrieve
memory_ids = [m["id"] for m in created]
response = requests.post(
    f"{BASE_URL}/v1/memories/batch/get",
    json={"memory_ids": memory_ids}
)
retrieved = response.json()
print(f"Retrieved {len(retrieved)} memories")

# Batch delete
response = requests.post(
    f"{BASE_URL}/v1/memories/batch/delete",
    json={"memory_ids": memory_ids}
)
print(f"Batch delete: {response.status_code == 200}")
```

#### Feature 10: Advanced Intelligence (REST API)

```python
import requests

BASE_URL = "http://localhost:8000"

# Store with full intelligence pipeline
response = requests.post(
    f"{BASE_URL}/v1/memories",
    json={
        "text": "Working on machine learning project with TensorFlow and PyTorch at NVIDIA",
        "user_id": "user123",
        "extract_entities": True,
        "extract_facts": True,
        "extract_relationships": True
    }
)
memory = response.json()

print("Facts:", memory.get("facts", []))
print("Entities:", memory.get("entities", {}))
print("Relationships:", memory.get("relationships", []))

# Get analytics
response = requests.get(
    f"{BASE_URL}/v1/memories/analytics",
    params={"user_id": "user123"}
)
analytics = response.json()
print(f"\nMemory Analytics:")
print(f"  Total memories: {analytics['total_memories']}")
print(f"  Average importance: {analytics['avg_importance']:.2f}")
print(f"  Most common tags: {analytics['top_tags']}")
```

#### Feature 11: Health and Monitoring (REST API)

```python
import requests

BASE_URL = "http://localhost:8000"

# Check server health
response = requests.get(f"{BASE_URL}/health")
health = response.json()
print(f"Server status: {health['status']}")

# Get API documentation
response = requests.get(f"{BASE_URL}/docs")
print(f"API docs available at: {BASE_URL}/docs")

# Get OpenAPI schema
response = requests.get(f"{BASE_URL}/openapi.json")
schema = response.json()
print(f"API version: {schema['info']['version']}")
```

#### Feature 12: JavaScript/TypeScript Client Example

```javascript
// Complete JavaScript client example
const axios = require('axios');

const BASE_URL = 'http://localhost:8000';

class HippocampAIClient {
    constructor(baseUrl = BASE_URL) {
        this.baseUrl = baseUrl;
    }

    async createMemory(text, userId, options = {}) {
        const response = await axios.post(`${this.baseUrl}/v1/memories`, {
            text,
            user_id: userId,
            ...options
        });
        return response.data;
    }

    async getMemory(memoryId) {
        const response = await axios.get(`${this.baseUrl}/v1/memories/${memoryId}`);
        return response.data;
    }

    async updateMemory(memoryId, updates) {
        const response = await axios.put(`${this.baseUrl}/v1/memories/${memoryId}`, updates);
        return response.data;
    }

    async deleteMemory(memoryId) {
        await axios.delete(`${this.baseUrl}/v1/memories/${memoryId}`);
        return true;
    }

    async recall(query, userId, options = {}) {
        const response = await axios.post(`${this.baseUrl}/v1/memories/recall`, {
            query,
            user_id: userId,
            ...options
        });
        return response.data;
    }

    async getMemories(userId, filters = {}) {
        const response = await axios.get(`${this.baseUrl}/v1/memories`, {
            params: {
                user_id: userId,
                ...filters
            }
        });
        return response.data;
    }
}

// Usage
async function main() {
    const client = new HippocampAIClient();

    // Create memory
    const memory = await client.createMemory(
        'User prefers TypeScript over JavaScript',
        'user123',
        { tags: ['preferences', 'programming'] }
    );
    console.log('Created:', memory.id);

    // Search
    const results = await client.recall(
        'programming language preferences',
        'user123'
    );
    console.log('Results:', results.length);
}

main();
```

---

## Mode Comparison

**Key Insight**: Same library (`UnifiedMemoryClient`), same API, just different backend connection!

| Aspect | Local Mode | Remote Mode |
|--------|------------|-------------|
| **Interface** | `UnifiedMemoryClient(mode="local")` | `UnifiedMemoryClient(mode="remote", api_url="...")` |
| **API Methods** | ✅ Same `remember()`, `recall()`, etc. | ✅ Same `remember()`, `recall()`, etc. |
| **Connection** | Direct to Qdrant/Redis/Ollama | HTTP to SaaS API |
| **Latency** | 🚀 5-15ms | 🌐 20-50ms (+ network) |
| **Throughput** | ~100-200 ops/sec | ~50-100 ops/sec |
| **Language Support** | Python only | Any language (via HTTP) |
| **Deployment** | Single process | Separate API server |
| **Scaling** | Vertical only | Horizontal + Vertical |
| **Use Case** | Python apps, max performance | Multi-language, microservices |

### When to Use Each Mode

**Use Local Mode When:**

- ✅ Building a Python application
- ✅ Performance is critical (5-15ms latency)
- ✅ Want maximum control over infrastructure
- ✅ Single-tenant deployment
- ✅ Embedding in existing Python app

**Use Remote Mode When:**

- ✅ Multi-language clients (JavaScript, Go, etc.)
- ✅ Microservices architecture
- ✅ True SaaS multi-tenant deployment
- ✅ Centralized memory service
- ✅ Need to scale horizontally

**Pro Tip**: You can even use BOTH modes in the same deployment! Internal Python services use local mode for speed, external clients use remote mode for flexibility.

---

## Troubleshooting

### Common Issues

#### 1. Services Not Running

**Error:** `Connection refused to localhost:6333`

**Solution:**

```bash
# Check if services are running
docker-compose ps

# Restart services
docker-compose restart

# View logs
docker-compose logs -f
```

#### 2. Ollama Model Not Found

**Error:** `Model llama3.2:3b not found`

**Solution:**

```bash
# Pull the model
docker exec -it $(docker ps -q -f name=ollama) ollama pull llama3.2:3b

# Verify
docker exec -it $(docker ps -q -f name=ollama) ollama list
```

#### 3. Import Error (Library Mode)

**Error:** `ModuleNotFoundError: No module named 'hippocampai'`

**Solution:**

```bash
# Reinstall in editable mode
cd HippocampAI
pip install -e .

# Verify
python -c "import hippocampai; print(hippocampai.__version__)"
```

#### 4. Port Already in Use (SaaS Mode)

**Error:** `Address already in use: 8000`

**Solution:**

```bash
# Find process using port 8000
lsof -ti:8000

# Kill the process
kill -9 $(lsof -ti:8000)

# Or use different port
uvicorn hippocampai.api.async_app:app --port 8001
```

#### 5. Connection Timeout

**Error:** `TimeoutError: Connection to Qdrant/Redis timeout`

**Solution:**

```bash
# Check service health
curl http://localhost:6333/healthz  # Qdrant
redis-cli ping                       # Redis
curl http://localhost:11434/api/tags # Ollama

# Restart services
docker-compose restart
```

#### 6. Memory Not Found

**Error:** `Memory not found: xxx-xxx-xxx`

**Solution:**

```python
# Check if memory exists
memory = client.get_memory(memory_id)
if memory is None:
    print("Memory does not exist")

# List all memories for user
memories = client.get_memories(user_id="user123")
print(f"Found {len(memories)} memories")
```

### Getting Help

1. **Check logs:** `docker-compose logs -f`
2. **API documentation:** <http://localhost:8000/docs> (SaaS mode)
3. **GitHub issues:** <https://github.com/rexdivakar/HippocampAI/issues>
4. **Discord community:** [Join our Discord](#)

---

## Next Steps

Now that you're familiar with both modes:

1. **Choose your mode:** Library for Python projects, SaaS for multi-language
2. **Explore examples:** Check the `examples/` directory
3. **Read architecture:** See `ARCHITECTURE.md` for technical details
4. **Deploy to production:** Follow `DEPLOYMENT_AND_USAGE_GUIDE.md`
5. **Customize:** Modify LLM providers, embedding models, etc.

Happy memory management with HippocampAI!
