# HippocampAI - Complete User Guide

## Table of Contents

1. [Quick Start](#quick-start)
2. [Installation & Setup](#installation--setup)
3. [Configuration](#configuration)
4. [Core Features](#core-features)
5. [Advanced Usage](#advanced-usage)
6. [API Reference](#api-reference)
7. [Deployment](#deployment)
8. [Troubleshooting](#troubleshooting)

---

## Quick Start

Get up and running with HippocampAI in 5 minutes:

### Installation

```bash
pip install hippocampai
```

### Basic Usage

```python
from hippocampai import MemoryClient

# Initialize client
client = MemoryClient()

# Store memories
memory = client.remember(
    text="I prefer oat milk in my coffee",
    user_id="alice",
    type="preference",
    tags=["beverages"]
)

# Retrieve memories
results = client.recall(
    query="How does Alice like her coffee?",
    user_id="alice",
    k=3
)

for result in results:
    print(f"{result.memory.text} (score: {result.score:.3f})")
```

### Memory Size Tracking

HippocampAI automatically tracks memory sizes for better resource management:

```python
# Memory size is calculated automatically
print(f"Memory: {memory.text_length} chars, {memory.token_count} tokens")

# Get user statistics
stats = client.get_memory_statistics(user_id="alice")
print(f"Total: {stats['total_memories']} memories")
print(f"Size: {stats['total_characters']} chars, {stats['total_tokens']} tokens")
```

---

## Installation & Setup

### Requirements

- Python 3.9+
- Docker (for local vector database)
- Redis (optional, for production)

### Installation Options

#### Option 1: Package Installation

```bash
pip install hippocampai
```

#### Option 2: Development Installation

```bash
git clone https://github.com/rexdivakar/HippocampAI.git
cd HippocampAI
pip install -e .
```

### Local Setup

#### Start Qdrant Vector Database

```bash
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant
```

#### Install Ollama (Optional - for local LLM)

```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a model
ollama pull qwen2.5:7b-instruct
```

### Cloud Setup

Set environment variables for your preferred providers:

```bash
# OpenAI
export OPENAI_API_KEY="your-key-here"

# Anthropic
export ANTHROPIC_API_KEY="your-key-here"

# Groq
export GROQ_API_KEY="your-key-here"
```

---

## Configuration

### Configuration Presets

HippocampAI provides several preset configurations:

```python
from hippocampai import MemoryClient

# Local development (fast, basic features)
client = MemoryClient.from_preset("development")

# Local production (full features, optimized)
client = MemoryClient.from_preset("local")

# Cloud LLM with local vector DB
client = MemoryClient.from_preset("cloud")

# Full production setup
client = MemoryClient.from_preset("production")
```

### Custom Configuration

```python
from hippocampai import MemoryClient, Config

config = Config(
    # Vector Database
    qdrant_url="http://localhost:6333",
    collection_facts="facts",
    collection_prefs="preferences",
    
    # Embedding Model
    embed_model="BAAI/bge-small-en-v1.5",
    embed_quantized=True,  # Use int8 quantization
    
    # Reranker
    reranker_model="BAAI/bge-reranker-base",
    
    # LLM Provider
    llm_provider="openai",  # or "anthropic", "groq", "ollama"
    llm_model="gpt-4o-mini",
    
    # Search Configuration
    top_k_final=20,
    weights={
        "sim": 0.55,      # Semantic similarity
        "rerank": 0.20,   # Reranker score
        "recency": 0.15,  # Time decay
        "importance": 0.10 # User-defined importance
    }
)

client = MemoryClient(config=config)
```

### Environment Variables

```bash
# Vector Database
QDRANT_URL="http://localhost:6333"
COLLECTION_FACTS="facts"
COLLECTION_PREFS="preferences"

# Models
EMBED_MODEL="BAAI/bge-small-en-v1.5"
EMBED_QUANTIZED="true"
RERANKER_MODEL="BAAI/bge-reranker-base"

# LLM Provider
LLM_PROVIDER="openai"
LLM_MODEL="gpt-4o-mini"
OPENAI_API_KEY="your-key"

# Search Settings
TOP_K_FINAL="20"
WEIGHT_SIM="0.55"
WEIGHT_RERANK="0.20"
WEIGHT_RECENCY="0.15"
WEIGHT_IMPORTANCE="0.10"

# Background Processing
CELERY_BROKER_URL="redis://localhost:6379/1"
CELERY_RESULT_BACKEND="redis://localhost:6379/2"
```

---

## Core Features

### 1. Memory Operations

#### Store Memories

```python
# Basic memory
memory = client.remember(
    text="Python is great for machine learning",
    user_id="alice"
)

# Memory with metadata
memory = client.remember(
    text="Meeting notes for Q1 planning",
    user_id="alice",
    type="fact",
    importance=8.5,
    tags=["work", "meetings", "q1"],
    ttl_days=90,  # Auto-expire in 90 days
    metadata={"project": "AI-Initiative", "department": "Engineering"}
)
```

#### Retrieve Memories

```python
# Semantic search
results = client.recall(
    query="What does Alice think about Python?",
    user_id="alice",
    k=5
)

# Advanced filtering
memories = client.get_memories_advanced(
    user_id="alice",
    filters={
        "tags": ["work"],
        "min_importance": 7.0,
        "type": "fact",
        "metadata": {"project": "AI-Initiative"}
    },
    sort_by="created_at",
    sort_order="desc",
    limit=10
)
```

#### Update and Delete

```python
# Update memory
updated = client.update_memory(
    memory_id="abc-123",
    text="Updated text content",
    importance=9.0,
    tags=["updated", "important"]
)

# Delete memory
deleted = client.delete_memory(
    memory_id="abc-123",
    user_id="alice"  # Optional verification
)
```

### 2. Batch Operations

```python
# Bulk create
memories = client.add_memories([
    {"text": "Memory 1", "user_id": "alice"},
    {"text": "Memory 2", "user_id": "alice", "type": "preference"},
    {"text": "Memory 3", "user_id": "alice", "tags": ["work"]}
])

# Bulk delete
deleted_count = client.delete_memories(
    user_id="alice",
    filters={"tags": ["temporary"]}
)
```

### 3. Context Injection

Automatically inject relevant memories into LLM prompts:

```python
# Inject context into prompts
enhanced_prompt = client.inject_context(
    prompt="What should I recommend for Alice?",
    query="Alice preferences recommendations",
    user_id="alice",
    k=5,
    template="detailed"  # or "minimal", "custom"
)

# The enhanced prompt includes relevant memories
print(enhanced_prompt)
# Output: "Based on what I know about Alice: [relevant memories]... What should I recommend for Alice?"
```

### 4. Conversation Extraction

Extract structured memories from conversations:

```python
conversation = """
Alice: I've been really enjoying Italian food lately
Bob: Have you tried that new pasta place?
Alice: Yes! I love their carbonara, but I'm trying to eat less meat
Bob: They have great vegetarian options too
Alice: Perfect, I'll definitely go back
"""

extracted = client.extract_from_conversation(
    conversation=conversation,
    user_id="alice",
    extract_preferences=True,
    extract_facts=True,
    auto_store=True  # Automatically store extracted memories
)

# Returns structured memories about preferences and facts
for memory in extracted:
    print(f"Type: {memory.type}, Text: {memory.text}")
```

### 5. Memory Statistics & Analytics

```python
# Get comprehensive statistics
stats = client.get_memory_statistics(user_id="alice")

print(f"Total memories: {stats['total_memories']}")
print(f"Total size: {stats['total_characters']} chars, {stats['total_tokens']} tokens")
print(f"Average size: {stats['avg_memory_size_chars']:.1f} chars")
print(f"By type: {stats['by_type']}")

# Memory access tracking
popular_memories = client.get_memories_advanced(
    user_id="alice",
    sort_by="access_count",
    sort_order="desc",
    limit=5
)
```

---

## Advanced Usage

### 1. Async Operations

```python
from hippocampai import AsyncMemoryClient
import asyncio

async def main():
    client = AsyncMemoryClient()
    
    # Async operations
    memory = await client.remember_async(
        text="Async memory",
        user_id="alice"
    )
    
    results = await client.recall_async(
        query="async",
        user_id="alice"
    )
    
    # Concurrent operations
    tasks = [
        client.remember_async(f"Memory {i}", user_id="alice")
        for i in range(10)
    ]
    memories = await asyncio.gather(*tasks)

asyncio.run(main())
```

### 2. Graph Relationships

```python
from hippocampai import RelationType

# Create memories
m1 = client.remember("Deep learning uses neural networks", user_id="alice")
m2 = client.remember("TensorFlow is a DL framework", user_id="alice")

# Add to memory graph
client.graph.add_memory(m1.id, "alice", {"topic": "ML"})
client.graph.add_memory(m2.id, "alice", {"topic": "ML"})

# Create relationships
client.add_relationship(m1.id, m2.id, RelationType.RELATED_TO)

# Find related memories
related = client.get_related_memories(m1.id, max_depth=2)
```

### 3. Memory Lifecycle Management

```python
# TTL and expiration
memory = client.remember(
    text="Temporary note",
    user_id="alice",
    ttl_days=7  # Auto-expire in 7 days
)

# Manual expiration
client.set_memory_ttl(memory.id, days=30)

# Check expiring memories
expiring = client.get_expiring_memories(
    user_id="alice",
    days_ahead=7  # Expiring in next 7 days
)
```

### 4. Version Control & Audit

```python
# Memory versioning
memory = client.remember("Version 1 text", user_id="alice")

# Update creates new version
updated = client.update_memory(
    memory.id,
    text="Version 2 text",
    importance=8.0
)

# View history
history = client.get_memory_history(memory.id)
for version in history:
    print(f"Version {version.version_number}: {version.change_summary}")

# Rollback
rollback = client.rollback_memory(memory.id, version_number=1)
```

### 5. Telemetry & Monitoring

```python
# Get operation metrics
metrics = client.get_telemetry_metrics()
print(f"Average recall latency: {metrics['recall_duration']['avg']:.2f}ms")
print(f"Success rate: {metrics['success_rate']:.1%}")

# View recent operations
operations = client.get_recent_operations(limit=10)
for op in operations:
    print(f"{op.operation}: {op.duration_ms:.2f}ms ({op.status})")

# Export telemetry data
telemetry_data = client.export_telemetry()
```

---

## API Reference

### Memory Model

```python
class Memory:
    id: str
    text: str
    user_id: str
    type: str = "fact"
    importance: float = 5.0
    tags: List[str] = []
    metadata: Dict[str, Any] = {}
    created_at: datetime
    updated_at: datetime
    access_count: int = 0
    last_accessed_at: Optional[datetime] = None
    text_length: int  # Automatically calculated
    token_count: int  # Automatically calculated
    ttl_days: Optional[int] = None
    expires_at: Optional[datetime] = None
```

### Client Methods

#### Core Operations

- `remember(text, user_id, **kwargs) -> Memory`
- `recall(query, user_id, k=5, **kwargs) -> List[RetrievalResult]`
- `update_memory(memory_id, **updates) -> Optional[Memory]`
- `delete_memory(memory_id, user_id=None) -> bool`
- `get_memories(user_id, limit=100, **kwargs) -> List[Memory]`

#### Batch Operations

- `add_memories(memories: List[Dict]) -> List[Memory]`
- `delete_memories(user_id, filters=None) -> int`

#### Advanced Features

- `get_memory_statistics(user_id) -> Dict[str, Any]`
- `inject_context(prompt, query, user_id, **kwargs) -> str`
- `extract_from_conversation(conversation, user_id, **kwargs) -> List[Memory]`

#### Async Variants

All methods have `_async` variants:
- `remember_async()`, `recall_async()`, etc.

---

## Deployment

### Local Development

```bash
# Start dependencies
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant
docker run -d --name redis -p 6379:6379 redis:alpine

# Install and run
pip install hippocampai
python -c "from hippocampai import MemoryClient; client = MemoryClient()"
```

### Docker Deployment

```yaml
# docker-compose.yml
version: '3.8'

services:
  hippocampai:
    image: hippocampai:latest
    environment:
      - QDRANT_URL=http://qdrant:6333
      - REDIS_URL=redis://redis:6379
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    depends_on:
      - qdrant
      - redis
    ports:
      - "8000:8000"

  qdrant:
    image: qdrant/qdrant
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage

  redis:
    image: redis:alpine
    ports:
      - "6379:6379"

volumes:
  qdrant_data:
```

### Production Setup

#### With Celery Background Tasks

```bash
# Start Celery worker
celery -A hippocampai.celery_app worker --loglevel=info

# Start Celery beat scheduler
celery -A hippocampai.celery_app beat --loglevel=info

# Start FastAPI server
uvicorn hippocampai.api.app:app --host 0.0.0.0 --port 8000
```

#### Environment Configuration

```bash
# Production settings
export ENVIRONMENT="production"
export DEBUG="false"

# Database URLs
export QDRANT_URL="https://your-qdrant-cluster.qdrant.tech"
export REDIS_URL="redis://your-redis-cluster:6379"

# API Keys
export OPENAI_API_KEY="your-production-key"

# Performance settings
export TOP_K_FINAL="50"
export EMBED_QUANTIZED="true"
export ENABLE_TELEMETRY="true"
```

---

## Troubleshooting

### Common Issues

#### 1. Connection Issues

**Problem:** Cannot connect to Qdrant
```
ConnectionError: Could not connect to Qdrant at http://localhost:6333
```

**Solution:**
```bash
# Check if Qdrant is running
docker ps | grep qdrant

# Start Qdrant if not running
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant
```

#### 2. Memory Size Issues

**Problem:** Token count calculation errors
```
RuntimeError: Token counting failed
```

**Solution:**
```python
# Disable token counting if having issues
client = MemoryClient(config=Config(
    calculate_token_count=False
))
```

#### 3. Performance Issues

**Problem:** Slow retrieval performance

**Solution:**
```python
# Optimize configuration
config = Config(
    embed_quantized=True,  # Use int8 quantization
    top_k_final=10,       # Reduce result count
    weights={
        "sim": 0.7,        # Focus on similarity
        "rerank": 0.0,     # Disable reranking
        "recency": 0.2,
        "importance": 0.1
    }
)
```

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable debug telemetry
from hippocampai import get_telemetry
telemetry = get_telemetry()
telemetry.set_debug_mode(True)

# View detailed operation traces
client = MemoryClient()
memory = client.remember("test", user_id="debug")
traces = telemetry.get_recent_traces(limit=1)
print(traces[0].events)  # Detailed step-by-step trace
```

### Performance Monitoring

```python
# Monitor system performance
metrics = client.get_telemetry_metrics()

print("Performance Metrics:")
print(f"  Remember: {metrics['remember_duration']['avg']:.2f}ms")
print(f"  Recall: {metrics['recall_duration']['avg']:.2f}ms")
print(f"  Success Rate: {metrics['success_rate']:.1%}")
print(f"  Total Operations: {metrics['total_operations']}")

# Check for slow operations
slow_ops = client.get_recent_operations(
    min_duration_ms=1000,  # Operations taking >1 second
    limit=10
)
```

---

## Support

- **Documentation:** [Full API Reference](API_REFERENCE.md)
- **Examples:** See `examples/` directory
- **Issues:** GitHub Issues
- **Discord:** [HippocampAI Community](https://discord.gg/hippocampai)

---

*This guide covers the essential features of HippocampAI. For complete API documentation, see [API_REFERENCE.md](API_REFERENCE.md).*