# HippocampAI — Autonomous Memory Engine for LLM Agents

[![PyPI version](https://badge.fury.io/py/hippocampai.svg)](https://pypi.org/project/hippocampai/)
[![Python Versions](https://img.shields.io/pypi/pyversions/hippocampai.svg)](https://pypi.org/project/hippocampai/)
[![Downloads](https://pepy.tech/badge/hippocampai)](https://pepy.tech/project/hippocampai)
[![Quality Gate Status](https://sonar.craftedbrain.com/api/project_badges/measure?project=rexdivakar_HippocampAI_6669aa8c-2e81-4016-9993-b29a3a78c475&metric=alert_status&token=sqb_dd0c0b1bf58646ce474b64a1fa8d83446345bccf)](https://sonar.craftedbrain.com/dashboard?id=rexdivakar_HippocampAI_6669aa8c-2e81-4016-9993-b29a3a78c475)

HippocampAI turns raw conversations into a curated long-term memory vault for your AI assistants. It extracts, scores, deduplicates, stores, and retrieves user memories so agents can stay personal, consistent, and context-aware across sessions.

- **Plug-and-play** `MemoryClient` API with built-in pipelines for extraction, dedupe, consolidation, and importance decay
- **Hybrid retrieval** that fuses dense vectors, BM25, reciprocal-rank fusion, reranking, recency, and importance signals
- **Self-hosted first** — works fully offline via Qdrant + Ollama or in the cloud via OpenAI with the same code paths
- **Production-ready** — automatic retry logic, structured JSON logging, request tracing, telemetry, typed models, and scheduled jobs
- **Fully customizable** — every component (extraction, retrieval, scoring) is extensible without vendor lock-in

**Current Release:** v1.0.0 — first major stable release of HippocampAI.

---

## ✨ Why HippocampAI

- **Persistent personalization** – store preferences, facts, goals, habits, and events per user with importance scoring and decay
- **Reliable retrieval** – hybrid ranking surfaces the right memories even when queries are vague or drift semantically
- **Automatic hygiene** – extractor, deduplicator, consolidator, and scorer keep the memory base uncluttered
- **Local-first** – run everything on your infra with open models, or flip a switch to activate OpenAI for higher quality
- **Built-in telemetry** – track all memory operations with detailed tracing and metrics (similar to Mem0 platform)
- **Extensible Python SDK** – customize every stage without being locked to a hosted API

---

## 🚀 Quick Start

### 1. Installation

**Install from PyPI (Recommended):**

```bash
# Install with core dependencies
pip install hippocampai

# Or install with additional providers
pip install "hippocampai[all]"  # All providers + API + Web
pip install "hippocampai[openai]"  # Just OpenAI
pip install "hippocampai[api]"  # FastAPI server
pip install "hippocampai[web]"  # Flask web interface
```

View on PyPI: [https://pypi.org/project/hippocampai/](https://pypi.org/project/hippocampai/)

**Or install from source:**

```bash
git clone https://github.com/rexdivakar/HippocampAI.git
cd HippocampAI

# Install with core dependencies
pip install -e .

# Or install with additional providers
pip install -e ".[all]"  # All providers + API + Web
pip install -e ".[openai]"  # Just OpenAI
pip install -e ".[api]"  # FastAPI server
pip install -e ".[web]"  # Flask web interface
```

### 2. Start Qdrant

```bash
docker run -p 6333:6333 qdrant/qdrant
```

### 3. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` and set your configuration:

```bash
# Basic settings
QDRANT_URL=http://localhost:6333
LLM_PROVIDER=ollama
LLM_MODEL=qwen2.5:7b-instruct

# For cloud providers (optional)
# LLM_PROVIDER=openai
# OPENAI_API_KEY=sk-your-key-here
# ALLOW_CLOUD=true
```

### 4. Initialize Collections

```bash
python -c "from hippocampai import MemoryClient; MemoryClient()"
```

---

## 💡 Basic Usage

```python
from hippocampai import MemoryClient

# Initialize client
client = MemoryClient()

# Store a memory
memory = client.remember(
    text="I prefer oat milk in my coffee",
    user_id="alice",
    type="preference",
    importance=8.0,
    tags=["beverages", "preferences"]
)

# Memory size is automatically tracked
print(f"Memory size: {memory.text_length} chars, {memory.token_count} tokens")

# Recall relevant memories
results = client.recall(
    query="How does Alice like her coffee?",
    user_id="alice",
    k=3
)

for result in results:
    print(f"{result.memory.text} (score: {result.score:.3f})")

# Get memory statistics
stats = client.get_memory_statistics(user_id="alice")
print(f"Total memories: {stats['total_memories']}")
print(f"Total size: {stats['total_characters']} chars, {stats['total_tokens']} tokens")
```

### Async Usage

```python
from hippocampai import AsyncMemoryClient
import asyncio

async def main():
    client = AsyncMemoryClient()

    # Store memories concurrently
    tasks = [
        client.remember_async(f"Memory {i}", user_id="alice")
        for i in range(10)
    ]
    memories = await asyncio.gather(*tasks)

    # Async recall
    results = await client.recall_async(
        query="What do you know about me?",
        user_id="alice",
        k=5
    )

asyncio.run(main())
```

---

## 🎯 Features

### Core Memory Operations

- **CRUD Operations** — remember, recall, update, delete with telemetry tracking
- **Automatic extraction** from conversations using heuristics or LLM
- **Deduplication** with semantic similarity and reranking
- **Importance scoring** with configurable decay
- **Multi-user isolation** — complete data separation per user
- **Memory size tracking** — automatic character and token counting
- **Semantic clustering & auto-categorization** — automatic topic detection, tag suggestion, and category assignment
- **Async support** — async variants of all core operations for high-performance apps

### Hybrid Retrieval

HippocampAI combines multiple retrieval strategies:

1. **Vector search** — semantic similarity using embeddings
2. **BM25** — keyword matching for precision
3. **Reciprocal rank fusion** — merges both signals
4. **Cross-encoder reranking** — precision refinement
5. **Score fusion** — combines similarity, reranking, recency, and importance

```python
results = client.recall(query="coffee preferences", user_id="alice", k=5)

for result in results:
    print(f"Score breakdown:")
    print(f"  Similarity: {result.breakdown['sim']:.3f}")
    print(f"  Rerank: {result.breakdown['rerank']:.3f}")
    print(f"  Recency: {result.breakdown['recency']:.3f}")
    print(f"  Importance: {result.breakdown['importance']:.3f}")
```

### Conversation Extraction

```python
conversation = """
User: I really enjoy drinking green tea in the morning.
Assistant: That's great! Green tea is healthy.
User: Yes, and I usually have it without sugar.
"""

memories = client.extract_from_conversation(
    conversation=conversation,
    user_id="bob",
    session_id="session_001"
)

print(f"Extracted {len(memories)} memories")
```

### Advanced Features

```python
# Batch operations
memories_data = [
    {"text": "I like Python", "tags": ["programming"]},
    {"text": "I prefer dark mode", "type": "preference"},
]
created = client.add_memories(memories_data, user_id="alice")

# Graph relationships
client.add_relationship(
    source_id=created[0].id,
    target_id=created[1].id,
    relation_type=RelationType.RELATED_TO
)

# Version control
history = client.get_memory_history(memory_id)
client.rollback_memory(memory_id, version_number=1)

# Context injection for LLMs
prompt = client.inject_context(
    prompt="What are my preferences?",
    query="user preferences",
    user_id="alice",
    k=5
)

# Memory statistics
stats = client.get_memory_statistics(user_id="alice")
print(f"Total: {stats['total_memories']} memories")
print(f"Size: {stats['total_characters']} chars, {stats['total_tokens']} tokens")
print(f"By type: {stats['by_type']}")

# Semantic clustering & categorization
clusters = client.cluster_user_memories(user_id="alice", max_clusters=10)
for cluster in clusters:
    print(f"Topic: {cluster.topic}, Memories: {len(cluster.memories)}")

tags = client.suggest_memory_tags(memory, max_tags=5)
topic_shift = client.detect_topic_shift(user_id="alice", window_size=10)
```

### Telemetry & Observability

```python
# Memory operations are automatically tracked
client.remember(text="I love Python", user_id="alice")
client.recall(query="What do I like?", user_id="alice")

# Access telemetry via client
metrics = client.get_telemetry_metrics()
print(f"Average recall time: {metrics['recall_duration']['avg']:.2f}ms")
print(f"Average memory size: {metrics['memory_size_chars']['avg']:.1f} chars")

# Get recent operations
operations = client.get_recent_operations(limit=10)
for op in operations:
    print(f"{op.operation.value}: {op.duration_ms:.2f}ms ({op.status})")
```

---

## 📚 Examples

Explore working examples in the `examples/` directory:

```bash
# Basic usage
python examples/01_basic_usage.py

# Conversation extraction
python examples/02_conversation_extraction.py

# Hybrid retrieval
python examples/03_hybrid_retrieval.py

# Custom configuration
python examples/04_custom_configuration.py

# Multi-user management
python examples/05_multi_user.py

# Batch operations
python examples/06_batch_operations.py

# Advanced features (graph, version control, context injection, etc.)
python examples/07_advanced_features_demo.py

# Memory consolidation scheduler
python examples/08_scheduler_demo.py

# Graph persistence (JSON export/import)
python examples/09_graph_persistence_demo.py

# Session management (hierarchical conversations, boundaries, summaries)
python examples/10_session_management_demo.py

# Semantic clustering & auto-categorization
python examples/11_semantic_clustering_demo.py

# Multi-agent memory management
python examples/12_multiagent_demo.py

# Production resilience (retry logic + structured logging)
python examples/example_resilience.py

# Run all examples
./run_examples.sh
```

---

## ⚙️ Configuration

### Environment Variables

See `.env.example` for all configuration options:

- **Qdrant** settings (URL, collections, HNSW tuning)
- **Embedding** model selection and parameters
- **LLM** provider (ollama, openai, anthropic, groq)
- **Retrieval** weights and parameters
- **Decay** half-lives per memory type
- **Jobs** scheduling (importance decay, consolidation)

### Custom Configuration

```python
from hippocampai import MemoryClient, Config

# Option 1: Override specific parameters
client = MemoryClient(
    qdrant_url="http://localhost:6333",
    weights={"sim": 0.6, "rerank": 0.2, "recency": 0.1, "importance": 0.1}
)

# Option 2: Create custom config object
config = Config(
    embed_model="BAAI/bge-small-en-v1.5",
    llm_provider="ollama",
    llm_model="qwen2.5:7b-instruct",
    top_k_final=10
)

client = MemoryClient(config=config)
```

---

## 📖 Documentation

### Getting Started

- **[Quick Start Guide](docs/QUICKSTART.md)** - Get up and running in 5 minutes
- **[Getting Started](docs/GETTING_STARTED.md)** - Detailed installation, setup, and first steps
- **[Usage Guide](docs/USAGE.md)** - Common usage patterns, chat clients, and integration examples

### Core Documentation

- **[API Reference](docs/API_REFERENCE.md)** - Complete API documentation for all methods and classes
- **[Features Guide](docs/FEATURES.md)** - Comprehensive feature documentation with examples and use cases
- **[Configuration Guide](docs/CONFIGURATION.md)** - All configuration options, presets, and environment variables
- **[Provider Setup](docs/PROVIDERS.md)** - Configure LLM providers (Ollama, OpenAI, Anthropic, Groq)

### Advanced Guides

- **[Resilience & Observability](docs/RESILIENCE.md)** - Automatic retry logic, error handling, and structured logging
- **[Telemetry Guide](docs/TELEMETRY.md)** - Observability, tracing, metrics, and performance monitoring
- **[Testing Guide](docs/TESTING_GUIDE.md)** - Complete testing guide with 117 tests and 100% pass rate

### Examples & Tutorials

- **[Examples Documentation](docs/examples.md)** - Guide to all working examples in the `examples/` directory
- See the [Examples section](#-examples) above for a complete list of runnable examples

### Developer Resources

- **[Package Summary](docs/PACKAGE_SUMMARY.md)** - Technical overview of the package architecture and structure
- **[Implementation Summary](docs/IMPLEMENTATION_SUMMARY.md)** - Implementation details of core features and components
- **[Contributing Guide](CONTRIBUTING.md)** - How to contribute to HippocampAI
- **[Changelog](docs/CHANGELOG.md)** - Version history, updates, and breaking changes

---

**Need help?** Join our community: [Discord](https://discord.gg/pPSNW9J7gB)

---

## 🆚 HippocampAI vs Mem0

| Feature | HippocampAI | Mem0 |
|---------|-------------|------|
| **Deployment** | ✅ Self-hosted first, cloud optional | ❌ SaaS-first |
| **Customization** | ✅ Full pipeline control | ❌ Limited API surface |
| **Retrieval** | ✅ Hybrid (vector + BM25 + rerank + decay) | ⚠️ Vector-first |
| **Memory hygiene** | ✅ Built-in extraction, dedupe, consolidation | ⚠️ Manual or implicit |
| **Telemetry** | ✅ Built-in tracing & metrics | ✅ Platform dashboard |
| **Data residency** | ✅ Your infra, your data | ❌ Managed cloud |
| **Setup complexity** | ⚠️ Requires Qdrant | ✅ Zero infra |
| **Cost** | ✅ Free (self-hosted) | ⚠️ Usage-based pricing |

**Choose HippocampAI when:**

- You need full control and customization
- Data residency is critical
- You want to avoid vendor lock-in
- You're building production systems requiring observability

**Choose Mem0 when:**

- You want zero infrastructure management
- You're prototyping quickly
- You don't mind managed services

---

## 🗺️ Roadmap

**Completed (v1.0.0):**

- [x] Configuration presets (`.from_preset("local")`, `.from_preset("cloud")`)
- [x] Built-in telemetry and observability
- [x] Automatic retry logic for Qdrant and LLM operations
- [x] Structured JSON logging with request ID tracking
- [x] Batch operations (add_memories, delete_memories)
- [x] Graph indexing and relationships
- [x] Version control and rollback
- [x] Context injection for LLM prompts
- [x] Memory access tracking
- [x] Advanced filtering and sorting
- [x] Snapshots and audit trail
- [x] KV store for fast lookups
- [x] Memory size tracking (text_length, token_count)
- [x] Async variants for all core operations
- [x] Memory consolidation scheduler (background jobs)
- [x] Persistent graph storage (JSON export/import)
- [x] Session management (hierarchical conversations, boundaries, summaries)
- [x] Smart memory updates (conflict resolution, quality refinement)
- [x] Semantic clustering & auto-categorization (topic detection, tag suggestion)
- [x] Multi-agent support (agent-specific memory spaces, permissions, transfers)

**Planned:**

- [ ] LangChain and LlamaIndex integrations
- [ ] Retrieval evaluators and A/B testing
- [ ] Multi-tenant RBAC and access policies
- [ ] Native TypeScript SDK
- [ ] Grafana/Prometheus metrics exporters
- [ ] WebSocket support for real-time updates

Contributions welcome! Open issues or PRs to shape HippocampAI's direction.

---

## 📄 License

Apache 2.0 — See [LICENSE](LICENSE) for details.

---

## 🤝 Contributing

We welcome contributions! Please see our [contributing guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`pytest`)
5. Commit your changes
6. Push to the branch
7. Open a Pull Request

---

## 🌟 Star History

If you find HippocampAI useful, please star the repo! It helps others discover the project.

---

## 📧 Contact

- **Issues**: [GitHub Issues](https://github.com/rexdivakar/HippocampAI/issues)
- **Discord**: [Join our community](https://discord.gg/pPSNW9J7gB)

Built with ❤️ by the HippocampAI team
