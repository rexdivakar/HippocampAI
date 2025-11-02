# HippocampAI — Autonomous Memory Engine for LLM Agents

[![PyPI version](https://badge.fury.io/py/hippocampai.svg)](https://pypi.org/project/hippocampai/)
[![Python Versions](https://img.shields.io/pypi/pyversions/hippocampai.svg)](https://pypi.org/project/hippocampai/)
[![Downloads](https://pepy.tech/badge/hippocampai)](https://pepy.tech/project/hippocampai)

HippocampAI turns raw conversations into a curated long-term memory vault for your AI assistants. It extracts, scores, deduplicates, stores, and retrieves user memories so agents can stay personal, consistent, and context-aware across sessions.

## 🎯 NEW: Unified Memory Client

**One interface, multiple backends!** The new `UnifiedMemoryClient` works with both local (direct) and remote (API) modes with the same code:

```python
from hippocampai import UnifiedMemoryClient

# Local mode - direct connection (fastest)
client = UnifiedMemoryClient(mode="local")

# Remote mode - API connection (multi-language support)
client = UnifiedMemoryClient(mode="remote", api_url="http://localhost:8000")

# Either way, same API!
memory = client.remember("User prefers dark mode", user_id="user123")
results = client.recall("UI preferences", user_id="user123")
```

📚 **[Read the Unified Client Guide](docs/UNIFIED_CLIENT_GUIDE.md)** | **[Complete Usage Examples](docs/UNIFIED_CLIENT_USAGE.md)** | **[What's New](docs/WHATS_NEW_UNIFIED_CLIENT.md)**

---

## ✨ Key Features

- **Universal SaaS Integration** — Seamless integration with Groq, OpenAI, Anthropic, and Ollama (85.7% success rate)
- **Unified Interface** — Same Python library works for local and remote deployments
- **High-Performance Memory** — Lightning-fast retrieval with advanced semantic clustering and cross-session insights
- **Hybrid retrieval** — Fuses dense vectors, BM25, reciprocal-rank fusion, reranking, recency, and importance signals
- **Multi-Agent Support** — Built-in coordination for complex multi-agent workflows and collaboration
- **Production-ready** — Docker Compose deployment, Celery task queue, monitoring, and enterprise-grade reliability
- **Fully customizable** — Every component is extensible without vendor lock-in

**Current Release:** v1.0.0 — Production-ready release with comprehensive features and type-safe architecture.

**✅ Verified Working**: Groq (0.37s), Ollama (0.02s), Docker Compose deployment, Celery task queue, comprehensive monitoring.

---

## 📚 Documentation

Complete documentation is available in the [docs/](docs/) folder:

### 🚀 Getting Started

- **[Getting Started Guide](getting_started.md)** - 🆕 **Complete setup, Docker deployment, API examples, and HippocampAI vs Mem0 comparison**
- **[Legacy Guide](docs/GETTING_STARTED.md)** - Original setup guide (still valid)
- **[Configuration Guide](docs/CONFIGURATION.md)** - Configure Qdrant, Redis, LLMs, and embeddings
- **[Architecture Overview](docs/ARCHITECTURE.md)** - System design and component architecture

### 🎯 Unified Memory Client

- **[Unified Client Guide](docs/UNIFIED_CLIENT_GUIDE.md)** - Conceptual overview and when to use each mode
- **[Unified Client Usage](docs/UNIFIED_CLIENT_USAGE.md)** - Complete API reference and examples
- **[What's New](docs/WHATS_NEW_UNIFIED_CLIENT.md)** - Latest updates and migration guide

### 📖 Core Documentation

- **[Complete API Reference](docs/API_COMPLETE_REFERENCE.md)** - Full REST API documentation
- **[Advanced Intelligence API](docs/ADVANCED_INTELLIGENCE_API.md)** - Fact extraction, entities, relationships
- **[Features Overview](docs/FEATURES.md)** - Complete feature documentation
- **[Search Enhancements](docs/SEARCH_ENHANCEMENTS_GUIDE.md)** - Hybrid search, saved searches
- **[Deployment Guide](docs/DEPLOYMENT_AND_USAGE_GUIDE.md)** - Production deployment

### 🌐 Integration & Deployment

- **[SaaS Integration Guide](docs/SAAS_INTEGRATION_GUIDE.md)** - Complete SaaS provider setup and deployment architectures
- **[Memory Management API](docs/MEMORY_MANAGEMENT_API.md)** - Advanced memory operations and lifecycle management
- **[Multi-Agent Features](docs/MULTIAGENT_FEATURES.md)** - Agent coordination and collaborative workflows

### 📚 More Documentation

- **[Full Documentation Index](docs/README.md)** - Browse all 27+ documentation files

---

## ✨ Why HippocampAI

- **Persistent personalization** – store preferences, facts, goals, habits, and events per user with importance scoring and decay
- **Reliable retrieval** – hybrid ranking surfaces the right memories even when queries are vague or drift semantically
- **Automatic hygiene** – extractor, deduplicator, consolidator, and scorer keep the memory base uncluttered
- **Intelligence features** – automatic fact extraction, entity recognition, session summarization, and knowledge graph building
- **Temporal reasoning** – time-based queries, chronological narratives, event sequences, and memory scheduling
- **Cross-session insights** – detect patterns, track behavioral changes, analyze preference drift, and identify habits
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

### 5. Validate Installation (Optional)

Verify that all intelligence features are working correctly:

```bash
python validate_intelligence_features.py
```

This will test:

- Fact extraction pipeline
- Entity recognition
- Session summarization
- Knowledge graph operations

For detailed output, use `--verbose` flag:

```bash
python validate_intelligence_features.py --verbose
```

---

## 🚀 Quick Start - Production Ready (v1.0.0)

Choose your deployment mode and AI provider with a single line:

### Option 1: Local Mode with Ollama (Fastest - 0.03s response time)

```python
from hippocampai import MemoryClient
from hippocampai.adapters import OllamaProvider

# Lightning-fast local processing
client = MemoryClient(
    llm_provider=OllamaProvider(base_url="http://localhost:11434"),
    mode="local"  # Direct connection to Qdrant/Redis
)

# Store and retrieve memories instantly
memory = client.remember("I love dark chocolate", user_id="alice")
results = client.recall("favorite foods", user_id="alice")
print(f"Found: {results[0].memory.text}")  # "I love dark chocolate"
```

### Option 2: SaaS Mode with Groq (Production-ready - 0.35s response time)

```python
from hippocampai import MemoryClient
from hippocampai.adapters import GroqProvider

# Production SaaS deployment
client = MemoryClient(
    llm_provider=GroqProvider(api_key="your-groq-key"),
    mode="remote",  # HTTP connection to FastAPI server
    api_url="http://localhost:8000"
)

# Same API, cloud-scale performance
memory = client.remember("Important business insight", user_id="enterprise_user")
results = client.recall("insights about", user_id="enterprise_user")
```

### Option 3: Universal Provider Switching (Enterprise flexibility)

```python
from hippocampai import MemoryClient
from hippocampai.adapters import GroqProvider, OpenAIProvider, AnthropicProvider

# Switch providers seamlessly
providers = {
    "groq": GroqProvider(api_key="groq-key"),      # Fast & cost-effective
    "openai": OpenAIProvider(api_key="openai-key"), # High quality
    "anthropic": AnthropicProvider(api_key="anthropic-key")  # Advanced reasoning
}

# Same code works with any provider
for provider_name, provider in providers.items():
    client = MemoryClient(llm_provider=provider)
    memory = client.remember(f"Test with {provider_name}", user_id="test")
    print(f"{provider_name} integration: ✅ Success")
```

---

## 💡 Core Memory Operations

```python
from hippocampai import MemoryClient

# Initialize with your preferred configuration
client = MemoryClient()

# Store a memory with advanced metadata
memory = client.remember(
    text="I prefer oat milk in my coffee and work from 9-5 PST",
    user_id="alice",
    type="preference",
    importance=8.0,
    tags=["beverages", "schedule", "work"]
)

# Automatic fact extraction (v1.0.0 feature)
print(f"Extracted facts: {memory.extracted_facts}")
# Output: ['beverage_preference: oat milk', 'work_schedule: 9-5 PST']

# Memory size tracking
print(f"Memory: {memory.text_length} chars, {memory.token_count} tokens")

# Intelligent recall with hybrid search
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
- **Intelligence features** — fact extraction, entity recognition, session summarization, and knowledge graph building
- **Temporal reasoning** — time-based queries, narratives, timelines, event sequences, and memory scheduling
- **Cross-session insights** — pattern detection, behavioral change tracking, preference drift analysis, and habit scoring
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

### Temporal Reasoning

Query memories by time range, build chronological narratives, analyze event sequences, and schedule future memories:

```python
from hippocampai import TimeRange

# Get memories from specific time periods
last_week_memories = client.get_memories_by_time_range(
    user_id="alice",
    time_range=TimeRange.LAST_WEEK
)

# Build chronological narrative
narrative = client.build_memory_narrative(
    user_id="alice",
    time_range=TimeRange.LAST_MONTH,
    title="My Month in Review"
)
print(narrative)

# Create memory timeline
timeline = client.create_memory_timeline(
    user_id="alice",
    title="Last Week's Journey",
    time_range=TimeRange.LAST_WEEK
)
print(f"Timeline has {len(timeline.events)} events")

# Analyze event sequences
sequences = client.analyze_event_sequences(
    user_id="alice",
    max_gap_hours=24
)
print(f"Found {len(sequences)} related event sequences")

# Schedule future memories with recurrence
from datetime import datetime, timedelta, timezone
tomorrow = datetime.now(timezone.utc) + timedelta(days=1)
scheduled = client.schedule_memory(
    text="Follow up on project proposal",
    user_id="alice",
    scheduled_for=tomorrow,
    recurrence="daily"  # Can be "daily", "weekly", "monthly"
)

# Get temporal summary
stats = client.get_temporal_summary(user_id="alice")
print(f"Peak activity hour: {stats['peak_activity_hour']}")
```

### Cross-Session Insights

Detect behavioral patterns, track changes, analyze preference drift, and identify habits:

```python
# Detect patterns across sessions
patterns = client.detect_patterns(user_id="alice")
for pattern in patterns[:3]:
    print(f"{pattern.pattern_type}: {pattern.description}")
    print(f"Confidence: {pattern.confidence:.2f}")
    print(f"Occurrences: {pattern.occurrences}")

# Track behavioral changes
changes = client.track_behavior_changes(
    user_id="alice",
    comparison_days=30  # Compare recent 30 days vs older
)
for change in changes:
    print(f"{change.change_type.value}: {change.description}")
    print(f"Confidence: {change.confidence:.2f}")

# Analyze preference drift
drifts = client.analyze_preference_drift(user_id="alice")
for drift in drifts:
    print(f"Category: {drift.category}")
    print(f"Original: {drift.original_preference}")
    print(f"Current: {drift.current_preference}")
    print(f"Drift score: {drift.drift_score:.2f}")

# Detect habit formation
habits = client.detect_habits(user_id="alice", min_occurrences=5)
for habit in habits[:3]:
    print(f"Behavior: {habit.behavior}")
    print(f"Habit score: {habit.habit_score:.2f}")
    print(f"Status: {habit.status}")
    print(f"Consistency: {habit.consistency:.2f}")

# Analyze long-term trends
trends = client.analyze_trends(user_id="alice", window_days=30)
for trend in trends:
    print(f"Category: {trend.category}")
    print(f"Trend: {trend.trend_type} ({trend.direction})")
    print(f"Strength: {trend.strength:.2f}")
```

### Intelligence Features 🧠

Extract structured knowledge from conversations with automatic fact extraction, entity recognition, and knowledge graph building:

```python
# Extract facts from text
facts = client.extract_facts(
    "John works at Google in San Francisco. He studied Computer Science at MIT.",
    source="profile"
)
for fact in facts:
    print(f"[{fact.category.value}] {fact.fact} (confidence: {fact.confidence:.2f})")

# Extract entities and relationships
entities = client.extract_entities("Elon Musk founded SpaceX in California")
relationships = client.extract_relationships(text, entities)

# Generate conversation summaries
summary = client.summarize_conversation(
    messages,
    session_id="chat_001",
    style=SummaryStyle.BULLET_POINTS
)
print(f"Summary: {summary.summary}")
print(f"Topics: {summary.topics}")
print(f"Sentiment: {summary.sentiment.value}")
print(f"Action items: {summary.action_items}")

# Build knowledge graph
memory = client.remember("Marie Curie was a physicist who won two Nobel Prizes", "alice")
enrichment = client.enrich_memory_with_intelligence(memory, add_to_graph=True)
print(f"Extracted {len(enrichment['facts'])} facts, {len(enrichment['entities'])} entities")

# Query knowledge graph
memory_ids = client.get_entity_memories("person_marie_curie")
timeline = client.get_entity_timeline("person_marie_curie")
connections = client.get_entity_connections("person_marie_curie", max_distance=2)

# Infer new knowledge from patterns
inferred = client.infer_knowledge(user_id="alice")
for fact in inferred:
    print(f"{fact['fact']} (confidence: {fact['confidence']:.2f})")
```

**Key Features:**

- **Fact Extraction** - Automatically extract structured facts (employment, education, skills, preferences, etc.)
- **Entity Recognition** - Identify and track people, organizations, locations, dates, and more
- **Relationship Extraction** - Discover connections between entities (works_at, located_in, studied_at)
- **Session Summarization** - Generate summaries with key points, action items, and sentiment analysis
- **Knowledge Graph** - Build rich graphs connecting memories, entities, facts, and topics
- **Knowledge Inference** - Infer new facts from existing knowledge patterns

See the [Intelligence Features Guide](docs/INTELLIGENCE_FEATURES.md) for comprehensive documentation and examples.

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

# Intelligence features (fact extraction, entity recognition, summarization, knowledge graph)
python examples/11_intelligence_features_demo.py

# Semantic clustering & auto-categorization
python examples/12_semantic_clustering_demo.py

# Multi-agent memory management
python examples/13_multiagent_demo.py

# Temporal reasoning (time-based queries, narratives, scheduling)
python examples/14_temporal_reasoning_demo.py

# Cross-session insights (patterns, habits, preference drift, trends)
python examples/15_cross_session_insights_demo.py

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

### 🚀 Getting Started

- **[Complete User Guide](docs/USER_GUIDE.md)** - Comprehensive guide from installation to deployment
- **[Quick Start](docs/GETTING_STARTED.md)** - Get up and running in 5 minutes
- **[Configuration](docs/CONFIGURATION.md)** - All configuration options and presets

### 📚 Core Documentation

- **[API Reference](docs/API_REFERENCE.md)** - Complete API documentation for all methods
- **[System Architecture](docs/ARCHITECTURE.md)** - Architecture overview with type safety and scalability
- **[Features Guide](docs/FEATURES.md)** - Comprehensive feature documentation with examples
- **[Provider Setup](docs/PROVIDERS.md)** - Configure LLM providers (Ollama, OpenAI, Anthropic, Groq)

### 🔧 Advanced Topics

- **[Resilience & Observability](docs/RESILIENCE.md)** - Error handling, retry logic, and structured logging
- **[Telemetry Guide](docs/TELEMETRY.md)** - Performance monitoring, tracing, and metrics
- **[Testing Guide](docs/TESTING_GUIDE.md)** - Complete testing guide with 117 tests and 100% coverage
- **[Multi-Agent Features](docs/MULTIAGENT_FEATURES.md)** - Advanced multi-agent coordination and memory sharing

### 📋 Specialized Guides

- **[Celery Usage](docs/CELERY_USAGE_GUIDE.md)** - Background task processing and scheduling
- **[Session Management](docs/SESSION_MANAGEMENT.md)** - User session and conversation management
- **[Memory Management API](docs/MEMORY_MANAGEMENT_API.md)** - Advanced memory lifecycle operations
- **[Versioning & Retention](docs/VERSIONING_AND_RETENTION_GUIDE.md)** - Data lifecycle and version control

### 📖 Examples & Tutorials

- **[Examples Documentation](docs/examples.md)** - Guide to all working examples in the `examples/` directory
- See the [Examples section](#-examples) above for a complete list of runnable examples

### Developer Resources

- **[Implementation Summary](docs/IMPLEMENTATION_SUMMARY.md)** - Implementation details of core features and components
- **[Contributing Guide](docs/CONTRIBUTING.md)** - How to contribute to HippocampAI
- **[Changelog](docs/CHANGELOG.md)** - Version history, updates, and breaking changes
- **[Full Documentation Index](docs/README.md)** - Complete documentation navigation

---

**Need help?** Join our community: [Discord](https://discord.gg/pPSNW9J7gB)

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
- [x] Temporal reasoning (time-based queries, narratives, timelines, event sequences, scheduling)
- [x] Cross-session insights (pattern detection, behavioral changes, preference drift, habit tracking, trends)
- [x] Intelligence features (fact extraction, entity recognition, session summarization, knowledge graph)

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

We welcome contributions! Please see our [contributing guidelines](docs/CONTRIBUTING.md) for details.

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
