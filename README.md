# HippocampAI — Enterprise Memory Engine for Intelligent AI Systems

[![PyPI version](https://badge.fury.io/py/hippocampai.svg)](https://pypi.org/project/hippocampai/)
[![Python Versions](https://img.shields.io/pypi/pyversions/hippocampai.svg)](https://pypi.org/project/hippocampai/)
[![Downloads](https://pepy.tech/badge/hippocampai)](https://pepy.tech/project/hippocampai)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-comprehensive-brightgreen.svg)](docs/)

**HippocampAI** is a production-ready, enterprise-grade memory engine that transforms how AI systems remember, reason, and learn from interactions. It provides persistent, intelligent memory capabilities that enable AI agents to maintain context across sessions, understand user preferences, detect behavioral patterns, and deliver truly personalized experiences.

> **The name "HippocampAI"** draws inspiration from the hippocampus - the brain region responsible for memory formation and retrieval - reflecting our mission to give AI systems human-like memory capabilities.

**Current Release:** v0.5.0 — Intelligent memory features: knowledge graph, graph-aware retrieval, feedback loops, triggers, procedural memory, and embedding migration.

---

## Package Structure

HippocampAI is organized into two main components for flexibility:

| Package | Description | Use Case |
|---------|-------------|----------|
| `hippocampai.core` | Core memory engine (no SaaS dependencies) | Library integration, embedded use |
| `hippocampai.platform` | SaaS platform (API, auth, Celery, monitoring) | Self-hosted SaaS deployment |

```python
# Core library only (lightweight)
from hippocampai.core import MemoryClient, Memory, Config

# SaaS platform features
from hippocampai.platform import run_api_server, AutomationController

# Or use the main package (includes everything, backward compatible)
from hippocampai import MemoryClient
```

---

## Quick Start

### Installation

```bash
# Core library (lightweight - 10 dependencies)
pip install hippocampai

# With SaaS features (API, auth, background tasks)
pip install "hippocampai[saas]"

# With specific LLM providers
pip install "hippocampai[openai]"     # OpenAI support
pip install "hippocampai[anthropic]"  # Anthropic Claude
pip install "hippocampai[groq]"       # Groq support

# Everything (development, all features)
pip install "hippocampai[all,dev]"
```

### Your First Memory (30 seconds)

```python
from hippocampai import MemoryClient

# Initialize client
client = MemoryClient()

# Store a memory
memory = client.remember(
    "I prefer oat milk in my coffee and work remotely on Tuesdays",
    user_id="alice",
    type="preference"
)

# Recall memories
results = client.recall("work preferences", user_id="alice")
print(f"Found: {results[0].memory.text}")
```

**That's it!** You now have intelligent memory for your AI application.

---

## Key Features

| Feature | Description | Learn More |
|---------|-------------|------------|
| **Intelligent Memory** | Hybrid search, importance scoring, semantic clustering | [Features Guide](docs/FEATURES.md) |
| **High Performance** | 50-100x faster with Redis caching, 500-1000+ RPS | [Performance](docs/ARCHITECTURE.md#performance) |
| **Advanced Search** | Vector + BM25 + reranking, temporal queries | [Search Guide](docs/FEATURES.md#hybrid-retrieval) |
| **Analytics** | Pattern detection, habit tracking, behavioral insights | [Analytics](docs/FEATURES.md#cross-session-insights) |
| **AI Integration** | Works with OpenAI, Anthropic, Groq, Ollama, local models | [Providers](docs/PROVIDERS.md) |
| **Session Management** | Conversation tracking, summaries, hierarchical sessions | [Sessions](docs/SESSION_MANAGEMENT.md) |
| **SaaS Platform** | Multi-tenant auth, rate limiting, background tasks | [SaaS Guide](docs/SAAS_GUIDE.md) |
| **Memory Quality** | Health monitoring, duplicate detection, quality tracking | [Memory Management](docs/MEMORY_MANAGEMENT.md) |
| **Background Tasks** | Celery-powered async operations, scheduled jobs | [Celery Guide](docs/CELERY_GUIDE.md) |
| **Memory Consolidation** ⭐ NEW | Sleep phase architecture with intelligent compaction | [Sleep Phase](docs/SLEEP_PHASE_ARCHITECTURE.md) |
| **Multi-Agent Collaboration** ⭐ NEW | Shared memory spaces for agent coordination | [Collaboration](docs/MULTIAGENT_FEATURES.md) |
| **React Dashboard** ⭐ NEW | Full-featured UI with analytics and visualization | [Frontend](frontend/README.md) |
| **Predictive Analytics** ⭐ NEW | Memory usage predictions and pattern forecasting | [New Features](docs/NEW_FEATURES_GUIDE.md) |
| **Auto-Healing** ⭐ NEW | Automatic detection and repair of memory issues | [New Features](docs/NEW_FEATURES_GUIDE.md) |
| **Knowledge Graph** NEW | Real-time entity/relationship extraction on every remember() | [Features](docs/FEATURES.md#real-time-incremental-knowledge-graph) |
| **Graph-Aware Retrieval** NEW | 3-way RRF fusion: vector + BM25 + graph | [Features](docs/FEATURES.md#graph-aware-retrieval) |
| **Relevance Feedback** NEW | User feedback loop with exponential decay scoring | [Features](docs/FEATURES.md#memory-relevance-feedback-loop) |
| **Memory Triggers** NEW | Event-driven webhooks, websocket, and log actions | [Features](docs/FEATURES.md#memory-triggers--event-driven-actions) |
| **Procedural Memory** NEW | Self-optimizing prompts via learned behavioral rules | [Features](docs/FEATURES.md#procedural-memory--prompt-self-optimization) |
| **Embedding Migration** NEW | Safe model migration with Celery background processing | [Features](docs/FEATURES.md#embedding-model-migration) |
| **Plugin System** | Custom processors, scorers, retrievers, filters | [New Features](docs/NEW_FEATURES.md#plugin-system) |
| **Memory Namespaces** | Hierarchical organization with permissions | [New Features](docs/NEW_FEATURES.md#memory-namespaces) |
| **Export/Import** | Portable formats (JSON, Parquet, CSV) for backup | [New Features](docs/NEW_FEATURES.md#exportimport-portability) |
| **Offline Mode** | Queue operations when backend unavailable | [New Features](docs/NEW_FEATURES.md#offline-mode) |
| **Tiered Storage** | Hot/warm/cold storage tiers for efficiency | [New Features](docs/NEW_FEATURES.md#tiered-storage) |
| **Framework Integrations** | LangChain & LlamaIndex adapters | [New Features](docs/NEW_FEATURES.md#framework-integrations) |
| **Bi-Temporal Facts** | Track facts with validity periods and time-travel queries | [Bi-Temporal Guide](docs/bi_temporal.md) |
| **Context Assembly** | Automated context pack generation with token budgeting | [Context Assembly](docs/context_assembly.md) |
| **Custom Schemas** | Define entity/relationship types without code changes | [Schema Guide](docs/custom_schema.md) |
| **Benchmarks** | Reproducible performance benchmarks | [Benchmarks](docs/benchmarks.md) |

---

## Why Choose HippocampAI?

### vs. Traditional Vector Databases
- **Built-in Intelligence**: Pattern detection, insights, behavioral analysis
- **Memory Types**: Facts, preferences, goals, habits, events (not just vectors)
- **Temporal Reasoning**: Native time-based queries and narratives

### vs. Other Memory Platforms
- **5-100x Faster**: Redis caching, optimized retrieval
- **Deployment Flexibility**: Local, self-hosted, or SaaS
- **Full Control**: Complete source access and customization

### vs. Building In-House
- **Ready in Minutes**: `pip install hippocampai`
- **102+ Methods**: Complete API covering all use cases
- **Production-Tested**: Battle-tested in real applications

[See detailed comparison →](docs/FEATURES.md#competitive-advantages)

---

## Documentation

**Complete documentation is available in the [docs/](docs/) directory.**

### Quick Links

| What do you want to do? | Go here |
|--------------------------|---------|
| **Get started in 5 minutes** | [Getting Started Guide](docs/GETTING_STARTED.md) \| [Quickstart](docs/QUICKSTART.md) |
| **Try interactive demo** | [Chat Demo Guide](docs/CHAT_DEMO_GUIDE.md) |
| **See all 102+ functions** | [API Reference](docs/API_REFERENCE.md) \| [Library Reference](docs/LIBRARY_COMPLETE_REFERENCE.md) |
| **Deploy as SaaS platform** | [SaaS Platform Guide](docs/SAAS_GUIDE.md) ⭐ NEW |
| **Monitor memory quality** | [Memory Management](docs/MEMORY_MANAGEMENT.md) ⭐ NEW |
| **Set up background tasks** | [Celery Guide](docs/CELERY_GUIDE.md) ⭐ NEW |
| **Deploy to production** | [User Guide](docs/USER_GUIDE.md) \| [Deployment](docs/DEPLOYMENT_READINESS_REPORT.md) |
| **Configure settings** | [Configuration Guide](docs/CONFIGURATION.md) \| [Providers](docs/PROVIDERS.md) |
| **Monitor & observe** | [Monitoring](docs/MONITORING_INTEGRATION_GUIDE.md) \| [Telemetry](docs/TELEMETRY.md) |
| **Troubleshoot issues** | [Troubleshooting](docs/TROUBLESHOOTING.md) |
| **Use new features** | [New Features Guide](docs/NEW_FEATURES.md) ⭐ NEW |
| **View all documentation** | [Documentation Hub](docs/README.md) |

### Documentation Index

**[Complete Documentation Index](docs/README.md)** - Browse all 26 documentation files organized by topic

**Core Documentation:**
- [API Reference](docs/API_REFERENCE.md) - All 102+ methods with examples
- [Features Overview](docs/FEATURES.md) - Comprehensive feature documentation
- [Architecture](docs/ARCHITECTURE.md) - System design and components
- [Configuration](docs/CONFIGURATION.md) - All configuration options

**Advanced Topics:**
- [New Features](docs/NEW_FEATURES.md) - Plugins, namespaces, export/import, offline mode, tiered storage
- [Multi-Agent Features](docs/MULTIAGENT_FEATURES.md) - Agent coordination
- [Intelligence Features](docs/FEATURES.md#intelligence-features) - Fact extraction, entity recognition
- [Temporal Analytics](docs/FEATURES.md#temporal-reasoning) - Time-based queries
- [Session Management](docs/SESSION_MANAGEMENT.md) - Conversation tracking

**Production & Operations:**
- [User Guide](docs/USER_GUIDE.md) - Production deployment
- [Security](docs/SECURITY.md) - Security best practices
- [Monitoring](docs/TELEMETRY.md) - Observability and metrics
- [Backup & Recovery](docs/BACKUP_RECOVERY.md) - Data protection

---

## Configuration

### Local Development

```bash
# .env file
QDRANT_URL=http://localhost:6333
LLM_PROVIDER=ollama
LLM_MODEL=qwen2.5:7b-instruct
```

### Cloud/Production

```python
from hippocampai import MemoryClient
from hippocampai.adapters import GroqLLM

client = MemoryClient(
    llm_provider=GroqLLM(api_key="your-key"),
    qdrant_url="https://your-qdrant-cluster.com",
    redis_url="redis://your-redis:6379"
)
```

[See all configuration options →](docs/CONFIGURATION.md)

---

## Deployment Options

### Local Development
```bash
docker run -d -p 6333:6333 qdrant/qdrant
pip install hippocampai
```

### Production Stack
```bash
git clone https://github.com/rexdivakar/HippocampAI.git
cd HippocampAI
docker-compose up -d  # Includes Qdrant, Redis, API, Celery, Monitoring
```

**Includes:**
- FastAPI server (port 8000)
- React Dashboard (port 3001)
- Celery workers with Beat scheduler
- Flower monitoring (port 5555)
- Prometheus metrics (port 9090)
- Grafana dashboards (port 3000)

### React Dashboard (New in v0.4.0)
```bash
cd frontend
npm install
npm run dev  # Development server on port 5173
```

[Production deployment guide →](docs/USER_GUIDE.md)

---

## Use Cases

**AI Agents & Chatbots**
- Personalized assistants with context across sessions
- Customer support with interaction history
- Educational tutoring that adapts to students

**Enterprise Applications**
- Knowledge management for teams
- CRM enhancement with interaction intelligence
- Compliance monitoring and audit trails

**Research & Analytics**
- Behavioral pattern analysis
- Long-term trend detection
- User experience personalization

[More use cases →](docs/FEATURES.md#use-cases)

---

## Performance

| Metric | Performance |
|--------|-------------|
| **Query Speed** | 50-100x faster with caching |
| **Throughput** | 500-1000+ requests/second |
| **Latency** | 1-2ms (cached), 5-15ms (uncached) |
| **Availability** | 99.9% uptime |

[See benchmarks →](docs/ARCHITECTURE.md#performance-benchmarks)

---

## Community & Support

- **Documentation**: [Complete guides](docs/)
- **Issues**: [GitHub Issues](https://github.com/rexdivakar/HippocampAI/issues)
- **Discussions**: [GitHub Discussions](https://github.com/rexdivakar/HippocampAI/discussions)
- **Discord**: [Join our community](https://discord.gg/pPSNW9J7gB)

---

## Examples

### Code Examples

Over 25 working examples in the `examples/` directory:

```bash
# Basic operations
python examples/01_basic_usage.py

# Advanced features
python examples/11_intelligence_features_demo.py
python examples/13_temporal_reasoning_demo.py
python examples/14_cross_session_insights_demo.py

# New v0.4.0 features
python examples/20_collaboration_demo.py      # Multi-agent collaboration
python examples/21_predictive_analytics_demo.py  # Predictive analytics
python examples/22_auto_healing_demo.py       # Auto-healing pipeline
python examples/consolidation_demo.py         # Memory consolidation
```

[View all examples →](examples/)

---

## Contributing

We welcome contributions! See our [Contributing Guide](docs/CONTRIBUTING.md) for details.

```bash
git clone https://github.com/rexdivakar/HippocampAI.git
cd HippocampAI
pip install -e ".[dev]"
pytest
```

---

## License

**Apache 2.0** - Use freely in commercial and open-source projects.

---

## Star History

If you find HippocampAI useful, please star the repo! It helps others discover the project.

---

**Built with by the HippocampAI team**

---

## Quick Reference Card

```python
from hippocampai import MemoryClient

client = MemoryClient()

# Core operations
memory = client.remember("text", user_id="alice")
results = client.recall("query", user_id="alice", k=5)
client.update_memory(memory_id, text="new text")
client.delete_memory(memory_id)

# Intelligence
facts = client.extract_facts("John works at Google")
entities = client.extract_entities("Elon Musk founded SpaceX")
patterns = client.detect_patterns(user_id="alice")

# Analytics
habits = client.detect_habits(user_id="alice")
changes = client.track_behavior_changes(user_id="alice")
stats = client.get_memory_statistics(user_id="alice")

# Sessions
session = client.create_session(user_id="alice", title="Planning")
client.complete_session(session.id, generate_summary=True)

# Bi-Temporal Facts (NEW)
from hippocampai.models.bitemporal import BiTemporalQuery
fact = client.store_bitemporal_fact(
    user_id="alice",
    subject="alice",
    predicate="works_at",
    object_value="Acme Corp",
    valid_from=datetime(2024, 1, 1),
)
facts = client.query_bitemporal_facts(BiTemporalQuery(
    user_id="alice",
    valid_at=datetime(2024, 6, 1),
))

# Context Assembly (NEW)
from hippocampai.context.models import ContextConstraints
context = client.assemble_context(
    user_id="alice",
    query="What are Alice's work preferences?",
    constraints=ContextConstraints(token_budget=4000),
)
print(context.final_context_text)

# Custom Schema Validation (NEW)
from hippocampai.schema import SchemaRegistry
registry = SchemaRegistry()
result = registry.validate_entity("person", {"name": "Alice"})

# Relevance Feedback (NEW v0.5.0)
client.rate_recall(
    memory_id=results[0].memory.id,
    user_id="alice",
    query="coffee preferences",
    feedback_type="relevant"
)

# See docs/LIBRARY_COMPLETE_REFERENCE.md for the full method reference
```

**[Full API Reference](docs/LIBRARY_COMPLETE_REFERENCE.md)** | **[REST API Reference](docs/SAAS_API_COMPLETE_REFERENCE.md)**
