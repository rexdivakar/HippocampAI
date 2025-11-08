# HippocampAI — Enterprise Memory Engine for Intelligent AI Systems

[![PyPI version](https://badge.fury.io/py/hippocampai.svg)](https://pypi.org/project/hippocampai/)
[![Python Versions](https://img.shields.io/pypi/pyversions/hippocampai.svg)](https://pypi.org/project/hippocampai/)
[![Downloads](https://pepy.tech/badge/hippocampai)](https://pepy.tech/project/hippocampai)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-comprehensive-brightgreen.svg)](docs/)

**HippocampAI** is a production-ready, enterprise-grade memory engine that transforms how AI systems remember, reason, and learn from interactions. It provides persistent, intelligent memory capabilities that enable AI agents to maintain context across sessions, understand user preferences, detect behavioral patterns, and deliver truly personalized experiences.

> **The name "HippocampAI"** draws inspiration from the hippocampus - the brain region responsible for memory formation and retrieval - reflecting our mission to give AI systems human-like memory capabilities.

**Current Release:** v0.2.5 — Production-ready with 102+ methods, 50+ API endpoints, and comprehensive monitoring.

---

## Quick Start

### Installation

```bash
# Install from PyPI
pip install hippocampai

# Or with all features
pip install "hippocampai[all]"
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
| **Background Tasks** | Celery-powered async operations, scheduled jobs | [Celery Guide](docs/CELERY_USAGE_GUIDE.md) |
| **Monitoring & Tracking** | Prometheus, Grafana dashboards, memory lifecycle tracking | [Monitoring](MONITORING_INTEGRATION_GUIDE.md) \| [Memory Tracking](MEMORY_TRACKING_GUIDE.md) |

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

### Quick Links

| What do you want to do? | Go here |
|--------------------------|---------|
| **Get started in 5 minutes** | [Getting Started Guide](docs/GETTING_STARTED.md) |
| **See all 102+ functions** | [Library Complete Reference](docs/LIBRARY_COMPLETE_REFERENCE.md) |
| **Use REST APIs** | [SaaS API Complete Reference](docs/SAAS_API_COMPLETE_REFERENCE.md) |
| **Deploy to production** | [User Guide](docs/USER_GUIDE.md) |
| **Configure settings** | [Configuration Guide](docs/CONFIGURATION.md) |
| **Optimize Celery workers** | [Celery Optimization & Tracing](docs/CELERY_OPTIMIZATION_AND_TRACING.md) |
| **Troubleshoot issues** | [Troubleshooting Guide](docs/TROUBLESHOOTING.md) |

### Documentation Index

**[Complete Documentation Index](docs/README.md)** - Browse all 26 documentation files organized by topic

**Core Documentation:**
- [API Reference](docs/API_REFERENCE.md) - All 102+ methods with examples
- [Features Overview](docs/FEATURES.md) - Comprehensive feature documentation
- [Architecture](docs/ARCHITECTURE.md) - System design and components
- [Configuration](docs/CONFIGURATION.md) - All configuration options

**Advanced Topics:**
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
- Celery workers with Beat scheduler
- Flower monitoring (port 5555)
- Prometheus metrics (port 9090)
- Grafana dashboards (port 3000)

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

### Interactive Chat Demo

Try the fully-functional chatbot with persistent memory:

```bash
# Set your Groq API key
export GROQ_API_KEY="your-api-key-here"

# Run the interactive chat
python chat.py
```

**Features:**
- Persistent memory across sessions
- Context-aware responses
- Pattern detection
- Memory search
- Session summaries

[Complete chat demo guide →](docs/CHAT_DEMO_GUIDE.md)

### Code Examples

Over 15 working examples in the `examples/` directory:

```bash
# Basic operations
python examples/01_basic_usage.py

# Advanced features
python examples/11_intelligence_features_demo.py
python examples/13_temporal_reasoning_demo.py
python examples/14_cross_session_insights_demo.py

# Run all examples
./run_examples.sh
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

# See docs/LIBRARY_COMPLETE_REFERENCE.md for all 102+ methods
```

**[Full API Reference](docs/LIBRARY_COMPLETE_REFERENCE.md)** | **[REST API Reference](docs/SAAS_API_COMPLETE_REFERENCE.md)**
