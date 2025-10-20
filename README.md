# HippocampAI — Autonomous Memory Engine for LLM Agents

[![Quality Gate Status](https://sonar.craftedbrain.com/api/project_badges/measure?project=rexdivakar_HippocampAI_6669aa8c-2e81-4016-9993-b29a3a78c475&metric=alert_status&token=sqb_dd0c0b1bf58646ce474b64a1fa8d83446345bccf)](https://sonar.craftedbrain.com/dashboard?id=rexdivakar_HippocampAI_6669aa8c-2e81-4016-9993-b29a3a78c475)

HippocampAI turns raw conversations into a curated long-term memory vault for your AI assistants. It extracts, scores, deduplicates, stores, and retrieves user memories so agents can stay personal, consistent, and context-aware across sessions.

- **Plug-and-play** `MemoryClient` API with built-in pipelines for extraction, dedupe, consolidation, and importance decay
- **Hybrid retrieval** that fuses dense vectors, BM25, reciprocal-rank fusion, reranking, recency, and importance signals
- **Self-hosted first** — works fully offline via Qdrant + Ollama or in the cloud via OpenAI with the same code paths
- **Production-ready** — ships with CLI and web chat UIs, telemetry, typed models, scheduled jobs, and comprehensive logging
- **Fully customizable** — every component (extraction, retrieval, scoring) is extensible without vendor lock-in

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
client.remember(
    text="I prefer oat milk in my coffee",
    user_id="alice",
    type="preference",
    importance=8.0
)

# Recall relevant memories
results = client.recall(
    query="How does Alice like her coffee?",
    user_id="alice",
    k=3
)

for result in results:
    print(f"{result.memory.text} (score: {result.score:.3f})")
```

---

## 🎯 Features

### Memory Management

- **Automatic extraction** from conversations using heuristics or LLM
- **Deduplication** with semantic similarity and reranking
- **Importance scoring** with configurable decay
- **Multi-user isolation** — complete data separation per user

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

### Telemetry & Observability

Track all memory operations with built-in telemetry (library access only):

```python
from hippocampai import MemoryClient

client = MemoryClient()

# Memory operations are automatically tracked
client.remember(text="I love Python", user_id="alice")
client.recall(query="What do I like?", user_id="alice")

# Access telemetry via client
metrics = client.get_telemetry_metrics()
print(f"Average recall time: {metrics['recall_duration']['avg']:.2f}ms")

# Get recent operations
operations = client.get_recent_operations(limit=10)
for op in operations:
    print(f"{op.operation.value}: {op.duration_ms:.2f}ms ({op.status})")

# Export telemetry data
exported = client.export_telemetry()

# Or access global telemetry instance
from hippocampai import get_telemetry
telemetry = get_telemetry()
traces = telemetry.get_recent_traces(limit=10)
```

---

## 🖥️ Interfaces

### CLI Chat

```bash
python cli_chat.py alice

# Commands:
# /help      - Show help
# /stats     - Show memory statistics
# /memories  - View stored memories
# /clear     - Clear session
# /quit      - Exit
```

### Web Interface

```bash
python web_chat.py
```

Then open http://localhost:5000

### FastAPI Server

```bash
python -m hippocampai.api.app

# Endpoints:
# POST /v1/memories:remember - Store a memory
# POST /v1/memories:recall   - Retrieve memories
# POST /v1/memories:extract  - Extract from conversation
# GET  /healthz              - Health check
```

**Note:** Telemetry data is accessed via library functions, not REST endpoints.

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

- [Quick Start Guide](docs/QUICKSTART.md) - Get up and running in 5 minutes
- [Configuration Guide](docs/CONFIGURATION.md) - All configuration options explained
- [Provider Setup](docs/PROVIDERS.md) - Configure Ollama, OpenAI, Anthropic, Groq
- [API Reference](docs/API.md) - Complete API documentation
- [Architecture](docs/ARCHITECTURE.md) - System design and components
- [Telemetry Guide](docs/TELEMETRY.md) - Observability and tracing

Need help? Join our community: [Discord](https://discord.gg/pPSNW9J7gB)

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

- [x] Configuration presets (`.from_preset("local")`, `.from_preset("cloud")`) ✅ v0.1.0
- [x] Built-in telemetry and observability ✅ v0.1.0
- [ ] LangChain and LlamaIndex integrations
- [ ] Retrieval evaluators and A/B testing
- [ ] Multi-tenant RBAC and access policies
- [ ] Native TypeScript SDK
- [ ] Grafana/Prometheus metrics exporters
- [ ] Memory versioning and time-travel queries

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
- **Email**: rexdivakar@hotmail.com

Built with ❤️ by the HippocampAI team
