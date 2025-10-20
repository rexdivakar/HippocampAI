# HippocampAI Examples

This directory contains working examples demonstrating HippocampAI's core features.

## Prerequisites

1. Install HippocampAI with dependencies:
   ```bash
   pip install -e ".[core]"
   ```

2. Start Qdrant:
   ```bash
   docker run -p 6333:6333 qdrant/qdrant
   ```

3. Configure environment:
   ```bash
   cp .env.example .env
   # Edit .env and set your configuration
   ```

## Examples

### 01_basic_usage.py
**Basic remember/recall workflow**

Demonstrates:
- Initializing MemoryClient
- Storing memories (preferences, facts, goals, events)
- Recalling memories with queries
- Viewing memory scores and importance

```bash
python examples/01_basic_usage.py
```

### 02_conversation_extraction.py
**Extract memories from conversations**

Demonstrates:
- Automatic memory extraction from conversation text
- Different memory types (preferences, facts, goals)
- Importance scoring
- Querying extracted memories

```bash
python examples/02_conversation_extraction.py
```

### 03_hybrid_retrieval.py
**Hybrid retrieval demonstration**

Demonstrates:
- Vector search + BM25 fusion
- Cross-encoder reranking
- Score breakdowns (similarity, rerank, recency, importance)
- Diverse memory types and queries

```bash
python examples/03_hybrid_retrieval.py
```

### 04_custom_configuration.py
**Custom configuration options**

Demonstrates:
- Default configuration
- Parameter overrides
- Custom Config objects
- Retrieval weight tuning
- Collection name customization

```bash
python examples/04_custom_configuration.py
```

### 05_multi_user.py
**Multi-user memory management**

Demonstrates:
- User isolation
- Per-user memory stores
- User-specific retrieval
- Memory statistics per user

```bash
python examples/05_multi_user.py
```

## Run All Examples

```bash
# From project root
./run_examples.sh
```

Or individually:
```bash
python examples/01_basic_usage.py
python examples/02_conversation_extraction.py
python examples/03_hybrid_retrieval.py
python examples/04_custom_configuration.py
python examples/05_multi_user.py
```

## Next Steps

- Explore the [CLI chat interface](../cli_chat.py)
- Try the [web interface](../web_chat.py)
- Read the [full documentation](../docs/)
- Check the [API reference](../docs/API.md)
