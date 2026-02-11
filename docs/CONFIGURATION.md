# Configuration Guide

HippocampAI uses **Pydantic BaseSettings** for configuration, loaded exclusively from environment variables and `.env` files. There is no YAML configuration.

## Quick Setup

Create a `.env` file from the template:

```bash
cp .env.example .env
```

Minimal configuration for local development:

```env
QDRANT_URL=http://localhost:6333
LLM_PROVIDER=ollama
LLM_MODEL=qwen2.5:7b-instruct
LLM_BASE_URL=http://localhost:11434
```

---

## Complete Configuration Reference

All fields are defined in `src/hippocampai/config.py` as a `Config(BaseSettings)` class. Environment variables are case-insensitive.

### Core — Qdrant Vector Database

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `QDRANT_URL` | str | `http://localhost:6333` | Qdrant server URL |
| `COLLECTION_FACTS` | str | `hippocampai_facts` | Collection name for facts |
| `COLLECTION_PREFS` | str | `hippocampai_prefs` | Collection name for preferences |

### HNSW Index Tuning

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `HNSW_M` | int | `48` | HNSW graph connectivity (higher = better recall, more memory) |
| `EF_CONSTRUCTION` | int | `256` | Build-time search depth (higher = better index quality) |
| `EF_SEARCH` | int | `128` | Query-time search depth (higher = better recall, slower) |

### Embeddings

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `EMBED_MODEL` | str | `BAAI/bge-small-en-v1.5` | Sentence-transformers embedding model |
| `EMBED_DIMENSION` | int | `384` | Embedding vector dimension |
| `EMBED_BATCH_SIZE` | int | `32` | Batch size for encoding |
| `EMBED_QUANTIZED` | bool | `false` | Use quantized model for lower memory |

### Reranker

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `RERANKER_MODEL` | str | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder reranker model |
| `RERANK_CACHE_TTL` | int | `86400` | Rerank cache TTL in seconds (24h) |

### BM25

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `BM25_BACKEND` | str | `rank-bm25` | BM25 implementation backend |

### LLM Providers

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `LLM_PROVIDER` | str | `ollama` | Provider: `ollama`, `openai`, `anthropic`, `groq` |
| `LLM_MODEL` | str | `qwen2.5:7b-instruct` | Model name for the selected provider |
| `LLM_BASE_URL` | str | `http://localhost:11434` | Base URL (used by Ollama) |
| `ALLOW_CLOUD` | bool | `false` | Allow cloud LLM providers |

Provider-specific keys (set as environment variables):
- **OpenAI**: `OPENAI_API_KEY`
- **Anthropic**: `ANTHROPIC_API_KEY`
- **Groq**: `GROQ_API_KEY`

### Retrieval Parameters

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `TOP_K_QDRANT` | int | `200` | Candidates from vector search |
| `TOP_K_FINAL` | int | `20` | Final results after reranking |
| `RRF_K` | int | `60` | Reciprocal rank fusion constant |

### Scoring Weights

All weights are used in the final score fusion. They are auto-normalized at runtime via `get_weights()`.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `WEIGHT_SIM` | float | `0.55` | Semantic similarity weight |
| `WEIGHT_RERANK` | float | `0.20` | Cross-encoder reranker weight |
| `WEIGHT_RECENCY` | float | `0.15` | Recency decay weight |
| `WEIGHT_IMPORTANCE` | float | `0.10` | Importance score weight |
| `WEIGHT_GRAPH` | float | `0.0` | Graph-aware retrieval weight (v0.5.0) |
| `WEIGHT_FEEDBACK` | float | `0.1` | Relevance feedback weight (v0.5.0) |

### Decay Half-Lives

Controls how quickly importance decays per memory type.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `HALF_LIFE_PREFS` | int | `90` | Preferences/goals/habits half-life (days) |
| `HALF_LIFE_FACTS` | int | `30` | Facts/context half-life (days) |
| `HALF_LIFE_EVENTS` | int | `14` | Events half-life (days) |
| `HALF_LIFE_PROCEDURAL` | int | `180` | Procedural rules half-life (days) (v0.5.0) |

### Redis

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `REDIS_URL` | str | `redis://localhost:6379` | Redis connection URL |
| `REDIS_DB` | int | `0` | Redis database number |
| `REDIS_CACHE_TTL` | int | `300` | Cache TTL in seconds (5 min) |
| `REDIS_MAX_CONNECTIONS` | int | `100` | Connection pool max size |
| `REDIS_MIN_IDLE` | int | `10` | Minimum idle connections |

### PostgreSQL

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `POSTGRES_HOST` | str | `localhost` | PostgreSQL host |
| `POSTGRES_PORT` | int | `5432` | PostgreSQL port |
| `POSTGRES_DB` | str | `hippocampai` | Database name |
| `POSTGRES_USER` | str | `hippocampai` | Database user |
| `POSTGRES_PASSWORD` | str | `hippocampai_secret` | Database password |

### Background Tasks & Scheduling

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `ENABLE_SCHEDULER` | bool | `true` | Enable job scheduler |
| `DECAY_CRON` | str | `0 2 * * *` | Importance decay schedule (2am daily) |
| `CONSOLIDATE_CRON` | str | `0 3 * * 0` | Consolidation schedule (3am Sunday) |
| `SNAPSHOT_CRON` | str | `0 * * * *` | Snapshot schedule (hourly) |
| `ENABLE_BACKGROUND_TASKS` | bool | `true` | Enable background task runner |
| `AUTO_DEDUP_ENABLED` | bool | `true` | Auto-deduplication |
| `AUTO_CONSOLIDATION_ENABLED` | bool | `false` | Auto-consolidation |
| `DEDUP_INTERVAL_HOURS` | int | `24` | Dedup interval |
| `CONSOLIDATION_INTERVAL_HOURS` | int | `168` | Consolidation interval (7 days) |
| `EXPIRATION_INTERVAL_HOURS` | int | `1` | Expiration check interval |
| `DEDUP_THRESHOLD` | float | `0.88` | Deduplication similarity threshold |
| `CONSOLIDATION_THRESHOLD` | float | `0.85` | Consolidation similarity threshold |

### Auto-Summarization

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `AUTO_SUMMARIZATION_ENABLED` | bool | `true` | Enable auto-summarization |
| `HIERARCHICAL_SUMMARIZATION_ENABLED` | bool | `true` | Enable hierarchical summaries |
| `SLIDING_WINDOW_ENABLED` | bool | `true` | Enable sliding window |
| `SLIDING_WINDOW_SIZE` | int | `10` | Window size |
| `SLIDING_WINDOW_KEEP_RECENT` | int | `5` | Recent items to keep |
| `MAX_TOKENS_PER_SUMMARY` | int | `150` | Max tokens per summary |
| `HIERARCHICAL_BATCH_SIZE` | int | `5` | Batch size for hierarchical summarization |
| `HIERARCHICAL_MAX_LEVELS` | int | `3` | Maximum summarization levels |

### Memory Tiering

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `HOT_THRESHOLD_DAYS` | int | `7` | Hot tier threshold |
| `WARM_THRESHOLD_DAYS` | int | `30` | Warm tier threshold |
| `COLD_THRESHOLD_DAYS` | int | `90` | Cold tier threshold |
| `HOT_ACCESS_COUNT_THRESHOLD` | int | `10` | Access count for hot tier |

### Importance Decay

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `IMPORTANCE_DECAY_ENABLED` | bool | `true` | Enable importance decay |
| `DECAY_FUNCTION` | str | `exponential` | Function: `linear`, `exponential`, `logarithmic`, `step`, `hybrid` |
| `DECAY_INTERVAL_HOURS` | int | `24` | Decay application interval |
| `MIN_IMPORTANCE_THRESHOLD` | float | `1.0` | Minimum importance floor |
| `ACCESS_BOOST_FACTOR` | float | `0.5` | Importance boost on access |

### Pruning

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `AUTO_PRUNING_ENABLED` | bool | `false` | Enable auto-pruning |
| `PRUNING_INTERVAL_HOURS` | int | `168` | Pruning interval (weekly) |
| `PRUNING_STRATEGY` | str | `comprehensive` | Strategy: `importance_only`, `age_based`, `access_based`, `comprehensive`, `conservative` |
| `MIN_HEALTH_THRESHOLD` | float | `3.0` | Minimum health score |
| `PRUNING_TARGET_PERCENTAGE` | float | `0.1` | Target prune percentage (10%) |

### Conflict Resolution

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `ENABLE_CONFLICT_RESOLUTION` | bool | `true` | Enable conflict resolution |
| `CONFLICT_RESOLUTION_STRATEGY` | str | `temporal` | Strategy: `temporal`, `confidence`, `importance`, `user_review`, `auto_merge`, `keep_both` |
| `CONFLICT_SIMILARITY_THRESHOLD` | float | `0.75` | Similarity threshold for conflict detection |
| `CONFLICT_CONTRADICTION_THRESHOLD` | float | `0.85` | Contradiction detection threshold |
| `AUTO_RESOLVE_CONFLICTS` | bool | `true` | Auto-resolve conflicts |
| `CONFLICT_CHECK_LLM` | bool | `true` | Use LLM for contradiction analysis |
| `CONFLICT_RESOLUTION_INTERVAL_HOURS` | int | `24` | Resolution check interval |

### Real-Time Knowledge Graph (v0.5.0)

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `ENABLE_REALTIME_GRAPH` | bool | `true` | Enable auto-extraction on `remember()` |
| `GRAPH_EXTRACTION_MODE` | str | `pattern` | Extraction mode: `pattern` or `llm` |
| `GRAPH_PERSISTENCE_PATH` | str | `data/knowledge_graph.json` | Path for graph persistence |
| `GRAPH_AUTO_SAVE_INTERVAL` | int | `300` | Auto-save interval in seconds |

### Graph-Aware Retrieval (v0.5.0)

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `ENABLE_GRAPH_RETRIEVAL` | bool | `false` | Enable graph-based retrieval scoring |
| `GRAPH_RETRIEVAL_MAX_DEPTH` | int | `2` | Maximum graph traversal depth |
| `WEIGHT_GRAPH` | float | `0.0` | Graph score weight in fusion (see Scoring Weights) |

### Memory Relevance Feedback (v0.5.0)

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `WEIGHT_FEEDBACK` | float | `0.1` | Feedback score weight in fusion (see Scoring Weights) |
| `FEEDBACK_WINDOW_DAYS` | int | `90` | Feedback lookback window in days |

### Memory Triggers (v0.5.0)

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `ENABLE_TRIGGERS` | bool | `true` | Enable event-driven triggers |
| `TRIGGER_WEBHOOK_TIMEOUT` | int | `10` | Webhook timeout in seconds |

### Procedural Memory (v0.5.0)

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `ENABLE_PROCEDURAL_MEMORY` | bool | `false` | Enable procedural memory rules |
| `PROCEDURAL_RULE_MAX_COUNT` | int | `50` | Maximum rules per user |
| `HALF_LIFE_PROCEDURAL` | int | `180` | Rule effectiveness half-life in days |

### Embedding Migration (v0.5.0)

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `EMBED_MODEL_VERSION` | str | `1` | Current embedding model version tag |

---

## Programmatic Access

```python
from hippocampai.config import Config, get_config

# Singleton access (recommended)
config = get_config()

# Direct instantiation (loads from .env and environment)
config = Config()

# Access fields
print(config.qdrant_url)          # "http://localhost:6333"
print(config.embed_model)         # "BAAI/bge-small-en-v1.5"
print(config.weight_sim)          # 0.55
print(config.enable_triggers)     # True

# Get all scoring weights as dict (includes graph and feedback)
weights = config.get_weights()
# {"sim": 0.55, "rerank": 0.2, "recency": 0.15, "importance": 0.1, "graph": 0.0, "feedback": 0.1}

# Get all half-lives as dict (includes procedural)
half_lives = config.get_half_lives()
# {"preference": 90, "goal": 90, "fact": 30, "event": 14, "context": 30, "habit": 90, "procedural": 180}
```

---

## Environment-Specific Examples

### Local Development

```env
QDRANT_URL=http://localhost:6333
REDIS_URL=redis://localhost:6379
LLM_PROVIDER=ollama
LLM_MODEL=qwen2.5:7b-instruct
LLM_BASE_URL=http://localhost:11434
ALLOW_CLOUD=false
ENABLE_SCHEDULER=false
ENABLE_BACKGROUND_TASKS=false
ENABLE_REALTIME_GRAPH=true
ENABLE_GRAPH_RETRIEVAL=false
ENABLE_PROCEDURAL_MEMORY=false
```

### Production

```env
QDRANT_URL=http://qdrant:6333
REDIS_URL=redis://redis:6379
LLM_PROVIDER=groq
LLM_MODEL=llama-3.3-70b-versatile
GROQ_API_KEY=gsk_your-key
ALLOW_CLOUD=true

# Scoring weights tuned for production
WEIGHT_SIM=0.40
WEIGHT_RERANK=0.25
WEIGHT_RECENCY=0.15
WEIGHT_IMPORTANCE=0.10
WEIGHT_GRAPH=0.05
WEIGHT_FEEDBACK=0.05

# Enable all features
ENABLE_REALTIME_GRAPH=true
GRAPH_EXTRACTION_MODE=llm
ENABLE_GRAPH_RETRIEVAL=true
ENABLE_TRIGGERS=true
ENABLE_PROCEDURAL_MEMORY=true

# Background tasks
ENABLE_SCHEDULER=true
ENABLE_BACKGROUND_TASKS=true
AUTO_DEDUP_ENABLED=true
```

---

## Configuration Hierarchy

Settings are loaded in this order (later overrides earlier):

1. Default values in `Config` class
2. `.env` file values
3. System environment variables

---

## Troubleshooting

### "Qdrant connection failed"

Check that Qdrant is running and `QDRANT_URL` points to the correct host:

```bash
curl http://localhost:6333/collections
```

### Scoring weights don't sum to 1.0

Weights are auto-normalized at runtime. You can set them to any relative values and the system will normalize them proportionally.

### New features not activating

Ensure the feature flag is enabled:

```env
ENABLE_REALTIME_GRAPH=true
ENABLE_GRAPH_RETRIEVAL=true
ENABLE_TRIGGERS=true
ENABLE_PROCEDURAL_MEMORY=true
```

---

**Source of truth:** `src/hippocampai/config.py`
