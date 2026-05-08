# HippocampAI Architecture

> Autonomous memory engine with hybrid retrieval, tiered storage, and real-time consolidation.

---

## System Overview

```
                                    HippocampAI Architecture
 ========================================================================================

   CLIENTS                     API GATEWAY                        BACKGROUND
  +----------+              +----------------+                 +-----------------+
  | React 18 |  WebSocket   |   Socket.IO    |                 | Celery Workers  |
  | Vite 5   |<------------>|   AsyncServer  |                 |  concurrency=4  |
  | Tailwind |              +--------+-------+                 +--------+--------+
  | RQ v5    |   REST/HTTP           |                                  |
  |----------|  (Axios)     +--------+-------+                 +--------+--------+
  | Port 81  |<------------>| FastAPI (8000) |<--------------->| Celery Beat     |
  | (nginx)  |  /api /v1    |   Uvicorn      |  task dispatch  | (scheduler)     |
  +----------+  /auth       +--------+-------+                 +-----------------+
                                     |
                            MIDDLEWARE STACK
                        (top-to-bottom execution)
                     +---------------------------+
                     | 1. CORS                   |
                     | 2. Prometheus Metrics      |
                     | 3. Auth + Rate Limiting    |
                     +---------------------------+
                                     |
                     +===============+===============+
                     |       20 ROUTE MODULES        |
                     +===============================+
```

---

## Detailed Layer Diagram

```
 ============================================================================================
  PRESENTATION LAYER
 ============================================================================================

  +-----------------------------------------------------------------------------------+
  |                          FRONTEND (React 18 + TypeScript 5)                       |
  |                                                                                   |
  |  +-------------+  +-------------+  +-----------+  +----------+  +-----------+     |
  |  | Dashboard   |  | Memories    |  | Sleep     |  | Graph    |  | 22 more   |     |
  |  | (stats,     |  | (CRUD,      |  | Phase     |  | View     |  | pages...  |     |
  |  |  charts,    |  |  filters,   |  | (consol-  |  | (force-  |  |           |     |
  |  |  activity)  |  |  detail)    |  |  idation) |  |  graph)  |  |           |     |
  |  +-------------+  +-------------+  +-----------+  +----------+  +-----------+     |
  |                                                                                   |
  |  +---------------------------+  +---------------------+  +--------------------+   |
  |  | React Query (v5)         |  | Socket.IO Client    |  | Axios (2 instances)|   |
  |  | - Server state cache     |  | - Real-time events  |  | - /api client      |   |
  |  | - 5min stale time        |  | - Auto-reconnect    |  | - /v1 client       |   |
  |  | - Optimistic updates     |  | - Room subscriptions|  | - Retry + backoff  |   |
  |  +---------------------------+  +---------------------+  +--------------------+   |
  +-----------------------------------------------------------------------------------+
       |              |                    |                         |
       | REST         | WebSocket          | REST                   | REST
       | /api/*       | /socket.io         | /v1/*                  | /auth/*
       v              v                    v                         v
 ============================================================================================
  API GATEWAY LAYER (FastAPI + Uvicorn, Port 8000)
 ============================================================================================

  +-----------------------------------------------------------------------------------+
  |                              MIDDLEWARE CHAIN                                      |
  |  +--------+    +-------------------+    +-----------------------------------+     |
  |  | CORS   |--->| Prometheus        |--->| AuthMiddleware                    |     |
  |  | (all   |    | (request count,   |    | - Bearer token validation         |     |
  |  |  methods)   |  latency buckets) |    | - API key lookup via AuthService  |     |
  |  +--------+    +-------------------+    | - Rate limiting (tier-based)      |     |
  |                                         | - Admin access check              |     |
  |                                         | - Usage logging                   |     |
  |                                         +-----------------------------------+     |
  +-----------------------------------------------------------------------------------+
       |
  +-----------------------------------------------------------------------------------+
  |                          ROUTE MODULES (20 Routers)                               |
  |                                                                                   |
  |  CORE MEMORY (/v1/memories/*)        AUTH (/auth/*)                               |
  |  - POST /v1/memories (create)        - POST /auth/signup                          |
  |  - POST /v1/memories/query           - POST /auth/validate                        |
  |  - POST /v1/memories/recall          - GET  /auth/session/{id}/exists             |
  |  - PATCH /v1/memories/{id}                                                        |
  |  - DELETE /v1/memories/{id}          ADMIN (/admin/*)                              |
  |  - POST /v1/memories/batch           - GET/PATCH /admin/users                     |
  |  - POST /v1/memories/extract         - POST/DELETE /admin/api-keys                |
  |  - POST /v1/classify                                                              |
  |                                                                                   |
  |  INTELLIGENCE (/v1/intelligence/*)   HEALING (/v1/healing/*)                      |
  |  - facts:extract                     - health, full-check                         |
  |  - entities:extract, :search         - cleanup, deduplication                     |
  |  - relationships:analyze             - tagging, importance                        |
  |  - clustering:analyze, :optimize     - health/stale, /duplicates, /gaps           |
  |  - temporal:analyze, :peak-times     - config (get/set)                           |
  |                                                                                   |
  |  PREDICTIONS (/v1/predictions/*)     COLLABORATION (/v1/collaboration/*)          |
  |  - patterns, anomalies, trends       - spaces CRUD                               |
  |  - forecast, recommendations         - collaborators, permissions                 |
  |  - insights, activity/peaks          - conflicts, notifications                   |
  |                                                                                   |
  |  BITEMPORAL (/v1/bitemporal/*)       CONTEXT (/v1/context/*)                      |
  |  - facts:store, :revise, :retract    - :assemble (JSON)                           |
  |  - facts:query, :latest, :history    - :assemble/text (plain)                     |
  |                                                                                   |
  |  FEEDBACK (/v1/feedback/*)           TRIGGERS (/v1/triggers/*)                    |
  |  - POST /memories/{id}/feedback      - CRUD + history                             |
  |  - GET  /feedback/stats              - webhook, log, websocket actions            |
  |                                                                                   |
  |  PROCEDURAL (/v1/procedural/*)       PROSPECTIVE (/v1/prospective/*)              |
  |  - rules, extract, inject            - intents CRUD, :parse                       |
  |  - consolidate, feedback             - evaluate, expire, consolidate              |
  |                                                                                   |
  |  OBSERVABILITY (/v1/observability/*) MIGRATION (/v1/admin/embeddings/*)           |
  |  - explain, visualize                - migrate, status, cancel                    |
  |  - heatmap, profile                                                               |
  |                                                                                   |
  |  CONSOLIDATION (/api/consolidation/) DASHBOARD (/api/dashboard/*)                 |
  |  - config, status, runs, trigger     - stats, recent-activity                     |
  |                                                                                   |
  |  SESSIONS (/api/sessions/*)          COMPACTION (/api/compaction/*)               |
  |  - list, stats, soft-delete, wipe    - compact, preview, history                  |
  |                                                                                   |
  |  AUDIT (/api/audit/*)               USAGE (/api/usage/*)                          |
  |  - logs, search, retention           - me, quotas, platform                       |
  |                                                                                   |
  |  CELERY (/celery/*)                  BACKGROUND (/v1/background/*)                |
  |  - task status, cancel, inspect      - dedup/trigger, consolidate/trigger         |
  +-----------------------------------------------------------------------------------+
       |
 ============================================================================================
  SERVICE LAYER
 ============================================================================================

  +-----------------------------------------------------------------------------------+
  |                        MEMORY MANAGEMENT SERVICE                                  |
  |                     (Central orchestrator, 2000+ lines)                           |
  |                                                                                   |
  |  +-------------------+  +-------------------+  +--------------------+             |
  |  | CRUD Operations   |  | Batch Operations  |  | Search & Retrieval |             |
  |  | - create_memory   |  | - batch_create    |  | - recall_memories  |             |
  |  | - get_memory      |  | - batch_update    |  | - get_memories     |             |
  |  | - update_memory   |  | - batch_delete    |  | - query (filters)  |             |
  |  | - delete_memory   |  | - batch_get       |  |                    |             |
  |  +-------------------+  +-------------------+  +--------------------+             |
  |                                                                                   |
  |  +-------------------+  +-------------------+  +--------------------+             |
  |  | Deduplication     |  | Consolidation     |  | Conflict Resolver  |             |
  |  | - check_duplicate |  | - consolidate_    |  | - detect conflicts |             |
  |  | - dedup_user      |  |   memories        |  | - resolve (LLM/    |             |
  |  | - threshold: 0.88 |  | - merge similar   |  |   newest/merge)    |             |
  |  +-------------------+  +-------------------+  +--------------------+             |
  |                                                                                   |
  |  +-------------------+  +-------------------+  +--------------------+             |
  |  | Lifecycle Manager |  | Quality Monitor   |  | Lock Manager       |             |
  |  | - tier migration  |  | - stale detection |  | - prevent races    |             |
  |  | - temperature     |  | - gap analysis    |  | - weak references  |             |
  |  |   scoring         |  | - health scoring  |  |                    |             |
  |  +-------------------+  +-------------------+  +--------------------+             |
  +-----------------------------------------------------------------------------------+
       |                          |                           |
       v                          v                           v
 ============================================================================================
  PIPELINE LAYER (Processing & Intelligence)
 ============================================================================================

  +-----------------------------------------------------------------------------------+
  |                                                                                   |
  |  INGESTION                    RETRIEVAL                   MAINTENANCE             |
  |  +-----------------------+    +-----------------------+   +--------------------+  |
  |  | Agentic Classifier    |    | Hybrid Retriever      |   | Importance Decay   |  |
  |  | - LLM-based type      |    |                       |   | - Exponential      |  |
  |  |   detection            |    |  Query                |   | - Half-lives:      |  |
  |  | - Pattern fallback    |    |    |                   |   |   pref: 90d        |  |
  |  +-----------------------+    |    v                   |   |   fact: 30d        |  |
  |                               |  +---+ BM25 (lexical) |   |   event: 14d       |  |
  |  Entity Recognition           |  |   | rank-bm25      |   +--------------------+  |
  |  - NER extraction             |  |   v                |                            |
  |  - Relationship mapping       |  | +---+ Vector Search|   Memory Lifecycle         |
  |  - Mention tracking           |  | |   | Qdrant top200|   - HOT    (< 7d, >10x)   |
  |                               |  | |   v              |   - WARM   (< 30d, 3-10x) |
  |  Fact Extraction              |  | | RRF Fusion (k=60)|   - COLD   (< 90d, <3x)   |
  |  - Pattern + LLM              |  | |   |              |   - ARCHIVED (> 90d)       |
  |  - Confidence scoring         |  | |   v              |   - HIBERNATED (> 365d)    |
  |                               |  | | CrossEncoder     |                            |
  |  Conversation Memory          |  | | Reranking        |   Auto-Healing             |
  |  - Turn tracking              |  | |   |              |   - Broken link repair     |
  |  - Summarization              |  | |   v              |   - Duplicate merge        |
  |                               |  | | Final Scoring:   |   - Gap filling            |
  |  Semantic Clustering          |  | | sim    = 0.55    |                            |
  |  - K-means grouping           |  | | rerank = 0.20    |   Temporal Analytics       |
  |  - Optimal k detection        |  | | recency= 0.15    |   - Time-series           |
  |                               |  | | import = 0.10    |   - Peak detection        |
  |  Predictive Analytics         |  | +---+ Top K=20     |   - Trend analysis        |
  |  - Pattern recognition        |  +-----------------------+                        |
  |  - Anomaly detection          |                                                   |
  |  - Forecasting                |    Consolidation Pipeline                          |
  |                               |    - LLM-based merge                              |
  |                               |    - Heuristic fallback                           |
  |                               |    - Tag combining                                |
  +-----------------------------------------------------------------------------------+
       |                    |                    |                    |
       v                    v                    v                    v
 ============================================================================================
  ML / EMBEDDING LAYER
 ============================================================================================

  +-----------------------------------------------------------------------------------+
  |                                                                                   |
  |  +-----------------------+  +-----------------------+  +------------------------+ |
  |  | Embedder              |  | Reranker              |  | LLM Adapters           | |
  |  | BAAI/bge-small-en-v1.5|  | ms-marco-MiniLM-L-6  |  |                        | |
  |  | Dimension: 384        |  | CrossEncoder          |  | +------------------+  | |
  |  | Batch size: 32        |  | Cache: MD5 keys, 24h  |  | | Ollama (local)   |  | |
  |  | Thread-safe (lock)    |  | Batch prediction      |  | | Groq (cloud)     |  | |
  |  | SentenceTransformer   |  | SentenceTransformer   |  | | OpenAI (cloud)   |  | |
  |  +-----------------------+  +-----------------------+  | | Anthropic (cloud) |  | |
  |                                                        | +------------------+  | |
  |  Knowledge Graph                                       | Auto-retry, backoff   | |
  |  - Entity/relationship graph                           | JSON serialization    | |
  |  - Pattern-based extraction                            +------------------------+ |
  |  - JSON persistence (data/knowledge_graph.json)                                   |
  |  - Auto-save every 300s                                                           |
  +-----------------------------------------------------------------------------------+
       |                    |                    |
       v                    v                    v
 ============================================================================================
  STORAGE LAYER
 ============================================================================================

  +---------------------+  +---------------------+  +-------------------------------+
  |  QDRANT             |  |  REDIS              |  |  PostgreSQL / SQLite          |
  |  (Vector Database)  |  |  (Cache & Broker)   |  |  (Auth Database)             |
  |                     |  |                     |  |                               |
  |  v1.15.1            |  |  7.2-alpine         |  |  postgres:15-alpine           |
  |  Port: 6333 (HTTP)  |  |  Port: 6379         |  |  Port: 5432                   |
  |  Port: 6334 (gRPC)  |  |                     |  |  (or SQLite file)             |
  |                     |  |  DB 0: Data cache   |  |                               |
  |  Collections:       |  |  DB 1: Celery broker|  |  Tables:                      |
  |  +----------------+ |  |  DB 2: Celery result|  |  - users                      |
  |  | hippocampai_   | |  |                     |  |  - api_keys                   |
  |  | facts          | |  |  Features:          |  |  - api_key_usage              |
  |  | (facts, events,| |  |  - Memory cache     |  |  - organizations              |
  |  |  context,      | |  |  - Reranker cache   |  |  - rate_limit_buckets         |
  |  |  summaries)    | |  |  - BM25 corpus      |  |  - sessions                   |
  |  +----------------+ |  |  - Rate limiters    |  |  - audit_log                  |
  |  +----------------+ |  |  - Session store    |  |                               |
  |  | hippocampai_   | |  |                     |  |  Dual backend:                |
  |  | prefs          | |  |  Max: 2GB           |  |  - DB_TYPE=postgres (prod)    |
  |  | (preferences,  | |  |  Policy: allkeys-lru|  |  - DB_TYPE=sqlite (dev)       |
  |  |  goals, habits,| |  |  TTL: 300s default  |  |  - Unified async interface    |
  |  |  procedural,   | |  |  Connections: 100   |  |                               |
  |  |  prospective)  | |  |  AOF persistence    |  |  Auth: asyncpg / aiosqlite   |
  |  +----------------+ |  +---------------------+  +-------------------------------+
  |                     |
  |  HNSW Config:       |
  |  - M=48             |
  |  - ef_construct=256 |
  |  - ef_search=128    |
  |  - Cosine distance  |
  +---------------------+
```

---

## Background Processing

```
 ============================================================================================
  BACKGROUND PROCESSING (Celery + Redis)
 ============================================================================================

  +-----------------------------------------------------------------------+
  |                         CELERY BEAT (Scheduler)                       |
  |                                                                       |
  |  +-------------------+  +-------------------+  +-------------------+  |
  |  | Daily 2:00 AM     |  | Daily 3:00 AM     |  | Every 24 Hours   |  |
  |  | Importance Decay   |  | Sleep Phase       |  | Deduplication    |  |
  |  | (all memories)     |  | Consolidation     |  | (if enabled)     |  |
  |  +-------------------+  +-------------------+  +-------------------+  |
  |                                                                       |
  |  +-------------------+  +-------------------+  +-------------------+  |
  |  | Hourly            |  | Hourly            |  | Every 5 Minutes   |  |
  |  | Expire Memories   |  | Collection        |  | Health Check      |  |
  |  | (TTL cleanup)     |  | Snapshots         |  | (system health)   |  |
  |  +-------------------+  +-------------------+  +-------------------+  |
  +-----------------------------------------------------------------------+
       |                         |                         |
       v                         v                         v
  +-----------------------------------------------------------------------+
  |                       CELERY WORKERS (x4 concurrent)                  |
  |                                                                       |
  |  Queues:                                                              |
  |  +----------+  +-------------+  +------------+  +----------+         |
  |  | default  |  | memory_ops  |  | background |  | scheduled|         |
  |  | (general)|  | (CRUD,      |  | (health,   |  | (periodic|         |
  |  |          |  |  embedding, |  |  monitoring)|  |  tasks)  |         |
  |  |          |  |  reranking) |  |            |  |          |         |
  |  +----------+  +-------------+  +------------+  +----------+         |
  |                                                                       |
  |  Config: acks_late=true, max_tasks_per_child=1000                     |
  |  Limits: soft=5min, hard=10min                                        |
  +-----------------------------------------------------------------------+
       |
       v
  +-----------------------------------------------------------------------+
  |  FLOWER (Celery Monitoring UI)                                        |
  |  Port: 5555  |  Basic Auth  |  Task/Worker/Queue visibility           |
  +-----------------------------------------------------------------------+
```

---

## Monitoring Stack

```
 ============================================================================================
  MONITORING & OBSERVABILITY
 ============================================================================================

  +-------------------+         +-------------------+         +-------------------+
  |    GRAFANA        |         |   PROMETHEUS      |         |   APPLICATION     |
  |    Port: 3002     |<--------|   Port: 9090      |<--------|   METRICS         |
  |                   |  query  |                   |  scrape |                   |
  |  Dashboards:      |         |  Scrape targets:  |         |  GET /metrics     |
  |  - API Latency    |         |  - hippocampai    |         |                   |
  |  - Memory Ops     |         |    :8000/metrics  |         |  Counters:        |
  |  - Cache Stats    |         |  - redis          |         |  - http_requests  |
  |  - Worker Health  |         |    :6379/metrics  |         |  - memory_ops     |
  |  - Queue Depth    |         |  - qdrant         |         |  - cache_hits     |
  |                   |         |    :6333/metrics  |         |  - llm_calls      |
  |  Retention: 30d   |         |  - prometheus     |         |                   |
  +-------------------+         |    :9090          |         |  Histograms:      |
                                |                   |         |  - request_dur    |
                                |  Interval: 15s    |         |  - retrieval_time |
                                +-------------------+         |  - embed_time     |
                                                              +-------------------+
```

---

## Data Flow: Memory Creation

```
  User types memory text
       |
       v
  +----------+     POST /v1/memories      +----------------+
  | Frontend  |-------------------------->| FastAPI Route   |
  | (React)   |                           |                 |
  +----------+                            +--------+--------+
                                                   |
                                          +--------v--------+
                                          | Auth Middleware  |
                                          | - Validate token|
                                          | - Rate limit    |
                                          +--------+--------+
                                                   |
                                          +--------v-----------+
                                          | MemoryMgmtService  |
                                          +--------+-----------+
                                                   |
                              +--------------------+--------------------+
                              |                    |                    |
                     +--------v--------+  +--------v--------+  +-------v--------+
                     | Agentic         |  | Embedder        |  | Deduplicator   |
                     | Classifier      |  | encode(text)    |  | check_dup()    |
                     | (detect type)   |  | -> vec[384]     |  | sim > 0.88?    |
                     +-----------------+  +--------+--------+  +-------+--------+
                                                   |                    |
                                                   |              (if unique)
                                                   |                    |
                                          +--------v--------------------v--------+
                                          |           QDRANT UPSERT              |
                                          | collection: facts or prefs           |
                                          | payload: {text, user_id, type,       |
                                          |   importance, tags, timestamps...}   |
                                          | vector: [384 floats]                 |
                                          +--------+----------------------------+
                                                   |
                              +--------------------+--------------------+
                              |                                         |
                     +--------v--------+                       +--------v--------+
                     | Redis Cache     |                       | BM25 Corpus     |
                     | SET memory:{id} |                       | (update index)  |
                     | TTL: 300s       |                       |                 |
                     +-----------------+                       +-----------------+
                              |
                     +--------v--------+
                     | WebSocket       |
                     | Emit:           |
                     | memory:created  |
                     +-----------------+
```

---

## Data Flow: Memory Recall (Hybrid Retrieval)

```
  User enters search query
       |
       v
  +----------+    POST /v1/memories/recall    +----------------+
  | Frontend  |------------------------------>| FastAPI Route   |
  | (React)   |                               +--------+--------+
  +----------+                                         |
                                              +--------v-----------+
                                              | HybridRetriever    |
                                              +--------+-----------+
                                                       |
                         +-----------------------------+-----------------------------+
                         |                                                           |
                +--------v--------+                                         +--------v--------+
                | BM25 Retrieval  |                                         | Vector Search   |
                | (lexical match) |                                         | (semantic match)|
                |                 |                                         |                 |
                | rank-bm25 lib   |                                         | Embedder.encode |
                | user's corpus   |                                         | -> Qdrant.search|
                | top_k = 20      |                                         | top_k = 200     |
                +--------+--------+                                         +--------+--------+
                         |                                                           |
                         +---------------------+-+-----------------------------------+
                                               |
                                      +--------v--------+
                                      | RRF Fusion      |
                                      | (k=60)          |
                                      | Combine BM25 +  |
                                      | vector ranks    |
                                      +--------+--------+
                                               |
                                      +--------v--------+
                                      | CrossEncoder    |
                                      | Reranking       |
                                      | ms-marco-MiniLM |
                                      | (cached 24h)    |
                                      +--------+--------+
                                               |
                                      +--------v-----------+
                                      | Final Scoring      |
                                      |                    |
                                      | similarity  = 0.55 |
                                      | rerank      = 0.20 |
                                      | recency     = 0.15 |
                                      | importance  = 0.10 |
                                      |                    |
                                      | + intent boost     |
                                      | + feedback weight  |
                                      +--------+-----------+
                                               |
                                      +--------v--------+
                                      | Top K=20        |
                                      | RetrievalResult |
                                      | [{memory,score, |
                                      |   breakdown}]   |
                                      +-----------------+
```

---

## Docker Compose Infrastructure

```
 ============================================================================================
  DOCKER COMPOSE SERVICES (bridge network: 172.28.0.0/16)
 ============================================================================================

  +-------------------+     +-------------------+     +-------------------+
  |   frontend        |     |   hippocampai     |     |   admin           |
  |   (nginx)         |     |   (FastAPI)       |     |   (nginx)         |
  |   Port: 81        |---->|   Port: 8000      |     |   Port: 3001      |
  |   React SPA       |     |   + Socket.IO     |     |   Admin UI        |
  +-------------------+     +---+---+---+-------+     +-------------------+
                                |   |   |
              +-----------------+   |   +-----------------+
              |                     |                     |
  +-----------v-----+   +----------v------+   +-----------v-----+
  |   qdrant        |   |   redis         |   |   postgres      |
  |   (Vector DB)   |   |   (Cache)       |   |   (Auth DB)     |
  |   v1.15.1       |   |   7.2-alpine    |   |   15-alpine     |
  |                 |   |                 |   |                 |
  |   6333 (HTTP)   |   |   6379          |   |   5432          |
  |   6334 (gRPC)   |   |   DB0: cache    |   |                 |
  |                 |   |   DB1: broker   |   |   qdrant_data   |
  |   qdrant_data   |   |   DB2: results  |   |   postgres_data |
  +-----------------+   |   redis_data    |   +-----------------+
                        +--------+--------+
                                 |
              +------------------+------------------+
              |                                     |
  +-----------v-----+                   +-----------v-----+
  |  celery-worker  |                   |  celery-beat    |
  |  (4 concurrent) |                   |  (scheduler)    |
  |  Task execution |                   |  Periodic tasks |
  +--------+--------+                   +-----------------+
           |
  +--------v--------+     +-------------------+     +-------------------+
  |  flower         |     |  prometheus       |     |  grafana          |
  |  (Celery UI)    |     |  (Metrics)        |     |  (Dashboards)     |
  |  Port: 5555     |     |  Port: 9090       |     |  Port: 3002       |
  +-----------------+     +-------------------+     +-------------------+
```

---

## Memory Tier Lifecycle

```
                    Access Pattern & Age Drive Tier Migration

  +============+     7 days      +============+     30 days     +============+
  |    HOT     | --------------> |    WARM    | -------------> |    COLD    |
  | In Redis + |    or < 10     | Qdrant     |   or < 3      | Qdrant     |
  | Qdrant     |    accesses    | only       |   accesses    | compressed |
  | < 7 days   |                | < 30 days  |               | < 90 days  |
  | > 10 access|                | 3-10 access|               | < 3 access |
  +============+                +============+               +============+
                                                                    |
                                                               90 days
                                                                    |
                                                              +============+
                                                              |  ARCHIVED  |
                                                              | gzip       |
                                                              | > 90 days  |
                                                              +============+
                                                                    |
                                                              365 days
                                                                    |
                                                              +============+
                                                              | HIBERNATED |
                                                              | deep       |
                                                              | archive    |
                                                              | > 365 days |
                                                              +============+

  Temperature Score = frequency(0.4) + recency(0.3) + importance(0.3)
```

---

## Retrieval Scoring Weights

```
  +------------------------------------------------------------------+
  |                    FINAL SCORE COMPOSITION                       |
  +------------------------------------------------------------------+
  |                                                                  |
  |  +-----------+  +-----------+  +-----------+  +-----------+     |
  |  | Semantic  |  | Reranker  |  | Recency   |  | Importance|     |
  |  | Similarity|  | Score     |  | Decay     |  | Score     |     |
  |  |           |  |           |  |           |  |           |     |
  |  |   0.55    |  |   0.20    |  |   0.15    |  |   0.10    |     |
  |  |           |  |           |  |           |  |           |     |
  |  | Qdrant    |  | Cross-    |  | Half-life |  | Memory    |     |
  |  | cosine    |  | Encoder   |  | based     |  | priority  |     |
  |  | distance  |  | ms-marco  |  | exp decay |  | (0-10)    |     |
  |  +-----------+  +-----------+  +-----------+  +-----------+     |
  |                                                                  |
  |  Optional:  +-------------+  +--------------+                   |
  |             | Feedback    |  | Graph Weight |                   |
  |             | Weight: 0.1 |  | Weight: 0.0  |                   |
  |             | (90d window)|  | (disabled)   |                   |
  |             +-------------+  +--------------+                   |
  |                                                                  |
  |  Intent Boost:                                                   |
  |  - Temporal queries  -> recency x 2.0                           |
  |  - Preference queries -> importance x 1.5                       |
  +------------------------------------------------------------------+
```

---

## Tech Stack Summary

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Frontend** | React 18, TypeScript 5, Vite 5, Tailwind CSS | SPA with 27 pages |
| **State** | TanStack React Query v5 | Server state, caching |
| **Real-time** | Socket.IO | Live memory updates |
| **Charts** | Recharts, React Force Graph 2D | Visualization |
| **API** | FastAPI, Uvicorn | Async REST + WebSocket |
| **Auth** | JWT, API Keys, Rate Limiting | Multi-tenant security |
| **Embedding** | BAAI/bge-small-en-v1.5 (384d) | Semantic vectors |
| **Reranking** | ms-marco-MiniLM-L-6-v2 | Cross-encoder scoring |
| **LLM** | Ollama/Groq/OpenAI/Anthropic | Classification, consolidation |
| **Vector DB** | Qdrant v1.15.1 | HNSW similarity search |
| **Cache** | Redis 7.2 | Hot tier, rate limits, sessions |
| **Auth DB** | PostgreSQL 15 / SQLite | Users, API keys, audit |
| **Queue** | Celery + Redis | Background task processing |
| **Monitoring** | Prometheus + Grafana | Metrics and dashboards |
| **Container** | Docker Compose (11 services) | Full-stack deployment |
