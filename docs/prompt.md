# Claude Prompt: Build `hippocampai` as a Python Package with Advanced Retrieval

**ROLE**  
You are a senior Python library author. Build a clean, minimal, production-ready **pip package** called `hippocampai` from my existing project (long-term memory engine). Keep code compact, readable, and dependency-light. Do **not** include chain-of-thought; just outputs (files and code).

**GOAL**  
1) Package the code so others can `pip install hippocampai` and run it as a **library, CLI, or FastAPI service**.  
2) Implement **Retrieval** upgrades:
   - **Hybrid search**: BM25 (`rank-bm25` or `tantivy`) + embeddings → fuse with **RRF**.  
   - **Two-stage ranking**: Qdrant top-K (e.g., 200) → local **cross-encoder rerank** (`cross-encoder/ms-marco-MiniLM-L-6-v2`) → final K (e.g., 20).  
   - **Typed routing**: keep separate Qdrant collections for `profile/preferences/goals` vs `facts/events`, with **tighter decay** for the first; route queries by type.  
   - **Qdrant tuning**: HNSW `M=32–64`, `ef_construction=200–512`; search with `ef_search=64–256` (configurable). Use **payload filters** (`user_id`, `type`) to shrink candidate set. Enable **WAL & snapshots**; define **payload schema**.  
   - **Embeddings & models**: batch embeddings; reuse a single model instance with a **thread-safe queue** and max batch size; allow **quantized int8/bfloat16** models (e.g., `FlagEmbedding/bge-small` int8).  
   - **Local LLM via Ollama** (7–8B instruct). Keep context short (system prompt + compressed memory bullets).  
   - **Cache**: memoize cross-encoder scores keyed by `(query_hash, memory_id)` for 24h.

3) Keep **token usage efficient** while preserving logic: compact prompts, short schemas, small defaults, selective rerank, and compression of memory context.

**INPUTS**  
Assume my current repo is like:  
- Embeddings via `sentence-transformers`  
- Vector store in **Qdrant**  
- Memory pipeline modules: extractor, deduplicator, consolidator, importance scorer (some direct Anthropic calls)  
- A simple `web_chat` used for testing

---

## WHAT TO BUILD

Create a complete, minimal package with this structure (adjust minor details if needed, but keep the spirit):

```
hippocampai/
  src/hippocampai/
    __init__.py
    config.py
    models/memory.py                 # Pydantic Memory model + enums
    adapters/
      llm_base.py                    # interface
      provider_openai.py             # optional
      provider_ollama.py             # local llm
    vector/
      qdrant_store.py                # schema, filters, hnsw params, wal, snapshots
    embed/
      embedder.py                    # shared model, batching, quantization flag
    retrieval/
      bm25.py                        # BM25 using rank-bm25 or tantivy
      rrf.py                         # Reciprocal Rank Fusion
      rerank.py                      # CrossEncoder reranker
      router.py                      # Typed routing: prefs/goals vs facts/events
      retriever.py                   # Hybrid→RRF→Qdrant topK→CrossEncoder→FinalK + scoring fusion
    pipeline/
      extractor.py                   # provider-agnostic; optional heuristic mode
      dedup.py                       # vector+rerank candidate selection + rules (no vendor lock)
      consolidate.py                 # optional local-LLM rewrite or heuristic merge
      importance.py                  # heuristic + optional LLM refine; half-life support
    api/
      app.py                         # FastAPI routers
      deps.py
    cli/
      main.py                        # Typer CLI: init, remember, recall, run-api, export/import
    jobs/
      scheduler.py                   # APScheduler jobs: decay, consolidate, snapshot
    utils/
      cache.py                       # 24h TTL cache for rerank scores
      scoring.py                     # sim/rerank/recency/importance weighted fusion
      time.py
  tests/
    test_retrieval.py
    test_api.py
  pyproject.toml
  README.md
  CHANGELOG.md
  LICENSE
  examples/
    web_chat/                        # demo that calls only the public API
      README.md
      app.py
      memory_inspector.html
```

---

## KEY REQUIREMENTS

### Public Python API (thin & stable)
```python
from hippocampai import MemoryClient

client = MemoryClient(
    qdrant_url="http://localhost:6333",
    collection_facts="hippocampai_facts",
    collection_prefs="hippocampai_prefs",
    embed_model="bge-small-en-v1.5",
    embed_quantized=True,
    reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    bm25_backend="rank-bm25",    # or "tantivy"
    llm_provider="ollama",
    llm_model="qwen2.5:7b-instruct",
    hnsw_M=48, ef_construction=256, ef_search=128,
    weights={"sim":0.55,"rerank":0.20,"recency":0.15,"importance":0.10},
    half_lives={"prefs":90, "facts":30, "events":14},
    allow_cloud=False
)

client.remember("I prefer South Indian vegetarian food.", user_id="u1", session_id="s1", type="preference")
hits = client.recall("What cuisine do they like?", user_id="u1", session_id="s1", k=5)
```

### Retrieval
- **Hybrid + RRF**: implement Reciprocal Rank Fusion with tunable `k` & `C`.
- **Two-stage**: Qdrant `topK=200` → CrossEncoder rerank → final `K=20` (configurable).
- **Typed routing**: route queries to `collection_prefs` or `collection_facts` (and union results) using a lightweight classifier (keyword rules + optional LLM refine if available).

### Qdrant
- Initialize collections with HNSW params (`M`, `ef_construction`); set `ef_search` per call.
- Define payload schema: `user_id`, `type`, `tags`, `created_at`, `importance`, `access_ct`.
- Enable **WAL & snapshots**.

### Embeddings & Models
- **Batch embeddings**: single shared model instance; queue requests; `batch_size` configurable.
- **Quantization**: allow int8/bfloat16 for embeddings to speed CPU; surface a flag and show how to load quantized models.
- **Local LLM (Ollama)**: default to 7–8B instruct; keep prompts & context short (system + compressed memory bullets).
- **Caching**: `cache.get((hash(query), memory_id))` for CrossEncoder scores with TTL 24h.

### Scoring
- Fusion formula (normalized to [0,1]):  
  `score = 0.55*sim + 0.20*rerank + 0.15*recency + 0.10*importance`  
- Recency uses half-life exponential decay.

### API & CLI
- **FastAPI**: `POST /v1/memories:remember`, `GET /v1/memories:recall`, `PATCH /v1/memories/{id}`, `POST /v1/feedback`, `GET /v1/sessions/{id}/summary`, `GET /metrics`, `GET /healthz`.
- **CLI (Typer)**: `hippocampai init`, `hippocampai remember`, `hippocampai recall`, `hippocampai-api`.

### Jobs
- APScheduler for **nightly importance decay**, **weekly consolidation**, **hourly snapshot**.

### Token Efficiency
- Keep prompts short & deterministic; default to heuristics if no LLM.
- Compress memory context into bullets; only rerank the top candidate set.

### Agentic (Optional)
- A small “memory-maintenance agent” that monitors growth, schedules dedup/consolidate, adjusts thresholds. Use local LLM only if available; otherwise heuristics.

### Docs
- 5-minute **Quickstart** (Docker Compose with Qdrant + API + Ollama).
- Library usage, Service usage, CLI, Config, Local-only mode, Production checklist.

### License
- Apache-2.0.

---

## IMPLEMENTATION NOTES
- Prefer **`rank-bm25`** by default; allow **`tantivy`** via flag.
- CrossEncoder default: `"cross-encoder/ms-marco-MiniLM-L-6-v2"`, but keep an interface for custom models.
- Keep `web_chat` as a **demo** calling only the public API; add a **Memory Inspector** showing retrieved items + score breakdown (sim, rerank, recency, importance).
- Provide small **unit tests** for: RRF fusion, half-life decay, router classification, retriever returns.
- Centralize knobs in `config.py` and allow env var overrides.

---

## OUTPUT FORMAT (what you return)
- A single code block containing a **bash-style virtual file tree** and then **full file contents** for each file (no ellipses).  
- Order: `pyproject.toml` → `src/hippocampai/...` (all files) → `tests/...` → `examples/web_chat/...` → `README.md` → `CHANGELOG.md` → `LICENSE`.  
- Keep comments concise. No explanations outside code.  
- Keep total output tight; minimal viable code that runs.

---

## ACCEPTANCE CRITERIA (I will verify)
- I can `pip install -e .` and run:  
  - `hippocampai init` then `hippocampai-api`  
  - `echo "I live in Raleigh" | hippocampai remember --user u1 --type fact`  
  - `hippocampai recall --user u1 --query "where does user live?" -k 5`  
- Hybrid+RRF and 2-stage rerank are **active by default**.  
- Typed routing works (rules + optional LLM refine).  
- Qdrant collections created with HNSW params & payload schema; WAL & snapshots enabled.  
- CrossEncoder scores cached for repeated queries.  
- Example `web_chat` works and shows **score breakdown**.  
- Ollama path is first-class; no vendor lock-in.  
- Tests pass with `pytest -q`.
