# HippocampAI — Agent Playbook

> The definitive reference for Claude agents working on this codebase.
> Read this BEFORE writing any code. It contains the architecture, coding standards,
> metrics, quality gates, and issue dispatch needed to work effectively.
>
> Last updated: 2026-05-08 | Version: 0.5.1

---

## 1. Project Identity

**HippocampAI** is an autonomous long-term memory engine with hybrid retrieval and cross-encoder reranking. It operates in three modes:

| Mode | Entry Point | Use Case |
|------|-------------|----------|
| **Library** | `pip install hippocampai` → `MemoryClient` | Embedded in Python apps |
| **API** | `async_app.py` → FastAPI on port 8000 | Standalone REST service |
| **SaaS** | `docker-compose.yml` → 11 services | Full platform with auth, monitoring |

---

## 2. Codebase Metrics

### Scale

| Metric | Value |
|--------|-------|
| Python source files | 211 |
| Python lines (src/) | 72,729 |
| Test files | 40 |
| Test lines | 18,989 |
| Test cases total | ~750 |
| Frontend TS/TSX files | 47 |
| Frontend lines | 14,955 |
| Git commits | 69 |
| Version | 0.5.1 |

### Code Composition

| Module | Lines | Files | Role |
|--------|-------|-------|------|
| `pipeline/` | 17,548 | 33 | Memory processing, lifecycle, conflict resolution |
| `api/` | 9,819 | 26 | FastAPI routes and middleware |
| `consolidation/` | 3,427 | 7 | Sleep-phase memory consolidation |
| `services/` | 3,306 | 5 | Business logic layer |
| `monitoring/` | 3,083 | 6 | Health checks, metrics, tracking |
| `models/` | 1,495 | 10 | Pydantic data models |
| `storage/` | 1,487 | 6 | Redis, bitemporal, KV stores |
| `graph/` | 1,357 | 4 | Knowledge graph |
| `auth/` | 1,165 | 4 | Auth service, DB abstraction, rate limiting |
| `retrieval/` | 865 | 7 | Vector search, BM25, reranking |
| `vector/` | 526 | 2 | Qdrant store |
| `embed/` | 316 | 3 | Embedding models |

### Quality Metrics (current state)

| Metric | Current | Target |
|--------|---------|--------|
| Functions with return types | 1,193 / 2,166 (55%) | 90%+ |
| Docstrings | ~2,034 | Match all public functions |
| `except Exception as e` (broad) | 355 | Reduce to <50 |
| `except SpecificError` (targeted) | 98 | Increase |
| `raise HTTPException` | 205 | Standardize format |
| Pydantic models | 71 files | Keep using for all I/O |
| f-string SQL queries | 11 | 0 (security risk) |
| Test coverage estimate | ~25% | 60%+ |

### API Surface

| Source | Endpoints | Status |
|--------|-----------|--------|
| `async_app.py` inline | 46 | Active |
| `intelligence_routes.py` | 12 | Registered |
| `celery_routes.py` | 11 | Registered |
| `admin_routes.py` | 16 | Conditional (ENABLE_ADMIN_ROUTES) |
| **15 unregistered route files** | **143 total** | **NOT wired — ISSUE-001** |
| `healing_routes.py` | 12 | NOT registered |
| `collaboration_routes.py` | 14 | NOT registered |
| `audit_routes.py` | 10 | NOT registered |
| `prospective_routes.py` | 9 | NOT registered |
| `prediction_routes.py` | 8 | NOT registered |
| `session_routes.py` | 8 | NOT registered |
| `consolidation_routes.py` | 7 | NOT registered |
| `bitemporal_routes.py` | 6 | NOT registered |
| `procedural_routes.py` | 5 | NOT registered |
| `trigger_routes.py` | 4 | NOT registered |
| `compaction_routes.py` | 4 | NOT registered |
| `usage_routes.py` | 4 | NOT registered |
| `migration_routes.py` | 3 | NOT registered |
| `feedback_routes.py` | 3 | NOT registered |
| `auth_routes.py` | 3 | NOT registered |
| `dashboard_routes.py` | 2 | NOT registered |
| `context_routes.py` | 2 | NOT registered |

### Test Coverage by Area

| Area | Test File | Test Count | Covered |
|------|-----------|------------|---------|
| Knowledge graph | `test_knowledge_graph_e2e.py` | 56 | Yes |
| Dual DB | `test_dual_db.py` | 52 | Yes |
| Priority fixes | `test_priority_fixes.py` | 36 | Yes |
| Advanced features | `test_advanced_features.py` | 34 | Yes |
| Prospective memory | `test_prospective_memory.py` | 33 | Yes |
| Compression | `test_advanced_compression.py` | 31 | Yes |
| Metrics | `test_metrics.py` | 30 | Yes |
| Custom schema | `test_custom_schema.py` | 28 | Yes |
| Agentic classifier | `test_agentic_classifier.py` | 28 | Yes |
| Health monitoring | `test_memory_health.py` | 25 | Yes |
| Auto-consolidation | `test_auto_consolidation.py` | 25 | Yes |
| Conflict resolution | `test_conflict_resolution.py` | 20 | Yes |
| **Auth middleware** | **MISSING** | **0** | **No** |
| **Healing routes** | **MISSING** | **0** | **No** |
| **Background tasks** | **MISSING** | **0** | **No** |
| **Rate limiting** | **MISSING** | **0** | **No** |
| **CLI** | **MISSING** | **0** | **No** |

### Hottest Files (most frequently changed)

| File | Changes | Lines | Risk Level |
|------|---------|-------|------------|
| `client.py` | 14 | 7,406 | HIGH — massive, change with care |
| `api/app.py` | 14 | legacy | LOW — being replaced by async_app |
| `__init__.py` | 14 | exports | MEDIUM — affects public API |
| `config.py` | 10 | 268 | MEDIUM — all features read this |
| `vector/qdrant_store.py` | 10 | 526 | HIGH — core storage |
| `models/memory.py` | 10 | ~200 | HIGH — 55 files import this |
| `api/async_app.py` | 8 | 1,973 | HIGH — primary entry point |

### Dependency Graph (import counts)

| Module | Imported By | Centrality |
|--------|-------------|------------|
| `models/memory.py` | 55 files | Highest — change breaks everything |
| `vector/qdrant_store.py` | 16 files | High |
| `config.py` | 14 files | High |
| `services/memory_service.py` | 4 files | Medium (accessed via `get_service()`) |
| `storage/redis_store.py` | 3 files | Low (accessed via global) |
| `auth/auth_service.py` | 2 files | Low (accessed via `app.state`) |

---

## 3. Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend (React 18 + TS)                   │
│  Pages: 27 TSX files  |  API client: services/api.ts         │
│  Axios: client (/api) + v1Client (/v1)  |  React Query       │
└──────────────────────────┬──────────────────────────────────┘
                           │ HTTP
┌──────────────────────────▼──────────────────────────────────┐
│                   FastAPI (async_app.py)                      │
│                                                              │
│  Middleware stack:                                            │
│    CORS → AuthMiddleware → RateLimiter → PrometheusMiddleware│
│                                                              │
│  Service access:                                             │
│    get_service() → global _service (MemoryManagementService) │
│    app.state.auth_service → AuthService                      │
│    app.state.rate_limiter → RateLimiter                      │
│    app.state.db_pool → SQLite or PostgreSQL                  │
│                                                              │
│  Routes: 46 inline + 3 registered routers                    │
│  (15 more route files exist but NOT registered — ISSUE-001)  │
└───┬──────────┬──────────┬──────────┬────────────────────────┘
    │          │          │          │
    ▼          ▼          ▼          ▼
┌────────┐ ┌────────┐ ┌────────┐ ┌────────────┐
│ Qdrant │ │ Redis  │ │ SQLite │ │ PostgreSQL │
│ Vector │ │ Cache  │ │  Auth  │ │   Auth     │
│ Store  │ │ + Rate │ │ (dev)  │ │  (prod)    │
│        │ │ Limit  │ │        │ │            │
└────────┘ └────────┘ └────┬───┘ └─────┬──────┘
                           │           │
                      ┌────▼───────────▼────┐
                      │   auth/db.py        │
                      │   create_db_pool()  │
                      │   DB_TYPE env var   │
                      └─────────────────────┘
```

### Key Entry Points

| What | File | When |
|------|------|------|
| Production API | `src/hippocampai/api/async_app.py` | `uvicorn hippocampai.api.async_app:app` |
| Legacy API | `src/hippocampai/api/app.py` | Has more routes but older patterns |
| Library client | `src/hippocampai/client.py` | `from hippocampai import MemoryClient` |
| Config | `src/hippocampai/config.py` | `get_config()` singleton |
| DB factory | `src/hippocampai/auth/db.py` | `create_db_pool(db_type=...)` |
| Background jobs | `src/hippocampai/services/background_tasks.py` | Started in async_app lifespan |
| CLI | `src/hippocampai/cli/main.py` | `hippocampai init/remember/recall/api` |

---

## 4. Coding Standards

These are the ENFORCED patterns in this codebase. Every agent MUST follow them.

### 4.1 Tooling

| Tool | Config | Command |
|------|--------|---------|
| Formatter | `black` (line-length=100, py39) | `black src/` |
| Linter | `ruff` (rules: E, F, I) | `ruff check src/` |
| Type checker | `mypy` (strict mode) | `mypy src/hippocampai/` |
| Tests | `pytest` + `pytest-asyncio` (strict) | `python -m pytest tests/ -v` |
| Frontend | `tsc` (strict) + Vite | `cd frontend && npx tsc --noEmit` |

### 4.2 Python Formatting

```
Line length:    100 characters
Target Python:  3.9 (no X | Y unions in annotations, use Optional[X])
Indent:         4 spaces
Quotes:         double quotes (black default)
Trailing comma: yes in multiline
```

### 4.3 Import Order

Enforced by `ruff` rule `I` (isort-compatible). Three groups, blank line between:

```python
# 1. Standard library
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional, cast
from uuid import UUID

# 2. Third-party
import bcrypt
from asyncpg import Pool
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

# 3. Local (absolute imports ONLY — no relative imports)
from hippocampai.auth.models import User, UserCreate
from hippocampai.config import get_config
from hippocampai.models.memory import Memory, MemoryType
```

**NEVER use relative imports.** Always `from hippocampai.x.y import Z`.

### 4.4 Naming Conventions

| Element | Convention | Examples |
|---------|-----------|----------|
| Files | `snake_case.py` | `auth_service.py`, `memory_lifecycle.py` |
| Classes | `PascalCase` | `MemoryService`, `AuthMiddleware` |
| Internal classes | `_PascalCase` | `_SQLitePool`, `_SQLiteConnection` |
| Functions/methods | `snake_case` | `create_user`, `validate_api_key` |
| Constants | `UPPER_SNAKE` | `PROMETHEUS_AVAILABLE`, `_PG_PARAM_RE` |
| Private | `_prefix` | `_service`, `_handle_returning` |
| Fixtures | `snake_case` | `sqlite_pool`, `auth_service` |
| Test classes | `TestPascalCase` | `TestAuthServiceSQLite` |
| Test methods | `test_snake_case` | `test_create_user_duplicate_email` |

### 4.5 Type Annotations

**Rule: ALL public functions MUST have parameter types and return type.**

```python
# CORRECT — fully typed
async def create_user(self, user_data: UserCreate) -> User:

# CORRECT — Optional explicit
async def get_user(self, user_id: UUID) -> Optional[User]:

# CORRECT — dict return with type hint
async def validate_api_key(self, api_key: str) -> Optional[dict]:

# WRONG — missing return type (seen in route handlers — fix when touching)
async def detect_stale_memories(request: Request):
```

**Per CLAUDE.md: `type: ignore[no-any-return]` — FIX the actual type issue, NEVER suppress.**

### 4.6 Docstrings

**Style: Google-style.** Consistent across `auth_service.py`, `client.py`, `memory_service.py`.

```python
async def create_user(self, user_data: UserCreate) -> User:
    """Create a new user.

    Args:
        user_data: User creation data

    Returns:
        Created user

    Raises:
        ValueError: If email already exists
    """
```

**Rules:**
- All public functions MUST have docstrings
- Module-level docstrings on every file (`"""Description of this module."""`)
- Args/Returns/Raises sections when non-obvious
- One-line docstrings for trivial helpers are fine

### 4.7 Logging

**Pattern — every module:**
```python
import logging
logger = logging.getLogger(__name__)
```

**Levels:**
| Level | Use For | Example |
|-------|---------|---------|
| `logger.info` | Startup, shutdown, connections, feature flags | `"AuthService initialized (db_type=sqlite)"` |
| `logger.warning` | Degraded state, fallback, optional feature missing | `"Could not load intelligence routes: ..."` |
| `logger.error` | Operation failed, will affect user | `"Failed to create user: ..."` |
| `logger.critical` | Security risk, data integrity | `"Admin routes enabled without auth!"` |
| `logger.debug` | Verbose tracing | Rarely used in this codebase |

**NEVER** use `print()`. **NEVER** use `logging.basicConfig()` in library code.

### 4.8 Error Handling

**Current dominant pattern (355 occurrences — being improved):**
```python
# EXISTING (acceptable for now, but too broad)
try:
    result = await service.do_thing()
    return {"success": True, "result": result}
except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))
```

**TARGET pattern (write this for new code):**
```python
# PREFERRED — catch specific errors, let global handler catch the rest
async def do_thing(request: Request, body: ThingRequest) -> dict:
    """Do a specific thing."""
    service = get_service()
    try:
        result = await service.do_thing(body.user_id)
    except MemoryNotFoundError:
        raise HTTPException(status_code=404, detail="Memory not found")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return {"success": True, "result": result}
    # Unexpected errors bubble to global exception handler (ISSUE-014)
```

**NEVER catch Exception just to log and re-raise unchanged.**
**NEVER swallow exceptions silently.**

### 4.9 Database Queries

Auth layer uses **asyncpg-style SQL** compatible with both PostgreSQL and SQLite via `auth/db.py`:

```python
# Positional params — $1, $2, ... (translated to ? for SQLite)
row = await conn.fetchrow(
    "SELECT * FROM users WHERE email = $1 AND is_active = true",
    email,
)

# RETURNING * — emulated on SQLite
row = await conn.fetchrow(
    "INSERT INTO users (email, hashed_password) VALUES ($1, $2) RETURNING *",
    email, hashed,
)

# ::jsonb casts — stripped on SQLite
await conn.fetchrow(
    "INSERT INTO api_keys (scopes) VALUES ($1::jsonb) RETURNING *",
    json.dumps(scopes),
)

# Command tags for mutations
result = await conn.execute("DELETE FROM users WHERE id = $1", user_id)
deleted = result.endswith("1")  # "DELETE 1" → True
```

**NEVER use f-strings in SQL.** There are 11 f-string SQL queries in the codebase — each is a security risk. Always use positional parameters.

### 4.10 Route File Structure

Every route file follows this template:

```python
"""REST API routes for <feature>."""

from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from hippocampai.<module> import <dependencies>

router = APIRouter(prefix="/v1/<feature>", tags=["<feature>"])


# ============================================================================
# REQUEST/RESPONSE MODELS (Pydantic, defined inline)
# ============================================================================

class MyRequest(BaseModel):
    user_id: str
    option: bool = True

class MyResponse(BaseModel):
    success: bool
    data: dict


# ============================================================================
# ENDPOINTS
# ============================================================================

@router.post("/action")
async def do_action(request: Request, body: MyRequest) -> dict:
    """Short description of what this does."""
    try:
        # ... logic
        return {"success": True, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

**Route registration in async_app.py:**
```python
try:
    from hippocampai.api.healing_routes import router as healing_router
    app.include_router(healing_router)
    logger.info("Healing routes registered successfully")
except ImportError as e:
    logger.warning(f"Could not load healing routes: {e}")
```

### 4.11 Config Access

**Always use the singleton:**
```python
from hippocampai.config import get_config

config = get_config()
if config.enable_procedural_memory:
    ...
```

**NEVER** read `os.environ` directly for values defined in `Config`. Exception: secrets in adapter modules (`OPENAI_API_KEY`, `GROQ_API_KEY`).

### 4.12 Pydantic Models

```python
from pydantic import BaseModel, ConfigDict, Field

class User(BaseModel):
    id: UUID
    email: EmailStr
    tier: UserTier = UserTier.FREE
    is_active: bool = True
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8)
    full_name: Optional[str] = None
```

**Rules:**
- All request/response models inherit from `BaseModel`
- Use `Field(...)` for validation constraints
- Use `ConfigDict(from_attributes=True)` for DB-backed models
- Enums: `class UserTier(str, Enum)` pattern
- Response format: `{"success": True, "result": {...}}` or `{"error": "...", "detail": "..."}`

### 4.13 Testing

**Framework:** pytest + pytest-asyncio (strict mode)

```python
import pytest
import pytest_asyncio

@pytest_asyncio.fixture
async def sqlite_pool():
    """Create a temporary SQLite pool."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        pool = await create_db_pool(db_type="sqlite", sqlite_path=db_path)
        yield pool
        await pool.close()

@pytest.mark.asyncio
class TestMyFeature:
    async def test_happy_path(self, sqlite_pool):
        result = await do_thing(sqlite_pool)
        assert result is not None
        assert result.status == "ok"

    async def test_error_case(self, sqlite_pool):
        with pytest.raises(ValueError, match="already exists"):
            await do_bad_thing(sqlite_pool)

    async def test_not_found_returns_none(self, sqlite_pool):
        result = await find_thing("nonexistent")
        assert result is None
```

**Rules:**
- File naming: `tests/test_<feature>.py`
- Class naming: `TestFeatureName`
- Method naming: `test_<scenario>` — be descriptive
- Use `@pytest_asyncio.fixture` for async fixtures
- Use `tempfile.TemporaryDirectory()` for test databases
- Always clean up resources in fixture teardown
- Run: `python -m pytest tests/test_<file>.py -v --tb=short`

### 4.14 Frontend (TypeScript/React)

```typescript
// API calls — two Axios instances in services/api.ts
const client = axios.create({ baseURL: '/api' });     // admin/dashboard
const v1Client = axios.create({ baseURL: '/v1' });    // memory API

// React Query for data fetching (111 hooks across 27 pages)
const { data, isLoading } = useQuery({
  queryKey: ['memories', userId],
  queryFn: () => api.getMemories(userId),
});

// Mutations
const mutation = useMutation({
  mutationFn: (data) => api.createMemory(data),
  onSuccess: () => queryClient.invalidateQueries(['memories']),
});

// Types in frontend/src/types/index.ts (shared)
// Pages in frontend/src/pages/<Name>Page.tsx
```

**Build check:** `cd frontend && npx tsc --noEmit` — must pass with zero errors.

### 4.15 Git & Commits

```
feat: add /healthz endpoint with dependency checks
fix: enforce user_id isolation in memory_service
refactor: remove duplicate in-memory rate limiter
test: add auth middleware integration tests
docs: add migration documentation
chore: update aiosqlite to 0.20
```

**Rules:**
- One logical change per commit
- Never commit secrets, .env files, or API keys
- Always run tests before committing
- Never amend published commits

---

## 5. Anti-Patterns to Avoid

These are specific mistakes found in the codebase. Agents must NOT introduce more.

| Anti-Pattern | Where Found | Correct Approach |
|---|---|---|
| f-string SQL | `audit/logger.py:304`, `consolidation/db.py:288,311` | Use positional params `$1, $2` |
| `except Exception as e: raise HTTPException(500, str(e))` | 355 occurrences across routes | Catch specific errors; let global handler catch rest |
| Route handler missing return type | `healing_routes.py` (0/12 typed) | Add `-> dict` or Pydantic response model |
| Global module-level initialization | `healing_routes.py:17-19` (creates Embedder at import) | Initialize in lifespan or lazy-load |
| Optional user_id validation | `memory_service.py:620` (`if user_id and ...`) | Always require user_id |
| Silent import failure | `async_app.py:239` (logs warning, continues) | Fail fast or report via status endpoint |
| Circular import via lazy import | `tasks.py:485` (`from hippocampai.api.deps import ...`) | Restructure deps or use protocol classes |
| In-memory state shared across workers | `api/deps.py:34` (`defaultdict(list)`) | Use Redis for distributed state |

---

## 6. Agent Routing

### Decision Tree

```
What kind of task is this?
│
├─ Writing/modifying Python backend code?
│  ├─ Memory pipeline (ingestion, decay, consolidation, recall)? → memory-core
│  ├─ Auth, secrets, CORS, rate limiting, sessions? → security
│  └─ Everything else (routes, services, CLI, models)? → core-code
│
├─ Docker, Compose, monitoring, metrics, logging? → observability-infra
├─ Writing or fixing tests? → test-reliability
├─ Documentation or release prep? → docs-release
├─ Search quality, ranking, recall/precision? → retrieval-eval
├─ Performance, latency, profiling? → performance-profiler
├─ Planning a multi-step task? → planner FIRST, then implementation agents
└─ Just completed a large change? → architecture-review
```

### Agent Capabilities

| Agent | Scope | Tools | Key Files |
|-------|-------|-------|-----------|
| **core-code** | Backend features, routes, services, CLI, models | All | `async_app.py`, `memory_service.py`, `client.py`, route files |
| **security** | Auth, secrets, CORS, rate limiting, middleware, sessions | All | `auth/`, `middleware.py`, `rate_limiter.py`, `.env` |
| **memory-core** | Memory ingestion, schemas, consolidation, recall, decay, triggers | All | `pipeline/`, `services/`, `models/memory.py`, `consolidation/` |
| **observability-infra** | Docker, monitoring, metrics, logging, health checks | All | `docker-compose.yml`, `monitoring/`, `Dockerfile` |
| **test-reliability** | Tests, coverage, regression, failure paths | All | `tests/`, `conftest.py` |
| **docs-release** | Docs, changelogs, migration guides, release prep | All | `docs/`, `CHANGELOG.md`, `README.md` |
| **planner** | Multi-step planning, sprint design, dependency ordering | Read-only | `issues.md`, `agents.md` |
| **architecture-review** | Post-change review for coupling, boundaries, imports | Read-only | Entire `src/` tree |
| **performance-profiler** | Latency, bottlenecks, resource profiling | All | `services/`, `retrieval/`, `embed/` |
| **retrieval-eval** | Search quality, ranking, benchmarks | All | `retrieval/`, `vector/`, `search/` |
| **hippocampai-release-validator** | Pre-release E2E validation across all modes | All | Everything |

---

## 7. Agent-Specific Instructions

### 7.1 `core-code`

**Before writing code:**
1. Read the target file completely
2. Check `config.py` for relevant feature flags
3. Check if feature exists in `app.py` (legacy) — port pattern to `async_app.py`
4. Check `models/` for existing Pydantic models before creating new ones

**After writing code:**
1. `python -m pytest tests/test_<relevant>.py -v --tb=short`
2. `cd frontend && npx tsc --noEmit` (if types/API changed)
3. Verify no new `type: ignore` suppressions added

**Quality gates:**
- All new functions have return types and docstrings
- All new routes use Pydantic request models
- All database queries use positional params
- No broad `except Exception` without re-raise

### 7.2 `security`

**Threat model (priority order):**
1. **Multi-tenancy** — user_id MUST be enforced on every data access path
2. **Input validation** — all user input via Pydantic before processing
3. **Secrets** — never in code; env vars only; rotate if exposed
4. **Auth bypass** — verify middleware covers all routes (check `PUBLIC_PATHS` in `middleware.py:23-35`)
5. **Rate limiting** — must work even when auth is disabled
6. **SQL injection** — 11 f-string SQL queries exist; parameterize all

**Current auth architecture:**
- API keys: bcrypt-hashed, validated per-request in middleware
- Sessions: plain-text tokens in DB (ISSUE-016: must hash)
- Bypass: `PUBLIC_PATHS` list in middleware
- Rate limiting: token-bucket via Redis (`auth/rate_limiter.py`)
- Default: `USER_AUTH_ENABLED=false` — everything open

### 7.3 `memory-core`

**Memory model:** `src/hippocampai/models/memory.py` (imported by 55 files — change carefully)
**Lifecycle tiers:** hot → warm → cold → archived → hibernated (`pipeline/memory_lifecycle.py`)

**Key operations:**
| Operation | File | Pattern |
|-----------|------|---------|
| Ingest | `memory_service.py:create_memory()` | Embed → Qdrant + Redis |
| Recall | `memory_service.py:recall()` | Vector + BM25 + Rerank + Graph |
| Dedup | `services/deduplication_service.py` | Cosine similarity threshold |
| Consolidate | `services/consolidation_service.py` | Merge similar memories |
| Decay | `pipeline/importance_decay.py` | 5 strategies: exponential/linear/logarithmic/step/hybrid |

### 7.4 `observability-infra`

**Docker service status:**

| Service | Status | Issue |
|---------|--------|-------|
| hippocampai | Working | — |
| frontend | Working | — |
| qdrant | Working | — |
| redis | Working | Eviction policy wrong (ISSUE-004) |
| postgres | Working | — |
| celery-worker | Working | — |
| celery-beat | Working | Redundant with APScheduler (ISSUE-006) |
| flower | Functional | — |
| admin | Functional | — |
| prometheus | Partial | Metrics defined but never called (ISSUE-005) |
| grafana | Partial | Empty dashboards directory (ISSUE-032) |

### 7.5 `test-reliability`

**Infrastructure:**
- Fixtures in `tests/conftest.py` (3 fixtures: `ensure_qdrant_collections`, `memory_client`, `user_id`)
- `.env` loaded into `os.environ` by conftest
- Use `tempfile.TemporaryDirectory()` for SQLite test DBs
- Async: `@pytest_asyncio.fixture` + `@pytest.mark.asyncio`

**Priority test files to create:**
1. `tests/test_auth_middleware.py` — auth flow, rate limiting, session validation
2. `tests/test_healing_routes.py` — all 12 healing endpoints
3. `tests/test_background_tasks.py` — task scheduling, error handling
4. `tests/test_cli.py` — CLI commands with mock services

### 7.6 `architecture-review`

**Known concerns to check:**
1. `async_app.py` vs `app.py` — two entry points, diverging route sets
2. `client.py` at 7,406 lines — single-file monolith
3. `pipeline/` has 33 files — potential circular dependencies
4. Three scheduling systems coexist: Celery Beat, APScheduler, BackgroundTaskManager
5. `models/memory.py` imported by 55 files — any change has massive blast radius
6. Module-level initialization in route files (creates services at import time)

---

## 8. Issue Dispatch Matrix

Full issue details in `issues.md`. This table maps each issue to its agent and scope.

### Phase 1 — Critical (7 issues, blocks production)

| ID | Agent | Title | Scope |
|----|-------|-------|-------|
| ISSUE-001 | core-code | Register 15 route files in async_app.py | ~100 lines in `async_app.py` |
| ISSUE-002 | security | Fix multi-tenancy isolation | 3 files, ~30 lines |
| ISSUE-003 | security | Enforce rate limiting always | 2 files, ~50 lines |
| ISSUE-004 | observability-infra | Fix Redis eviction policy | 1 line in `docker-compose.yml` |
| ISSUE-005 | observability-infra | Instrument Prometheus metrics | 3 files, ~40 lines |
| ISSUE-006 | observability-infra | Remove duplicate scheduler | 3 files, ~100 lines removed |
| ISSUE-007 | security | Rotate committed API key | Config only |

### Phase 2 — High (16 issues)

| ID | Agent | Title |
|----|-------|-------|
| ISSUE-008 | core-code | Persist healing config to DB |
| ISSUE-009 | memory-core | Automate per-user dedup/consolidation |
| ISSUE-010 | memory-core | Init prospective memory manager |
| ISSUE-011 | memory-core | Wire TMS into pipeline |
| ISSUE-012 | memory-core | Wire Knowledge Graph |
| ISSUE-013 | core-code | Replace silent import failures |
| ISSUE-014 | core-code | Global exception handler |
| ISSUE-015 | core-code | Fix Celery task name mismatches |
| ISSUE-016 | security | Hash session tokens |
| ISSUE-017 | core-code | Implement /healthz endpoint |
| ISSUE-018 | observability-infra | Fix Qdrant prometheus target |
| ISSUE-019 | core-code | Move heavy deps to optional |
| ISSUE-020 | core-code | Create py.typed marker |
| ISSUE-021 | core-code | Eager validation in MemoryClient |
| ISSUE-022 | core-code | Custom exception hierarchy |
| ISSUE-023 | core-code | Fix core version skew |

### Phase 3 — Medium (12 issues)

| ID | Agent | Title |
|----|-------|-------|
| ISSUE-024 | core-code | Redis connection retry with backoff |
| ISSUE-025 | security | Fix CORS defaults |
| ISSUE-026 | core-code | Standardize error response format |
| ISSUE-027 | core-code | LLM/embedder call timeouts |
| ISSUE-028 | observability-infra | Structured JSON logging |
| ISSUE-029 | core-code | CLI status/config/export commands |
| ISSUE-030 | core-code | In-memory vector store fallback |
| ISSUE-031 | core-code | Remove duplicate rate limiter |
| ISSUE-032 | observability-infra | Grafana dashboard templates |
| ISSUE-033 | core-code | Memory.text max_length validation |
| ISSUE-034 | test-reliability | Expand test coverage to 60%+ |
| ISSUE-035 | docs-release | Migration documentation |

### Phase 4 — Low (5 issues)

| ID | Agent | Title |
|----|-------|-------|
| ISSUE-036 | core-code | Lock manager cleanup |
| ISSUE-037 | core-code | DB pool warm-up test |
| ISSUE-038 | core-code | __init__.pyi stubs |
| ISSUE-039 | docs-release | Migration guides from mem0/Zep |
| ISSUE-040 | docs-release | Document partition creation |

---

## 9. Execution Playbooks

### Fix a single issue
```
Read issues.md and find ISSUE-<N>. Read agents.md section 6 for which agent,
section 4 for coding standards, and section 5 for anti-patterns to avoid.
```

### Fix all Phase 1 criticals (recommended order)
```
1. ISSUE-007 → security       (rotate key — immediate, no code)
2. ISSUE-004 → observability   (one-line Redis fix)
3. ISSUE-002 → security       (user_id isolation — 3 files)
4. ISSUE-003 → security       (rate limiting — 2 files)
5. ISSUE-006 → observability   (remove dup scheduler)
6. ISSUE-001 → core-code      (register 15 routes — largest change)
7. ISSUE-005 → observability   (instrument metrics — needs routes)
```

### Fix by area
```
Read issues.md. Fix all issues tagged [saas]. Use agents.md for routing.
```

### Plan a sprint
```
Read issues.md and agents.md. Plan a sprint for Phase 1 + Phase 2.
Group by agent. Give dependency-ordered task list.
```

### Post-implementation review
```
I just completed ISSUE-001 (registered 15 route files). Use architecture-review
to check for circular imports, endpoint conflicts, and coupling issues.
```

### Pre-release validation
```
Use hippocampai-release-validator. Validate: library install, API server startup,
Docker Compose up, all endpoints responding, auth flow, rate limiting.
```
