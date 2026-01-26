# claude.md — HippocampAI (library + SaaS) implementation guide

You are **Claude Code** working inside the **HippocampAI** repository.

## Runtime + Environment
- Conda environment name: **hippocamp**
- Prefer Python **3.11+**
- Static typing: **mypy** (strict where feasible)
- Formatting/linting: keep existing repo tooling; if missing, add **ruff** + **black** (only if repo doesn’t already standardize something else)

## Non-negotiable rules (anti-hallucination)
1. **Do not invent files, APIs, classes, endpoints, env vars, or behaviors.**
2. Before making changes:
   - Inspect the repo structure, README/docs, and current public APIs.
   - Identify existing modules for: storage, memory objects, ingestion, retrieval, ranking, API server.
3. After each change:
   - Run tests (or add them if missing).
   - Run `mypy` and fix errors.
   - Ensure `docker compose up` works end-to-end.
4. If something is unclear:
   - Search the codebase first.
   - If still ambiguous, implement in a minimal, backwards-compatible way and document assumptions in `docs/` and in PR notes.
5. Keep changes small and verifiable:
   - Add unit tests for each new feature.
   - Add at least one integration test for SaaS API endpoints (if present).
6. Use .env files for the credentials
---

## What we are implementing (skip Graph DB for now)
Implement these features in **both** the Python library and the SaaS surface area (API/config/docs):

### (2) Bi-temporal fact tracking
Add bi-temporal modeling for “facts” / memory assertions:
- **event_time**: when the fact occurred (or was stated)
- **valid_time**: the real-world validity interval of the fact (start/end)
- **system_time**: when HippocampAI stored/observed it (append-only ledger)
- Support corrections (superseding facts) without deleting history.
- Queries should support:
  - “As of system_time T, what did we believe?”
  - “What was valid during [valid_start, valid_end]?”
  - “Latest known valid fact” for an entity/property.

Deliverables:
- Data model changes (Pydantic/dataclass + storage schema)
- Storage migrations (if any)
- Retrieval filters + query API
- Tests:
  - inserting fact revisions
  - querying by valid_time
  - querying by system_time (“as-of”)
- Docs: how to use it + examples

### (3) Automated context assembly
Agents should not manually pick memory items; HippocampAI should assemble a context pack:
- Input: `user_query`, optional `session_id`, optional constraints (token_budget, recency_bias, entity_focus, allow_summaries)
- Steps:
  1. Retrieve candidates (vector/BM25/whatever repo already supports)
  2. Re-rank (existing cross-encoder/reranker if present)
  3. Deduplicate
  4. Apply time filters (including bi-temporal where relevant)
  5. Compress/summarize when token budget exceeded (use existing summarization utilities if present; do not invent)
  6. Output a **ContextPack**:
     - `final_context_text`
     - `citations` (memory IDs)
     - `selected_items` (structured)
     - `dropped_items` with reasons (optional, for debugging)
- Provide a single call like `assemble_context(...)` with sane defaults.

Deliverables:
- New module/function(s) for context assembly
- Token budgeting utility (estimate tokens; if no tokenizer dependency exists, implement a conservative heuristic)
- Tests:
  - deterministic selection under fixed seed
  - budget trimming
  - citations match selected items
- Docs: “Using Context Assembly”

### (4) Custom schema support
Let users define entity/relationship types *without* rewriting code:
- Provide a schema definition mechanism:
  - Option A: JSONSchema/YAML file
  - Option B: Python dict config
- Must support:
  - Entity types + allowed attributes
  - Relationship types + allowed endpoints + attributes
  - Validation of memory objects against schema
  - Backwards-compatible default schema if none provided

Deliverables:
- `SchemaRegistry` (or similar) with:
  - load/validate schema
  - validate entity/relationship payloads
- Integration points:
  - ingestion validates payloads
  - context pack includes typed entities when available
- Tests:
  - valid schema loads
  - invalid schema rejected with clear errors
  - invalid entity rejected
- Docs: schema format + examples

### (5) Benchmarks (publishable + reproducible)
Add a benchmark suite to prove performance claims:
- Must run locally via one command (Makefile or scripts)
- Benchmarks should cover:
  - Ingestion throughput (memories/sec)
  - Retrieval latency (p50/p95)
  - Reranking overhead (if applicable)
  - Context assembly latency + context quality proxy metrics (e.g., recall@k using a labeled mini dataset if feasible)
- Include a small synthetic dataset generator if no dataset exists (do not claim it reflects real-world accuracy unless proven).
- Output results as:
  - JSON artifact
  - Markdown summary table under `docs/benchmarks.md`

Deliverables:
- `bench/` folder with runnable scripts
- `docker compose` profile for benchmark infra (Qdrant/Redis/Postgres/etc if used)
- Docs for running benchmarks + interpreting results

### (6) MCP server integration
Add an **MCP server** so modern agent frameworks can call HippocampAI tools:
- Provide MCP tools such as:
  - `memory_add`
  - `memory_search`
  - `context_assemble`
  - `memory_get`
  - `memory_update` (append-only where possible; revisions for bi-temporal)
- Must be configurable by `.env`
- Must run via `docker compose up` (service `mcp-server`)
- Keep it minimal; do not add unnecessary dependencies.

Deliverables:
- `mcp_server/` (or repo-appropriate location)
- Tool definitions mapped to internal library calls
- Basic auth or API key from env (if SaaS already has auth, reuse pattern)
- Tests: at least one happy-path test for tool invocation

---

## Implementation plan (follow this order)
1. **Repo audit**
   - Identify current memory object model(s) and storage backends.
   - Identify retrieval + reranking pipeline.
   - Identify SaaS API layer and how it calls the library.
   - Identify existing docker-compose and infra services.
   - Summarize findings in `docs/architecture_current.md` (short, factual).

2. **Bi-temporal fact tracking**
   - Implement data model + storage adaptations.
   - Add query filters and tests.
   - Update docs.

3. **Automated context assembly**
   - Implement `ContextPack` + assembly pipeline.
   - Add unit/integration tests.
   - Update docs.

4. **Custom schema support**
   - Implement schema loader + validation.
   - Wire into ingestion + relevant APIs.
   - Add tests + docs.

5. **MCP server**
   - Implement MCP service exposing core tools.
   - Add docker compose service.
   - Add smoke tests + docs.

6. **Benchmarks**
   - Add bench harness, dataset generator, and docs output.
   - Add a CI job (optional) that runs a lightweight benchmark subset.

7. **Cleanup**
   - Remove dead code / outdated docs.
   - Ensure docs match actual behavior.
   - Ensure `.env.example` is updated and referenced everywhere.

---

## Docker + Compose requirements
- Provide a single top-level `docker-compose.yml` (or keep existing) that can run:
  - `api` (SaaS)
  - `worker` (if used)
  - `qdrant` (if used)
  - `redis` (if used)
  - `postgres` (if used)
  - `mcp-server`
- All credentials/config must be from `.env`
- Provide `.env.example` with safe defaults:
  - `QDRANT_URL=...`
  - `POSTGRES_DSN=...`
  - `REDIS_URL=...`
  - `HIPPOCAMPAI_API_KEY=...` (if needed)
  - `MCP_HOST=0.0.0.0`
  - `MCP_PORT=...`
- Never commit real secrets.

---

## Typing + mypy requirements
- Add/adjust `pyproject.toml` (or `mypy.ini`) for mypy:
  - Prefer `strict = true` for new modules
  - If strict breaks legacy code, scope strictness to new packages first
- All new public functions must have type hints.
- Prefer `pydantic` models only if repo already uses them; otherwise use `dataclasses`.

---

## Documentation requirements
Update docs as features land:
- `docs/bi_temporal.md`
- `docs/context_assembly.md`
- `docs/custom_schema.md`
- `docs/mcp.md`
- `docs/benchmarks.md`
Also update:
- Main `README.md` feature list
- Any API reference docs
- Add “Quickstart (Docker)” section with copy-paste commands

Docs must include:
- Minimal working code snippets
- Expected outputs
- Common errors + fixes
- Versioning note if this is a breaking change

---

## Testing requirements
Minimum test coverage for each feature:
- Unit tests for core logic
- Integration test for SaaS endpoints (if present)
- For MCP: a smoke test that starts server (or mocks) and calls one tool
- Ensure tests run in CI if CI exists; otherwise add a minimal GitHub Actions workflow

Commands (adapt to repo tooling):
- `pytest -q`
- `mypy .`
- `docker compose up -d && <smoke test> && docker compose down`

---

## Output format for your work
For each major step, provide:
1. What changed (file list)
2. How to run it (exact commands)
3. What tests were added and how to run
4. Any migration notes
5. Any assumptions (explicit, minimal)

---

## Guardrails: what NOT to do
- Do not add Graph DB support in this iteration.
- Do not introduce heavy new dependencies unless absolutely required.
- Do not break existing public APIs unless you:
  - provide a compatibility layer, AND
  - document breaking changes clearly.
- Do not claim “benchmarks prove accuracy” unless you have labeled evaluation data.

---

## Success criteria checklist
- [ ] Bi-temporal facts supported + tested + documented
- [ ] Context assembly available as one call + token budgeting + citations
- [ ] Custom schema load/validate + integrated into ingestion
- [ ] MCP server runs via docker compose and exposes core tools
- [ ] Benchmarks runnable + results generated + docs updated
- [ ] `mypy` passes (or strict on new modules) + tests pass
- [ ] Docs are current; remove outdated sections

END.
