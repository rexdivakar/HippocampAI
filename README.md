# HippocampAI — Autonomous Memory Engine for LLM Agents

[![Quality Gate Status](https://sonar.craftedbrain.com/api/project_badges/measure?project=rexdivakar_HippocampAI_6669aa8c-2e81-4016-9993-b29a3a78c475&metric=alert_status&token=sqb_dd0c0b1bf58646ce474b64a1fa8d83446345bccf)](https://sonar.craftedbrain.com/dashboard?id=rexdivakar_HippocampAI_6669aa8c-2e81-4016-9993-b29a3a78c475)

HippocampAI turns raw conversations into a curated long-term memory vault for your AI assistants. It extracts, scores, deduplicates, stores, and retrieves user memories so agents can stay personal, consistent, and context-aware across sessions.

- Plug-and-play `MemoryClient` API with built-in pipelines for extraction, dedupe, consolidation, and importance decay
- Hybrid retrieval that fuses dense vectors, BM25, reciprocal-rank fusion, reranking, recency, and importance signals
- Works fully self-hosted via Qdrant + Ollama or in the cloud via OpenAI (and other providers) with the same code paths
- Ships with CLI and web chat UIs, typed models, scheduled maintenance jobs, and production-grade logging/config

---

## Why HippocampAI

- **Persistent personalization** – store preferences, facts, goals, habits, and events per user with importance scoring and decay
- **Reliable retrieval** – hybrid ranking surfaces the right memories even when queries are vague or drift semantically
- **Automatic hygiene** – extractor, deduplicator, consolidator, and scorer keep the memory base uncluttered
- **Local-first** – run everything on your infra with open models, or flip a switch to activate OpenAI for higher quality
- **Extensible Python SDK** – customize every stage without being locked to a hosted API

---

## Quick Start

```bash
git clone https://github.com/rexdivakar/HippocampAI.git
cd HippocampAI
python -m venv .venv
source .venv/bin/activate
pip install -e .
python setup_initial.py  # creates .env, initializes Qdrant collections
```

Configure the generated `.env`:

```bash
OPENAI_API_KEY=sk-...
QDRANT_URL=http://localhost:6333
LLM_PROVIDER=ollama       # or openai
LLM_MODEL=qwen2.5:7b-instruct
```

Start the chat clients:

```bash
python -m hippocampai.cli_chat alice
python -m hippocampai.web_chat        # open http://localhost:5000
```

Run the interactive example suite:

```bash
./run_example.sh
```

---

## Embed HippocampAI in Your Agent

```python
from hippocampai import MemoryClient

client = MemoryClient(user_id="user_123")

client.remember(
    text="I prefer oat milk in my coffee.",
    user_id="user_123",
    session_id="morning_chat",
    type="preference"
)

memories = client.recall(
    query="How should I serve their coffee?",
    user_id="user_123",
    session_id="morning_chat",
    k=3
)

for item in memories:
    print(item.memory.text, item.score)
```

- `remember` scores importance, checks for duplicates, stores vectors + payload in Qdrant.
- `recall` rebuilds BM25 on demand, routes queries, fuses vector + lexical hits, reranks, and blends final scores.

---

## Feature Highlights

- **Memory extraction** – heuristics + optional LLM convert conversation logs into structured memories.
- **Deduplication & consolidation** – detect near duplicates via embeddings + rerankers; merge clusters into concise summaries.
- **Importance decay** – configurable half-life per memory type plus recency/access boosts keep results fresh.
- **Hybrid retrieval** – vector search + BM25 + reciprocal-rank fusion + cross-encoder rerank → final recency/importance weighted set.
- **Session management** – track sessions, keep summaries, and bias retrieval toward the active conversation.
- **Jobs & tooling** – built-in cron schedules for decay/consolidation/snapshots; CLI commands for maintenance.

Full API and architecture references live under `docs/`.

---

## Usage Documentation

- [docs/USAGE.md](docs/USAGE.md) – end-to-end setup, CLI/web walkthroughs, API snippets, and troubleshooting.
- [docs/CHAT_INTEGRATION.md](docs/CHAT_INTEGRATION.md) – connect HippocampAI memories to your custom chat front end.
- [docs/CONFIGURATION.md](docs/CONFIGURATION.md) – environment variables, YAML overrides, deployment tips.
- [docs/PROVIDERS.md](docs/PROVIDERS.md) – switch between Ollama, OpenAI, Anthropic, Groq.
- [docs/TOOLS.md](docs/TOOLS.md) – extend agents with external tool calls.
- Need help or want to collaborate? Join the community Discord: https://discord.gg/pPSNW9J7gB

---

## Dependency Management

- We follow a "latest minor" pinning strategy: each requirement is constrained with `>=` and `<` to stay current without taking untested major releases.
- Install optional groups with extras, e.g. `pip install .[core]` for provider SDKs, `pip install .[test]` for the pytest stack, and `pip install .[docs]` for MkDocs authoring.
- Run `pip install .[dev]` to pull in tooling such as Ruff, MyPy, `pip-audit`, and `liccheck`.
- The `dependency-health` GitHub workflow runs weekly (and on relevant PRs) to audit vulnerabilities via `pip-audit` and enforce license compatibility with `liccheck`.

---

## Testing

- Execute the unit suite with `./scripts/run-tests.sh`; pass extra pytest arguments as needed (for example `./scripts/run-tests.sh -k retrieval`).
- The helper script exports `PYTHONPATH=src` automatically, so no additional environment tweaks are required.
- CI runs the same command; please mirror it locally before opening a PR.

---

## Contributing

- Start with [CONTRIBUTING.md](CONTRIBUTING.md) for environment setup, coding conventions, and pull request expectations.
- Run `ruff check` and `pytest` locally before opening a PR; add tests for any behavior changes.
- All timestamps are stored as timezone-aware UTC (`+00:00`); please normalize new datetime fields via `hippocampai.utils.time` helpers.

---

## HippocampAI vs Mem0 and Others

| Capability                | HippocampAI                                         | Mem0 / hosted memory APIs                    |
|---------------------------|----------------------------------------------------|----------------------------------------------|
| Deployment                | Self-hosted first (Qdrant + Ollama), cloud optional| SaaS-first, hosted vector memory             |
| Customization             | Full Python pipeline control, extend every stage   | Limited to exposed API surface               |
| Retrieval                 | Hybrid fusion (vector + BM25 + rerank + decay)     | Typically vector-first with optional rerank  |
| Memory hygiene            | Built-in extraction, dedupe, consolidation, decay  | Generally manual or implicit                 |
| Data residency & control  | Your infra, your compliance boundaries             | Stored in provider cloud                     |
| Setup effort              | Requires Qdrant + (optional) local models          | Minimal—call REST API                        |

HippocampAI is ideal when you need ownership, customization, or offline capability. Mem0 (and similar SaaS tools) win when you want zero infrastructure and accept managed trade-offs.

---

## Roadmap

- LangChain/LlamaIndex adapters
- Retrieval evaluators and telemetry dashboards
- Multi-tenant access policies and RBAC
- Native TypeScript SDK

Contributions welcome—open issues or PRs to shape the direction of HippocampAI.

---

## License

Apache 2.0. See [LICENSE](LICENSE) for details.
