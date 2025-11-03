# HippocampAI Usage Guide

This guide walks through the common ways to run, integrate, and extend HippocampAI in your agents and applications.

---

## 1. Prerequisites

- Python 3.9+
- Qdrant vector database (Docker or managed)
- Optional: Ollama for local LLMs, or API keys for OpenAI/Anthropic/Groq

Spin up Qdrant locally:

```bash
docker run -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  qdrant/qdrant
```

Install dependencies and initialize HippocampAI:

```bash
pip install -r requirements.txt
python setup_initial.py
```

`setup.py` bootstraps config files, validates connections, and creates the `hippocampai_facts` and `hippocampai_prefs` collections in Qdrant.

---

## 2. Running the Chat Clients

### CLI Chat

```bash
python cli_chat.py alice
```

- Navigate conversations with commands such as `/stats`, `/memories`, `/help`, `/quit`.
- Memories extracted from the chat are stored automatically and surfaced in responses.

### Web Chat

```bash
python web_chat.py
```

Open `http://localhost:5000` to access the web UI:

- Responsive layout with conversation history and live memory stats
- Memory viewer shows what HippocampAI stored from the session
- Supports tool-calling extensions configured in `docs/TOOLS.md`

---

## 3. Embedding HippocampAI in Python Agents

```python
from hippocampai import MemoryClient

client = MemoryClient()

client.remember(
    text="I am traveling to Berlin next week.",
    user_id="user_456",
    session_id="travel_planning",
    type="event"
)

results = client.recall(
    query="Where is the user going?",
    user_id="user_456",
    session_id="travel_planning"
)

for result in results:
    print(result.memory.text, result.score)
```

- Memories are scored, deduplicated, vectorized, and persisted in Qdrant.
- Retrieval blends vector similarity, BM25, reranking, recency, and importance.
- Use `filters={"type": "preference"}` to limit by memory type.

---

## 4. Scheduling Maintenance Jobs

HippocampAI ships with background jobs (see `src/hippocampai/jobs/`) to keep the memory store fresh:

- **Decay**: apply exponential decay to importance scores.
- **Consolidation**: merge clusters of similar memories.
- **Snapshots**: create periodic Qdrant snapshots for backup.

Enable or disable jobs in `.env` or `config/config.yaml`:

```yaml
jobs:
  enable_scheduler: true
  decay_cron: "0 2 * * *"
  consolidate_cron: "0 3 * * 0"
  snapshot_cron: "0 * * * *"
```

Run the scheduler:

```bash
python -m hippocampai.jobs.scheduler
```

---

## 5. Extending HippocampAI

- **Providers**: Implement custom LLM providers by extending `hippocampai.adapters.llm_base.BaseLLM`.
- **Embeddings**: Swap embedding models via environment variables (`EMBED_MODEL`) without touching code.
- **Tools**: Add new functions for the agent to call through the tool framework described in `docs/TOOLS.md`.
- **Pipelines**: Override extractor/deduplicator/consolidator classes when instantiating `MemoryClient` for bespoke logic.

---

## 6. HippocampAI vs Mem0 and Other Platforms

| Scenario                              | Choose HippocampAI when…                                      | Choose Mem0/others when…                           |
|--------------------------------------|---------------------------------------------------------------|----------------------------------------------------|
| Hosting                               | You need on-prem/offline control over data and models.        | You prefer a managed SaaS with no infra overhead.  |
| Custom pipelines                      | You want to tweak extraction, scoring, and retrieval logic.   | Default behavior is sufficient and fixed.          |
| Compliance & data residency           | You must keep data within your environment.                   | Cloud storage is acceptable.                       |
| Time to first response                | You can invest in running Qdrant + optional local LLMs.       | You need instant setup via REST APIs.              |
| Cost control                          | You want to leverage open weights and local inference.        | You’re comfortable with provider pricing tiers.    |

HippocampAI delivers deeper control and extensibility; Mem0 and similar hosted platforms trade that flexibility for turnkey convenience.

---

## 7. Troubleshooting

- **Qdrant connection errors** – ensure ports 6333/6334 are open and credentials match `.env`.
- **Missing embeddings** – confirm the embedding model is downloadable (requires network or pre-downloaded model).
- **LLM failures** – if running locally via Ollama, verify the model is pulled; for cloud APIs, check rate limits.
- **Duplicate memories** – adjust the dedupe similarity threshold (`DEDUP_SIMILARITY_THRESHOLD`) in `.env`.

Need more help? Open an issue or check the example scripts in `examples/` for working reference flows.
You can also join the community Discord for live support and collaboration: <https://discord.gg/pPSNW9J7gB>
