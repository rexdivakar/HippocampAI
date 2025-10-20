# Changelog

All notable changes to HippocampAI will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-10-20

### Added

#### Core Features
- Initial release of HippocampAI memory engine
- `MemoryClient` with `remember()`, `recall()`, and `extract_from_conversation()` methods
- Hybrid retrieval system combining vector search, BM25, RRF, and cross-encoder reranking
- Multi-user isolation with per-user collections
- Memory type support: preference, fact, goal, habit, event, context
- Importance scoring with configurable decay

#### Configuration
- Configuration presets for easy deployment:
  - `local` - Fully self-hosted (Ollama + local Qdrant)
  - `cloud` - Cloud LLM with local vector DB (OpenAI + Qdrant)
  - `production` - Optimized settings for production workloads
  - `development` - Fast settings for development/testing
- Pydantic-based configuration with environment variable support
- `.env.example` with all configuration options documented

#### Telemetry & Observability
- Comprehensive telemetry system similar to Mem0 platform
- Automatic operation tracking for all memory operations
- Performance metrics with P50, P95, P99 latencies
- Detailed trace inspection with events timeline
- Operation filtering by type (remember, recall, extract)
- Export functionality for external tools (JSON format)
- **Library-level access only** (no REST API endpoints for telemetry):
  - `client.get_telemetry_metrics()` - Get performance metrics
  - `client.get_recent_operations()` - Get recent operations with traces
  - `client.export_telemetry()` - Export telemetry data
  - `get_telemetry()` - Access global telemetry instance

#### Interfaces
- FastAPI REST API with endpoints (memory operations only):
  - `POST /v1/memories:remember` - Store memories
  - `POST /v1/memories:recall` - Retrieve memories
  - `POST /v1/memories:extract` - Extract from conversations
  - `GET /healthz` - Health check
  - **Note:** Telemetry accessed via library functions, not REST endpoints
- CLI chat interface (`cli_chat.py`)
- Web chat interface (`web_chat.py`)
- Entry point commands: `hippocampai`, `hippocampai-api`

#### Pipeline Components
- Memory extraction from conversations (heuristic and LLM-based)
- Semantic deduplication with configurable thresholds
- Memory consolidation for related memories
- Importance decay with per-type half-lives
- Scheduled jobs for maintenance

#### Retrieval Features
- HNSW vector search via Qdrant
- BM25 keyword search with rank-bm25
- Reciprocal rank fusion (RRF)
- Cross-encoder reranking
- Score fusion with configurable weights:
  - Semantic similarity
  - Reranker score
  - Recency
  - Importance
- Session-aware filtering

#### Provider Support
- **Embeddings:** sentence-transformers (BAAI/bge-small-en-v1.5 default)
- **Vector DB:** Qdrant
- **LLM Providers:**
  - Ollama (local, self-hosted)
  - OpenAI (cloud)
  - Anthropic (cloud)
  - Groq (cloud)
- **Reranker:** cross-encoder/ms-marco-MiniLM-L-6-v2

#### Examples & Documentation
- 5 working examples in `examples/` directory:
  - Basic usage
  - Conversation extraction
  - Hybrid retrieval
  - Custom configuration
  - Multi-user management
- Comprehensive documentation in `docs/`:
  - Quick Start Guide
  - Configuration Guide
  - Provider Setup
  - API Reference
  - Architecture Overview
  - Telemetry Guide
- Interactive example runner (`run_examples.sh`)
- Demo scripts: `demo_telemetry.py`, `test_new_features.py`

#### Testing & Quality
- Test suite with pytest
- 28 tests covering:
  - Installation
  - Configuration loading
  - BM25 scoring
  - RRF fusion
  - Importance decay
  - Score combinations
  - Cache functionality
  - Pydantic validation
- Type hints throughout codebase
- MyPy configuration
- Black and Ruff for code formatting
- Pre-commit hooks configuration

#### Development Tools
- Package structure following best practices (`src/` layout)
- Optional dependency groups:
  - `[all]` - All providers + API + Web
  - `[api]` - FastAPI server
  - `[web]` - Flask interface
  - `[ollama]` - Ollama provider
  - `[openai]` - OpenAI provider
  - `[anthropic]` - Anthropic provider
  - `[groq]` - Groq provider
  - `[dev]` - Development tools
  - `[test]` - Testing tools
  - `[docs]` - Documentation tools

### Technical Details

#### Dependencies (Core)
- pydantic >= 2.6, < 3.0
- sentence-transformers >= 2.2, < 3.0
- qdrant-client >= 1.7, < 2.0
- rank-bm25 >= 0.2, < 1.0
- httpx >= 0.25, < 1.0
- apscheduler >= 3.10, < 4.0
- cachetools >= 5.3, < 6.0
- typer[all] >= 0.9, < 1.0

#### Python Support
- Python 3.9+
- Python 3.10
- Python 3.11
- Python 3.12

#### Architecture
- Modular design with clear separation of concerns:
  - `client.py` - Public API
  - `config.py` - Configuration management
  - `telemetry.py` - Observability
  - `embed/` - Embedding generation
  - `vector/` - Qdrant integration
  - `retrieval/` - Hybrid retrieval
  - `pipeline/` - Memory processing
  - `models/` - Data models
  - `api/` - REST API
  - `cli/` - Command-line interface

### Known Issues

- 1 test failing: `test_memory_type_routing` (minor, non-blocking)
- Pydantic deprecation warnings for `env` parameter in Field (works but deprecated)

### Breaking Changes

None (initial release)

### Migration Guide

Not applicable (initial release)

---

## [Unreleased]

### Planned Features
- [ ] LangChain integration
- [ ] LlamaIndex integration
- [ ] Retrieval evaluators and A/B testing
- [ ] Multi-tenant RBAC
- [ ] Native TypeScript SDK
- [ ] Grafana/Prometheus exporters
- [ ] Memory versioning
- [ ] Time-travel queries

### Planned Fixes
- [ ] Fix `test_memory_type_routing` test
- [ ] Update to Pydantic V2 style (remove deprecated `env` parameter)
- [ ] Improve test coverage to 90%+

---

[0.1.0]: https://github.com/rexdivakar/HippocampAI/releases/tag/v0.1.0
