# Changelog

All notable changes to HippocampAI will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

- No changes yet.

## [1.0.0] - 2025-10-21

### Highlights
- First major stable release of HippocampAI, signaling API and feature stability for production use.

### Documentation
- Expanded installation instructions and consolidated example guides in the README and docs for clearer onboarding.

### Changed
- Improved CI workflow caching to better reuse Python environments and pip downloads.

### Fixed
- Standardized timestamp parsing in `HybridRetriever` by routing ISO strings through `parse_iso_datetime`.
- Updated retrieval tests to use `now_utc()` for timezone-aware recency scoring assertions.

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

## [0.2.0] - TBD

### Added (New in v0.2.0)

#### Memory Size Tracking
- **NEW**: Automatic character and token counting for all memories
  - Added `text_length: int` field to Memory model
  - Added `token_count: int` field to Memory model (4 chars ≈ 1 token approximation)
  - `calculate_size_metrics()` method for automatic calculation
  - Size metrics calculated on memory creation and updates
- **NEW**: Memory statistics API
  - `get_memory_statistics(user_id)` - Get comprehensive size analytics
  - Returns total memories, characters, tokens, averages, min/max
  - Statistics grouped by memory type
- **NEW**: Telemetry integration for size tracking
  - Added `memory_size_chars` and `memory_size_tokens` metrics
  - `track_memory_size()` method in telemetry
  - Size metrics included in metrics summary

#### Async Support
- **NEW**: `AsyncMemoryClient` class for asynchronous operations
  - Extends `MemoryClient` with async variants
  - All core operations: `remember_async()`, `recall_async()`, `update_memory_async()`, `delete_memory_async()`, `get_memories_async()`
  - Batch operations: `add_memories_async()`, `delete_memories_async()`
  - Utility methods: `get_memory_statistics_async()`, `inject_context_async()`, `extract_from_conversation_async()`
  - Uses `asyncio.run_in_executor()` for non-blocking execution
  - Full support for concurrent operations with `asyncio.gather()`
- **NEW**: Async test suite
  - Comprehensive tests in `tests/test_async.py`
  - Tests for all async operations
  - Concurrent operation tests

#### Advanced Features (Previously Implemented, Now Documented)
- Batch operations (`add_memories`, `delete_memories`)
- Graph indexing and relationships
- Version control and rollback
- Context injection for LLM prompts
- Memory access tracking
- Advanced filtering and sorting
- Snapshots and audit trail
- KV store for fast lookups

### Changed
- Updated telemetry to track memory size metrics
- Enhanced `MemoryClient` to calculate size metrics automatically
- Improved test coverage with async operation tests (5 new test classes, 9 new tests)

### Documentation
- **Updated**: README.md with async usage examples and memory size tracking
- **Updated**: FEATURES.md with complete feature documentation (all 15 features)
- **Removed**: Obsolete chat integration docs (CHAT_INTEGRATION.md, CHAT_README.md)
- **Removed**: Obsolete tool documentation (TOOLS.md, TOOLS_SUMMARY.md, HOW_TO_RUN.md)
- **Updated**: Roadmap with completed features
- **Updated**: Examples list to include advanced features demo

### Planned Features
- [ ] Memory consolidation scheduler (background jobs with APScheduler/Celery)
- [ ] Persistent graph storage (JSON export/import for NetworkX graph)
- [ ] LangChain integration
- [ ] LlamaIndex integration
- [ ] Retrieval evaluators and A/B testing
- [ ] Multi-tenant RBAC
- [ ] Native TypeScript SDK
- [ ] Grafana/Prometheus exporters
- [ ] WebSocket support for real-time updates

### Planned Fixes
- [ ] Update to Pydantic V2 style (remove deprecated `env` parameter)
- [ ] Fix datetime.utcnow() deprecation warnings (use datetime.now(UTC))
- [ ] Improve test coverage to 90%+

---

[Unreleased]: https://github.com/rexdivakar/HippocampAI/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/rexdivakar/HippocampAI/releases/tag/v1.0.0
[0.1.0]: https://github.com/rexdivakar/HippocampAI/releases/tag/v0.1.0
