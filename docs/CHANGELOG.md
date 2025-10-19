# Changelog

All notable changes to HippocampAI will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `scripts/run-tests.sh` convenience wrapper for running the pytest suite with the correct environment.
- Contributor onboarding guide (`CONTRIBUTING.md`) and refreshed testing documentation.
- Automated PyPI publishing via GitHub Actions
- TestPyPI workflow for release candidates
- MANIFEST.in for package distribution control

### Changed
- Standardized all runtime timestamps to timezone-aware UTC and updated settings to use modern Pydantic v2 patterns.
- README/TESTING instructions now document local unit test execution.
- Renamed `setup.py` to `setup_initial.py` to avoid build conflicts
- Updated all documentation to reference new setup script name

### Fixed
- Removed legacy `datetime.utcnow()` usage that surfaced deprecation warnings during tests.
- Fixed circular dependency during `pip install -e .` by removing setup.py from build process

## [0.1.0] - 2025-01-06

### Added
- Hybrid retrieval (BM25 + embeddings + RRF)
- Two-stage ranking with cross-encoder reranking
- Typed routing for preferences vs facts
- Qdrant HNSW optimization
- Embedder with batching and quantization support
- Ollama and OpenAI LLM adapters
- Memory extraction, deduplication, consolidation
- FastAPI REST API
- Typer CLI interface
- Background scheduler for decay/consolidation/snapshots
- Score caching with 24h TTL
- Comprehensive configuration system
