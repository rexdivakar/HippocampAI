# Changelog

All notable changes to HippocampAI will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Migrated all runtime code under `src/hippocampai/` (single import namespace) and moved CLI/Web front ends into the package.
- New `CONTRIBUTING.md`, updated README/docs, and helper `scripts/run-tests.sh` for local CI parity.
- Optional extras in `pyproject.toml` (`core`, `dev`, `test`, `docs`) plus `liccheck.ini` and a scheduled dependency-health GitHub workflow.
- Lazy import of `MemoryClient` so base installs work without vector/db extras.

### Changed

- Standardised import ordering (ruff/isort), added docstrings to package `__init__` files, and normalised timestamps to timezone-aware UTC.
- Refreshed test scripts (`test_install.py`, `test_functional.py`) to add `src/` to `PYTHONPATH` and skip optional checks when dependencies are absent.
- Updated packaging metadata to include HTML assets via `MANIFEST.in` and `package-data`.

### Fixed

- Removed duplicate modules in the project root that shadowed the packaged implementations.
- Addressed deprecation warnings in time utilities and recency scoring.
- Ensured install/functional tests pass in minimal environments (optional deps reported as warnings, not hard failures).

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
