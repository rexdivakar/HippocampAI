# Changelog

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
