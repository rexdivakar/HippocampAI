# HippocampAI Documentation

Welcome to the comprehensive documentation for HippocampAI - the enterprise-grade memory engine for intelligent AI systems.

**Last Updated**: 2026-02-11

---

## 🚀 Quick Start (New Users Start Here!)

**New to HippocampAI?** Start with these essential guides:

1. **[Quick Start - Simple API](QUICK_START_SIMPLE.md)** ⭐ **(30 seconds!)**
   - Fastest way to get started
   - Multiple API styles available
   - Zero configuration required

2. **[Unified Guide](UNIFIED_GUIDE.md)** - Complete overview
   - All API styles explained
   - Testing guide
   - Deployment options

---

## Quick Navigation

### Getting Started

- **[Quick Start - Simple API](QUICK_START_SIMPLE.md)** ⭐ - 30-second quickstart with simple API
- **[Unified Guide](UNIFIED_GUIDE.md)** - Complete guide: testing, API, deployment
- **[Getting Started Guide](GETTING_STARTED.md)** - Complete setup, configuration, and first steps
- **[Chat Demo Guide](CHAT_DEMO_GUIDE.md)** - Interactive chatbot demo with persistent memory

### Core Documentation

- **[Package Structure](PACKAGE_STRUCTURE.md)** ⭐ - Core vs Platform separation guide
- **[API Reference](API_REFERENCE.md)** - Complete API documentation for all 102+ methods
- **[Features Overview](FEATURES.md)** - Complete feature documentation
- **[Architecture Overview](ARCHITECTURE.md)** - System design and component architecture
- **[User Guide](USER_GUIDE.md)** - Production deployment and operations

### Specialized Guides

- **[New Features Guide](NEW_FEATURES.md)** - Plugins, namespaces, export/import, offline mode, tiered storage, integrations
- **[SaaS Platform Guide](SAAS_GUIDE.md)** - Multi-tenant SaaS deployment and authentication
- **[Memory Management](MEMORY_MANAGEMENT.md)** - Health monitoring, duplicate detection, quality tracking
- **[Celery Guide](CELERY_GUIDE.md)** - Background task processing with Celery
- **[Configuration](CONFIGURATION.md)** - All configuration options
- **[Providers](PROVIDERS.md)** - LLM provider setup (Ollama, OpenAI, Anthropic, Groq)
- **[MCP Server](MCP_SERVER.md)** ⭐ NEW - Zero-config memory for Claude Code, Cursor, and MCP agents
- **[Quality Evaluation](EVALUATION.md)** ⭐ NEW - LOCOMO/LongMemEval harness: recall@k, MRR, nDCG, LLM-judge QA accuracy

### v0.5.0 Features

- **[Knowledge Graph](FEATURES.md#real-time-incremental-knowledge-graph)** NEW - Real-time entity/relationship extraction
- **[Graph-Aware Retrieval](FEATURES.md#graph-aware-retrieval)** NEW - 3-way RRF fusion (vector + BM25 + graph)
- **[Relevance Feedback](FEATURES.md#memory-relevance-feedback-loop)** NEW - User feedback loop with decay scoring
- **[Memory Triggers](FEATURES.md#memory-triggers--event-driven-actions)** NEW - Event-driven webhooks and actions
- **[Procedural Memory](FEATURES.md#procedural-memory--prompt-self-optimization)** NEW - Self-optimizing prompts
- **[Embedding Migration](FEATURES.md#embedding-model-migration)** NEW - Safe model migration with Celery

### Advanced Features

- **[Multi-Agent Features](MULTIAGENT_FEATURES.md)** - Agent coordination and collaboration
- **[Session Management](SESSION_MANAGEMENT.md)** - Conversation organization and boundaries
- **[Versioning & Retention](VERSIONING_AND_RETENTION_GUIDE.md)** - Version control and data lifecycle

### Operations & Deployment

- **[Deployment Guide](DEPLOYMENT_READINESS_REPORT.md)** - Production deployment checklist
- **[Monitoring](MONITORING_INTEGRATION_GUIDE.md)** - Observability and metrics
- **[Security](SECURITY.md)** - Best practices and security hardening
- **[Backup & Recovery](BACKUP_RECOVERY.md)** - Data protection strategies
- **[Troubleshooting](TROUBLESHOOTING.md)** - Common issues and solutions

### Development

- **[Testing Guide](TESTING_GUIDE.md)** - Comprehensive testing strategies
- **[Contributing](CONTRIBUTING.md)** - Development setup and guidelines

### About & Vision

- **[Why We Built HippocampAI](WHY_WE_BUILT_HIPPOCAMPAI.md)** ⭐ - Our story, vision, and what makes us unique
- **[Competitive Advantages](COMPETITIVE_ADVANTAGES.md)** ⭐ - How HippocampAI compares to mem0, Zep, Letta, Cognee, and LangMem

### Project Status

- **[Implementation Complete](IMPLEMENTATION_COMPLETE.md)** - Recent implementation summary

### Additional Resources

- **[Telemetry](TELEMETRY.md)** - Metrics and observability
- **[Resilience](RESILIENCE.md)** - Error handling and retry logic
- **[Project Overview](PROJECT_OVERVIEW.md)** - High-level project overview

---

## Documentation by Use Case

### I want to

#### Get Started Quickly

1. Read **[Quick Start - Simple API](QUICK_START_SIMPLE.md)** ⭐ (30 seconds!)
2. Read [Unified Guide](UNIFIED_GUIDE.md) (complete overview)
3. Try [Getting Started Guide](GETTING_STARTED.md)
4. Try [Chat Demo](CHAT_DEMO_GUIDE.md)

#### Deploy to Production

1. Review [User Guide](USER_GUIDE.md)
2. Follow [Deployment Guide](DEPLOYMENT_READINESS_REPORT.md)
3. Set up [Monitoring](MONITORING_INTEGRATION_GUIDE.md)
4. Configure [Security](SECURITY.md)
5. Plan [Backup & Recovery](BACKUP_RECOVERY.md)

#### Deploy as SaaS Platform

1. Read [SaaS Platform Guide](SAAS_GUIDE.md)
2. Configure [Celery](CELERY_GUIDE.md) for background tasks
3. Set up authentication and rate limiting
4. Configure monitoring and observability

#### Optimize Memory Quality

1. Review [Memory Management](MEMORY_MANAGEMENT.md)
2. Set up health monitoring
3. Configure duplicate detection
4. Implement cleanup workflows

#### Integrate with My App

1. Check [API Reference](API_REFERENCE.md)
2. Review [Features](FEATURES.md)
3. Configure [Providers](PROVIDERS.md)
4. Read [Configuration](CONFIGURATION.md)

#### Build Multi-Agent System

1. Read [Multi-Agent Features](MULTIAGENT_FEATURES.md)
2. Review [Session Management](SESSION_MANAGEMENT.md)
3. Check [Architecture](ARCHITECTURE.md)

#### Troubleshoot Issues

1. Check [Troubleshooting](TROUBLESHOOTING.md)
2. Review [Testing Guide](TESTING_GUIDE.md)
3. Examine [Resilience](RESILIENCE.md)

---

## Documentation Structure

```
docs/
├── README.md (this file)
│
├── Getting Started
│   ├── GETTING_STARTED.md
│   ├── QUICKSTART.md
│   ├── CHAT_DEMO_GUIDE.md
│   └── CONFIGURATION.md
│
├── Core Documentation
│   ├── API_REFERENCE.md
│   ├── FEATURES.md
│   ├── ARCHITECTURE.md
│   ├── USER_GUIDE.md
│   └── PROVIDERS.md
│
├── Specialized Guides
│   ├── SAAS_GUIDE.md (new - consolidated SaaS docs)
│   ├── MEMORY_MANAGEMENT.md (new - consolidated memory health)
│   ├── CELERY_GUIDE.md (new - consolidated Celery docs)
│   ├── MULTIAGENT_FEATURES.md
│   ├── SESSION_MANAGEMENT.md
│   └── VERSIONING_AND_RETENTION_GUIDE.md
│
├── Operations
│   ├── DEPLOYMENT_READINESS_REPORT.md
│   ├── MONITORING_INTEGRATION_GUIDE.md
│   ├── SECURITY.md
│   ├── BACKUP_RECOVERY.md
│   ├── TELEMETRY.md
│   └── RESILIENCE.md
│
├── Development
│   ├── TESTING_GUIDE.md
│   ├── CONTRIBUTING.md
│   └── TROUBLESHOOTING.md
│
├── Advanced Topics
│   ├── ADVANCED_COMPRESSION_GUIDE.md
│   ├── AUTO_SUMMARIZATION_GUIDE.md
│   ├── MEMORY_CONFLICT_RESOLUTION_GUIDE.md
│   ├── MEMORY_QUALITY_AND_OBSERVABILITY.md
│   └── LIBRARY_COMPLETE_REFERENCE.md
│
```

---

## Recent Changes

### 2026-02-11: v0.5.0 — Intelligent Memory Features

**6 New Features:**

1. **Real-Time Knowledge Graph** — Auto-extraction of entities, facts, and relationships on every `remember()` call
2. **Graph-Aware Retrieval** — 3-way RRF fusion combining vector, BM25, and graph-based scoring
3. **Memory Relevance Feedback** — User feedback loop with exponentially-weighted scoring
4. **Memory Triggers** — Event-driven actions (webhooks, websocket, log) on memory lifecycle events
5. **Procedural Memory** — Self-optimizing prompts via learned behavioral rules
6. **Embedding Migration** — Safe model migration with Celery background processing

**Also added:** 16 new config fields, 15 new REST API endpoints, 2 new Celery tasks

See the [Changelog](../CHANGELOG.md) for full details.

### 2026-02-11: Documentation Reorganization

**New Essential Guides:**

1. **[QUICK_START_SIMPLE.md](QUICK_START_SIMPLE.md)** - 30-second quickstart with simple API
2. **[UNIFIED_GUIDE.md](UNIFIED_GUIDE.md)** - Complete overview: testing, API, deployment

**Organizational Changes:**

- Moved all root `.md` files (except README.md and CHANGELOG.md) to `docs/`
- Updated all internal documentation links
- Created simplified API compatible with mem0 and zep
- Added unified test runner (`tests/run_all_tests.py`)
- 99%+ test pass rate (81/82 tests)

### Previous Version (v0.2.5 - 2026-02-11)

**Consolidated Guides Created:**

- [SAAS_GUIDE.md](SAAS_GUIDE.md) - Unified SaaS documentation
- [MEMORY_MANAGEMENT.md](MEMORY_MANAGEMENT.md) - Unified memory health docs
- [CELERY_GUIDE.md](CELERY_GUIDE.md) - Unified Celery documentation

All source files from these consolidations have been removed in v0.3.0

---

## Quick Reference Table

| Task | Documentation |
|------|---------------|
| **Get started in 30 seconds** ⭐ | [Quick Start - Simple API](QUICK_START_SIMPLE.md) |
| **Package structure** ⭐ | [Package Structure](PACKAGE_STRUCTURE.md) |
| **New features** ⭐ | [New Features Guide](NEW_FEATURES.md) |
| **Why HippocampAI?** ⭐ | [Why We Built HippocampAI](WHY_WE_BUILT_HIPPOCAMPAI.md) |
| **Compare to alternatives** ⭐ | [Competitive Advantages](COMPETITIVE_ADVANTAGES.md) |
| **Complete overview** | [Unified Guide](UNIFIED_GUIDE.md) |
| **Try interactive demo** | [Chat Demo Guide](CHAT_DEMO_GUIDE.md) |
| **First time setup** | [Getting Started](GETTING_STARTED.md) |
| **Configure LLM provider** | [Providers](PROVIDERS.md) |
| **Deploy to production** | [User Guide](USER_GUIDE.md) + [Deployment](DEPLOYMENT_READINESS_REPORT.md) |
| **Deploy as SaaS** | [SaaS Guide](SAAS_GUIDE.md) |
| **Secure your deployment** | [Security](SECURITY.md) |
| **Monitor performance** | [Monitoring](MONITORING_INTEGRATION_GUIDE.md) + [Telemetry](TELEMETRY.md) |
| **Manage memory quality** | [Memory Management](MEMORY_MANAGEMENT.md) |
| **Set up background tasks** | [Celery Guide](CELERY_GUIDE.md) |
| **Troubleshoot issues** | [Troubleshooting](TROUBLESHOOTING.md) |
| **Run tests** | [Testing Guide](TESTING_GUIDE.md) |
| **Contribute code** | [Contributing](CONTRIBUTING.md) |
| **Understand architecture** | [Architecture](ARCHITECTURE.md) |
| **Learn all features** | [Features](FEATURES.md) |
| **API reference** | [API Reference](API_REFERENCE.md) |

---

## Documentation Quality Standards

All HippocampAI documentation follows these standards:

✅ **Complete Examples** - Every feature includes working code examples
✅ **Step-by-Step** - Clear, numbered steps for all procedures
✅ **Production-Ready** - Focus on real-world deployment scenarios
✅ **Cross-Referenced** - Related documentation is linked
✅ **Up-to-Date** - Regular updates with version information
✅ **Beginner-Friendly** - Assumes minimal prior knowledge
✅ **Advanced Coverage** - Deep dives for complex topics

---

## Contributing to Documentation

Found an error or want to improve the documentation? See our [Contributing Guide](CONTRIBUTING.md) for how to help.

Specific documentation improvements:

- Fix typos or unclear explanations
- Add more examples
- Improve diagrams
- Expand troubleshooting sections
- Add use cases
- Update for new features

---

## Need Help?

- **GitHub Issues**: [Report bugs or request features](https://github.com/rexdivakar/HippocampAI/issues)
- **Discord**: [Join our community](https://discord.gg/pPSNW9J7gB)
- **Main README**: [Project overview](../README.md)
- **Changelog**: [Version history](../CHANGELOG.md)

---

## External Resources

- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [Redis Documentation](https://redis.io/documentation)
- [Celery Documentation](https://docs.celeryproject.org/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Groq API Documentation](https://console.groq.com/docs)
- [OpenAI API Documentation](https://platform.openai.com/docs)

---

**Built with ❤️ by the HippocampAI community**
