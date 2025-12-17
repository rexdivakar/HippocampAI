# HippocampAI New Features

This document covers the new features added to HippocampAI for enhanced memory management.

## Table of Contents

1. [Plugin System](#plugin-system)
2. [Memory Namespaces](#memory-namespaces)
3. [Export/Import (Portability)](#exportimport-portability)
4. [Offline Mode](#offline-mode)
5. [Tiered Storage](#tiered-storage)
6. [Framework Integrations](#framework-integrations)

---

## Plugin System

The plugin system allows custom memory processors, scorers, retrievers, and filters to be registered and used throughout the memory pipeline.

### Usage

```python
from hippocampai.plugins import PluginRegistry, MemoryProcessor, MemoryScorer

# Create a custom processor
class SentimentProcessor(MemoryProcessor):
    name = "sentiment"
    
    def process(self, memory, context=None):
        # Add sentiment analysis to memory
        memory.metadata["sentiment"] = analyze_sentiment(memory.text)
        return memory

# Create a custom scorer
class RecencyScorer(MemoryScorer):
    name = "recency"
    weight = 0.2
    
    def score(self, memory, query, context=None):
        age_days = (datetime.now() - memory.created_at).days
        return max(0, 1 - age_days / 365)

# Register plugins
registry = PluginRegistry()
registry.register_processor(SentimentProcessor())
registry.register_scorer(RecencyScorer())

# Plugins are automatically applied during memory operations
```

### Plugin Types

- **MemoryProcessor**: Modify memories before storage or after retrieval
- **MemoryScorer**: Contribute to relevance scoring during retrieval
- **MemoryRetriever**: Implement custom retrieval strategies
- **MemoryFilter**: Filter memories based on custom criteria

---

## Memory Namespaces

Namespaces provide hierarchical organization of memories into projects, contexts, or any logical grouping.

### Usage

```python
from hippocampai.namespaces import NamespaceManager, NamespacePermission

manager = NamespaceManager()

# Create namespaces
work_ns = manager.create("work", user_id="alice", description="Work memories")
project_ns = manager.create("work/project-x", user_id="alice")

# Grant permissions
manager.grant_permission("work", "bob", NamespacePermission.READ, "alice")

# Check permissions
ns = manager.get("work")
print(ns.can_read("bob"))  # True
print(ns.can_write("bob"))  # False

# List namespaces
namespaces = manager.list("alice")
```

### Features

- Hierarchical organization (like folders)
- Permission inheritance
- Quota management
- Public/private namespaces

---

## Export/Import (Portability)

Export and import memories for backup, migration, or data portability compliance.

### Export

```python
from hippocampai.portability import MemoryExporter, ExportOptions, ExportFormat

exporter = MemoryExporter(client)

# Export to JSON
stats = exporter.export_json("backup.json", user_id="alice")

# Export to Parquet (columnar format)
stats = exporter.export_parquet("backup.parquet", user_id="alice")

# Export with options
options = ExportOptions(
    format=ExportFormat.JSONL,
    compress=True,
    include_vectors=False,
    filter_types=["fact", "preference"],
    anonymize=True,  # Remove PII
)
stats = exporter.export_jsonl("backup.jsonl.gz", user_id="alice", options=options)
```

### Import

```python
from hippocampai.portability import MemoryImporter, ImportOptions

importer = MemoryImporter(client)

# Import from JSON
stats = importer.import_json("backup.json", user_id="alice")

# Import with options
options = ImportOptions(
    merge_strategy="skip",  # skip, replace, merge
    skip_duplicates=True,
    dry_run=True,  # Preview without importing
)
stats = importer.import_json("backup.json", options=options)
```

### Supported Formats

- JSON (full export with metadata)
- JSONL (streaming, line-by-line)
- Parquet (columnar, efficient for large datasets)
- CSV (simple tabular format)

---

## Offline Mode

Queue operations when Qdrant/Redis are unavailable and sync when connectivity is restored.

### Usage

```python
from hippocampai import MemoryClient
from hippocampai.offline import OfflineClient

client = MemoryClient()
offline = OfflineClient(client, auto_sync=True)

# Operations work even if backend is down
offline.remember("Important fact", user_id="alice")  # Queued if offline

# Check status
print(offline.is_online())
print(offline.get_queue_stats())

# Manual sync
stats = offline.sync()
print(f"Synced {stats.successful} operations")
```

### Features

- SQLite-backed persistent queue
- Automatic retry with configurable limits
- Background auto-sync
- Priority-based operation ordering

---

## Tiered Storage

Implement hot/warm/cold storage tiers for efficient memory management.

### Usage

```python
from hippocampai.tiered import TieredStorageManager, TierConfig

manager = TieredStorageManager(client)

# Configure tiers
manager.configure_tiers(
    hot_days=30,      # Keep in hot for 30 days
    warm_days=90,     # Move to warm after 30 days
    cold_days=365,    # Move to cold after 90 days
)

# Run tier migration
stats = manager.migrate_tiers(user_id="alice")
print(f"Moved {stats.hot_to_warm} memories to warm tier")
print(f"Saved {stats.bytes_saved} bytes through compression")

# Get tier statistics
tier_stats = manager.get_tier_stats(user_id="alice")
for tier, stats in tier_stats.items():
    print(f"{tier.value}: {stats.memory_count} memories")
```

### Tiers

- **Hot**: Frequently accessed, full fidelity, fast retrieval
- **Warm**: Less frequent, light compression
- **Cold**: Archived, heavy compression (summarized)
- **Frozen**: Long-term archive, metadata only

---

## Framework Integrations

### LangChain

```python
from langchain.chains import ConversationChain
from hippocampai import MemoryClient
from hippocampai.integrations.langchain import HippocampMemory, HippocampRetriever

client = MemoryClient()

# Use as conversation memory
memory = HippocampMemory(client, user_id="alice")
chain = ConversationChain(llm=llm, memory=memory)

# Use as retriever
retriever = HippocampRetriever(client, user_id="alice", k=5)
docs = retriever.get_relevant_documents("coffee preferences")
```

### LlamaIndex

```python
from hippocampai import MemoryClient
from hippocampai.integrations.llamaindex import HippocampRetriever, HippocampMemoryStore

client = MemoryClient()

# Use as retriever
retriever = HippocampRetriever(client, user_id="alice", k=5)
nodes = retriever.retrieve("What do I like?")

# Use as document store
store = HippocampMemoryStore(client, user_id="alice")
store.add_documents([doc1, doc2])
```

---

## Installation

All new features are included in the main package:

```bash
pip install hippocampai
```

For framework integrations, install the optional dependencies:

```bash
# LangChain integration
pip install hippocampai langchain

# LlamaIndex integration
pip install hippocampai llama-index
```

For Parquet export support:

```bash
pip install hippocampai pyarrow
```
