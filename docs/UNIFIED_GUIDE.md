# HippocampAI: Unified Guide - Testing, API, and Usage

Complete guide for using, testing, and deploying HippocampAI - as easy as mem0 and zep!

## 📑 Table of Contents

1. [Quick Start (30 seconds)](#quick-start)
2. [API Options (3 styles)](#api-options)
3. [Testing Guide](#testing-guide)
4. [Examples](#examples)
5. [Deployment](#deployment)
6. [Comparison with Competitors](#comparison)

---

## 🚀 Quick Start

### Simplest Possible Usage (mem0-style)

```python
from hippocampai import SimpleMemory as Memory

# One line to initialize
m = Memory()

# Store and search
m.add("I prefer dark mode", user_id="alice")
results = m.search("preferences", user_id="alice")
```

**Time to first memory: 30 seconds!**

---

## 🎯 API Options

HippocampAI offers **3 different API styles** - choose what works best for you:

### 1. Simple API (mem0-compatible)

**Best for**: Quick prototyping, mem0 migration, simple use cases

```python
from hippocampai import SimpleMemory as Memory

m = Memory()
m.add("text", user_id="alice")           # Store
results = m.search("query", user_id="alice")  # Retrieve
m.update(memory_id, text="new text")    # Update
m.delete(memory_id)                     # Delete
```

📖 [Full Simple API Guide](QUICK_START_SIMPLE.md)

### 2. Session API (zep-compatible)

**Best for**: Chatbots, conversation apps, zep migration

```python
from hippocampai import SimpleSession as Session

session = Session(session_id="conv_123", user_id="alice")
session.add_message("user", "Hello!")
session.add_message("assistant", "Hi there!")
summary = session.get_summary()
```

📖 [Full Session API Guide](QUICK_START_SIMPLE.md#session-api)

### 3. Native API (HippocampAI)

**Best for**: Advanced features, fine control, cognitive metaphors

```python
from hippocampai import MemoryClient

client = MemoryClient()
memory = client.remember("text", user_id="alice", type="preference")
results = client.recall("query", user_id="alice")
patterns = client.detect_patterns(user_id="alice")
```

📖 [Full Native API Guide](API_REFERENCE.md)

---

## 🧪 Testing Guide

### Unified Test Runner

We provide a comprehensive test runner that organizes all tests by category:

```bash
# Run all unit tests
python tests/run_all_tests.py

# Run specific category
python tests/run_all_tests.py --category core
python tests/run_all_tests.py --category scheduler
python tests/run_all_tests.py --category intelligence

# Quick smoke test
python tests/run_all_tests.py --quick

# List all categories
python tests/run_all_tests.py --list

# Check service availability
python tests/run_all_tests.py --check-services
```

### Test Categories

| Category | Tests | Purpose |
|----------|-------|---------|
| **core** | 4 tests | Basic functionality (remember/recall/async) |
| **scheduler** | 4 tests | Auto-consolidation, decay, summarization |
| **intelligence** | 2 tests | Pattern detection, entity recognition |
| **memory_management** | 4 tests | Health monitoring, compression |
| **multiagent** | 2 tests | Multi-agent coordination |
| **monitoring** | 2 tests | Metrics and telemetry |
| **integration** | 2 tests | End-to-end integration tests |

### Quick Test Commands

```bash
# Core functionality only (fastest)
python tests/run_all_tests.py --category core

# Everything except integration
python tests/run_all_tests.py --all

# Integration tests (run separately)
python tests/test_all_features_integration.py
python tests/test_library_saas_integration.py
```

📖 [Full Testing Guide](../tests/README_TESTING.md)

---

## 📚 Examples

### Quick Examples

#### Example 1: Simple Memory Store
```python
from hippocampai import SimpleMemory as Memory

m = Memory()
m.add("I prefer oat milk", user_id="alice")
m.add("I work at TechCorp", user_id="alice")

results = m.search("work", user_id="alice")
print(results[0].memory.text)  # "I work at TechCorp"
```

#### Example 2: Conversation Bot
```python
from hippocampai import SimpleSession as Session

session = Session(session_id="chat_123")
session.add_message("user", "What's the weather?")
session.add_message("assistant", "It's sunny today!")

history = session.get_messages()
print(f"Conversation has {len(history)} messages")
```

#### Example 3: Pattern Detection
```python
from hippocampai import MemoryClient

client = MemoryClient()

# Store work habits
client.remember("Had standup at 9am", user_id="alice", type="event")
client.remember("Had standup at 9am yesterday", user_id="alice", type="event")
client.remember("Daily standup at 9am", user_id="alice", type="event")

# Detect patterns
patterns = client.detect_patterns(user_id="alice")
print(patterns[0].description)  # "Daily standup meetings at 9am"
```

### Full Example Scripts

| Script | Description | API Style |
|--------|-------------|-----------|
| `examples/simple_api_mem0_style.py` | mem0-compatible Simple API | Simple |
| `examples/simple_api_session_style.py` | zep-compatible Session API | Session |
| `examples/01_basic_usage.py` | Basic remember/recall | Native |
| `examples/02_conversation_extraction.py` | Auto-extract from conversations | Native |
| `examples/03_hybrid_retrieval.py` | Advanced search | Native |
| `examples/07_advanced_features_demo.py` | Pattern detection, analytics | Native |
| `examples/12_multiagent_demo.py` | Multi-agent coordination | Native |

📂 [All Examples](../examples)

---

## 🚀 Deployment

### Local Development

```bash
# 1. Install dependencies
pip install hippocampai

# 2. Start services
docker run -p 6333:6333 qdrant/qdrant
docker run -p 6379:6379 redis

# 3. Run your app
python your_app.py
```

### Remote/SaaS Mode

```bash
# 1. Start API server
uvicorn hippocampai.api.async_app:app --port 8000

# 2. Use remote mode in your app
```

```python
from hippocampai import SimpleMemory as Memory

# Connect to remote API
m = Memory(api_url="http://localhost:8000", api_key="your-key")
m.add("text", user_id="alice")
```

### Docker Deployment

```bash
# Full stack deployment
docker-compose up -d
```

📖 [Deployment Guide](USER_GUIDE.md)

---

## 🆚 Comparison with Competitors

### Feature Comparison

| Feature | HippocampAI | mem0 | zep |
|---------|-------------|------|-----|
| **Simple API** | ✅ mem0-compatible | ✅ | ❌ |
| **Session API** | ✅ zep-compatible | ❌ | ✅ |
| **Cognitive Metaphors** | ✅ remember/recall | ❌ | ❌ |
| **Memory Types** | ✅ 6 types | ❌ Untyped | ❌ Message-based |
| **Hybrid Search** | ✅ Vector+BM25+Rerank | ❌ Vector only | ❌ Vector only |
| **Pattern Detection** | ✅ Built-in | ❌ Custom | ❌ Custom |
| **Multi-agent** | ✅ Built-in | ❌ Limited | ❌ Session-based |
| **Open Source** | ✅ No lock-in | ⚠️ Cloud-first | ⚠️ Cloud-first |
| **Local + Remote** | ✅ Unified API | ⚠️ Different APIs | ⚠️ Different packages |

### API Comparison

```python
# HippocampAI - mem0 style
from hippocampai import SimpleMemory as Memory
m = Memory()
m.add("text", user_id="alice")

# mem0
from mem0 import Memory
m = Memory()
m.add("text", user_id="alice")

# SAME API! ✅
```

```python
# HippocampAI - zep style
from hippocampai import SimpleSession as Session
session = Session(session_id="123")
session.add_message("user", "text")

# zep
from zep_cloud.client import Zep
client = Zep()
messages = [Message(role="user", content="text")]
client.memory.add(session_id="123", messages=messages)

# Similar patterns! ✅
```

📖 [Full Comparison](COMPETITIVE_COMPARISON.md)

---

## 📖 Documentation Structure

```
HippocampAI/
├── README.md                       # Project README
├── CHANGELOG.md                    # Version history
├── tests/
│   ├── run_all_tests.py           # Unified test runner
│   └── [23+ test files]
├── docs/
│   ├── QUICK_START_SIMPLE.md      # 30-second quickstart (BEST PLACE TO START!)
│   ├── UNIFIED_GUIDE.md           # This file - complete overview
│   ├── COMPETITIVE_COMPARISON.md  # vs mem0, zep, others
│   ├── API_REFERENCE.md           # Complete API docs (102 methods)
│   ├── USER_GUIDE.md              # Deployment and production
│   ├── FEATURES.md                # All features explained
│   └── [48+ more docs]
└── examples/
    ├── simple_api_mem0_style.py   # mem0-compatible example
    ├── simple_api_session_style.py # zep-compatible example
    └── [25+ more examples]
```

---

## 🎓 Learning Path

### Beginner (0-30 minutes)
1. Read [QUICK_START_SIMPLE.md](QUICK_START_SIMPLE.md)
2. Run `examples/simple_api_mem0_style.py`
3. Try `examples/simple_api_session_style.py`
4. Build your first memory app!

### Intermediate (30 minutes - 2 hours)
1. Explore [examples/01_basic_usage.py](../examples/01_basic_usage.py)
2. Learn about memory types and importance
3. Try hybrid search with `examples/03_hybrid_retrieval.py`
4. Run tests: `python tests/run_all_tests.py --quick`

### Advanced (2+ hours)
1. Study [docs/API_REFERENCE.md](docs/API_REFERENCE.md)
2. Explore pattern detection and analytics
3. Learn multi-agent coordination
4. Deploy to production with SaaS mode

---

## 🔗 Quick Links

| Resource | Link | Description |
|----------|------|-------------|
| **Quick Start** | [QUICK_START_SIMPLE.md](QUICK_START_SIMPLE.md) | 30-second start guide |
| **API Docs** | [API_REFERENCE.md](API_REFERENCE.md) | All 102 methods |
| **Testing** | [Testing Guide](TESTING_GUIDE.md) | How to run tests |
| **Examples** | [Examples](../examples) | 25+ working examples |
| **Comparison** | [COMPETITIVE_COMPARISON.md](COMPETITIVE_COMPARISON.md) | vs mem0, zep |
| **GitHub** | [GitHub Repo](https://github.com/yourusername/HippocampAI) | Source code |

---

## 🆘 Support

- 📖 **Documentation**: This guide + `/docs` folder
- 💻 **Examples**: `/examples` folder (25+ examples)
- 🧪 **Tests**: `python tests/run_all_tests.py`
- 🐛 **Issues**: GitHub Issues
- 💬 **Discussions**: GitHub Discussions

---

## ⭐ Why HippocampAI?

1. **🚀 Easiest to start**: 30 seconds to first memory
2. **🔄 Compatible**: Works with mem0 and zep patterns
3. **🧠 Cognitive**: remember/recall metaphors feel natural
4. **💪 Powerful**: 102 methods, 6 memory types, hybrid search
5. **🏢 Production-ready**: Battle-tested, well-documented
6. **🆓 Open source**: No vendor lock-in

**Choose HippocampAI for the best of all worlds!** 🎉

---

**Made with ❤️ for the developer community**

Start building memory into your apps today! 🚀
