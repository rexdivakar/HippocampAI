# HippocampAI Quick Start - Simple API

Get started with HippocampAI in 30 seconds! This guide shows the simplest possible API for memory management.

## Installation

```bash
pip install hippocampai
```

## 🚀 30-Second Quickstart

### Option 1: Simple Memory API

```python
from hippocampai import SimpleMemory as Memory

# Initialize (auto-configures everything)
m = Memory()

# Store memories
m.add("I prefer oat milk in my coffee", user_id="alice")
m.add("I work as a software engineer", user_id="alice")
m.add("I want to learn machine learning", user_id="alice")

# Search memories
results = m.search("work", user_id="alice", limit=3)

# Print results
for result in results:
    print(f"Score {result.score:.2f}: {result.memory.text}")
```

### Option 2: Session-Based API

```python
from hippocampai import SimpleSession as Session

# Initialize session
session = Session(session_id="conversation_123", user_id="alice")

# Add messages
session.add_message("user", "Hello! I'm looking for coffee recommendations")
session.add_message("assistant", "I'd be happy to help! What's your preference?")
session.add_message("user", "I prefer oat milk")

# Search within session
results = session.search("coffee preferences")

# Get session summary
summary = session.get_summary()
print(summary)
```

### Option 3: Native API (Cognitive Metaphors)

```python
from hippocampai import MemoryClient

# Initialize
client = MemoryClient()

# Use cognitive metaphors (more intuitive!)
memory = client.remember("I prefer oat milk", user_id="alice", type="preference")
results = client.recall("coffee preferences", user_id="alice")

# Advanced features available
patterns = client.detect_patterns(user_id="alice")
habits = client.detect_habits(user_id="alice")
```

---

## 📚 Complete API Comparison

| Operation | Simple API | Native API | Session API |
|-----------|------------|------------|-------------|
| **Store** | `m.add(text, user_id)` | `client.remember(text, user_id)` | `session.add_message(role, content)` |
| **Retrieve** | `m.search(query, user_id)` | `client.recall(query, user_id)` | `session.search(query)` |
| **Get by ID** | `m.get(memory_id)` | `client.get_memory(memory_id)` | - |
| **Update** | `m.update(memory_id, text=...)` | `client.update_memory(memory_id, text=...)` | - |
| **Delete** | `m.delete(memory_id)` | `client.delete_memory(memory_id)` | `session.clear()` |
| **Get All** | `m.get_all(user_id)` | `client.get_memories(user_id)` | `session.get_messages()` |

---

## 🎯 Choose Your API Style

### Use Simple API when:
- ✅ You want the simplest possible interface
- ✅ You don't need advanced features
- ✅ You prefer generic method names (add/search)
- ✅ You're building a basic memory system

### Use Session API when:
- ✅ You're building a chatbot/conversation app
- ✅ You want automatic session management
- ✅ You need conversation summaries
- ✅ You work with message-based interactions

### Use Native API when:
- ✅ You want cognitive metaphors (remember/recall)
- ✅ You need advanced features (patterns, habits, analytics)
- ✅ You want fine-grained control
- ✅ You're building complex memory systems

---

## 🔄 Remote Mode (SaaS API)

All three APIs work in remote mode too!

```python
# Simple API - Remote mode
from hippocampai import SimpleMemory as Memory
m = Memory(api_url="http://localhost:8000", api_key="your-key")
m.add("text", user_id="alice")

# Session API - Remote mode
from hippocampai import SimpleSession as Session
session = Session(session_id="conv_123")
session.add_message("user", "Hello!")

# Native API - Remote mode
from hippocampai import UnifiedMemoryClient
client = UnifiedMemoryClient(mode="remote", api_url="http://localhost:8000")
client.remember("text", user_id="alice")
```

---

## 💡 Examples

### Example 1: Simple Memory Store

```python
from hippocampai import SimpleMemory as Memory

m = Memory()

# Store preferences
m.add("I prefer dark mode", user_id="alice")
m.add("I like large fonts", user_id="alice")
m.add("I use Python daily", user_id="alice")

# Search by topic
results = m.search("UI preferences", user_id="alice")
for r in results:
    print(f"{r.score:.2f}: {r.memory.text}")

# Get all memories
all_memories = m.get_all(user_id="alice")
print(f"Total: {len(all_memories)} memories")

# Update a memory
memory_id = all_memories[0].id
m.update(memory_id, text="I strongly prefer dark mode")

# Delete a memory
m.delete(memory_id)
```

### Example 2: Conversation Management

```python
from hippocampai import SimpleSession as Session

# Create session
session = Session(session_id="support_chat_123", user_id="customer_456")

# Simulate conversation
session.add_message("user", "I need help with my order")
session.add_message("assistant", "I'd be happy to help! What's your order number?")
session.add_message("user", "It's ORDER-789")
session.add_message("assistant", "Let me look that up...")

# Search conversation history
results = session.search("order number")
print(results[0].memory.text)  # "It's ORDER-789"

# Get conversation summary
summary = session.get_summary()
print(summary)

# Clear conversation
count = session.clear()
print(f"Cleared {count} messages")
```

### Example 3: Cognitive Memory System

```python
from hippocampai import MemoryClient

client = MemoryClient()

# Store with rich metadata
client.remember(
    text="I prefer morning meetings",
    user_id="alice",
    type="preference",
    importance=8.0,
    tags=["work", "scheduling"]
)

client.remember(
    text="I had a meeting with John yesterday",
    user_id="alice",
    type="event",
    importance=6.0,
    tags=["work", "meetings"]
)

# Recall with filters
results = client.recall(
    query="work meetings",
    user_id="alice",
    filters={"tags": ["work"], "type": "preference"}
)

# Detect patterns
patterns = client.detect_patterns(user_id="alice")
for pattern in patterns:
    print(f"{pattern.pattern_type}: {pattern.description}")

# Detect habits
habits = client.detect_habits(user_id="alice")
for habit in habits:
    print(f"{habit.action}: {habit.frequency} times, {habit.confidence*100}% confident")
```

---

## 🔧 Configuration

### Auto-Configuration (Recommended)

By default, all APIs auto-configure from environment variables:

```bash
# Optional environment variables
export QDRANT_URL="http://localhost:6333"
export REDIS_URL="redis://localhost:6379"
export LLM_PROVIDER="groq"  # or "openai" or "ollama"
export GROQ_API_KEY="your-key"  # if using Groq
```

Then just:
```python
from hippocampai import SimpleMemory as Memory
m = Memory()  # Auto-configures everything!
```

### Manual Configuration

```python
from hippocampai import SimpleMemory as Memory

# Local mode with custom config
m = Memory(config={
    "qdrant_url": "http://localhost:6333",
    "redis_url": "redis://localhost:6379",
    "llm_provider": "groq",
    "llm_model": "llama-3.1-8b-instant"
})

# Remote mode
m = Memory(
    api_url="http://localhost:8000",
    api_key="your-api-key"
)
```

---

## 🐳 Quick Docker Setup

```bash
# Start required services
docker run -d -p 6333:6333 qdrant/qdrant
docker run -d -p 6379:6379 redis

# Optional: Start Ollama for local LLM
docker run -d -p 11434:11434 ollama/ollama
```

---

## ⚡ Performance Tips

### 1. Batch Operations

```python
# Slow: Multiple individual adds
for text in texts:
    m.add(text, user_id="alice")

# Fast: Single batch operation (if available)
# (Use native API for batch operations)
from hippocampai import MemoryClient
client = MemoryClient()
memories = client.add_memories(memories_data, user_id="alice")
```

### 2. Use Filters to Narrow Search

```python
# Generic search (slower)
results = m.search("preferences", user_id="alice")

# Filtered search (faster - use native API)
from hippocampai import MemoryClient
client = MemoryClient()
results = client.recall(
    "preferences",
    user_id="alice",
    filters={"type": "preference", "tags": ["ui"]}
)
```

---

## 🆘 Troubleshooting

### "Failed to connect to Qdrant"

```bash
# Make sure Qdrant is running
docker run -p 6333:6333 qdrant/qdrant

# Test connection
curl http://localhost:6333/health
```

### "No module named 'hippocampai'"

```bash
pip install hippocampai
```

### "Remote API not responding"

```bash
# Start the API server
cd /path/to/hippocampai
uvicorn hippocampai.api.async_app:app --port 8000

# Test
curl http://localhost:8000/health
```

---

## 🎓 Next Steps

1. **Try the examples above** - Copy-paste and run!
2. **Read the full docs** - `docs/API_REFERENCE.md`
3. **Explore advanced features** - `docs/FEATURES.md`
4. **Join the community** - GitHub Discussions

---

## 💬 Support

- 📖 **Documentation**: `/docs` folder
- 💻 **Examples**: `/examples` folder
- 🐛 **Issues**: GitHub Issues
- 💡 **Discussions**: GitHub Discussions

**Happy memory building!** 🧠✨
