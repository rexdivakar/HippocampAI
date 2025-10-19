# HippocampAI Chat Integration Guide

Complete guide for using the memory-enhanced AI chat assistant.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [CLI Chat](#cli-chat)
- [Web Chat](#web-chat)
- [Python Integration](#python-integration)
- [API Reference](#api-reference)
- [Architecture](#architecture)
- [Customization](#customization)

---

## Overview

HippocampAI provides a full-featured AI chat assistant that integrates all memory management capabilities:

### Features

✅ **Smart Memory Retrieval**
- Retrieves relevant memories before each response
- Multi-factor ranking (similarity, importance, recency)
- Context-aware search

✅ **Automatic Memory Extraction**
- Extracts memories from conversations
- AI-powered classification and importance scoring
- Deduplication to prevent redundant storage

✅ **Session Management**
- Tracks conversation sessions
- Auto-generates summaries at session end
- Historical session retrieval

✅ **Memory Consolidation**
- Periodic cleanup of similar memories
- Importance decay over time
- Conflict resolution

✅ **Multiple Interfaces**
- Command-line interface (CLI)
- Web interface with beautiful UI
- Python API for custom integration

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs Flask and Flask-CORS in addition to existing dependencies.

### 2. Configure API Key

Make sure your `.env` file has an API key configured:

```env
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

### 3. Run CLI Chat

```bash
python cli_chat.py
```

Or specify a user ID:

```bash
python cli_chat.py alice
```

### 4. Run Web Chat

```bash
python web_chat.py
```

Then open: http://localhost:5000

---

## CLI Chat

### Starting CLI Chat

```bash
# Default user (cli_user)
python cli_chat.py

# Specific user
python cli_chat.py john_doe
```

### Available Commands

| Command | Description |
|---------|-------------|
| `/help` | Show help message |
| `/stats` | Display memory statistics |
| `/memories` | View stored memories |
| `/clear` | Clear current session (no summary) |
| `/end` | End session and save summary |
| `/quit` | Exit chat |

### Example CLI Session

```
🧠 HippocampAI - Memory-Enhanced AI Chat
==================================================
User: alice

Commands:
  /help      - Show this help
  /stats     - Show memory statistics
  /memories  - View stored memories
  /clear     - Clear current session
  /end       - End session and save summary
  /quit      - Exit chat

==================================================

You: Hi! I love hiking and outdoor photography.

[System] Thinking...
Assistant: Hello! It's great to meet someone who enjoys hiking and outdoor photography. Those are wonderful hobbies...

You: /stats

📊 Memory Statistics

  Total Memories: 2
  Average Importance: 7.5/10
  Recent (7 days): 2

  By Type:
    - preference: 2

You: /quit
[System] Ending session...
[System] Goodbye! 👋
```

### CLI Features

- **Colored Output**: User messages in blue, AI in green, system in cyan
- **Auto Memory Extraction**: Learns from every conversation
- **Session Persistence**: Memories saved across sessions
- **Graceful Shutdown**: Saves summary on Ctrl+C or /quit

---

## Web Chat

### Starting Web Server

```bash
python web_chat.py
```

Output:
```
============================================================
  HippocampAI Memory-Enhanced Chat Server
============================================================
Provider: anthropic
Qdrant: 192.168.1.120:6333

Server starting at: http://localhost:5000

Endpoints:
  - Web Interface: http://localhost:5000/
  - API Docs: See web_chat.py docstrings

Press Ctrl+C to stop
============================================================
```

### Web Interface Features

**Left Sidebar:**
- User ID configuration
- Real-time memory statistics
- View memories button
- End session button
- Clear chat button

**Chat Area:**
- Beautiful gradient design
- Typing indicators
- Message timestamps
- Smooth animations
- Responsive layout

**How to Use:**
1. Enter your user ID (default: demo_user)
2. Type messages in the input box
3. Press Enter or click Send
4. View stats updating in real-time
5. Click "View Memories" to see what AI remembers

### Web Chat Screenshots

```
┌─────────────────────────────────────────────────────────┐
│ 🧠 HippocampAI Chat                           Ready           │
├──────────┬──────────────────────────────────────────────┤
│          │                                              │
│ User ID  │  Welcome to HippocampAI! 👋                       │
│ ┌──────┐ │                                              │
│ │alice │ │  I'm your AI assistant with memory...       │
│ └──────┘ │                                              │
│          │                                              │
│ Stats    │  ┌────────────────────────────────────────┐ │
│ ┌──────┐ │  │ You: I love hiking                    │ │
│ │  15  │ │  │ 2:30 PM                                │ │
│ │Total │ │  └────────────────────────────────────────┘ │
│ └──────┘ │                                              │
│          │  ┌────────────────────────────────────────┐ │
│ Buttons  │  │ Assistant: That's wonderful! ...       │ │
│ [View  ] │  │ 2:30 PM                                │ │
│ [End   ] │  └────────────────────────────────────────┘ │
│ [Clear ] │                                              │
│          │  [Type your message...            ] [Send]  │
└──────────┴──────────────────────────────────────────────┘
```

---

## Python Integration

### Basic Usage

```python
from src.ai_chat import MemoryEnhancedChat

# Create chat instance
chat = MemoryEnhancedChat(
    user_id="alice",
    auto_extract_memories=True,
    auto_consolidate=True,
    memory_retrieval_limit=10
)

# Send message
response = chat.send_message("Hello! I'm planning a trip to Japan.")
print(response)

# Continue conversation
response = chat.send_message("What should I pack?")
print(response)

# End session
chat.end_conversation()
```

### Advanced Usage

```python
from src.ai_chat import MemoryEnhancedChat

# Initialize with custom settings
chat = MemoryEnhancedChat(
    user_id="bob",
    auto_extract_memories=True,
    auto_consolidate=False,  # Manual consolidation
    memory_retrieval_limit=15,
    system_prompt="You are a helpful travel assistant..."
)

# Send message with context hint
response = chat.send_message(
    "Best restaurants in Tokyo?",
    context_type="personal"  # or "work", "casual"
)

# Get conversation history
history = chat.get_conversation_history()
for msg in history:
    print(f"{msg['role']}: {msg['content']}")

# Get user's memories
memories = chat.get_user_memories(
    memory_type="preference",  # Filter by type
    limit=20
)

# Get memory statistics
stats = chat.get_memory_stats()
print(f"Total memories: {stats['total_memories']}")
print(f"By type: {stats['by_type']}")

# Manual consolidation
chat._consolidate_memories()

# Clear session without summary
chat.clear_session()
```

### Custom System Prompt

```python
custom_prompt = """You are an expert Python tutor with memory.

You remember the student's skill level, topics they've learned,
questions they've asked, and areas where they struggle.

Be encouraging and adapt your teaching style based on what you
know about the student."""

chat = MemoryEnhancedChat(
    user_id="student_123",
    system_prompt=custom_prompt
)
```

---

## API Reference

### MemoryEnhancedChat Class

#### Constructor

```python
MemoryEnhancedChat(
    user_id: str,
    auto_extract_memories: bool = True,
    auto_consolidate: bool = False,
    memory_retrieval_limit: int = 10,
    system_prompt: Optional[str] = None
)
```

**Parameters:**
- `user_id`: Unique identifier for the user
- `auto_extract_memories`: Extract memories after each message (default: True)
- `auto_consolidate`: Auto-consolidate every 20 messages (default: False)
- `memory_retrieval_limit`: Number of memories to retrieve (default: 10)
- `system_prompt`: Custom system prompt (default: friendly assistant)

#### Methods

##### send_message()

```python
send_message(
    message: str,
    context_type: Optional[str] = None
) -> str
```

Send a message and get AI response with memory context.

**Parameters:**
- `message`: User's message
- `context_type`: Optional hint ("work", "personal", "casual")

**Returns:** AI response string

**Example:**
```python
response = chat.send_message("What are my goals?")
```

##### end_conversation()

```python
end_conversation() -> Optional[str]
```

End the current session and save summary.

**Returns:** Summary memory ID if session was active

**Example:**
```python
summary_id = chat.end_conversation()
print(f"Summary saved: {summary_id}")
```

##### get_conversation_history()

```python
get_conversation_history() -> List[Dict[str, str]]
```

Get current conversation history.

**Returns:** List of messages with role, content, timestamp

**Example:**
```python
history = chat.get_conversation_history()
for msg in history:
    print(f"{msg['role']}: {msg['content']}")
```

##### get_user_memories()

```python
get_user_memories(
    memory_type: Optional[str] = None,
    limit: int = 20
) -> List[Dict[str, Any]]
```

Get user's stored memories.

**Parameters:**
- `memory_type`: Filter by type (preference, fact, goal, etc.)
- `limit`: Maximum number to return

**Returns:** List of memory objects

**Example:**
```python
preferences = chat.get_user_memories(memory_type="preference")
for pref in preferences:
    print(pref['text'])
```

##### get_memory_stats()

```python
get_memory_stats() -> Dict[str, Any]
```

Get statistics about user's memories.

**Returns:** Dictionary with counts and averages

**Example:**
```python
stats = chat.get_memory_stats()
print(f"Total: {stats['total_memories']}")
print(f"By type: {stats['by_type']}")
print(f"Average importance: {stats['avg_importance']}")
```

##### clear_session()

```python
clear_session() -> None
```

Clear current session without saving summary.

**Example:**
```python
chat.clear_session()
```

---

## REST API Endpoints

### POST /api/chat/message

Send message and get response.

**Request:**
```json
{
  "user_id": "alice",
  "message": "Hello!",
  "context_type": "personal"
}
```

**Response:**
```json
{
  "success": true,
  "response": "Hi Alice! How are you?",
  "timestamp": "2025-10-05T15:30:00"
}
```

### GET /api/chat/history

Get conversation history.

**Query Params:** `user_id=alice`

**Response:**
```json
{
  "success": true,
  "history": [
    {
      "role": "user",
      "content": "Hello!",
      "timestamp": "2025-10-05T15:30:00"
    },
    {
      "role": "assistant",
      "content": "Hi Alice!",
      "timestamp": "2025-10-05T15:30:01"
    }
  ]
}
```

### GET /api/chat/memories

Get user's memories.

**Query Params:**
- `user_id=alice`
- `type=preference` (optional)
- `limit=20` (optional)

**Response:**
```json
{
  "success": true,
  "memories": [
    {
      "text": "User loves hiking",
      "memory_type": "preference",
      "importance": 8,
      "category": "personal",
      "timestamp": "2025-10-05T15:00:00"
    }
  ],
  "count": 1
}
```

### GET /api/chat/stats

Get memory statistics.

**Query Params:** `user_id=alice`

**Response:**
```json
{
  "success": true,
  "stats": {
    "total_memories": 42,
    "by_type": {
      "preference": 15,
      "fact": 20,
      "goal": 7
    },
    "by_category": {
      "personal": 25,
      "work": 17
    },
    "avg_importance": 7.2,
    "recent_count": 12
  }
}
```

### POST /api/chat/end

End session and save summary.

**Request:**
```json
{
  "user_id": "alice"
}
```

**Response:**
```json
{
  "success": true,
  "summary_id": "uuid-here",
  "message": "Session ended and summary saved"
}
```

### POST /api/chat/clear

Clear current session.

**Request:**
```json
{
  "user_id": "alice"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Session cleared"
}
```

---

## Architecture

### How It Works

```
User Message
    ↓
┌─────────────────────────────────────────┐
│ 1. Retrieve Relevant Memories           │
│    - Query: "What should I pack?"       │
│    - Retrieves: Japan trip, preferences │
│    - Limit: 10 most relevant            │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ 2. Build Context from Memories          │
│    === CONTEXT ABOUT USER ===           │
│    Facts:                               │
│      - Planning trip to Japan           │
│    Preferences:                         │
│      - Loves hiking and photography     │
│    === END CONTEXT ===                  │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ 3. Generate AI Response                 │
│    - System prompt + context + history  │
│    - LLM generates personalized response│
│    - Uses memories for relevant answer  │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ 4. Extract New Memories                 │
│    - User: "What should I pack?"        │
│    - AI: "For hiking in Japan..."       │
│    - Extracted: User interested in      │
│      hiking equipment for Japan         │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ 5. Deduplicate & Store                  │
│    - Check for similar memories         │
│    - AI decides: store/skip/merge       │
│    - Update importance scores           │
└─────────────────────────────────────────┘
    ↓
Response to User
```

### Memory Retrieval Process

1. **Smart Search**: Uses multi-factor ranking
   - Similarity to query: 50%
   - Importance score: 30%
   - Recency: 20%
   - Access frequency boost

2. **Context Building**: Groups by type
   - Preferences
   - Facts
   - Goals
   - Recent Events
   - Background

3. **Context Injection**: Added to system prompt
   - AI knows user context
   - Personalized responses
   - Natural references to past conversations

### Memory Extraction Process

1. **Conversation Analysis**: LLM analyzes turn
   - User message
   - AI response
   - Extracts structured memories

2. **Classification**: Each memory gets
   - Type: preference/fact/goal/habit/event/context
   - Importance: 1-10 scale
   - Category: work/personal/learning/etc.
   - Confidence: 0.0-1.0

3. **Deduplication**: Before storing
   - Find similar memories (vector similarity)
   - AI decides action: replace/merge/skip/store
   - Prevents redundant storage

4. **Storage**: Save to Qdrant
   - Embedding generated
   - Metadata attached
   - Indexed for search

### Session Management

1. **Session Start**: Creates session ID
   - Tracks all messages
   - Records metadata (start time, topics)

2. **Message Tracking**: Each turn logged
   - User messages
   - AI responses
   - Timestamps

3. **Session End**: Generates summary
   - AI summarizes conversation
   - Extracts key points
   - Stores as memory
   - Archives session

---

## Customization

### Custom LLM Provider

Change in `.env`:
```env
LLM_PROVIDER=openai  # or anthropic, groq
OPENAI_API_KEY=sk-your-key
```

### Custom Memory Retrieval Limit

```python
chat = MemoryEnhancedChat(
    user_id="user",
    memory_retrieval_limit=20  # Retrieve more memories
)
```

### Disable Auto Features

```python
chat = MemoryEnhancedChat(
    user_id="user",
    auto_extract_memories=False,  # Manual extraction
    auto_consolidate=False         # Manual consolidation
)

# Manual memory extraction
memories = chat.extractor.extract_memories(
    conversation_text="...",
    user_id="user"
)

# Manual consolidation
chat._consolidate_memories()
```

### Custom Context Filtering

```python
# Retrieve only specific types
preferences = chat.retriever.search_memories(
    query="",
    filters={
        "user_id": "alice",
        "memory_type": "preference",
        "category": "work"
    },
    limit=10
)
```

### Web Server Configuration

Edit `web_chat.py`:

```python
# Change host/port
app.run(
    host='0.0.0.0',  # Allow external connections
    port=8080,        # Custom port
    debug=False       # Production mode
)
```

### Web UI Customization

Edit `web/chat.html`:

- Change colors in `<style>` section
- Modify layout (sidebar width, chat area)
- Add custom features (file upload, voice, etc.)

---

## Production Deployment

### Redis Session Storage

For production, replace in-memory sessions with Redis:

```python
import redis
from src.ai_chat import MemoryEnhancedChat

redis_client = redis.Redis(host='localhost', port=6379)

# Store chat session
chat = MemoryEnhancedChat(user_id="alice")
redis_client.set(f"chat:alice", pickle.dumps(chat))

# Retrieve chat session
chat = pickle.loads(redis_client.get(f"chat:alice"))
```

### Environment Variables

Production `.env`:
```env
# LLM
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-prod-...

# Qdrant
QDRANT_HOST=qdrant.production.com
QDRANT_PORT=6333
QDRANT_API_KEY=your-prod-key

# Logging
LOG_LEVEL=WARNING
LOG_FILE=/var/log/hippocampai/chat.log

# Rate Limiting
RATE_LIMIT_INTERVAL_MS=200
MAX_RETRIES=3
```

### Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "web_chat.py"]
```

### NGINX Reverse Proxy

```nginx
server {
    listen 80;
    server_name chat.example.com;

    location / {
        proxy_pass http://localhost:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    location /api {
        proxy_pass http://localhost:5000/api;
    }
}
```

---

## Troubleshooting

### Issue: "API key required"

**Solution:** Check `.env` file has correct API key for your provider:
```bash
grep API_KEY .env
```

### Issue: "Module 'src' not found"

**Solution:** Set PYTHONPATH:
```bash
export PYTHONPATH=/path/to/HippocampAI:$PYTHONPATH
python cli_chat.py
```

### Issue: Web server won't start

**Solution:** Install Flask dependencies:
```bash
pip install flask flask-cors
```

### Issue: Slow responses

**Solutions:**
1. Reduce memory_retrieval_limit:
   ```python
   chat = MemoryEnhancedChat(memory_retrieval_limit=5)
   ```

2. Use faster LLM provider (Groq):
   ```env
   LLM_PROVIDER=groq
   ```

3. Disable auto-consolidation:
   ```python
   chat = MemoryEnhancedChat(auto_consolidate=False)
   ```

### Issue: Too many memories stored

**Solution:** Run manual consolidation:
```python
from src.memory_consolidator import MemoryConsolidator

consolidator = MemoryConsolidator(retriever, updater, embeddings)
clusters = consolidator.find_similar_clusters(user_id="alice")
for cluster in clusters:
    consolidator.consolidate_cluster(cluster, user_id="alice")
```

---

## Examples

### Example 1: Personal Assistant

```python
from src.ai_chat import MemoryEnhancedChat

chat = MemoryEnhancedChat(
    user_id="john",
    system_prompt="You are John's personal assistant."
)

# Day 1
chat.send_message("I have a meeting tomorrow at 2pm")
chat.send_message("I need to buy groceries")
chat.end_conversation()

# Day 2 (new session)
response = chat.send_message("What did I need to do?")
# AI remembers: meeting at 2pm, buy groceries
```

### Example 2: Learning Tutor

```python
tutor = MemoryEnhancedChat(
    user_id="student_456",
    system_prompt="You are a Python tutor who remembers student progress."
)

# Lesson 1
tutor.send_message("I'm struggling with loops")
tutor.end_conversation()

# Lesson 2
response = tutor.send_message("Can you explain recursion?")
# AI knows: student struggled with loops, adapts teaching style
```

### Example 3: Customer Support

```python
support = MemoryEnhancedChat(
    user_id="customer_789",
    system_prompt="You are a customer support agent."
)

# Previous conversation remembered
support.send_message("My order still hasn't arrived")
# AI recalls: previous order issues, shipping address, product
```

---

## Next Steps

1. **Try CLI Chat**: `python cli_chat.py`
2. **Launch Web Interface**: `python web_chat.py`
3. **Read API Docs**: See docstrings in `src/ai_chat.py`
4. **Customize**: Modify system prompts, retrieval limits
5. **Deploy**: Use Docker or cloud platforms

Happy chatting with memory! 🧠✨
