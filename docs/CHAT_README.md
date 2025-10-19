# 💬 HippocampAI Chat - Quick Reference

## 🚀 Quick Start

### 1. CLI Chat (Command Line)
```bash
python cli_chat.py
```

**Features:**
- ✅ Colored terminal output
- ✅ Auto memory extraction
- ✅ Commands: /stats, /memories, /help, /quit
- ✅ Session management

### 2. Web Chat (Browser)
```bash
python web_chat.py
```
Then open: **http://localhost:5000**

**Features:**
- ✅ Beautiful gradient UI
- ✅ Real-time stats sidebar
- ✅ View stored memories
- ✅ Typing indicators
- ✅ Mobile responsive

### 3. Python API
```python
from src.ai_chat import MemoryEnhancedChat

chat = MemoryEnhancedChat(user_id="alice")
response = chat.send_message("Hello!")
print(response)
```

---

## 📋 What You Get

### Memory-Enhanced Chat Features

**Before Each Response:**
1. Retrieves relevant memories (preferences, facts, goals)
2. Builds context from memories
3. Injects context into AI prompt
4. AI generates personalized response

**After Each Response:**
5. Extracts new memories from conversation
6. Deduplicates (prevents storing same thing twice)
7. Stores important memories
8. Updates importance scores

**Session Management:**
- Tracks full conversation
- Auto-generates summary when session ends
- Stores session as memory
- Retrieves past sessions

**Smart Features:**
- Multi-factor memory ranking
- Importance decay over time
- Memory consolidation (merges similar memories)
- Conflict resolution

---

## 🎯 Use Cases

### Personal Assistant
```python
chat = MemoryEnhancedChat(
    user_id="john",
    system_prompt="You are John's personal assistant."
)

# Day 1
chat.send_message("Meeting tomorrow at 2pm")
chat.end_conversation()

# Day 2
response = chat.send_message("What's on my schedule?")
# AI remembers the meeting!
```

### Learning Tutor
```python
tutor = MemoryEnhancedChat(
    user_id="student",
    system_prompt="You are a Python tutor."
)

# Lesson 1
tutor.send_message("I struggle with loops")

# Lesson 2
response = tutor.send_message("Explain recursion")
# AI knows student struggled with loops, adapts teaching
```

### Customer Support
```python
support = MemoryEnhancedChat(
    user_id="customer_123",
    system_prompt="You are a support agent."
)

support.send_message("My order hasn't arrived")
# AI recalls: order number, shipping address, previous issues
```

---

## 🛠️ Key Files Created

| File | Purpose |
|------|---------|
| `src/ai_chat.py` | Core memory-enhanced chat class (650+ lines) |
| `web_chat.py` | Flask web server with REST API |
| `web/chat.html` | Beautiful web UI with gradient design |
| `cli_chat.py` | Command-line chat interface |
| `docs/CHAT_INTEGRATION.md` | Complete documentation (900+ lines) |

---

## 🔌 REST API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/chat/message` | POST | Send message, get response |
| `/api/chat/history` | GET | Get conversation history |
| `/api/chat/memories` | GET | Get user's memories |
| `/api/chat/stats` | GET | Get memory statistics |
| `/api/chat/end` | POST | End session & save summary |
| `/api/chat/clear` | POST | Clear current session |

---

## 📊 Memory Types

- **preference**: User likes/dislikes
- **fact**: Personal information
- **goal**: User objectives
- **habit**: Recurring behaviors
- **event**: Important occurrences
- **context**: Background information

---

## 🎨 Customization

### Custom System Prompt
```python
chat = MemoryEnhancedChat(
    user_id="user",
    system_prompt="You are an expert travel guide..."
)
```

### Adjust Memory Retrieval
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
```

---

## 🚦 How It Works

```
User: "What should I pack for Japan?"
  ↓
[Retrieve Memories]
  - User planning trip to Japan
  - User loves hiking
  - User interested in photography
  ↓
[Build Context]
  === CONTEXT ABOUT USER ===
  Facts:
    - Planning trip to Japan
  Preferences:
    - Loves hiking and photography
  ↓
[Generate Response]
  System: "You are a helpful assistant..."
  Context: "User loves hiking and photography..."
  History: [previous messages]
  User: "What should I pack?"
  ↓
AI: "For your Japan trip, since you love hiking and
     photography, I'd recommend..."
  ↓
[Extract Memories]
  - User interested in packing for hiking
  - User asking about Japan travel prep
  ↓
[Deduplicate & Store]
  - Check if similar memory exists
  - Store if new, merge if similar
```

---

## 📦 Installation

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up API key in .env
echo "ANTHROPIC_API_KEY=sk-ant-your-key" >> .env

# 3. Run setup
python setup_initial.py

# 4. Start chatting!
python cli_chat.py
# or
python web_chat.py
```

---

## 📖 Full Documentation

See **[docs/CHAT_INTEGRATION.md](docs/CHAT_INTEGRATION.md)** for:
- Complete API reference
- Architecture details
- Production deployment
- Troubleshooting
- Advanced examples

---

## ✨ What Makes This Special

**Traditional Chatbots:**
- No memory between sessions
- Can't learn from conversations
- Generic responses
- Forgets context

**HippocampAI Chat:**
- ✅ Remembers across sessions
- ✅ Learns from every conversation
- ✅ Personalized responses
- ✅ Maintains long-term context
- ✅ Smart memory management
- ✅ Automatic deduplication
- ✅ Importance scoring
- ✅ Session summaries

---

**Enjoy chatting with memory! 🧠✨**

For questions or issues, see the full documentation or check the examples.
