# Interactive Chat Demo Guide

**File**: `chat.py`
**Purpose**: Fully-functional chatbot demonstrating HippocampAI's memory capabilities

This guide explains how to use the interactive chat demo that showcases HippocampAI's memory features with Groq's fast LLM responses.

---

## 🎯 What It Demonstrates

The `chat.py` script is a complete example showing:

1. **Persistent Memory**: Conversations are remembered across sessions
2. **Context Retrieval**: Relevant memories are automatically recalled
3. **Memory Extraction**: Important information is extracted and stored
4. **Session Management**: Conversations are organized into sessions
5. **Pattern Detection**: Behavioral patterns are identified
6. **Memory Search**: Full-text search across all memories
7. **Rich Memory Types**: Facts, preferences, goals, events, and habits

---

## 📋 Prerequisites

### 1. Qdrant Vector Database

**Option A: Docker (Recommended)**
```bash
docker run -d -p 6333:6333 qdrant/qdrant
```

**Option B: Local Installation**
```bash
# Download from https://github.com/qdrant/qdrant/releases
./qdrant
```

**Verify it's running:**
```bash
curl http://localhost:6333/collections
```

### 2. Groq API Key

1. Sign up at [console.groq.com](https://console.groq.com)
2. Navigate to [API Keys](https://console.groq.com/keys)
3. Create a new API key
4. Set environment variable:

```bash
# Linux/macOS
export GROQ_API_KEY="your-api-key-here"

# Windows (PowerShell)
$env:GROQ_API_KEY="your-api-key-here"

# Or add to ~/.bashrc or ~/.zshrc for persistence
echo 'export GROQ_API_KEY="your-api-key-here"' >> ~/.bashrc
```

### 3. Python Dependencies

```bash
# Install HippocampAI with all dependencies
pip install "hippocampai[all]"

# Or minimal installation
pip install hippocampai openai  # openai package required for Groq
```

---

## 🚀 Quick Start

### Basic Usage

```bash
# Run the chat demo
python chat.py

# Follow the prompts
Enter your name (or press Enter for 'demo_user'): alice

# Start chatting!
alice> Hi! I'm Alice and I love coffee and hiking.
🤖 Assistant> Nice to meet you, Alice! It's great to know...
```

### First Conversation Example

```
alice> I'm a software engineer at TechCorp
🤖 Assistant> That's great! What kind of software do you work on?

alice> I work on machine learning models. I want to learn more about LLMs.
🤖 Assistant> Excellent goal! LLMs are fascinating...

alice> I prefer to have my coffee with oat milk every morning
🤖 Assistant> Noted! I'll remember that you like oat milk...

alice> What do you know about me?
🤖 Assistant> Based on our conversation, I know that:
- You're a software engineer at TechCorp
- You work on machine learning models
- You want to learn more about LLMs
- You prefer oat milk in your coffee
```

### Second Conversation (Later)

```bash
# Run again with the same user ID
python chat.py
Enter your name: alice

alice> What's my favorite coffee addition?
🤖 Assistant> You prefer oat milk in your coffee!

alice> Where do I work again?
🤖 Assistant> You work as a software engineer at TechCorp, focusing on machine learning models.
```

---

## 📖 Available Commands

### `/help` - Show Help
```
alice> /help

💡 AVAILABLE COMMANDS
  /help       - Show this help message
  /stats      - Show memory statistics
  /memories   - Show recent memories
  /patterns   - Detect behavioral patterns
  /search     - Search memories
  /clear      - Clear screen
  /exit       - Exit chat (with session summary)
```

### `/stats` - Memory Statistics
```
alice> /stats

📊 MEMORY STATISTICS
Total memories: 12
Memory types: {'preference': 4, 'fact': 5, 'goal': 3}
```

### `/memories` - Recent Memories
```
alice> /memories

🧠 RECENT MEMORIES (Last 10)

1. [PREFERENCE] I prefer oat milk in my coffee
   Importance: 8/10 | Created: 2026-02-11 10:15

2. [FACT] I work as a software engineer at TechCorp
   Importance: 7/10 | Created: 2026-02-11 10:14

3. [GOAL] I want to learn more about LLMs
   Importance: 9/10 | Created: 2026-02-11 10:14
```

### `/search` - Search Memories
```
alice> /search coffee

🔎 SEARCH RESULTS for: 'coffee'

1. I prefer oat milk in my coffee
   Type: preference | Score: 0.95
   Importance: 8/10

2. I drink coffee every morning
   Type: habit | Score: 0.78
   Importance: 6/10
```

### `/patterns` - Detect Patterns
```
alice> /patterns

🔍 DETECTED PATTERNS

1. Morning coffee routine with oat milk
   Confidence: 0.87
   Description: User consistently mentions morning coffee with oat milk

2. Career focus on machine learning
   Confidence: 0.92
   Description: Strong interest in ML and LLMs, works in the field
```

### `/exit` - Exit with Summary
```
alice> /exit

Completing session and generating summary...

📝 SESSION SUMMARY
In this session, we discussed Alice's role as a software engineer
at TechCorp, her interest in learning about LLMs, and her coffee
preference for oat milk...

Goodbye! Your memories are saved for next time. 👋
```

---

## 🔧 Configuration

### Using Different LLM Models

Edit `chat.py` line 49 to use different Groq models:

```python
self.llm = GroqLLM(
    api_key=api_key,
    model="llama-3.3-70b-versatile"  # Higher quality, slower
    # model="llama-3.1-8b-instant"   # Faster (default)
    # model="mixtral-8x7b-32768"     # Large context window
)
```

### Using Different User IDs

```bash
# Each user ID has separate memories
python chat.py
Enter your name: alice   # alice's memories

python chat.py
Enter your name: bob     # bob's separate memories
```

### Custom Qdrant URL

Edit `chat.py` initialization:

```python
self.memory_client = MemoryClient(
    llm_provider=self.llm,
    qdrant_url="http://localhost:6333"  # Default
    # qdrant_url="https://your-cloud-instance.com"  # Cloud
)
```

### Enable Redis Caching (Optional)

```bash
# Start Redis
docker run -d -p 6379:6379 redis:7.2-alpine

# Set environment variable
export REDIS_URL="redis://localhost:6379"

# Edit chat.py initialization
self.memory_client = MemoryClient(
    llm_provider=self.llm,
    redis_url=os.getenv("REDIS_URL")
)
```

---

## 💡 Usage Tips

### 1. Share Personal Information
The more you share, the better the memory system works:

```
✅ Good:
"I'm a software engineer specializing in Python and Go"
"I prefer working remotely on Mondays and Wednesdays"
"My goal is to launch a SaaS product by Q2 2026"

❌ Too vague:
"I like coding"
"I work from home sometimes"
```

### 2. Ask Follow-up Questions
Test the memory by asking about previous topics:

```
alice> What did I tell you about my work?
alice> What are my career goals?
alice> What do you remember about my preferences?
```

### 3. Use Pattern Detection
After chatting for a while:

```
alice> /patterns

# See what behavioral patterns the system detected
```

### 4. Search Your Memories
Find specific information:

```
alice> /search work
alice> /search coffee
alice> /search goals
```

### 5. Check Statistics
Monitor how much the system has learned:

```
alice> /stats

# Shows memory count and distribution by type
```

---

## 🎓 Educational Use Cases

### For Developers

**Learn how to:**
- Integrate HippocampAI into chat applications
- Use context retrieval for relevant responses
- Extract and store memories from conversations
- Manage sessions and generate summaries
- Implement pattern detection

**Code sections to study:**
- `get_relevant_context()` - Context retrieval (line 56)
- `extract_and_store_memories()` - Memory extraction (line 75)
- `chat()` - Main conversation loop (line 100)

### For Students

**Understand:**
- How AI memory systems work
- Vector similarity search
- Session management
- Pattern detection algorithms
- Context-aware responses

### For Product Managers

**Explore:**
- Memory-powered user experiences
- Personalization capabilities
- Cross-session continuity
- Behavioral insights
- Privacy-preserving memory

---

## 🔍 How It Works

### 1. Initialization
```python
# Creates memory client with Groq LLM
self.memory_client = MemoryClient(llm_provider=self.llm)

# Creates a session for this conversation
self.session = self.memory_client.create_session(user_id=user_id)
```

### 2. Context Retrieval
```python
# For each user message, recall relevant memories
results = self.memory_client.recall(query=query, user_id=user_id, k=5)

# Add to LLM context
context = format_memories(results)
```

### 3. Response Generation
```python
# Build prompt with memories + conversation history
messages = [system_prompt, *conversation_history, user_message]

# Generate response with Groq
response = self.llm.chat(messages)
```

### 4. Memory Extraction
```python
# Extract important information from conversation
memories = self.memory_client.extract_from_conversation(
    conversation=f"User: {user_msg}\nAssistant: {response}",
    user_id=user_id
)
```

### 5. Session Tracking
```python
# Track messages in session
self.memory_client.track_session_message(
    session_id=self.session.id,
    role="user",
    content=user_message
)
```

---

## 🐛 Troubleshooting

### Error: "GROQ_API_KEY environment variable not set"

**Solution:**
```bash
export GROQ_API_KEY="your-api-key-here"

# Verify
echo $GROQ_API_KEY
```

### Error: "Failed to connect to HippocampAI"

**Solution:**
```bash
# Check if Qdrant is running
curl http://localhost:6333/collections

# If not, start it
docker run -d -p 6333:6333 qdrant/qdrant
```

### Error: "openai package required"

**Solution:**
```bash
pip install openai
```

### Slow Responses

**Solutions:**
1. Use faster model: `llama-3.1-8b-instant` (default)
2. Enable Redis caching
3. Reduce `k` parameter in recall (fewer memories)

### Memory Not Persisting

**Check:**
1. Using same user_id across sessions
2. Qdrant data directory has write permissions
3. Qdrant container is persistent (not `--rm` flag)

---

## 📈 Advanced Features

### Custom Memory Types

Edit the extraction to specify types:

```python
memory = client.remember(
    text="My favorite programming language is Python",
    user_id=user_id,
    type="preference",
    importance=8.0,
    metadata={"category": "programming"}
)
```

### Multi-Agent Conversations

Create separate agents:

```python
agent1 = client.create_agent(
    agent_id="assistant",
    name="Main Assistant"
)

agent2 = client.create_agent(
    agent_id="researcher",
    name="Research Specialist"
)
```

### Temporal Queries

Find memories from specific time periods:

```python
from datetime import datetime, timedelta

# Memories from last week
recent = client.get_memories_by_time_range(
    user_id=user_id,
    start_time=datetime.now() - timedelta(days=7),
    end_time=datetime.now()
)
```

---

## 🎨 Customization Ideas

### 1. Add Voice Input/Output
```python
# Use speech recognition
import speech_recognition as sr

# Use text-to-speech
import pyttsx3
```

### 2. Web Interface
```python
# Convert to Flask/FastAPI app
@app.post("/chat")
async def chat_endpoint(message: str, user_id: str):
    return bot.chat(message)
```

### 3. Slack/Discord Bot
```python
# Integrate with messaging platforms
@bot.command()
async def ask(ctx, *, question):
    response = memory_bot.chat(question)
    await ctx.send(response)
```

### 4. Memory Visualization
```python
# Export memories as graph
graph = client.export_graph_to_json(user_id)

# Visualize with networkx/plotly
```

---

## 📚 Related Documentation

- [Getting Started Guide](GETTING_STARTED.md) - Basic HippocampAI setup
- [Library Complete Reference](LIBRARY_COMPLETE_REFERENCE.md) - All 102+ methods
- [Session Management](SESSION_MANAGEMENT.md) - Session features
- [Intelligence Features](FEATURES.md#intelligence-features) - Fact extraction
- [Examples Directory](../examples/) - 15+ code examples

---

## 🤝 Contributing

Found a bug or want to improve the chat demo?

1. Fork the repository
2. Make your changes to `chat.py`
3. Test thoroughly
4. Submit a pull request

---

## 📝 License

This demo is part of HippocampAI and is licensed under Apache 2.0.

---

**Happy Chatting!** 🚀

For questions or issues:
- [GitHub Issues](https://github.com/rexdivakar/HippocampAI/issues)
- [Discord Community](https://discord.gg/pPSNW9J7gB)
