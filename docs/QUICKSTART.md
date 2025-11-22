# HippocampAI Quick Start

Get started with HippocampAI in 5 minutes!

## 🚀 Try the Interactive Chat Demo

The fastest way to see HippocampAI in action:

### 1. Install Dependencies

```bash
pip install "hippocampai[all]"
```

### 2. Start Qdrant (Vector Database)

```bash
docker run -d -p 6333:6333 qdrant/qdrant
```

### 3. Set Your Groq API Key

Get a free API key from [console.groq.com](https://console.groq.com/keys)

```bash
export GROQ_API_KEY="your-api-key-here"
```

### 4. Run the Chat Demo

```bash
python chat.py
```

That's it! Start chatting and watch how the bot remembers everything you tell it.

## 💡 What to Try

```
# Tell it about yourself
> Hi! I'm Alice and I work as a software engineer at TechCorp

# Share preferences
> I prefer oat milk in my coffee and I work remotely on Mondays

# Set goals
> I want to learn machine learning this year

# Later, test the memory
> What do you know about me?
> Where do I work?
> What are my goals?

# Use commands
> /stats       # See memory statistics
> /memories    # View recent memories
> /patterns    # Detect behavioral patterns
> /search work # Search your memories
```

## 📚 Full Documentation

- **[Chat Demo Guide](docs/CHAT_DEMO_GUIDE.md)** - Complete chat demo documentation
- **[Getting Started](docs/GETTING_STARTED.md)** - Detailed setup guide
- **[Library Reference](docs/LIBRARY_COMPLETE_REFERENCE.md)** - All 102+ methods
- **[API Reference](docs/SAAS_API_COMPLETE_REFERENCE.md)** - All 56 REST endpoints

## 🔧 Use in Your Code

```python
from hippocampai import MemoryClient

# Initialize
client = MemoryClient()

# Store a memory
memory = client.remember(
    "I prefer oat milk in my coffee",
    user_id="alice",
    type="preference"
)

# Recall memories
results = client.recall("coffee preferences", user_id="alice")
print(results[0].memory.text)  # "I prefer oat milk in my coffee"
```

## 🐛 Troubleshooting

**"GROQ_API_KEY not set"**
```bash
export GROQ_API_KEY="your-key"
```

**"Failed to connect to HippocampAI"**
```bash
# Make sure Qdrant is running
docker ps | grep qdrant

# If not, start it
docker run -d -p 6333:6333 qdrant/qdrant
```

**"openai package required"**
```bash
pip install openai
```

## 🤝 Get Help

- [GitHub Issues](https://github.com/rexdivakar/HippocampAI/issues)
- [Discord Community](https://discord.gg/pPSNW9J7gB)
- [Full Documentation](docs/)

---

**Happy Building! 🚀**
