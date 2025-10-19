# HippocampAI Tools - Web Search & More

## Overview

HippocampAI chat now supports **tool calling** (function calling), allowing the AI assistant to search the web, perform calculations, and more!

## Available Tools

### 1. Web Search 🔍

Search the web for current information, news, facts, and answers.

**Supported Engines:**
- **DuckDuckGo** (default, free, no API key needed) ✅
- **Google Custom Search** (requires API key)
- **Brave Search** (requires API key)

**Example queries:**
- "What's the weather in Tokyo today?"
- "Latest news about AI"
- "Who won the 2024 Olympics?"
- "Current price of Bitcoin"

**How it works:**
1. AI detects when web search is needed
2. Automatically calls web_search tool
3. Gets search results
4. Uses results to answer your question

### 2. Calculator 🧮

Perform mathematical calculations.

**Supported operations:**
- Basic arithmetic: +, -, *, /, **
- Functions: abs, round, min, max, sum, pow

**Example queries:**
- "What is 234 * 567?"
- "Calculate the square root of 144"
- "What's 15% of 350?"

---

## Setup

### Quick Start (DuckDuckGo - No API Key)

The chat works out of the box with DuckDuckGo search! No configuration needed.

```bash
python web_chat.py
# or
python cli_chat.py
```

That's it! Try asking: "What's the current temperature in Paris?"

### Google Custom Search (Optional)

For Google search, you need API credentials:

1. **Get API Key:**
   - Go to: https://console.cloud.google.com/
   - Create project → Enable Custom Search API
   - Get API key

2. **Get Custom Search Engine ID (CX):**
   - Go to: https://programmablesearchengine.google.com/
   - Create search engine
   - Get CX ID

3. **Add to `.env`:**
   ```env
   GOOGLE_API_KEY=your_api_key_here
   GOOGLE_CX=your_cx_id_here
   ```

4. **Use Google search:**
   ```python
   from src.tools import WebSearchTool

   search = WebSearchTool(search_engine="google")
   ```

### Brave Search (Optional)

For Brave search:

1. **Get API Key:**
   - Go to: https://brave.com/search/api/
   - Sign up for API access
   - Get API key

2. **Add to `.env`:**
   ```env
   BRAVE_API_KEY=your_api_key_here
   ```

3. **Use Brave search:**
   ```python
   from src.tools import WebSearchTool

   search = WebSearchTool(search_engine="brave")
   ```

---

## Usage

### Automatic (Recommended)

The AI automatically uses tools when needed:

```python
from src.ai_chat import MemoryEnhancedChat

chat = MemoryEnhancedChat(user_id="alice", enable_tools=True)

# AI will automatically search the web when needed
response = chat.send_message("What's the latest news about SpaceX?")
print(response)
```

### Manual Tool Execution

You can also call tools directly:

```python
from src.tools import create_default_registry

# Get tool registry
registry = create_default_registry()

# Execute web search
result = registry.execute_tool(
    "web_search",
    query="Latest AI developments",
    num_results=5
)

print(result["results"])

# Execute calculator
result = registry.execute_tool(
    "calculator",
    expression="2 ** 10"
)

print(result["result"])  # 1024
```

---

## How It Works

### Tool Calling Flow

```
User: "What's the weather in Tokyo?"
  ↓
[AI detects need for current information]
  ↓
[Calls web_search tool]
  query: "weather in Tokyo"
  num_results: 3
  ↓
[Tool executes DuckDuckGo search]
  ↓
[Returns search results]
  - "Tokyo weather: 22°C, sunny..."
  - "Current conditions in Tokyo..."
  ↓
[AI uses results to formulate answer]
  ↓
AI: "The current weather in Tokyo is 22°C and sunny..."
```

### Tool Schema

Tools are defined with schemas that tell the AI:
- Tool name
- Description (when to use it)
- Parameters (what inputs it needs)
- Types (validation)

Example schema:
```json
{
  "name": "web_search",
  "description": "Search the web for current information",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "Search query"
      },
      "num_results": {
        "type": "integer",
        "description": "Number of results",
        "default": 5
      }
    },
    "required": ["query"]
  }
}
```

---

## Examples

### Example 1: Current Events

```python
chat = MemoryEnhancedChat("user1", enable_tools=True)

response = chat.send_message("What are today's top tech news?")
# AI searches web and provides current news
```

### Example 2: Research

```python
response = chat.send_message("Compare Python vs JavaScript for web development")
# AI searches for comparisons and summarizes findings
```

### Example 3: Calculations with Context

```python
response = chat.send_message("I have $1000. If I invest at 5% annual return for 10 years, how much will I have?")
# AI uses calculator: 1000 * (1.05 ** 10)
```

### Example 4: Mixed Query

```python
response = chat.send_message("What's the current Bitcoin price and calculate 10% of that")
# AI:
# 1. Searches web for Bitcoin price
# 2. Uses calculator for 10% calculation
# 3. Provides complete answer
```

---

## Creating Custom Tools

You can add your own tools!

```python
from src.tools import BaseTool, ToolRegistry
from typing import Dict, Any

class WeatherTool(BaseTool):
    """Get weather information."""

    def __init__(self):
        super().__init__(
            name="get_weather",
            description="Get current weather for a location"
        )

    def execute(self, location: str) -> Dict[str, Any]:
        # Your implementation
        # Call weather API, etc.
        return {
            "success": True,
            "location": location,
            "temperature": "22°C",
            "conditions": "Sunny"
        }

    def get_schema(self) -> Dict[str, Any]:
        return {
            "name": "get_weather",
            "description": "Get current weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name or location"
                    }
                },
                "required": ["location"]
            }
        }

# Register your tool
registry = ToolRegistry()
registry.register_tool(WeatherTool())

# Use in chat
chat = MemoryEnhancedChat("user", enable_tools=True)
chat.tool_registry = registry  # Replace with custom registry
```

---

## Configuration

### Enable/Disable Tools

```python
# Enable tools (default)
chat = MemoryEnhancedChat("user", enable_tools=True)

# Disable tools
chat = MemoryEnhancedChat("user", enable_tools=False)
```

### Search Engine Selection

Edit `src/tools.py`:

```python
# Default search engine
def create_default_registry() -> ToolRegistry:
    registry = ToolRegistry()

    # Change to "google" or "brave" if you have API keys
    registry.register_tool(WebSearchTool(search_engine="duckduckgo"))

    registry.register_tool(CalculatorTool())

    return registry
```

---

## Troubleshooting

### "Search failed"

**DuckDuckGo:**
- Check internet connection
- DuckDuckGo might be rate-limiting
- Try again in a few seconds

**Google:**
- Verify GOOGLE_API_KEY in .env
- Verify GOOGLE_CX in .env
- Check API quota (Google has daily limits)

**Brave:**
- Verify BRAVE_API_KEY in .env
- Check API quota

### Tool not being called

**Possible reasons:**
1. Tools disabled: `enable_tools=False`
2. AI doesn't think tool is needed
3. Try being more explicit: "Search the web for..."

**Solutions:**
- Enable tools: `enable_tools=True`
- Be explicit in your query
- Check logs: Look for "Executing tool:" messages

### Calculator errors

**Issue:** "Calculation failed"

**Causes:**
- Unsafe expression (only basic math allowed)
- Syntax error

**Examples:**
- ✅ "2 + 2"
- ✅ "pow(3, 2)"
- ✅ "max(10, 20, 30)"
- ❌ "import os"  (not allowed, security)
- ❌ "2 +"  (syntax error)

---

## API Reference

### WebSearchTool

```python
class WebSearchTool(search_engine: str = "duckduckgo")
```

**Parameters:**
- `search_engine`: "duckduckgo", "google", or "brave"

**Methods:**
- `execute(query: str, num_results: int = 5)` → Dict

**Returns:**
```python
{
    "success": True,
    "query": "search query",
    "engine": "duckduckgo",
    "results": [
        {
            "title": "Result title",
            "snippet": "Description...",
            "url": "https://...",
            "source": "DuckDuckGo"
        }
    ],
    "timestamp": "2025-10-05T16:00:00"
}
```

### CalculatorTool

```python
class CalculatorTool()
```

**Methods:**
- `execute(expression: str)` → Dict

**Returns:**
```python
{
    "success": True,
    "expression": "2 + 2",
    "result": 4
}
```

### ToolRegistry

```python
class ToolRegistry()
```

**Methods:**
- `register_tool(tool: BaseTool)` - Add a tool
- `get_tool(name: str)` - Get tool by name
- `get_all_tools()` - List all tools
- `get_tools_schema()` - Get schemas for LLM
- `execute_tool(name: str, **kwargs)` - Execute tool

---

## Best Practices

### 1. Let AI Decide

Don't force tool usage. The AI knows when to use tools:

```python
# ✅ Good - Natural query
"What's the weather today?"

# ❌ Unnecessary - AI can answer directly
"Use web search to tell me what 2+2 is"
```

### 2. Combine with Memory

Tools + memory = powerful combination:

```python
chat.send_message("I'm interested in machine learning")
# Stores: User interested in machine learning

chat.send_message("What's the latest news?")
# AI searches for: "latest machine learning news"
# (Uses memory to personalize search)
```

### 3. Verify Critical Information

For important decisions, verify web search results:

```python
response = chat.send_message("What's the recommended dosage for aspirin?")
# AI provides info from web search
# ⚠️ Always consult professionals for medical/legal/financial advice
```

---

## Future Tools (Coming Soon)

Planned additions:
- 📧 Email sender
- 📅 Calendar integration
- 🗺️ Maps & directions
- 📰 News aggregator
- 💱 Currency converter
- 🌤️ Weather API
- 📊 Data visualization
- 🔐 API integrations

---

## Summary

✅ **Web search** - Get current information (DuckDuckGo, Google, Brave)
✅ **Calculator** - Perform math calculations
✅ **Auto tool calling** - AI decides when to use tools
✅ **Free tier** - DuckDuckGo works without API keys
✅ **Extensible** - Easy to add custom tools
✅ **Memory integration** - Tools + memory = smarter responses

**Get started:**
```bash
python web_chat.py
```

Ask: "What's the latest news about AI?"

The AI will automatically search the web and give you current information! 🚀
