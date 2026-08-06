# Framework Integrations

HippocampAI ships official adapters for six agent frameworks under
`hippocampai.integrations.*`. Each adapter imports its framework **lazily** —
nothing breaks if you don't have the framework installed; you'll just see a clear
`ImportError` when you try to use the adapter.

The MCP server (see [MCP_SERVER.md](MCP_SERVER.md)) also covers any host that
speaks Model Context Protocol Claude Code, Cursor, Windsurf, Zed, Codex CLI, etc.
The adapters below are for direct in-process integration when you're writing
agents in Python.

## At a glance

| Framework | Module | Primary surface |
|---|---|---|
| **LangChain** | `hippocampai.integrations.langchain` | `HippocampMemory`, `HippocampRetriever` |
| **LlamaIndex** | `hippocampai.integrations.llamaindex` | `HippocampRetriever`, `HippocampMemoryStore` |
| **CrewAI** | `hippocampai.integrations.crewai` | `HippocampAIMemoryStorage` (CrewAI storage protocol) |
| **LangGraph** | `hippocampai.integrations.langgraph` | `HippocampAIMemory` (context helper + LangChain tool factory) |
| **Pydantic AI** | `hippocampai.integrations.pydantic_ai` | `HippocampDeps`, `register_memory_tools(agent)` |
| **AutoGen** | `hippocampai.integrations.autogen` | `HippocampAIMemory`, `register_memory_tools(agent, memory)` |

## CrewAI

CrewAI's memory system supports pluggable storage backends. `HippocampAIMemoryStorage`
implements that contract (`save(value, metadata, agent=None)` / `search(query, ...)` /
`reset()`) so an entire crew shares HippocampAI long-term memory.

```python
from crewai.memory import LongTermMemory
from hippocampai import MemoryClient
from hippocampai.integrations.crewai import HippocampAIMemoryStorage

storage = HippocampAIMemoryStorage(client=MemoryClient(), user_id="alice")
long_term = LongTermMemory(storage=storage)
```

`reset()` is intentionally a no-op HippocampAI memories are durable. Use the
engine's per-user delete APIs (or the MCP `delete_memory` tool) to actually wipe state.

## LangGraph

LangGraph runs short-lived workflows; durable knowledge belongs in long-term memory.
`HippocampAIMemory` has two integration modes:

**As a node helper** drop a context pack into the state:

```python
from hippocampai import MemoryClient
from hippocampai.integrations.langgraph import HippocampAIMemory

mem = HippocampAIMemory(client=MemoryClient(), user_id="alice")

def planner(state: dict) -> dict:
    ctx = mem.context_for(state["question"], token_budget=2000)
    return {"memory_context": ctx.text, **state}
```

**As LLM-bound tools** give a model inside a graph node `remember`, `recall`,
and `assemble_context` tools:

```python
tools = mem.tools()                # list of LangChain StructuredTool
llm_with_tools = llm.bind_tools(tools)
```

(`.tools()` requires `langchain-core`; the node helper does not.)

## Pydantic AI

Pydantic AI agents declare a typed dependency. `HippocampDeps` is a dataclass with
the client and the current user; `register_memory_tools(agent)` adds three async
tools to the agent.

```python
from pydantic_ai import Agent
from hippocampai import MemoryClient
from hippocampai.integrations.pydantic_ai import HippocampDeps, register_memory_tools

agent = Agent("groq:llama-3.1-8b-instant", deps_type=HippocampDeps)
register_memory_tools(agent)

result = await agent.run(
    "What do I prefer for coffee?",
    deps=HippocampDeps(client=MemoryClient(), user_id="alice"),
)
```

## AutoGen

Supports both `pyautogen` (v0.2) and `autogen-agentchat` (v0.4+).
`HippocampAIMemory` gives you a shared store for the crew; `register_memory_tools`
wires `remember`/`recall` into the agent's function map (v0.2) or tool list (v0.4+).

```python
from autogen import AssistantAgent
from hippocampai import MemoryClient
from hippocampai.integrations.autogen import HippocampAIMemory, register_memory_tools

mem = HippocampAIMemory(client=MemoryClient(), user_id="alice")
assistant = AssistantAgent("planner", llm_config={...})
register_memory_tools(assistant, mem)
```

You can also enrich a message manually before sending:

```python
enriched = mem.enrich_message("plan my week", token_budget=1500)
assistant.send(enriched, recipient=...)
```

## LangChain and LlamaIndex

These have shipped longer; see [NEW_FEATURES.md](NEW_FEATURES.md#framework-integrations)
for the original write-up. The classes are still imported from
`hippocampai.integrations.langchain` and `hippocampai.integrations.llamaindex`.

## When to use an adapter vs. the MCP server

| Scenario | Use |
|---|---|
| Writing a Python agent that imports the framework directly | The framework's adapter above |
| Plugging memory into Claude Code, Cursor, etc. (any MCP host) | The MCP server ([docs](MCP_SERVER.md)) |
| Remote / web client speaking JSON-RPC | The MCP server's `streamable-http` transport |
| Both Python integration + IDE/host integration | Both they share the same engine |
