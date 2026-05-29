# MCP Server

HippocampAI ships a [Model Context Protocol](https://modelcontextprotocol.io) (MCP)
server that exposes the memory engine as a set of tools any MCP-compatible agent can
call — Claude Code, Claude Desktop, Cursor, and the growing ecosystem of MCP clients.

## Why this matters

In 2026, MCP is the standard way agents plug into external capabilities. Without an
MCP server, using HippocampAI from an agent means writing and maintaining custom glue
code for every host. With it:

- **Zero-glue integration.** Point an MCP client at one command and the agent gains
  persistent long-term memory — no SDK wiring, no bespoke tool definitions.
- **Distribution.** MCP is the channel where agents live. A server makes HippocampAI
  installable into Claude Code/Cursor the same way every other memory engine is.
- **Portability.** The same server works across every MCP host, so memory behaves
  consistently whether you're in an IDE, a desktop app, or a custom agent runtime.

It turns "a Python library you integrate" into "a capability your agent already has."

## Install

```bash
pip install "hippocampai[mcp]"
```

This pulls in the `mcp` package alongside HippocampAI. (It is also included in the
`[all]` extra.)

## Universal by design

This is a **standard MCP server** — it implements the protocol, not any single vendor's
API. It works with **any MCP client**: Claude Code, Claude Desktop, Cursor, Windsurf,
Zed, Continue, custom agents built on the MCP SDKs, and remote/web clients over HTTP.
There is nothing Claude-specific about it.

## Run

The server supports all three MCP transports via `--transport`:

```bash
# stdio (default) — the zero-config transport local IDE/agent hosts launch automatically
hippocampai-mcp
python -m hippocampai.mcp

# streamable-http — for remote / web clients (binds 127.0.0.1:8765 by default)
hippocampai-mcp --transport streamable-http --host 0.0.0.0 --port 8765

# sse — legacy HTTP transport for older clients
hippocampai-mcp --transport sse --port 8765
```

The HTTP default port is **8765** to avoid clashing with the REST API (8000). The server
starts instantly and connects to Qdrant lazily on the first tool call, so launching it
never blocks on infrastructure.

## Configuration via environment

The `env`/environment configures the underlying `MemoryClient` exactly like a normal
HippocampAI deployment — see [Configuration](CONFIGURATION.md) for all variables. At
minimum set `QDRANT_URL`; set an LLM provider if you want extraction/enrichment on write:

```bash
QDRANT_URL=http://localhost:6333
LLM_PROVIDER=groq
LLM_MODEL=llama-3.1-8b-instant
GROQ_API_KEY=your-key-here
```

## Register with a client

### Claude Code

This repo ships a ready-to-use **`.mcp.json`** at the project root, so opening the
project in Claude Code auto-detects the `hippocampai` server. To use it elsewhere, drop
this into your project's `.mcp.json` (or the global Claude Code MCP config):

```jsonc
{
  "mcpServers": {
    "hippocampai": {
      "command": "hippocampai-mcp",
      "env": { "QDRANT_URL": "http://localhost:6333" }
    }
  }
}
```

### Claude Desktop

Add to `claude_desktop_config.json` (same shape):

```jsonc
{
  "mcpServers": {
    "hippocampai": {
      "command": "hippocampai-mcp",
      "env": { "QDRANT_URL": "http://localhost:6333" }
    }
  }
}
```

### Cursor / Windsurf / Zed / Continue

Any host supporting stdio MCP servers uses the same `command` + `env` shape; consult
your host's MCP docs for where its config file lives.

### Remote / web clients (HTTP)

Start the server with an HTTP transport and point the client at the endpoint:

```bash
hippocampai-mcp --transport streamable-http --host 0.0.0.0 --port 8765
# endpoint: http://<host>:8765/mcp        (sse transport uses /sse)
```

```jsonc
{
  "mcpServers": {
    "hippocampai": { "url": "http://your-host:8765/mcp" }
  }
}
```

> When exposing an HTTP transport beyond localhost, put it behind TLS and
> authentication (e.g. a reverse proxy) — the server itself does not authenticate
> callers.

### Any MCP SDK (programmatic)

Because it is a standard server, any MCP client SDK can connect — see the
[verification snippet](#verify-it-works) below for a Python example.

## Tools

The server exposes five tools wrapping the core `MemoryClient` operations:

| Tool | Purpose | Key arguments |
|------|---------|---------------|
| `remember` | Store a memory for a user | `text`, `user_id` (required); `session_id`, `type`, `importance`, `tags` |
| `recall` | Hybrid search for relevant memories | `query`, `user_id` (required); `k`, `session_id` |
| `assemble_context` | Build a token-budgeted context pack for a prompt | `query`, `user_id` (required); `token_budget`, `max_items`, `session_id` |
| `get_memory` | Fetch a single memory by id | `memory_id` |
| `delete_memory` | Delete a memory by id | `memory_id`; `user_id` |

### Tool details

**`remember(text, user_id, session_id=None, type="fact", importance=None, tags=None)`**
Stores content and returns the created memory (including its generated `id`). `type`
is one of `fact`, `preference`, `goal`, `habit`, `event`, `context`. Passing an
explicit `type` skips the LLM classification step.

**`recall(query, user_id, k=5, session_id=None)`**
Runs hybrid retrieval (vector + BM25 + optional graph, with reranking) and returns a
ranked list of memories, each annotated with a relevance `score`.

**`assemble_context(query, user_id, token_budget=4000, max_items=20, session_id=None)`**
Retrieves, dedupes, and fits memories to a token budget, returning `context` (ready to
drop into a prompt), `citations` (memory ids), and `num_items` selected.

**`get_memory(memory_id)`** — returns the memory as an object, or `{}` if not found.

**`delete_memory(memory_id, user_id=None)`** — returns `{"deleted": true|false, "memory_id": ...}`.

## Verify it works

A quick round-trip against a running Qdrant:

```python
import asyncio, os, json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    params = StdioServerParameters(
        command="hippocampai-mcp",
        env={**os.environ, "QDRANT_URL": "http://localhost:6333"},
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            print("tools:", [t.name for t in tools.tools])
            await session.call_tool("remember", {
                "text": "I prefer dark roast coffee", "user_id": "alice", "type": "preference"})
            res = await session.call_tool("recall", {"query": "coffee", "user_id": "alice", "k": 3})
            print(res.structuredContent["result"])

asyncio.run(main())
```

> List-returning tools (`recall`) deliver their array under
> `result.structuredContent["result"]`; object-returning tools return the object
> directly in `structuredContent`.

## Troubleshooting

- **`ImportError: requires the 'mcp' package`** — install the extra: `pip install "hippocampai[mcp]"`.
- **Tool calls hang or error on first use** — the client connects to Qdrant lazily;
  ensure `QDRANT_URL` is reachable from the server's environment.
- **Writes are slow / rate-limited** — `remember` runs LLM enrichment by default. Pass
  an explicit `type` to skip classification, or point `LLM_PROVIDER` at a self-hosted
  model. See [Providers](PROVIDERS.md).

## See also

- [Configuration](CONFIGURATION.md) — all environment variables
- [Providers](PROVIDERS.md) — LLM provider setup
- [Context Assembly](context_assembly.md) — the engine behind `assemble_context`
