"""HippocampAI MCP server (stdio transport).

Wraps a single :class:`hippocampai.MemoryClient` and exposes its core operations
as MCP tools. The client is created lazily on first tool call so the server
process starts instantly and only connects to Qdrant when actually used.
"""

from __future__ import annotations

from typing import Any, Optional

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as exc:  # pragma: no cover - guarded by optional extra
    raise ImportError(
        "The MCP server requires the 'mcp' package. Install it with: "
        "pip install hippocampai[mcp]"
    ) from exc

mcp = FastMCP("hippocampai")

_client: Any = None


def get_client() -> Any:
    """Return a lazily-initialized MemoryClient (reused across calls)."""
    global _client
    if _client is None:
        from hippocampai import MemoryClient

        _client = MemoryClient()
    return _client


def _memory_to_dict(memory: Any) -> dict[str, Any]:
    """Serialize a Memory model to a compact JSON-safe dict."""
    if memory is None:
        return {}
    if hasattr(memory, "model_dump"):
        data = memory.model_dump(mode="json")
        return {
            "id": data.get("id"),
            "text": data.get("text"),
            "type": data.get("type"),
            "user_id": data.get("user_id"),
            "session_id": data.get("session_id"),
            "importance": data.get("importance"),
            "tags": data.get("tags"),
            "created_at": data.get("created_at"),
        }
    return {"id": getattr(memory, "id", None), "text": getattr(memory, "text", None)}


@mcp.tool()
def remember(
    text: str,
    user_id: str,
    session_id: Optional[str] = None,
    type: str = "fact",
    importance: Optional[float] = None,
    tags: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Store a memory for a user.

    Args:
        text: The content to remember.
        user_id: Identifier of the user this memory belongs to.
        session_id: Optional conversation/session identifier.
        type: Memory type: fact, preference, goal, habit, event, or context.
        importance: Optional importance score 0-10 (auto-scored if omitted).
        tags: Optional list of tags for categorization.

    Returns:
        The stored memory as a dict (including its generated id).
    """
    memory = get_client().remember(
        text=text,
        user_id=user_id,
        session_id=session_id,
        type=type,
        importance=importance,
        tags=tags,
    )
    return _memory_to_dict(memory)


@mcp.tool()
def recall(
    query: str,
    user_id: str,
    k: int = 5,
    session_id: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Retrieve the most relevant memories for a query (hybrid search).

    Args:
        query: Natural-language query to search memories with.
        user_id: Identifier of the user whose memories to search.
        k: Number of memories to return (default 5).
        session_id: Optional session scope.

    Returns:
        A ranked list of memories, each with a relevance ``score``.
    """
    results = get_client().recall(query=query, user_id=user_id, k=k, session_id=session_id)
    out: list[dict[str, Any]] = []
    for r in results:
        item = _memory_to_dict(r.memory)
        item["score"] = r.score
        out.append(item)
    return out


@mcp.tool()
def assemble_context(
    query: str,
    user_id: str,
    token_budget: int = 4000,
    max_items: int = 20,
    session_id: Optional[str] = None,
) -> dict[str, Any]:
    """Assemble a ready-to-use context pack for a prompt, fit to a token budget.

    Args:
        query: Query to gather relevant context for.
        user_id: Identifier of the user.
        token_budget: Maximum tokens of context to return (default 4000).
        max_items: Maximum number of memory items (default 20).
        session_id: Optional session scope.

    Returns:
        A dict with ``context`` (the assembled text), ``citations`` (memory ids),
        and ``num_items`` selected.
    """
    pack = get_client().assemble_context(
        query=query,
        user_id=user_id,
        session_id=session_id,
        token_budget=token_budget,
        max_items=max_items,
    )
    return {
        "context": getattr(pack, "final_context_text", ""),
        "citations": list(getattr(pack, "citations", []) or []),
        "num_items": len(getattr(pack, "selected_items", []) or []),
    }


@mcp.tool()
def get_memory(memory_id: str) -> dict[str, Any]:
    """Fetch a single memory by its id. Returns an empty object if not found."""
    return _memory_to_dict(get_client().get_memory(memory_id))


@mcp.tool()
def delete_memory(memory_id: str, user_id: Optional[str] = None) -> dict[str, Any]:
    """Delete a memory by id. Returns ``{"deleted": true|false}``."""
    deleted = get_client().delete_memory(memory_id, user_id=user_id)
    return {"deleted": bool(deleted), "memory_id": memory_id}


def main() -> None:
    """Run the MCP server.

    Defaults to stdio (the zero-config transport that local MCP hosts such as Claude
    Code, Claude Desktop, Cursor, Windsurf and Zed launch automatically). Pass
    ``--transport streamable-http`` or ``--transport sse`` to serve remote/web clients
    over HTTP.
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="hippocampai-mcp", description="HippocampAI Model Context Protocol server"
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http", "sse"],
        default="stdio",
        help="MCP transport. stdio (default) for local IDE/agent hosts; "
        "streamable-http or sse for remote/web clients.",
    )
    parser.add_argument(
        "--host", default="127.0.0.1", help="Bind host for HTTP transports (default 127.0.0.1)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Bind port for HTTP transports (default 8765; the REST API uses 8000)",
    )
    args = parser.parse_args()

    if args.transport != "stdio":
        mcp.settings.host = args.host
        mcp.settings.port = args.port

    mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()
