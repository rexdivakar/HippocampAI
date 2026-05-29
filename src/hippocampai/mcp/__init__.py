"""Model Context Protocol (MCP) server for HippocampAI.

Exposes the memory engine as MCP tools so agents (Claude Code, Cursor, etc.) can
store and retrieve long-term memory with zero glue code.

Run it over stdio (the default transport, zero-config for Claude Code/Cursor):

    python -m hippocampai.mcp

Requires the optional ``mcp`` extra: ``pip install hippocampai[mcp]``.
"""

from hippocampai.mcp.server import main, mcp

__all__ = ["main", "mcp"]
