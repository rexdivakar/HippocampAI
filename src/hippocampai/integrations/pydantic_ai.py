"""Pydantic AI integration: typed memory tools for Pydantic AI agents.

Pydantic AI agents declare a dependency type (``deps_type``) and tool functions
that receive a ``RunContext[Deps]``. This adapter provides:

- :class:`HippocampDeps`: a typed dataclass holding a HippocampAI client and the
  current ``user_id``, designed to be the agent's ``deps_type``.
- :func:`register_memory_tools`: a one-call helper that registers ``remember``,
  ``recall``, and ``assemble_context`` tools on an existing ``Agent``.

Install Pydantic AI yourself: ``pip install pydantic-ai``. Imports are lazy.

Example:

    from pydantic_ai import Agent
    from hippocampai import MemoryClient
    from hippocampai.integrations.pydantic_ai import HippocampDeps, register_memory_tools

    agent = Agent("groq:llama-3.1-8b-instant", deps_type=HippocampDeps)
    register_memory_tools(agent)

    result = await agent.run(
        "What do I prefer for coffee?",
        deps=HippocampDeps(client=MemoryClient(), user_id="alice"),
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


def _require_pydantic_ai() -> None:
    try:
        import pydantic_ai  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "The Pydantic AI adapter requires the 'pydantic-ai' package. Install "
            "it with: pip install pydantic-ai"
        ) from exc


@dataclass
class HippocampDeps:
    """Typed deps for a Pydantic AI agent: a memory client and the active user.

    Pass an instance via ``deps=`` to ``agent.run``/``agent.run_sync``; the
    registered tools read both fields from the run context.
    """

    client: Any
    user_id: str


def register_memory_tools(agent: Any) -> None:
    """Register ``remember`` / ``recall`` / ``assemble_context`` tools on ``agent``.

    The agent must be constructed with ``deps_type=HippocampDeps`` (or a subclass).
    Tools are typed via Pydantic AI's standard tool decorator so they show up as
    proper tool calls to the underlying model.
    """
    _require_pydantic_ai()
    from pydantic_ai import RunContext

    @agent.tool
    async def remember(
        ctx: RunContext[HippocampDeps],
        text: str,
        type: str = "fact",
        importance: Optional[float] = None,
    ) -> str:
        """Store a memory for the current user. Returns the new memory id."""
        memory = ctx.deps.client.remember(
            text=text, user_id=ctx.deps.user_id, type=type, importance=importance
        )
        return str(memory.id)

    @agent.tool
    async def recall(
        ctx: RunContext[HippocampDeps], query: str, k: int = 5
    ) -> list[dict[str, Any]]:
        """Retrieve the top-k relevant memories for the current user."""
        results = ctx.deps.client.recall(
            query=query, user_id=ctx.deps.user_id, k=k
        )
        return [
            {"id": r.memory.id, "text": r.memory.text, "score": r.score}
            for r in results
        ]

    @agent.tool
    async def assemble_context(
        ctx: RunContext[HippocampDeps], query: str, token_budget: int = 2000
    ) -> str:
        """Return a token-budgeted context string for the query."""
        pack = ctx.deps.client.assemble_context(
            query=query, user_id=ctx.deps.user_id, token_budget=token_budget
        )
        return getattr(pack, "final_context_text", "") or ""
