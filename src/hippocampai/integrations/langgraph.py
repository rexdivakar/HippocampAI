"""LangGraph integration: long-term memory across a LangGraph state graph.

LangGraph runs short-lived workflows on a per-thread state; durable knowledge that
must survive across threads/runs belongs in long-term memory. This adapter exposes
HippocampAI as both a callable node-helper and a tool factory you can bind to a
LangChain ``ChatModel`` used inside a graph.

Install LangGraph yourself: ``pip install langgraph langchain-core``. Imports are
lazy.

Example -- inject memory into a graph node:

    from hippocampai import MemoryClient
    from hippocampai.integrations.langgraph import HippocampAIMemory

    mem = HippocampAIMemory(client=MemoryClient(), user_id="alice")

    def planner(state: dict) -> dict:
        context = mem.context_for(state["question"], token_budget=2000)
        return {"memory_context": context.text, **state}

Example -- expose memory as tools to a graph-bound LLM:

    tools = mem.tools()         # remember, recall, assemble_context
    llm_with_tools = llm.bind_tools(tools)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


def _require_langchain_core() -> None:
    try:
        import langchain_core  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "The LangGraph adapter requires 'langchain-core' (LangGraph itself is "
            "optional unless you bind tools). Install: pip install langchain-core"
        ) from exc


@dataclass
class HippocampContext:
    """Token-budgeted context pack returned by :meth:`HippocampAIMemory.context_for`."""

    text: str
    citations: list[str]
    num_items: int


class HippocampAIMemory:
    """LangGraph-friendly façade over a :class:`~hippocampai.MemoryClient`.

    Provides three call styles that fit naturally in a graph:

    - :meth:`context_for(query, token_budget=...)` -- build a context pack to drop
      into a node's state (the most common pattern in graphs).
    - :meth:`tools()` -- a list of LangChain ``StructuredTool``s an LLM bound into
      a graph node can call (remember, recall, assemble_context).
    - :meth:`remember` / :meth:`recall` -- direct passthroughs for in-node use.
    """

    def __init__(self, client: Any, user_id: str) -> None:
        self.client = client
        self.user_id = user_id

    # ---- direct ops -----------------------------------------------------------

    def remember(self, text: str, **kwargs: Any) -> Any:
        return self.client.remember(text=text, user_id=self.user_id, **kwargs)

    def recall(self, query: str, k: int = 5, **kwargs: Any) -> list[Any]:
        return self.client.recall(query=query, user_id=self.user_id, k=k, **kwargs)

    def context_for(
        self,
        query: str,
        token_budget: int = 2000,
        max_items: int = 10,
        session_id: Optional[str] = None,
    ) -> HippocampContext:
        """Build a token-budgeted context pack for a query (ideal for graph nodes)."""
        pack = self.client.assemble_context(
            query=query,
            user_id=self.user_id,
            token_budget=token_budget,
            max_items=max_items,
            session_id=session_id,
        )
        return HippocampContext(
            text=getattr(pack, "final_context_text", ""),
            citations=list(getattr(pack, "citations", []) or []),
            num_items=len(getattr(pack, "selected_items", []) or []),
        )

    # ---- LangChain tool factory ----------------------------------------------

    def tools(self) -> list[Any]:
        """Return ``StructuredTool``s an LLM-bound graph node can call."""
        _require_langchain_core()
        from langchain_core.tools import StructuredTool

        def _remember(text: str, type: str = "fact") -> str:
            m = self.remember(text=text, type=type)
            return f"stored: {m.id}"

        def _recall(query: str, k: int = 5) -> str:
            results = self.recall(query=query, k=k)
            if not results:
                return "no memories found"
            return "\n".join(f"- {r.memory.text} (score={r.score:.2f})" for r in results)

        def _assemble_context(query: str, token_budget: int = 2000) -> str:
            return self.context_for(query, token_budget=token_budget).text or "no context"

        return [
            StructuredTool.from_function(_remember, name="remember"),
            StructuredTool.from_function(_recall, name="recall"),
            StructuredTool.from_function(_assemble_context, name="assemble_context"),
        ]
