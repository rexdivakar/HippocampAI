"""AutoGen integration: shared long-term memory for multi-agent conversations.

AutoGen agents (``ConversableAgent``, ``AssistantAgent``, ``UserProxyAgent``)
exchange messages but have no built-in persistent memory across runs. This
adapter exposes HippocampAI as a shared memory store and as registerable tool
functions any AutoGen agent can call.

Install AutoGen yourself: ``pip install pyautogen`` (or ``autogen-agentchat`` for
the newer v0.4+ split). Imports are lazy.

Example -- register memory tools on an AssistantAgent:

    from autogen import AssistantAgent
    from hippocampai import MemoryClient
    from hippocampai.integrations.autogen import HippocampAIMemory, register_memory_tools

    mem = HippocampAIMemory(client=MemoryClient(), user_id="alice")
    assistant = AssistantAgent("planner", llm_config={...})
    register_memory_tools(assistant, mem)

Example -- enrich a message manually before sending:

    enriched = mem.enrich_message("plan my week", token_budget=1500)
    assistant.send(enriched, recipient=...)
"""

from __future__ import annotations

from typing import Any, Callable, Optional


def _require_autogen() -> None:
    """Accept either pyautogen (v0.2) or autogen-agentchat (v0.4)."""
    try:
        import autogen  # type: ignore  # noqa: F401
        return
    except ImportError:
        pass
    try:
        import autogen_agentchat  # type: ignore  # noqa: F401
        return
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "The AutoGen adapter requires 'pyautogen' (v0.2) or 'autogen-agentchat' "
            "(v0.4+). Install one of: pip install pyautogen / pip install autogen-agentchat"
        ) from exc


class HippocampAIMemory:
    """Shared memory for an AutoGen crew, scoped to one user."""

    def __init__(self, client: Any, user_id: str) -> None:
        self.client = client
        self.user_id = user_id

    def remember(self, text: str, **kwargs: Any) -> Any:
        return self.client.remember(text=text, user_id=self.user_id, **kwargs)

    def recall(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        results = self.client.recall(query=query, user_id=self.user_id, k=k)
        return [
            {"text": r.memory.text, "score": r.score, "id": r.memory.id}
            for r in results
        ]

    def enrich_message(
        self,
        message: str,
        token_budget: int = 1500,
        prefix: str = "Relevant memory:\n",
    ) -> str:
        """Prepend memory context to a message, fit to a token budget."""
        pack = self.client.assemble_context(
            query=message, user_id=self.user_id, token_budget=token_budget
        )
        context = getattr(pack, "final_context_text", "") or ""
        if not context.strip():
            return message
        return f"{prefix}{context}\n\n---\n\n{message}"


def register_memory_tools(
    agent: Any,
    memory: HippocampAIMemory,
    name_prefix: str = "",
) -> dict[str, Callable[..., Any]]:
    """Register ``remember`` / ``recall`` tools on an AutoGen agent.

    Uses the v0.2 ``register_function`` API when present, otherwise registers as
    callable attributes (matching the v0.4 agent contract). Returns the dict of
    bound callables so the caller can also expose them via UserProxyAgent's
    function_map.
    """
    _require_autogen()

    def remember(text: str, type: str = "fact") -> str:
        m = memory.remember(text=text, type=type)
        return f"stored memory {m.id}"

    def recall(query: str, k: int = 5) -> list[dict[str, Any]]:
        return memory.recall(query=query, k=k)

    tools = {
        f"{name_prefix}remember": remember,
        f"{name_prefix}recall": recall,
    }

    # v0.2 path: agent.register_function(function_map=...)
    register = getattr(agent, "register_function", None)
    if callable(register):
        register(function_map=tools)
        return tools

    # v0.4 path / generic: attach as attributes; caller wires into agent tools list
    for name, fn in tools.items():
        setattr(agent, name, fn)
    return tools


def shared_memory_factory(
    client: Any, user_id: str
) -> Callable[[Optional[str]], HippocampAIMemory]:
    """Factory that produces per-agent memory wrappers sharing the same user.

    Useful in multi-agent setups where each agent should write/read under one
    tenant identity but you want a per-agent handle for clarity.
    """
    def _make(_agent_name: Optional[str] = None) -> HippocampAIMemory:
        return HippocampAIMemory(client=client, user_id=user_id)

    return _make
