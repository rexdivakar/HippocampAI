"""CrewAI integration: persistent long-term memory for CrewAI crews.

CrewAI's memory system supports pluggable storage. This adapter exposes
HippocampAI's :class:`~hippocampai.MemoryClient` as a CrewAI-compatible storage
backend so an entire crew can share long-term memory across runs.

Install CrewAI yourself: ``pip install crewai``. This module only imports it when
the adapter classes are instantiated, so the rest of the package stays usable
without CrewAI installed.

Example:

    from hippocampai import MemoryClient
    from hippocampai.integrations.crewai import HippocampAIMemoryStorage
    from crewai.memory import LongTermMemory

    client = MemoryClient()
    storage = HippocampAIMemoryStorage(client=client, user_id="alice")
    long_term = LongTermMemory(storage=storage)
"""

from __future__ import annotations

from typing import Any, Optional


def _require_crewai() -> None:
    try:
        import crewai  # noqa: F401
    except ImportError as exc:  # pragma: no cover - guarded by optional install
        raise ImportError(
            "The CrewAI adapter requires the 'crewai' package. Install it with: "
            "pip install crewai"
        ) from exc


class HippocampAIMemoryStorage:
    """CrewAI-compatible storage backend backed by HippocampAI.

    Implements the ``save(value, metadata, agent=None) / search(query, ...) /
    reset()`` contract CrewAI's memory classes expect. Each saved value becomes a
    HippocampAI memory under the configured ``user_id``; ``search`` issues a
    hybrid recall and returns CrewAI-shaped result dicts.

    Args:
        client: A :class:`~hippocampai.MemoryClient`.
        user_id: Identifier under which all crew memories are stored.
        memory_type: Default engine type for stored memories (``fact`` is the
            sensible default for crew-level long-term memory).
    """

    def __init__(
        self, client: Any, user_id: str, memory_type: str = "fact"
    ) -> None:
        _require_crewai()
        self.client = client
        self.user_id = user_id
        self.memory_type = memory_type

    def save(
        self,
        value: Any,
        metadata: Optional[dict[str, Any]] = None,
        agent: Optional[str] = None,
    ) -> None:
        """Store a crew memory."""
        text = value if isinstance(value, str) else str(value)
        meta = metadata or {}
        tags = list(meta.get("tags", []))
        if agent:
            tags.append(f"agent:{agent}")
        self.client.remember(
            text=text,
            user_id=self.user_id,
            type=meta.get("type", self.memory_type),
            tags=tags or None,
            session_id=meta.get("session_id"),
            agent_id=agent,
        )

    def search(
        self,
        query: str,
        limit: int = 5,
        score_threshold: float = 0.0,
        **_: Any,
    ) -> list[dict[str, Any]]:
        """Retrieve crew memories. Returns the CrewAI-shaped result list."""
        results = self.client.recall(query=query, user_id=self.user_id, k=limit)
        out: list[dict[str, Any]] = []
        for r in results:
            if r.score < score_threshold:
                continue
            out.append({
                "context": r.memory.text,
                "metadata": {
                    "id": r.memory.id,
                    "type": getattr(r.memory.type, "value", str(r.memory.type)),
                    "tags": r.memory.tags,
                    "created_at": r.memory.created_at.isoformat() if r.memory.created_at else None,
                },
                "score": r.score,
            })
        return out

    def reset(self) -> None:
        """Reset is a no-op: HippocampAI memories are durable. Use the engine's
        per-user deletion tools (``client.delete_memories(...)``) if you really
        want to wipe state -- silently dropping crew memory here would be a
        footgun.
        """
        return None
