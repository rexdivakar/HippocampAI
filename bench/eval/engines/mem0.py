"""Mem0 engine adapter for the workbench harness.

Wraps `Mem0 <https://docs.mem0.ai/>`_'s ``Memory`` class so the same harness can
score it on the same dataset HippocampAI is scored on.

Install: ``pip install mem0ai``.  Configuration is via Mem0's own env vars
(``OPENAI_API_KEY`` etc.) -- see https://docs.mem0.ai/configuration.

Caveats:

- Mem0's ``add()`` may extract zero memories from a single short turn (its LLM
  decides what's worth storing). When that happens, the adapter records a
  synthetic placeholder id so the harness can still track the source message --
  retrieval metrics for those questions will be 0 (the gold evidence won't match
  any actual stored memory), which is the *correct* honest score.
- Mem0's score scale is similarity-based (0..1). Used as-is.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from typing import Any


@dataclass
class _Mem:
    id: str
    text: str


@dataclass
class _Result:
    memory: _Mem
    score: float


class Mem0Adapter:
    """Map Mem0's ``Memory`` to the harness's ``remember`` / ``recall`` shape."""

    def __init__(self, client: Any) -> None:
        self.client = client

    def remember(
        self,
        text: str,
        user_id: str,
        session_id: str | None = None,
        type: str = "fact",
        **_: Any,
    ) -> _Mem:
        # Mem0's add() takes either a string or a messages list. The list form is
        # the documented happy path; some versions also accept a string.
        try:
            result = self.client.add(
                messages=[{"role": "user", "content": text}], user_id=user_id
            )
        except TypeError:
            result = self.client.add(text, user_id=user_id)

        # Newer SDK returns {"results": [{"id": ..., "memory": ..., "event": ...}, ...]}
        items: list[dict[str, Any]] = []
        if isinstance(result, dict) and isinstance(result.get("results"), list):
            items = result["results"]
        elif isinstance(result, list):
            items = result

        if items:
            first = items[0]
            mem_id = str(first.get("id") or first.get("memory_id") or uuid.uuid4())
            mem_text = str(first.get("memory") or first.get("text") or text)
            return _Mem(id=mem_id, text=mem_text)

        # Mem0 didn't extract anything from this turn -- record a synthetic id so
        # the source-message tracking in the harness still works (retrieval
        # metrics for evidence pointing here will correctly score 0).
        return _Mem(id=f"mem0:skipped:{uuid.uuid4()}", text=text)

    def recall(
        self,
        query: str,
        user_id: str,
        k: int = 5,
        session_id: str | None = None,
        **_: Any,
    ) -> list[_Result]:
        result = self.client.search(query=query, user_id=user_id, limit=k)
        items: list[dict[str, Any]] = []
        if isinstance(result, dict) and isinstance(result.get("results"), list):
            items = result["results"]
        elif isinstance(result, list):
            items = result

        out: list[_Result] = []
        for r in items:
            mem_id = str(r.get("id") or r.get("memory_id") or uuid.uuid4())
            mem_text = str(r.get("memory") or r.get("text") or "")
            score = float(r.get("score", 0.0) or 0.0)
            out.append(_Result(memory=_Mem(id=mem_id, text=mem_text), score=score))
        return out


def build() -> Mem0Adapter:
    """Construct a Mem0 ``Memory`` and wrap it for the harness."""
    try:
        from mem0 import Memory
    except ImportError as exc:  # pragma: no cover - guarded by optional install
        raise ImportError(
            "The Mem0 engine adapter requires the 'mem0ai' package. Install it "
            "with: pip install mem0ai. Then configure provider keys per "
            "https://docs.mem0.ai/configuration (e.g. OPENAI_API_KEY)."
        ) from exc

    # Honor an optional config dict if MEM0_CONFIG_JSON is set.
    cfg_json = os.getenv("MEM0_CONFIG_JSON")
    if cfg_json:
        import json

        client = Memory.from_config(json.loads(cfg_json))
    else:
        client = Memory()
    return Mem0Adapter(client=client)
