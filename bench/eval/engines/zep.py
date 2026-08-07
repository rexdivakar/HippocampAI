"""Zep engine adapter for the workbench harness.

Wraps `Zep <https://www.getzep.com/>`_'s Cloud client so the same harness scores
it on the same dataset.

Install: ``pip install zep-cloud``.  Set ``ZEP_API_KEY`` in the environment.

Caveats:

- Zep's data model is User -> Session -> Message. The harness's ``user_id`` maps
  to a Zep user; ``session_id`` (or a deterministic per-user default) maps to a
  Zep session. The adapter creates both lazily.
- Search returns either ``Message`` results (raw turns) or ``MemorySearchResult``
  (summaries/facts depending on the Zep tier). The adapter prefers the message
  representation; if only a summary is present, that's what we score against.
- Zep extracts entities/facts asynchronously. Brand-new memories may not be
  fully indexed for a short window. The harness ingests then queries within the
  same sample, so very fresh writes occasionally miss; if you see lower-than-
  expected recall, retry with a small per-sample sleep (``ZEP_INDEX_DELAY_S``).
"""

from __future__ import annotations

import os
import time
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


class ZepAdapter:
    """Map Zep's Cloud client to the harness's ``remember`` / ``recall`` shape."""

    def __init__(self, client: Any, index_delay_s: float = 0.0) -> None:
        self.client = client
        self.index_delay_s = index_delay_s
        self._users_seen: set[str] = set()
        self._sessions_seen: set[str] = set()

    # ---- internal helpers ----

    def _ensure_user(self, user_id: str) -> None:
        if user_id in self._users_seen:
            return
        try:
            self.client.user.add(user_id=user_id)
        except Exception:  # noqa: BLE001 - already-exists / version drift
            pass
        self._users_seen.add(user_id)

    def _ensure_session(self, user_id: str, session_id: str) -> None:
        if session_id in self._sessions_seen:
            return
        try:
            self.client.memory.add_session(session_id=session_id, user_id=user_id)
        except Exception:  # noqa: BLE001
            pass
        self._sessions_seen.add(session_id)

    # ---- harness contract ----

    def remember(
        self,
        text: str,
        user_id: str,
        session_id: str | None = None,
        type: str = "fact",
        **_: Any,
    ) -> _Mem:
        self._ensure_user(user_id)
        sid = session_id or f"{user_id}__default"
        self._ensure_session(user_id, sid)
        try:
            # zep-cloud Message takes content + role + role_type
            response = self.client.memory.add(
                session_id=sid,
                messages=[{"role": "user", "role_type": "user", "content": text}],
            )
        except Exception:  # noqa: BLE001 - surface as zero-id placeholder
            return _Mem(id=f"zep:error:{uuid.uuid4()}", text=text)

        # Zep returns either a message UUID directly or a list of messages
        mem_id: str | None = None
        if hasattr(response, "uuid"):
            mem_id = str(response.uuid)
        elif isinstance(response, dict):
            mem_id = str(response.get("uuid") or response.get("message_uuid") or "")
        elif isinstance(response, list) and response:
            first = response[0]
            mem_id = str(getattr(first, "uuid", None) or "")
        if not mem_id:
            mem_id = f"zep:added:{uuid.uuid4()}"
        if self.index_delay_s:
            time.sleep(self.index_delay_s)
        return _Mem(id=mem_id, text=text)

    def recall(
        self,
        query: str,
        user_id: str,
        k: int = 5,
        session_id: str | None = None,
        **_: Any,
    ) -> list[_Result]:
        # Prefer the new user-scoped search; fall back to per-session search.
        results: list[Any] = []
        try:
            graph_search = getattr(self.client.memory, "search_sessions", None)
            if callable(graph_search):
                resp = graph_search(user_id=user_id, text=query, limit=k)
                results = list(getattr(resp, "results", None) or resp or [])
            else:
                raise AttributeError("no search_sessions")
        except Exception:  # noqa: BLE001
            sid = session_id or f"{user_id}__default"
            try:
                resp = self.client.memory.search(session_id=sid, text=query, limit=k)
                results = list(getattr(resp, "results", None) or resp or [])
            except Exception:  # noqa: BLE001
                return []

        out: list[_Result] = []
        for r in results:
            message = getattr(r, "message", None)
            summary = getattr(r, "summary", None)
            fact = getattr(r, "fact", None)
            mem_text = (
                getattr(message, "content", None)
                or getattr(summary, "content", None)
                or getattr(fact, "fact", None)
                or ""
            )
            mem_id = (
                getattr(message, "uuid", None)
                or getattr(summary, "uuid", None)
                or getattr(fact, "uuid", None)
                or uuid.uuid4()
            )
            score = float(getattr(r, "score", 0.0) or 0.0)
            out.append(_Result(memory=_Mem(id=str(mem_id), text=str(mem_text)), score=score))
        return out


def build() -> ZepAdapter:
    """Construct a Zep Cloud client and wrap it for the harness."""
    try:
        from zep_cloud.client import Zep
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "The Zep engine adapter requires the 'zep-cloud' package. Install it "
            "with: pip install zep-cloud. Then set ZEP_API_KEY in the environment."
        ) from exc

    api_key = os.getenv("ZEP_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ZEP_API_KEY is not set. Get one at https://app.getzep.com/ and export it."
        )
    delay = float(os.getenv("ZEP_INDEX_DELAY_S", "0") or 0.0)
    return ZepAdapter(client=Zep(api_key=api_key), index_delay_s=delay)
