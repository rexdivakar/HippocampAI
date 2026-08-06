"""HippocampAI engine adapter -- the reference / default for the workbench harness.

The ``MemoryClient`` already exposes ``remember(text, user_id, ...)`` returning a
``Memory`` with ``.id``/``.text`` and ``recall(query, user_id, k=)`` returning a
list of ``RetrievalResult`` with ``.memory`` and ``.score``. No translation needed
-- this module just constructs the client.
"""

from __future__ import annotations

from typing import Any


def build() -> Any:
    """Return a configured :class:`hippocampai.MemoryClient`."""
    from hippocampai import MemoryClient

    return MemoryClient()
