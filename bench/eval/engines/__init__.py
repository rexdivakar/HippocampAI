"""Pluggable engine adapters for the HippocampAI workbench harness.

The harness drives any object that exposes two methods:

    remember(text, user_id, session_id=None, type="fact", **kwargs) -> memory
        memory.id   : str
        memory.text : str

    recall(query, user_id, k=5, session_id=None) -> list[result]
        result.memory.id   : str
        result.memory.text : str
        result.score       : float

That's it. Each adapter in this package wraps a third-party engine to fit that
shape so the **same harness, same dataset, same metrics** can score every engine.

Registered engines (lazy-imported -- nothing breaks if the SDK isn't installed):

- ``hippocampai`` -- the local HippocampAI ``MemoryClient`` (default)
- ``mem0`` -- Mem0's ``Memory`` class via ``pip install mem0ai``
- ``zep`` -- Zep Cloud via ``pip install zep-cloud`` (set ``ZEP_API_KEY``)
"""

from __future__ import annotations

from typing import Any, Callable


def _build_hippocampai() -> Any:
    from bench.eval.engines.hippocampai import build as build_fn

    return build_fn()


def _build_mem0() -> Any:
    from bench.eval.engines.mem0 import build as build_fn

    return build_fn()


def _build_zep() -> Any:
    from bench.eval.engines.zep import build as build_fn

    return build_fn()


# Engine name -> zero-arg factory. Add a row to register a new engine.
ENGINES: dict[str, Callable[[], Any]] = {
    "hippocampai": _build_hippocampai,
    "mem0": _build_mem0,
    "zep": _build_zep,
}


def build_engine(name: str) -> Any:
    """Build a harness-compatible client for the named engine."""
    name = name.lower()
    if name not in ENGINES:
        raise ValueError(
            f"Unknown engine '{name}'. Choices: {sorted(ENGINES)}. "
            f"Add a new one in bench/eval/engines/."
        )
    return ENGINES[name]()
