from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if SRC_PATH.exists():
    sys.path.insert(0, str(SRC_PATH))


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


import pytest  # noqa: E402


@pytest.mark.parametrize(
    "module_name, message",
    [
        ("hippocampai", None),
        ("hippocampai.models.memory", None),
    ],
)
def test_core_imports(module_name: str, message: str | None) -> None:
    try:
        __import__(module_name)
    except ModuleNotFoundError as exc:  # pragma: no cover
        pytest.fail(message or str(exc))


OPTIONAL_DEPENDENCIES: List[tuple[str, str]] = [
    ("hippocampai", "Install `hippocampai[core]` for full functionality"),
]


# Only run MemoryClient import if optional deps are present
@pytest.mark.skipif(
    not all(
        has_module(name)
        for name in ["cachetools", "qdrant_client.http", "sentence_transformers", "rank_bm25"]
    ),
    reason="Optional runtime dependencies missing",
)
def test_memory_client_import() -> None:
    from hippocampai import MemoryClient

    assert MemoryClient is not None
