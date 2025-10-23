"""Test configuration for HippocampAI."""

from __future__ import annotations

import sys
from pathlib import Path
import pytest
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

# Ensure the project root does not shadow third-party packages like qdrant_client.
while str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))

if str(SRC) not in sys.path:
    sys.path.append(str(SRC))


@pytest.fixture
def memory_client():
    """Create a MemoryClient instance for testing."""
    from hippocampai import MemoryClient
    test_id = uuid4().hex[:8]
    return MemoryClient(
        collection_facts=f"test_facts_{test_id}",
        collection_prefs=f"test_prefs_{test_id}"
    )


@pytest.fixture
def user_id():
    """Generate a unique user ID for testing."""
    return f"test_user_{uuid4().hex[:8]}"
