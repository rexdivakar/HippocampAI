"""Test configuration for HippocampAI."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from uuid import uuid4

import pytest
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

# Load .env into os.environ so os.getenv() picks up QDRANT_URL and other vars.
# This must happen before any hippocampai imports that read os.environ directly.
_env_file = ROOT / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _key, _, _val = _line.partition("=")
            _key = _key.strip()
            # Strip inline comments (e.g. "300  # seconds" → "300")
            _val = _val.split("#")[0].strip()
            # Only set if not already present (os.environ takes precedence)
            if _key and _key not in os.environ:
                os.environ[_key] = _val

# Ensure the project root does not shadow third-party packages like qdrant_client.
while str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))

if str(SRC) not in sys.path:
    sys.path.append(str(SRC))


@pytest.fixture(scope="session", autouse=True)
def ensure_qdrant_collections():
    """Ensure test collections exist before any memory tests run."""
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    # Parse host and port from URL
    stripped = qdrant_url
    if stripped.startswith("http://"):
        stripped = stripped[7:]
    elif stripped.startswith("https://"):
        stripped = stripped[8:]

    if ":" in stripped:
        host, port_str = stripped.split(":", 1)
        port = int(port_str)
    else:
        host = stripped
        port = 6333

    try:
        client = QdrantClient(host=host, port=port, timeout=5)

        # Create both test collections needed for advanced features tests
        test_collections = [
            ("test_facts_advanced", 384),  # Default embed_dimension for BAAI/bge-small-en-v1.5
            ("test_prefs_advanced", 384),
        ]

        for collection_name, vector_size in test_collections:
            if not client.collection_exists(collection_name=collection_name):
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
                )
    except Exception as e:
        # Don't fail tests that don't need Qdrant
        import warnings

        warnings.warn(f"Could not connect to Qdrant: {e}")

    yield


@pytest.fixture
def memory_client():
    """Create a MemoryClient instance for testing."""
    from hippocampai import MemoryClient

    test_id = uuid4().hex[:8]
    return MemoryClient(
        collection_facts=f"test_facts_{test_id}", collection_prefs=f"test_prefs_{test_id}"
    )


@pytest.fixture
def user_id():
    """Generate a unique user ID for testing."""
    return f"test_user_{uuid4().hex[:8]}"
