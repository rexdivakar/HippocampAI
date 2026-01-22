"""Test configuration for HippocampAI."""

from __future__ import annotations

import sys
from pathlib import Path
from uuid import uuid4

import pytest
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

# Ensure the project root does not shadow third-party packages like qdrant_client.
while str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))

if str(SRC) not in sys.path:
    sys.path.append(str(SRC))


@pytest.fixture(scope="session", autouse=True)
def ensure_qdrant_collections():
    """Ensure test collections exist before any memory tests run."""
    import os

    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    # Parse host and port from URL
    if qdrant_url.startswith("http://"):
        qdrant_url = qdrant_url[7:]
    elif qdrant_url.startswith("https://"):
        qdrant_url = qdrant_url[8:]

    if ":" in qdrant_url:
        host, port = qdrant_url.split(":")
        port = int(port)
    else:
        host = qdrant_url
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
