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


# CI sets this to something like "ci-<run_id>-<python-version>" so concurrent
# matrix jobs against the same Qdrant Cloud cluster never share collections.
# Empty locally, which keeps existing local collection names unchanged.
QDRANT_TEST_NAMESPACE = os.getenv("QDRANT_TEST_NAMESPACE", "")


def namespaced_collection(name: str) -> str:
    """Prefix a fixed test collection name with the CI namespace, if any.

    Only use this for collections with a fixed/shared name across test runs
    (e.g. "test_facts_advanced"). Collections already suffixed with a random
    id (e.g. via uuid4()) are unique per test and don't need this.
    """
    return f"{QDRANT_TEST_NAMESPACE}_{name}" if QDRANT_TEST_NAMESPACE else name


@pytest.fixture(scope="session", autouse=True)
def ensure_qdrant_collections():
    """Ensure test collections exist before any memory tests run."""
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")

    test_collections = [
        (namespaced_collection("test_facts_advanced"), 384),
        (namespaced_collection("test_prefs_advanced"), 384),
    ]

    try:
        client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, timeout=5)

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
        return

    yield

    # Clean up the collections this fixture created so they don't linger on
    # a shared Qdrant Cloud cluster across CI runs.
    try:
        for collection_name, _ in test_collections:
            client.delete_collection(collection_name=collection_name)
    except Exception:
        pass


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
