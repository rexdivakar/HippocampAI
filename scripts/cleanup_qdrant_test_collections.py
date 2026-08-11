#!/usr/bin/env python3
"""Delete ephemeral test collections from a Qdrant instance.

Run after the test suite (pass or fail) so the shared CI Qdrant Cloud
cluster doesn't accumulate one set of collections per run. Only deletes
collections matching known-safe test-only naming patterns; anything else
is left untouched. Never raises - cleanup must never fail the CI job.

Usage:
    python scripts/cleanup_qdrant_test_collections.py [--dry-run]

Env:
    QDRANT_URL, QDRANT_API_KEY  - required to connect
    QDRANT_TEST_NAMESPACE       - if set, also matches "<namespace>_*"
"""

import os
import sys

# Prefixes safe to delete unconditionally: only ever produced by test fixtures
# (see tests/conftest.py and tests/test_bitemporal.py), never by real usage.
SAFE_PREFIXES = ("test_facts_", "test_prefs_", "ci-")
# Fixed collection names test fixtures reuse across runs without a random
# suffix (e.g. BiTemporalStore's default). Only removed when a CI namespace
# is set, so a local dev run never nukes real local data.
CI_ONLY_FIXED_NAMES = ("hippocampai_bitemporal_facts",)


def main() -> int:
    dry_run = "--dry-run" in sys.argv

    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")
    namespace = os.getenv("QDRANT_TEST_NAMESPACE", "")

    if not url:
        print("QDRANT_URL not set, nothing to clean up.")
        return 0

    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(url=url, api_key=api_key, timeout=10)
        collections = [c.name for c in client.get_collections().collections]
    except Exception as e:
        print(f"Could not list Qdrant collections, skipping cleanup: {e}")
        return 0

    to_delete = [c for c in collections if c.startswith(SAFE_PREFIXES)]
    if namespace:
        to_delete += [c for c in collections if c.startswith(f"{namespace}_") and c not in to_delete]
        to_delete += [c for c in CI_ONLY_FIXED_NAMES if c in collections and c not in to_delete]

    if not to_delete:
        print("No stale test collections found.")
        return 0

    print(f"{'Would delete' if dry_run else 'Deleting'} {len(to_delete)} collection(s):")
    for name in to_delete:
        print(f"  - {name}")
        if not dry_run:
            try:
                client.delete_collection(collection_name=name)
            except Exception as e:
                print(f"    failed to delete {name}: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
