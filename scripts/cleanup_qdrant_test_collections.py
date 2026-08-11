#!/usr/bin/env python3
"""Delete ephemeral test collections from a Qdrant instance.

Run after the test suite (pass or fail) so the shared CI Qdrant Cloud
cluster doesn't accumulate one set of collections per run. Never raises -
cleanup must never fail the CI job.

Default behavior only deletes collections prefixed with this job's own
QDRANT_TEST_NAMESPACE (e.g. "ci-<run_id>-<python-version>_*"). Matrix jobs
in the same workflow run concurrently and share one Qdrant Cloud cluster,
so anything broader risks one job deleting another still-running job's
collections out from under it.

Use --all-stale for a manual, out-of-band sweep of the wider set of
known-safe ephemeral patterns (unnamespaced test fixtures, the fixed
bi-temporal collection name) that accumulate across many runs over time.
Do not wire --all-stale into the per-job CI step.

Usage:
    python scripts/cleanup_qdrant_test_collections.py [--dry-run] [--all-stale]

Env:
    QDRANT_URL, QDRANT_API_KEY  - required to connect
    QDRANT_TEST_NAMESPACE       - this job's own namespace prefix
"""

import os
import sys

# Prefixes only ever produced by unnamespaced test fixtures (see
# tests/conftest.py, tests/test_bitemporal.py). Global, so only ever matched
# under --all-stale, never in the per-job namespace-scoped default.
GLOBAL_SAFE_PREFIXES = ("test_facts_", "test_prefs_")
GLOBAL_FIXED_NAMES = ("hippocampai_bitemporal_facts",)


def main() -> int:
    dry_run = "--dry-run" in sys.argv
    all_stale = "--all-stale" in sys.argv

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

    to_delete: list[str] = []
    if namespace:
        to_delete += [c for c in collections if c.startswith(f"{namespace}_")]
    if all_stale:
        to_delete += [
            c for c in collections if c.startswith(GLOBAL_SAFE_PREFIXES) and c not in to_delete
        ]
        to_delete += [c for c in GLOBAL_FIXED_NAMES if c in collections and c not in to_delete]

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
