"""Diagnose the memories UI bug."""

import sys

from qdrant_client import QdrantClient

sys.path.insert(0, "/Users/rexdivakar/workspace/HippocampAI/src")

# Connect to Qdrant
client = QdrantClient(url="http://100.113.229.40:6333")
session_id = "234ccf7f-7eea-40fe-88bc-1c57dea88bac"

print(f"Searching for session_id: {session_id}\n")

# Search for records with this session_id
search_conditions = {
    "must": [
        {"key": "session_id", "match": {"value": session_id}}
    ]
}

collections = ["hippocampai_facts", "hippocampai_prefs"]

for collection_name in collections:
    try:
        results, _ = client.scroll(
            collection_name=collection_name,
            scroll_filter=search_conditions,
            limit=5,
            with_payload=True,
        )

        if results:
            print(f"{'='*70}")
            print(f"Collection: {collection_name}")
            print(f"Found {len(results)} record(s)")
            print(f"{'='*70}")

            for i, point in enumerate(results, 1):
                payload = point.payload
                print(f"\nRecord {i}:")
                print(f"  user_id: {payload.get('user_id')}")
                print(f"  session_id: {payload.get('session_id')}")
                print(f"  type: {payload.get('type')}")
                print(f"  text: {payload.get('text', 'N/A')[:80]}...")

            print()
        else:
            print(f"{collection_name}: No results found\n")

    except Exception as e:
        print(f"{collection_name}: Error - {e}\n")

print("\n" + "="*70)
print("DIAGNOSIS:")
print("="*70)
print("The UI is failing because:")
print("1. CLI demo generates a random user_id when you only pass --session-id")
print("2. The frontend sends session_id as the user_id parameter")
print("3. Backend queries Qdrant by user_id FIRST, gets 0 results")
print("4. session_id filter never runs because there's nothing to filter")
print("\nSOLUTION: Add session_id to the Qdrant query, not just Python filtering")
