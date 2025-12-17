"""Check if there's a record where user_id == session_id."""

import sys

from qdrant_client import QdrantClient

sys.path.insert(0, "/Users/rexdivakar/workspace/HippocampAI/src")

client = QdrantClient(url="http://100.113.229.40:6333")
session_id = "234ccf7f-7eea-40fe-88bc-1c57dea88ba2"

print(f"Checking if user_id == {session_id} exists anywhere\n")

# Search where user_id equals the session_id
search_conditions = {
    "must": [
        {"key": "user_id", "match": {"value": session_id}},
    ]
}

collections = ["hippocampai_facts", "hippocampai_prefs", "personal_facts", "hippocampai_sessions"]

for collection_name in collections:
    try:
        results, _ = client.scroll(
            collection_name=collection_name,
            scroll_filter=search_conditions,
            limit=10,
            with_payload=True,
        )

        if results and len(results) > 0:
            print(f"{'='*70}")
            print(f"⚠️  Found {len(results)} record(s) in {collection_name} where user_id == session_id")
            print(f"{'='*70}")
            for point in results:
                print(f"Point ID: {point.id}")
                print(f"user_id: {point.payload.get('user_id')}")
                print(f"session_id: {point.payload.get('session_id')}")
                print(f"text: {point.payload.get('text', 'N/A')[:100]}...")
                print()

    except Exception:
        continue

print("\n" + "="*70)
print("Summary:")
print("="*70)
print("If records were found above, that's the problem!")
print("The validation function finds these records FIRST and returns the wrong user_id.")
