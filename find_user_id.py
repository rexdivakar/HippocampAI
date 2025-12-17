"""Find actual user_id for the new session."""

import sys

from qdrant_client import QdrantClient

sys.path.insert(0, "/Users/rexdivakar/workspace/HippocampAI/src")

client = QdrantClient(url="http://100.113.229.40:6333")
session_id = "234ccf7f-7eea-40fe-88bc-1c57dea88ba2"

print(f"Searching for session_id: {session_id}\n")

search_conditions = {
    "should": [
        {"key": "user_id", "match": {"value": session_id}},
        {"key": "session_id", "match": {"value": session_id}},
        {"key": "metadata.session_id", "match": {"value": session_id}},
    ]
}

for collection_name in ["hippocampai_facts", "hippocampai_prefs"]:
    try:
        results, _ = client.scroll(
            collection_name=collection_name,
            scroll_filter=search_conditions,
            limit=5,
            with_payload=True,
        )

        if results:
            print(f"✅ Found in {collection_name}:")
            for point in results:
                print(f"  user_id: {point.payload.get('user_id')}")
                print(f"  session_id: {point.payload.get('session_id')}")
                print(f"  text: {point.payload.get('text')[:80]}...")
                print()
            break
    except Exception:
        continue
