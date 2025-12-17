"""Test script to debug user_id lookup from session_id."""

import json
import sys

from qdrant_client import QdrantClient

sys.path.insert(0, "/Users/rexdivakar/workspace/HippocampAI/src")

client = QdrantClient(url="http://100.113.229.40:6333")
session_id = "234ccf7f-7eea-40fe-88bc-1c57dea88ba2"

print(f"Looking for session_id: {session_id}\n")

search_conditions = {
    "should": [
        {"key": "user_id", "match": {"value": session_id}},
        {"key": "session_id", "match": {"value": session_id}},
        {"key": "metadata.session_id", "match": {"value": session_id}},
    ]
}

collections = ["hippocampai_facts", "hippocampai_prefs", "personal_facts", "hippocampai_sessions"]

for collection_name in collections:
    try:
        print(f"\n{'='*70}")
        print(f"Checking: {collection_name}")
        print(f"{'='*70}")

        results, _ = client.scroll(
            collection_name=collection_name,
            scroll_filter=search_conditions,
            limit=1,
            with_payload=True,
        )

        if results and len(results) > 0:
            print(f"✅ Found {len(results)} result(s)")
            for point in results:
                print(f"\nPoint ID: {point.id}")
                print(f"Payload keys: {list(point.payload.keys())}")
                print(f"user_id: {point.payload.get('user_id')}")
                print(f"session_id (top-level): {point.payload.get('session_id')}")
                if 'metadata' in point.payload:
                    print(f"metadata.session_id: {point.payload['metadata'].get('session_id')}")
                print("\nFull payload:")
                print(json.dumps(point.payload, indent=2, default=str))
        else:
            print("❌ No results found")

    except Exception as e:
        print(f"⚠️  Error: {e}")
