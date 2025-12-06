"""Check what's stored for the user's session."""

import sys
sys.path.insert(0, "/Users/rexdivakar/workspace/HippocampAI/src")

from qdrant_client import QdrantClient

client = QdrantClient(url="http://100.113.229.40:6333")

# The user's session and user ID from the demo output
user_id = "1adda8b3-b8eb-45"  # Truncated in output, let's search for partial match
session_id = "234ccf7f-7eea-40fe-88bc-1c57dea88ba1"

print(f"Searching for:")
print(f"  User ID pattern: {user_id}*")
print(f"  Session ID: {session_id}\n")

collections = client.get_collections()

for collection in collections.collections:
    coll_name = collection.name
    print(f"\n{'='*70}")
    print(f"Collection: {coll_name}")
    print(f"{'='*70}")

    try:
        # Get recent records
        results, _ = client.scroll(
            collection_name=coll_name,
            limit=10,
            with_payload=True,
        )

        if results:
            print(f"Found {len(results)} recent records. Checking structure...")
            for point in results[:3]:
                print(f"\n  Record ID: {point.id}")
                print(f"  Fields in payload: {list(point.payload.keys())}")

                # Show user_id if exists
                if 'user_id' in point.payload:
                    print(f"  user_id: {point.payload['user_id']}")

                # Show session_id if exists (top-level or in metadata)
                if 'session_id' in point.payload:
                    print(f"  session_id (top-level): {point.payload['session_id']}")
                if 'metadata' in point.payload and isinstance(point.payload['metadata'], dict):
                    if 'session_id' in point.payload['metadata']:
                        print(f"  session_id (in metadata): {point.payload['metadata']['session_id']}")

                # Check if this matches our session
                matches_user = ('user_id' in point.payload and
                              user_id in str(point.payload['user_id']))
                matches_session = (
                    ('session_id' in point.payload and point.payload['session_id'] == session_id) or
                    ('metadata' in point.payload and
                     isinstance(point.payload['metadata'], dict) and
                     point.payload['metadata'].get('session_id') == session_id)
                )

                if matches_user or matches_session:
                    print(f"  ✅ MATCHES our session!")
                    print(f"  Full payload: {point.payload}")
        else:
            print("  No records found")

    except Exception as e:
        print(f"  Error: {e}")
