"""Find where the session ID is stored in Qdrant."""

import sys
sys.path.insert(0, "/Users/rexdivakar/workspace/HippocampAI/src")

from qdrant_client import QdrantClient

def find_session():
    """Search all collections for the session ID."""
    client = QdrantClient(url="http://100.113.229.40:6333")
    session_id = "234ccf7f-7eea-40fe-88bc-1c57dea88bac"

    # List all collections
    collections = client.get_collections()
    print(f"📚 Available collections:")
    for col in collections.collections:
        print(f"  - {col.name}")

    print(f"\n🔍 Searching for session_id: {session_id}\n")

    # Check each collection
    for collection in collections.collections:
        coll_name = collection.name
        print(f"\n{'='*60}")
        print(f"Collection: {coll_name}")
        print(f"{'='*60}")

        # Get total count
        try:
            total = client.count(collection_name=coll_name)
            print(f"Total memories: {total.count}")
        except Exception as e:
            print(f"Could not count: {e}")
            continue

        # Try multiple search patterns
        search_patterns = [
            ("user_id", session_id),
            ("session_id", session_id),
            ("metadata.session_id", session_id),
            ("id", session_id),
        ]

        for key, value in search_patterns:
            try:
                count = client.count(
                    collection_name=coll_name,
                    count_filter={
                        "should": [
                            {"key": key, "match": {"value": value}}
                        ]
                    }
                )
                if count.count > 0:
                    print(f"✅ Found {count.count} matches for {key} = {value}")

                    # Get sample
                    results = client.scroll(
                        collection_name=coll_name,
                        scroll_filter={
                            "should": [
                                {"key": key, "match": {"value": value}}
                            ]
                        },
                        limit=2,
                        with_payload=True,
                    )

                    if results[0]:
                        for point in results[0]:
                            print(f"\n  Sample memory:")
                            print(f"    ID: {point.id}")
                            print(f"    user_id: {point.payload.get('user_id', 'N/A')}")
                            print(f"    session_id: {point.payload.get('session_id', 'N/A')}")
                            print(f"    metadata.session_id: {point.payload.get('metadata', {}).get('session_id', 'N/A')}")
                            print(f"    text: {point.payload.get('text', 'N/A')[:100]}...")
                            break

            except Exception as e:
                pass  # Field might not exist

        # Show sample records to understand structure
        try:
            print(f"\n📝 Sample records from {coll_name}:")
            results = client.scroll(
                collection_name=coll_name,
                limit=2,
                with_payload=True,
            )
            if results[0]:
                for i, point in enumerate(results[0], 1):
                    print(f"\n  Record {i}:")
                    print(f"    user_id: {point.payload.get('user_id', 'N/A')}")
                    print(f"    session_id: {point.payload.get('session_id', 'N/A')}")
                    print(f"    metadata: {point.payload.get('metadata', {})}")
        except Exception as e:
            print(f"Error getting samples: {e}")


if __name__ == "__main__":
    find_session()
