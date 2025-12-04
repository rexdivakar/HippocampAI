#!/usr/bin/env python3
"""
Query Qdrant Directly for Memories

This script shows how to query Qdrant directly to see stored memories
for a specific user or session.
"""

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

# Configuration
QDRANT_URL = "http://100.113.229.40:6333"
COLLECTION_FACTS = "hippocampai_facts"
COLLECTION_PREFS = "hippocampai_prefs"

# Your session details
USER_ID = "9299e742-1db9-4fd6-baa4-66d8ecf1750c"
SESSION_ID = "c7b19354-47f4-44f5-b41d-73fa534193b2"


def query_memories_by_session(qdrant_url: str, user_id: str, session_id: str):
    """Query all memories for a specific session."""
    client = QdrantClient(url=qdrant_url)

    print(f"Connecting to Qdrant at {qdrant_url}")
    print(f"User ID: {user_id}")
    print(f"Session ID: {session_id}")
    print("\n" + "="*80)

    # Query both collections
    for collection_name in [COLLECTION_FACTS, COLLECTION_PREFS]:
        print(f"\n📚 Collection: {collection_name}")
        print("-" * 80)

        try:
            # Check if collection exists
            collections = client.get_collections().collections
            collection_names = [c.name for c in collections]

            if collection_name not in collection_names:
                print(f"⚠️  Collection '{collection_name}' does not exist yet")
                continue

            # Get collection info
            collection_info = client.get_collection(collection_name)
            print(f"Total points in collection: {collection_info.points_count}")

            # Query by session_id
            results = client.scroll(
                collection_name=collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="session_id",
                            match=MatchValue(value=session_id)
                        )
                    ]
                ),
                limit=100,
                with_payload=True,
                with_vectors=False
            )

            points = results[0]  # First element is list of points

            if not points:
                print(f"No memories found for session: {session_id}")

                # Try querying by user_id instead
                print(f"\nTrying to find memories for user: {user_id[:16]}...")
                results = client.scroll(
                    collection_name=collection_name,
                    scroll_filter=Filter(
                        must=[
                            FieldCondition(
                                key="user_id",
                                match=MatchValue(value=user_id)
                            )
                        ]
                    ),
                    limit=100,
                    with_payload=True,
                    with_vectors=False
                )

                points = results[0]
                if not points:
                    print(f"No memories found for user either")
                    continue

            print(f"\n✅ Found {len(points)} memories:\n")

            for i, point in enumerate(points, 1):
                payload = point.payload
                print(f"Memory {i}:")
                print(f"  ID: {point.id}")
                print(f"  Text: {payload.get('text', 'N/A')}")
                print(f"  Type: {payload.get('metadata', {}).get('type', 'N/A')}")
                print(f"  Tags: {payload.get('tags', [])}")
                print(f"  Importance: {payload.get('importance', 'N/A')}")
                print(f"  Timestamp: {payload.get('metadata', {}).get('timestamp', 'N/A')}")
                print(f"  Session ID: {payload.get('session_id', 'N/A')}")
                print()

        except Exception as e:
            print(f"❌ Error querying {collection_name}: {e}")

    print("="*80)


def list_all_collections(qdrant_url: str):
    """List all available collections in Qdrant."""
    client = QdrantClient(url=qdrant_url)

    print(f"\n📋 All Collections in Qdrant at {qdrant_url}:")
    print("-" * 80)

    try:
        collections = client.get_collections().collections

        if not collections:
            print("No collections found")
            return

        for collection in collections:
            info = client.get_collection(collection.name)
            print(f"  • {collection.name}")
            print(f"    Points: {info.points_count}")
            print(f"    Vector size: {info.config.params.vectors.size}")
            print()

    except Exception as e:
        print(f"❌ Error listing collections: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Query Qdrant for stored memories")
    parser.add_argument("--qdrant-url", default=QDRANT_URL, help="Qdrant server URL")
    parser.add_argument("--user-id", default=USER_ID, help="User ID to query")
    parser.add_argument("--session-id", default=SESSION_ID, help="Session ID to query")
    parser.add_argument("--list-collections", action="store_true", help="List all collections")

    args = parser.parse_args()

    if args.list_collections:
        list_all_collections(args.qdrant_url)
    else:
        query_memories_by_session(args.qdrant_url, args.user_id, args.session_id)
