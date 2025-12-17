"""Deep search for session ID in Qdrant - checks ALL fields."""

import json
import sys

from qdrant_client import QdrantClient

sys.path.insert(0, "/Users/rexdivakar/workspace/HippocampAI/src")

def deep_search():
    """Search every collection and every field for the session ID."""
    client = QdrantClient(url="http://100.113.229.40:6333")
    session_id = "234ccf7f-7eea-40fe-88bc-1c57dea88bac"

    print(f"🔍 DEEP SEARCH for: {session_id}")
    print("📍 Qdrant: http://100.113.229.40:6333\n")
    print("=" * 70)

    # Get all collections
    collections = client.get_collections()

    for collection in collections.collections:
        coll_name = collection.name
        print(f"\n🔎 Searching: {coll_name}")
        print("-"*70)

        try:
            # Get ALL records (scroll through entire collection)
            offset = None
            batch_size = 100
            found_in_collection = False

            while True:
                results, offset = client.scroll(
                    collection_name=coll_name,
                    limit=batch_size,
                    offset=offset,
                    with_payload=True,
                )

                if not results:
                    break

                # Check each record
                for point in results:
                    # Convert payload to JSON string to search everywhere
                    payload_str = json.dumps(point.payload, default=str).lower()

                    # Check if session ID appears anywhere in the payload
                    if session_id.lower() in payload_str:
                        found_in_collection = True
                        print(f"\n✅ FOUND in collection: {coll_name}")
                        print(f"   Point ID: {point.id}")
                        print("   Full Payload:")
                        print(f"   {json.dumps(point.payload, indent=6, default=str)}")
                        print("\n" + "="*70)
                        return True  # Found it!

                if offset is None:
                    break

            if not found_in_collection:
                print(f"   ❌ Not found in {coll_name}")

        except Exception as e:
            print(f"   ⚠️  Error searching {coll_name}: {e}")

    print(f"\n❌ Session ID '{session_id}' NOT FOUND in any collection")
    return False


if __name__ == "__main__":
    found = deep_search()

    if not found:
        print("\n" + "=" * 70)
        print("💡 SUGGESTION:")
        print("=" * 70)
        print("The session ID doesn't exist in Qdrant yet.")
        print("Options:")
        print("  1. Use the 'Create New Account' button to generate a new session")
        print("  2. Verify you're checking the correct Qdrant instance")
        print("  3. Check if the ID format is different than expected")
