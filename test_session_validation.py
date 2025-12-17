"""Test session validation for existing session ID."""

import sys

from hippocampai.vector.qdrant_store import QdrantStore

sys.path.insert(0, "/Users/rexdivakar/workspace/HippocampAI/src")

def check_session(unique_id: str):
    """Check if session exists using new multi-field search."""
    try:
        qdrant_url = "http://100.113.229.40:6333"
        store = QdrantStore(url=qdrant_url)

        # Search for the unique_id in multiple places
        search_conditions = {
            "should": [
                {"key": "user_id", "match": {"value": unique_id}},
                {"key": "metadata.session_id", "match": {"value": unique_id}},
                {"key": "session_id", "match": {"value": unique_id}},
            ]
        }

        print(f"🔍 Searching for ID: {unique_id}")
        print(f"📍 Qdrant URL: {qdrant_url}\n")

        # Check facts collection
        facts_count = store.client.count(
            collection_name=store.collection_facts,
            count_filter=search_conditions,
        )
        print(f"✅ Facts collection: {facts_count.count} matches")

        # Check prefs collection
        prefs_count = store.client.count(
            collection_name=store.collection_prefs,
            count_filter=search_conditions,
        )
        print(f"✅ Prefs collection: {prefs_count.count} matches")

        total = facts_count.count + prefs_count.count
        print(f"\n📊 Total matches: {total}")

        if total > 0:
            print("\n✅ SESSION VALID - ID found in Qdrant!")

            # Show some sample memories
            results = store.client.scroll(
                collection_name=store.collection_facts,
                scroll_filter=search_conditions,
                limit=3,
                with_payload=True,
            )

            if results[0]:
                print("\n📝 Sample memories:")
                for i, point in enumerate(results[0][:3], 1):
                    print(f"\n{i}. {point.payload.get('text', '')[:100]}...")
                    print(f"   user_id: {point.payload.get('user_id')}")
                    print(f"   session_id (metadata): {point.payload.get('metadata', {}).get('session_id')}")

            return True
        else:
            print("\n❌ SESSION INVALID - ID not found in Qdrant")
            return False

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    session_id = "234ccf7f-7eea-40fe-88bc-1c57dea88bac"
    check_session(session_id)

    print("\n" + "="*60)
    print("Testing invalid IDs:")
    print("="*60 + "\n")

    for invalid_id in ["1", "2", "invalid_session"]:
        print(f"\nTesting: {invalid_id}")
        result = check_session(invalid_id)
        print(f"Result: {'ALLOWED' if result else 'BLOCKED'}")
        print("-" * 40)
