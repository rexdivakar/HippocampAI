#!/usr/bin/env python
"""Sync existing Qdrant sessions to DuckDB for admin management."""

import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from qdrant_client import QdrantClient

from hippocampai.storage import Session, User, UserStore


def sync_sessions():
    """Sync sessions from Qdrant to DuckDB."""
    qdrant_url = os.getenv("QDRANT_URL", "http://100.113.229.40:6333")
    client = QdrantClient(url=qdrant_url)
    store = UserStore("data/users.duckdb")
    
    print(f"🔄 Syncing sessions from Qdrant ({qdrant_url}) to DuckDB...")
    
    collections = ["hippocampai_facts", "hippocampai_prefs", "personal_facts"]
    users_seen = set()
    sessions_seen = set()
    
    for collection in collections:
        try:
            if not client.collection_exists(collection):
                continue
            
            print(f"  Scanning {collection}...")
            offset = None
            
            while True:
                results, offset = client.scroll(
                    collection_name=collection,
                    limit=1000,
                    offset=offset,
                    with_payload=True,
                )
                
                if not results:
                    break
                
                for point in results:
                    payload = point.payload
                    user_id = payload.get("user_id")
                    session_id = payload.get("session_id")
                    
                    if user_id and user_id not in users_seen:
                        users_seen.add(user_id)
                        try:
                            existing = store.get_user(user_id, include_deleted=True)
                            if not existing:
                                user = User(
                                    user_id=user_id,
                                    username=payload.get("metadata", {}).get("username"),
                                    created_at=datetime.fromisoformat(payload.get("created_at", datetime.utcnow().isoformat()).replace("Z", "+00:00"))
                                )
                                store.create_user(user)
                                print(f"    + User: {user_id}")
                        except Exception:
                            pass  # User might already exist
                    
                    if session_id and session_id not in sessions_seen:
                        sessions_seen.add(session_id)
                        try:
                            existing = store.get_session(session_id, include_deleted=True)
                            if not existing:
                                session = Session(
                                    session_id=session_id,
                                    user_id=user_id or "unknown",
                                    created_at=datetime.fromisoformat(payload.get("created_at", datetime.utcnow().isoformat()).replace("Z", "+00:00")),
                                    memory_count=1
                                )
                                store.create_session(session)
                                print(f"    + Session: {session_id[:8]}...")
                            else:
                                # Update memory count
                                store.update_session_memory_count(session_id, existing.memory_count + 1)
                        except Exception:
                            pass  # Session might already exist
                
                if offset is None:
                    break
                    
        except Exception as e:
            print(f"  ⚠️ Error scanning {collection}: {e}")
    
    stats = store.get_stats()
    print("\n✅ Sync complete!")
    print(f"   Users: {stats['total_users']}")
    print(f"   Sessions: {stats['total_sessions']}")
    print(f"   Memories: {stats['total_memories']}")
    
    store.close()


if __name__ == "__main__":
    sync_sessions()
