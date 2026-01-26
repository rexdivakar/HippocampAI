#!/usr/bin/env python3
"""Generate synthetic chat data for testing the UI."""

import os
import sys

# Add local source to path
_script_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.join(os.path.dirname(_script_dir), "src")
if os.path.exists(_src_dir) and _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from hippocampai import UnifiedMemoryClient  # noqa: E402

# Session details
SESSION_ID = "234ccf7f-7eea-40fe-88bc-1c57dea88bac"
USER_ID = f"user_{SESSION_ID[:8]}"

# More synthetic conversations to add
MORE_CONVERSATIONS = [
    {
        "user": "Can you help me understand how transformers work in NLP?",
        "assistant": "Transformers use self-attention mechanisms to process sequences in parallel. The key innovation is the attention mechanism that allows the model to weigh the importance of different parts of the input.",
        "type": "context",
        "importance": 7.0,
        "tags": ["learning", "nlp", "transformers", "ai"],
    },
    {
        "user": "I prefer using VS Code over PyCharm for Python development.",
        "assistant": "VS Code is a great choice! It is lightweight, has excellent Python support through extensions, and integrates well with Git. Many developers appreciate its flexibility.",
        "type": "preference",
        "importance": 6.0,
        "tags": ["preference", "ide", "development", "tools"],
    },
    {
        "user": "My team is working on a recommendation system using collaborative filtering.",
        "assistant": "Collaborative filtering is a powerful approach! Are you using user-based or item-based filtering? Matrix factorization techniques like SVD can also improve recommendations.",
        "type": "fact",
        "importance": 7.5,
        "tags": ["work", "project", "ml", "recommendations"],
    },
    {
        "user": "I want to improve my public speaking skills this year.",
        "assistant": "That is a valuable goal! Consider joining a Toastmasters club or practicing with smaller groups first. Recording yourself can also help identify areas for improvement.",
        "type": "goal",
        "importance": 7.0,
        "tags": ["goal", "personal-development", "skills"],
    },
    {
        "user": "Every Friday I review my weekly accomplishments and plan for next week.",
        "assistant": "Weekly reviews are excellent for productivity! This habit helps maintain focus and ensures continuous progress on your goals.",
        "type": "habit",
        "importance": 6.5,
        "tags": ["habit", "productivity", "planning"],
    },
    {
        "user": "I attended a great conference on AI safety last month.",
        "assistant": "AI safety is becoming increasingly important! What were the key takeaways from the conference? Were there any particular talks that stood out?",
        "type": "event",
        "importance": 6.0,
        "tags": ["event", "conference", "ai", "safety"],
    },
    {
        "user": "I find that I learn best through hands-on projects rather than reading documentation.",
        "assistant": "Learning by doing is very effective! Building projects helps solidify concepts and gives you practical experience to reference later.",
        "type": "preference",
        "importance": 7.0,
        "tags": ["preference", "learning", "style"],
    },
    {
        "user": "My favorite programming language is Python, but I also enjoy Rust for systems programming.",
        "assistant": "Great combination! Python for rapid development and ML, Rust for performance-critical systems. Both have strong communities and excellent tooling.",
        "type": "preference",
        "importance": 7.5,
        "tags": ["preference", "programming", "languages"],
    },
]


def main():
    print(f"Generating chat data for session: {SESSION_ID}")
    print(f"User ID: {USER_ID}")

    # Initialize client
    client = UnifiedMemoryClient(mode="local", qdrant_url="http://localhost:6333")

    print(f"\nAdding {len(MORE_CONVERSATIONS)} more conversation memories...")

    for i, conv in enumerate(MORE_CONVERSATIONS):
        try:
            # Create conversation memory
            conv_text = f"User: {conv['user']}\nAssistant: {conv['assistant']}"

            client.remember(
                text=conv_text,
                user_id=USER_ID,
                session_id=SESSION_ID,
                importance=conv["importance"],
                tags=conv["tags"],
            )

            print(f"  ✓ Created memory {i + 1}: {conv['type']} - {conv['user'][:40]}...")

        except Exception as e:
            print(f"  ✗ Failed to create memory {i + 1}: {e}")

    # Show total memories
    try:
        all_memories = client.get_memories(user_id=USER_ID, limit=100)
        print(f"\n✅ Total memories for user {USER_ID}: {len(all_memories)}")
    except Exception as e:
        print(f"\n⚠️ Could not count memories: {e}")

    print("\nTo test in UI:")
    print("  1. Go to the login page")
    print(f"  2. Enter Session ID: {SESSION_ID}")
    print(f"  3. Or enter User ID: {USER_ID}")
    print('  4. Click "Search" to login')


if __name__ == "__main__":
    main()
