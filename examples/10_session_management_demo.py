"""Session management demonstration.

This example shows how to use HippocampAI's session management features:
- Creating and tracking conversation sessions
- Automatic session summarization
- Entity and fact extraction
- Session search and hierarchical sessions
- Automatic boundary detection
"""

from hippocampai import MemoryClient, SessionStatus

print("=" * 70)
print("  HippocampAI - Session Management Demo")
print("=" * 70)

# Initialize client
print("\nInitializing MemoryClient...")
client = MemoryClient()

user_id = "alice"

# ============================================
# 1. CREATE A NEW SESSION
# ============================================
print("\n" + "=" * 70)
print("1. Creating a new session")
print("=" * 70)

session = client.create_session(
    user_id=user_id,
    title="ML Project Discussion",
    tags=["work", "machine-learning"],
    metadata={"project": "sentiment-analysis"}
)

print(f"✓ Created session: {session.id}")
print(f"  Title: {session.title}")
print(f"  Status: {session.status.value}")
print(f"  Started at: {session.started_at}")

# ============================================
# 2. TRACK MESSAGES IN SESSION
# ============================================
print("\n" + "=" * 70)
print("2. Tracking messages in the session")
print("=" * 70)

# Simulate a conversation
messages = [
    "I'm working on a sentiment analysis project using Python",
    "We're using TensorFlow and BERT for the model",
    "The dataset has about 100k customer reviews",
    "Main challenge is handling sarcasm and context",
    "I prefer PyTorch but team decided on TensorFlow",
]

for i, message in enumerate(messages, 1):
    print(f"\nMessage {i}: {message[:50]}...")
    session = client.track_session_message(
        session_id=session.id,
        text=message,
        user_id=user_id,
        type="fact",
        auto_boundary_detect=False,  # Disable for this demo
    )
    print(f"✓ Tracked. Session has {session.message_count} messages")

# ============================================
# 3. SESSION STATISTICS
# ============================================
print("\n" + "=" * 70)
print("3. Session statistics")
print("=" * 70)

stats = client.get_session_statistics(session.id)
print(f"\nSession Statistics:")
print(f"  Messages: {stats['message_count']}")
print(f"  Memories: {stats['memory_count']}")
print(f"  Duration: {stats['duration_seconds']:.1f} seconds")
print(f"  Entities extracted: {stats['entity_count']}")
print(f"  Facts extracted: {stats['fact_count']}")
print(f"  Avg importance: {stats['avg_importance']:.2f}")

if stats['top_entities']:
    print("\n  Top entities:")
    for entity in stats['top_entities']:
        print(f"    - {entity['name']} ({entity['type']}): {entity['mentions']} mentions")

# ============================================
# 4. SESSION SUMMARIZATION
# ============================================
print("\n" + "=" * 70)
print("4. Generating session summary")
print("=" * 70)

summary = client.summarize_session(session.id)
if summary:
    print(f"\n✓ Summary generated:")
    print(f"  {summary}")
else:
    print("  ℹ LLM not available - summary not generated")

# ============================================
# 5. EXTRACT FACTS AND ENTITIES
# ============================================
print("\n" + "=" * 70)
print("5. Extracting facts and entities")
print("=" * 70)

facts = client.extract_session_facts(session.id)
if facts:
    print(f"\n✓ Extracted {len(facts)} facts:")
    for fact in facts[:3]:
        print(f"  - {fact.fact} (confidence: {fact.confidence:.2f})")
else:
    print("  ℹ LLM not available - facts not extracted")

entities = client.extract_session_entities(session.id)
if entities:
    print(f"\n✓ Extracted {len(entities)} entities:")
    for name, entity in list(entities.items())[:5]:
        print(f"  - {name} ({entity.type}): {entity.mentions} mentions")
else:
    print("  ℹ Using fallback entity extraction")
    # Session should still have some entities from auto-extraction
    session_updated = client.get_session(session.id)
    if session_updated.entities:
        print(f"  Found {len(session_updated.entities)} entities:")
        for name, entity in list(session_updated.entities.items())[:5]:
            print(f"    - {name} ({entity.type})")

# ============================================
# 6. GET SESSION MEMORIES
# ============================================
print("\n" + "=" * 70)
print("6. Retrieving session memories")
print("=" * 70)

memories = client.get_session_memories(session.id)
print(f"\n✓ Retrieved {len(memories)} memories from session:")
for i, mem in enumerate(memories[:3], 1):
    print(f"  {i}. {mem.text[:60]}...")

# ============================================
# 7. CREATE CHILD SESSION (HIERARCHICAL)
# ============================================
print("\n" + "=" * 70)
print("7. Creating a child session (hierarchical)")
print("=" * 70)

child_session = client.create_session(
    user_id=user_id,
    title="Deep Dive: Sarcasm Detection",
    parent_session_id=session.id,
    tags=["work", "deep-dive"],
    metadata={"parent_project": "sentiment-analysis"}
)

print(f"✓ Created child session: {child_session.id}")
print(f"  Parent: {child_session.parent_session_id}")

# Add some messages to child session
client.track_session_message(
    session_id=child_session.id,
    text="Let's focus on sarcasm detection approaches",
    user_id=user_id,
    auto_boundary_detect=False,
)

# Get child sessions
children = client.get_child_sessions(session.id)
print(f"\n✓ Parent session has {len(children)} child sessions")
for child in children:
    print(f"  - {child.title} (ID: {child.id[:8]}...)")

# ============================================
# 8. AUTOMATIC BOUNDARY DETECTION
# ============================================
print("\n" + "=" * 70)
print("8. Automatic session boundary detection")
print("=" * 70)

# Create a new session for boundary detection demo
boundary_session = client.create_session(
    user_id=user_id,
    title="Boundary Detection Demo"
)

# Add some messages about one topic
for msg in [
    "Python is great for data science",
    "I use numpy and pandas daily",
]:
    client.track_session_message(
        session_id=boundary_session.id,
        text=msg,
        user_id=user_id,
        auto_boundary_detect=False,
    )

print(f"\n✓ Added messages to session {boundary_session.id[:8]}...")

# Now add a message with completely different topic
# This would trigger boundary detection if auto_boundary_detect=True
result = client.track_session_message(
    session_id=boundary_session.id,
    text="Let's switch topics and discuss cloud infrastructure",
    user_id=user_id,
    auto_boundary_detect=True,  # Enable boundary detection
)

if result.id != boundary_session.id:
    print(f"✓ Boundary detected! New session created: {result.id[:8]}...")
    print(f"  Previous session completed")
else:
    print(f"  No boundary detected, continuing in same session")

# ============================================
# 9. SESSION SEARCH
# ============================================
print("\n" + "=" * 70)
print("9. Searching sessions by semantic similarity")
print("=" * 70)

search_results = client.search_sessions(
    query="machine learning and tensorflow",
    user_id=user_id,
    k=5
)

print(f"\n✓ Found {len(search_results)} relevant sessions:")
for i, result in enumerate(search_results, 1):
    print(f"\n  {i}. {result.session.title}")
    print(f"     Score: {result.score:.3f}")
    print(f"     Messages: {result.session.message_count}")
    if result.session.summary:
        print(f"     Summary: {result.session.summary[:60]}...")

# ============================================
# 10. GET USER SESSIONS
# ============================================
print("\n" + "=" * 70)
print("10. Getting all user sessions")
print("=" * 70)

all_sessions = client.get_user_sessions(user_id=user_id, limit=10)
print(f"\n✓ User has {len(all_sessions)} total sessions:")
for sess in all_sessions:
    print(f"  - {sess.title} ({sess.status.value})")
    print(f"    {sess.message_count} messages, started {sess.started_at}")

# Filter by status
active_sessions = client.get_user_sessions(
    user_id=user_id,
    status=SessionStatus.ACTIVE,
    limit=10
)
print(f"\n✓ {len(active_sessions)} active sessions")

# ============================================
# 11. COMPLETE SESSION
# ============================================
print("\n" + "=" * 70)
print("11. Completing a session")
print("=" * 70)

completed = client.complete_session(
    session_id=session.id,
    generate_summary=True  # Generate final summary
)

if completed:
    print(f"\n✓ Session completed: {completed.id}")
    print(f"  Status: {completed.status.value}")
    print(f"  Duration: {completed.duration_seconds():.1f} seconds")
    print(f"  Ended at: {completed.ended_at}")
    if completed.summary:
        print(f"  Final summary: {completed.summary}")

# ============================================
# 12. SESSION UPDATE
# ============================================
print("\n" + "=" * 70)
print("12. Updating session metadata")
print("=" * 70)

updated = client.update_session(
    session_id=child_session.id,
    title="Sarcasm Detection - Updated",
    tags=["work", "nlp", "sarcasm"],
    metadata={"priority": "high", "reviewed": True}
)

if updated:
    print(f"\n✓ Session updated: {updated.title}")
    print(f"  Tags: {updated.tags}")
    print(f"  Metadata: {updated.metadata}")

# ============================================
# Summary
# ============================================
print("\n" + "=" * 70)
print("  Demo Complete!")
print("=" * 70)

print("\nSession management features demonstrated:")
print("  ✓ Session creation and tracking")
print("  ✓ Automatic message tracking with boundary detection")
print("  ✓ Session summarization (LLM-based)")
print("  ✓ Entity and fact extraction")
print("  ✓ Hierarchical sessions (parent-child)")
print("  ✓ Semantic session search")
print("  ✓ Session statistics and analytics")
print("  ✓ Session lifecycle management (active → completed)")
print("  ✓ Session metadata and tagging")

print("\nKey benefits:")
print("  • Organize conversations into logical sessions")
print("  • Automatic topic change detection")
print("  • Track entities and facts across conversations")
print("  • Search past sessions semantically")
print("  • Hierarchical organization for complex discussions")
print("  • Rich metadata for filtering and analytics")

print("\n" + "=" * 70)
