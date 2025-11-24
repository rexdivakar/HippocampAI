"""
Example: Using HippocampAI with mem0-compatible Simple API

This example shows how easy it is to use HippocampAI with the
simplified API that's compatible with mem0 patterns.
"""

from hippocampai import SimpleMemory as Memory

# Initialize - that's it! Auto-configures everything
print("=== HippocampAI Simple API (mem0-compatible) ===\n")
m = Memory()

print("✅ Memory client initialized\n")

# Store memories using simple add() method
print("1. Storing memories with add()...")
mem1 = m.add("I prefer oat milk in my coffee", user_id="alice")
print(f"   Added: {mem1.text}")

mem2 = m.add("I work as a software engineer at TechCorp", user_id="alice")
print(f"   Added: {mem2.text}")

mem3 = m.add("I want to learn machine learning this year", user_id="alice")
print(f"   Added: {mem3.text}\n")

# Search memories using simple search() method
print("2. Searching with search()...")
results = m.search("work and career", user_id="alice", limit=3)
print(f"   Found {len(results)} results:")
for i, result in enumerate(results, 1):
    print(f"   {i}. Score {result.score:.3f}: {result.memory.text}")
print()

# Get a specific memory
print("3. Get memory by ID...")
retrieved = m.get(mem1.id)
if retrieved:
    print(f"   Retrieved: {retrieved.text}\n")

# Update a memory
print("4. Updating memory...")
updated = m.update(mem1.id, text="I strongly prefer oat milk in my coffee")
if updated:
    print(f"   Updated: {updated.text}\n")

# Get all memories for user
print("5. Get all memories...")
all_memories = m.get_all(user_id="alice")
print(f"   Total memories for alice: {len(all_memories)}\n")

# Delete a memory
print("6. Deleting a memory...")
deleted = m.delete(mem2.id)
if deleted:
    print(f"   Deleted memory {mem2.id}\n")

# Verify deletion
all_memories_after = m.get_all(user_id="alice")
print(f"   Memories after deletion: {len(all_memories_after)}\n")

print("✅ Simple API demo complete!")
print("\n💡 This API is compatible with mem0 - easy migration!")
