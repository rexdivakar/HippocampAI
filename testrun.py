from hippocampai import MemoryClient

# Initialize client
# client = MemoryClient(qdrant_url = '192.168.1.120', embed_model = '')
client = MemoryClient()

# Store a memory
memory = client.remember(
	text = "I prefer oat milk in my coffee",
	user_id = "alice",
	type = "preference",
	importance = 8.0,
	tags = ["beverages", "preferences"]
)

# Memory size is automatically tracked
print(f"Memory size: {memory.text_length} chars, {memory.token_count} tokens")

# Recall relevant memories
results = client.recall(
	query = "How does Alice like her coffee?",
	user_id = "alice",
	k = 3
)

for result in results:
	print(f"{result.memory.text} (score: {result.score:.3f})")

# Get memory statistics
stats = client.get_memory_statistics(user_id = "alice")
print(f"Total memories: {stats['total_memories']}")
print(f"Total size: {stats['total_characters']} chars, {stats['total_tokens']} tokens")
