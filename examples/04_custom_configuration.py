"""Custom configuration example.

This demonstrates how to customize HippocampAI configuration.
"""

from hippocampai import MemoryClient
from hippocampai.config import Config

print("=" * 60)
print("  HippocampAI - Custom Configuration Example")
print("=" * 60)

# Option 1: Use default config
print("\n1. Using default configuration...")
client_default = MemoryClient()
print(f"   Qdrant URL: {client_default.config.qdrant_url}")
print(f"   Embed Model: {client_default.config.embed_model}")
print(f"   LLM Provider: {client_default.config.llm_provider}")

# Option 2: Override specific parameters
print("\n2. Overriding specific parameters...")
client_custom = MemoryClient(
    collection_facts="my_custom_facts",
    collection_prefs="my_custom_prefs",
    weights={"sim": 0.6, "rerank": 0.2, "recency": 0.1, "importance": 0.1},
)
print(f"   Facts Collection: {client_custom.config.collection_facts}")
print(f"   Prefs Collection: {client_custom.config.collection_prefs}")
print(f"   Weights: {client_custom.config.get_weights()}")

# Option 3: Create custom config object
print("\n3. Using custom Config object...")
custom_config = Config(
    qdrant_url="http://localhost:6333",
    embed_model="BAAI/bge-small-en-v1.5",
    llm_provider="ollama",
    llm_model="qwen2.5:7b-instruct",
    top_k_final=10,
    weight_sim=0.5,
    weight_rerank=0.25,
    weight_recency=0.15,
    weight_importance=0.10,
)

client_config = MemoryClient(config=custom_config)
print(f"   Top K Final: {client_config.config.top_k_final}")
print(f"   Embedding Model: {client_config.config.embed_model}")

# Test with custom configuration
print("\n4. Testing custom configuration...")
user_id = "dave"

memory = client_custom.remember(
    text="I prefer working remotely", user_id=user_id, type="preference", importance=8.0
)
print(f"   ✓ Stored memory: {memory.id[:8]}...")

results = client_custom.recall(query="How does Dave like to work?", user_id=user_id, k=1)
if results:
    print(f"   ✓ Retrieved: {results[0].memory.text}")

print("\n" + "=" * 60)
print("  Example Complete!")
print("=" * 60)
print("\nConfiguration options:")
print("  • Qdrant connection settings")
print("  • Collection names")
print("  • Embedding model selection")
print("  • LLM provider and model")
print("  • Retrieval weights and parameters")
print("  • HNSW index tuning")
print("  • Half-life decay settings")
