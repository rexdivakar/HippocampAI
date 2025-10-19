"""Example usage of the EmbeddingService."""

import sys
sys.path.append('..')

from src.embedding_service import EmbeddingService


def main():
    # Initialize the embedding service
    print("Initializing EmbeddingService...")
    embedding_service = EmbeddingService(model_name="all-MiniLM-L6-v2", cache_size=1000)

    # Check embedding dimension
    dim = embedding_service.get_embedding_dimension()
    print(f"Embedding dimension: {dim}\n")

    # Generate single embedding
    print("--- Single Embedding ---")
    text = "This is a test sentence for embedding."
    embedding = embedding_service.generate_embedding(text)
    print(f"Text: {text}")
    print(f"Embedding shape: {embedding.shape}")
    print(f"First 5 values: {embedding[:5]}\n")

    # Generate same embedding again (should hit cache)
    print("--- Cache Test ---")
    embedding2 = embedding_service.generate_embedding(text)
    print(f"Same text embedded again (cache hit)")
    print(f"Embeddings are identical: {(embedding == embedding2).all()}\n")

    # Generate batch embeddings
    print("--- Batch Embeddings ---")
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is fascinating.",
        "Python is a great programming language.",
        "This is a test sentence for embedding."  # Duplicate, should hit cache
    ]

    batch_embeddings = embedding_service.generate_batch_embeddings(texts)
    print(f"Generated {len(batch_embeddings)} embeddings")
    for i, (text, emb) in enumerate(zip(texts, batch_embeddings)):
        print(f"  {i+1}. '{text[:50]}...' -> shape {emb.shape}")

    # Cache statistics
    print("\n--- Cache Statistics ---")
    stats = embedding_service.get_cache_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")

    # Switch model example (commented out to avoid downloading another model)
    # print("\n--- Switch Model ---")
    # embedding_service.switch_model("paraphrase-MiniLM-L6-v2")
    # print(f"New embedding dimension: {embedding_service.get_embedding_dimension()}")


if __name__ == "__main__":
    main()
