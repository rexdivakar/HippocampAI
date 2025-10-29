"""Example: UnifiedMemoryClient configuration options.

This example shows all configuration options for both local and remote modes.
"""

from hippocampai import UnifiedMemoryClient


def example_local_mode_basic() -> None:
    """Basic local mode - uses defaults from environment or config."""
    print("1. LOCAL MODE - Basic (uses default config)")
    print("-" * 50)

    client = UnifiedMemoryClient(mode="local")

    print(f"Mode: {client.mode}")
    print("Uses default Qdrant, Redis, Ollama settings from .env\n")


def example_local_mode_custom() -> None:
    """Local mode with custom configuration."""
    print("2. LOCAL MODE - Custom Configuration")
    print("-" * 50)

    client = UnifiedMemoryClient(
        mode="local",
        # Custom Qdrant
        qdrant_url="http://custom-qdrant:6333",
        # Custom collections
        collection_facts="my_facts",
        collection_prefs="my_prefs",
        # Custom embedding model
        embed_model="all-MiniLM-L6-v2",
        # Custom LLM
        llm_provider="ollama",
        llm_model="llama3.2:3b",
        # HNSW optimization
        hnsw_M=48,
        ef_construction=256,
        ef_search=128,
    )

    print(f"Mode: {client.mode}")
    print("Using custom Qdrant, collections, and models\n")


def example_remote_mode_basic() -> None:
    """Basic remote mode."""
    print("3. REMOTE MODE - Basic")
    print("-" * 50)

    client = UnifiedMemoryClient(mode="remote", api_url="http://localhost:8000")

    print(f"Mode: {client.mode}")
    print("Connects to API at http://localhost:8000")
    print("No local dependencies needed\n")


def example_remote_mode_with_auth() -> None:
    """Remote mode with authentication."""
    print("4. REMOTE MODE - With Authentication")
    print("-" * 50)

    client = UnifiedMemoryClient(
        mode="remote",
        api_url="https://api.hippocampai.com",
        api_key="your-api-key-here",
        timeout=60,  # Custom timeout
    )

    print(f"Mode: {client.mode}")
    print("Connects to production API with authentication")
    print("API Key: ***hidden***")
    print("Timeout: 60 seconds\n")


def example_env_based_config() -> None:
    """Configuration based on environment variables."""
    print("5. ENVIRONMENT-BASED Configuration")
    print("-" * 50)

    import os

    # Read from environment
    mode = os.getenv("HIPPOCAMP_MODE", "local")
    api_url = os.getenv("HIPPOCAMP_API_URL", "http://localhost:8000")
    api_key = os.getenv("HIPPOCAMP_API_KEY")

    if mode == "remote":
        client = UnifiedMemoryClient(mode="remote", api_url=api_url, api_key=api_key)
    else:
        client = UnifiedMemoryClient(mode="local")

    print(f"Mode: {client.mode}")
    print("Configuration loaded from environment variables:")
    print(f"  HIPPOCAMP_MODE={mode}")
    if mode == "remote":
        print(f"  HIPPOCAMP_API_URL={api_url}")
        print(f"  HIPPOCAMP_API_KEY={'set' if api_key else 'not set'}\n")


def example_development_vs_production() -> None:
    """Different configurations for development vs production."""
    print("6. DEVELOPMENT vs PRODUCTION")
    print("-" * 50)

    import os

    environment = os.getenv("ENVIRONMENT", "development")

    if environment == "production":
        # Production: Use remote SaaS API
        client = UnifiedMemoryClient(
            mode="remote",
            api_url="https://api.hippocampai.com",
            api_key=os.getenv("HIPPOCAMP_API_KEY"),
            timeout=30,
        )
        print("Environment: PRODUCTION")
        print("Using: Remote SaaS API")
        print("Benefits: Managed, scalable, reliable\n")
    else:
        # Development: Use local connection
        client = UnifiedMemoryClient(mode="local", qdrant_url="http://localhost:6333")
        print("Environment: DEVELOPMENT")
        print(f"Using: Local Qdrant/Redis/Ollama (mode: {client.mode})")
        print("Benefits: Fast, no network latency, easy debugging\n")


def example_fallback_strategy() -> None:
    """Fallback from remote to local if remote fails."""
    print("7. FALLBACK Strategy")
    print("-" * 50)

    try:
        # Try remote first
        client = UnifiedMemoryClient(mode="remote", api_url="http://localhost:8000", timeout=5)
        # Test connection
        client.health_check()
        print("Mode: REMOTE")
        print("Successfully connected to API server\n")
    except Exception as e:
        # Fallback to local
        print(f"Remote connection failed: {e}")
        print("Falling back to LOCAL mode...\n")
        client = UnifiedMemoryClient(mode="local")
        print("Mode: LOCAL")
        print("Using local Qdrant/Redis/Ollama\n")


def main() -> None:
    """Run all configuration examples."""
    print("\n" + "=" * 60)
    print("UnifiedMemoryClient - Configuration Examples")
    print("=" * 60 + "\n")

    example_local_mode_basic()
    example_local_mode_custom()
    example_remote_mode_basic()
    example_remote_mode_with_auth()
    example_env_based_config()
    example_development_vs_production()
    example_fallback_strategy()

    print("=" * 60)
    print("✓ All configuration examples shown")
    print("=" * 60)
    print("\nKey Points:")
    print("  1. Local mode: Direct connection, max performance")
    print("  2. Remote mode: HTTP API, multi-language support")
    print("  3. Same API for both modes - easy to switch")
    print("  4. Use environment variables for flexibility")
    print("  5. Consider dev/prod environments")
    print("  6. Implement fallback strategies for reliability")


if __name__ == "__main__":
    main()
