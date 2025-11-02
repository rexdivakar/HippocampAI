"""Example: Switching between LOCAL and REMOTE modes.

This example demonstrates how easy it is to switch between local and remote modes
by simply changing the connection configuration. The API remains exactly the same!
"""

import os

from hippocampai import UnifiedMemoryClient


def run_demo(client: UnifiedMemoryClient, mode_name: str) -> None:
    """Run a demo with the given client."""
    print(f"\n{'=' * 60}")
    print(f"Running demo in {mode_name} mode")
    print(f"{'=' * 60}\n")

    # Store a memory
    memory = client.remember(
        text=f"Test memory created in {mode_name} mode", user_id="demo_user", tags=["demo"]
    )
    print(f"✓ Created memory: {memory.id}")

    # Search
    results = client.recall(query="test memory", user_id="demo_user", limit=1)
    if results:
        print(f"✓ Found {len(results)} result(s)")
        print(f"  Text: {results[0].memory.text}")
        print(f"  Score: {results[0].score:.3f}")

    # Cleanup
    client.delete_memory(memory.id)
    print("✓ Cleaned up memory\n")


def main() -> None:
    """Demonstrate mode switching."""
    print("=== UnifiedMemoryClient - Mode Switching Demo ===\n")
    print("This demo shows how to use the SAME code with DIFFERENT backends\n")

    # Configuration - easily switch between modes
    USE_REMOTE = os.getenv("USE_REMOTE_MODE", "false").lower() == "true"

    if USE_REMOTE:
        # Remote mode - connect to SaaS API
        print("Initializing REMOTE mode...")
        print("API URL: http://localhost:8000")
        try:
            client = UnifiedMemoryClient(mode="remote", api_url="http://localhost:8000")
            # Verify server is accessible
            health = client.health_check()
            print(f"Server health: {health['status']}\n")
            run_demo(client, "REMOTE")
        except Exception as e:
            print(f"⚠ Remote mode failed: {e}")
            print("Please start the server:")
            print("uvicorn hippocampai.api.async_app:app --port 8000\n")
    else:
        # Local mode - direct connection
        print("Initializing LOCAL mode...")
        print("Direct connection to Qdrant/Redis/Ollama\n")
        client = UnifiedMemoryClient(mode="local")
        run_demo(client, "LOCAL")

    print("\n" + "=" * 60)
    print("✓ Demo complete!")
    print("=" * 60)
    print("\nKey Takeaway:")
    print("  - Same code works with both local and remote backends")
    print("  - Switch modes by changing just ONE parameter")
    print("  - No need to rewrite application logic")
    print("\nTo switch modes:")
    print("  Local:  python unified_client_mode_switching.py")
    print("  Remote: USE_REMOTE_MODE=true python unified_client_mode_switching.py")
    print("\nOr in your code:")
    print('  client = UnifiedMemoryClient(mode="local")   # Local')
    print('  client = UnifiedMemoryClient(mode="remote", api_url="...")  # Remote')


if __name__ == "__main__":
    main()
