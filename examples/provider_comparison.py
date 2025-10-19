"""Example comparing different LLM providers."""

import sys
import time

sys.path.append("..")

from hippocampai.llm_provider import LLMProvider, get_llm_client


def test_provider(provider_name: str, test_prompt: str) -> dict:
    """Test a single provider."""
    try:
        print(f"\n{'=' * 60}")
        print(f"Testing: {provider_name.upper()}")
        print(f"{'=' * 60}")

        start_time = time.time()

        # Get client
        client = get_llm_client(provider=provider_name)

        print(f"Model: {client.model}")
        print(f"Prompt: {test_prompt}\n")

        # Generate response
        response = client.generate(test_prompt)

        elapsed = time.time() - start_time

        print(f"Response: {response}\n")
        print(f"⏱️  Time: {elapsed:.2f}s")

        return {
            "provider": provider_name,
            "model": client.model,
            "response": response,
            "time": elapsed,
            "success": True,
            "error": None,
        }

    except Exception as e:
        print(f"❌ Error: {e}")
        return {"provider": provider_name, "success": False, "error": str(e), "time": 0}


def main():
    print("=" * 60)
    print("  LLM Provider Comparison Test")
    print("=" * 60)

    # Test prompt
    test_prompt = "Explain what a vector database is in one sentence."

    # Test all providers
    providers = [LLMProvider.ANTHROPIC.value, LLMProvider.OPENAI.value, LLMProvider.GROQ.value]

    results = []

    for provider in providers:
        result = test_provider(provider, test_prompt)
        results.append(result)
        time.sleep(1)  # Brief pause between tests

    # Summary
    print(f"\n{'=' * 60}")
    print("  SUMMARY")
    print(f"{'=' * 60}\n")

    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    if successful:
        print("✅ Successful providers:")
        for r in successful:
            print(f"   - {r['provider']:12} ({r['model']:30}) {r['time']:.2f}s")

        # Speed comparison
        fastest = min(successful, key=lambda x: x["time"])
        print(f"\n🚀 Fastest: {fastest['provider']} ({fastest['time']:.2f}s)")

    if failed:
        print("\n❌ Failed providers:")
        for r in failed:
            print(f"   - {r['provider']:12} Error: {r['error']}")

    # Recommendations
    print(f"\n{'=' * 60}")
    print("  RECOMMENDATIONS")
    print(f"{'=' * 60}\n")

    print("📝 To use a specific provider, set in .env:")
    print("")
    print("   LLM_PROVIDER=anthropic   # For quality")
    print("   LLM_PROVIDER=openai      # For balance")
    print("   LLM_PROVIDER=groq        # For speed")
    print("")

    if failed:
        print("⚠️  Failed providers need API keys in .env:")
        for r in failed:
            provider = r["provider"].upper()
            print(f"   {provider}_API_KEY=your_key_here")

    print(f"\n{'=' * 60}\n")


if __name__ == "__main__":
    main()
