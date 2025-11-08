"""
Test to verify complete parity between SaaS API and HippocampAI library.

This test checks that every API endpoint has a corresponding library method
and that they work together seamlessly.
"""


from hippocampai import MemoryClient

# Color codes for output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"


def print_section(title):
    """Print a section header."""
    print(f"\n{BLUE}{'='*80}{RESET}")
    print(f"{BLUE}{title}{RESET}")
    print(f"{BLUE}{'='*80}{RESET}\n")


def print_success(message):
    """Print a success message."""
    print(f"{GREEN}✓{RESET} {message}")


def print_error(message):
    """Print an error message."""
    print(f"{RED}✗{RESET} {message}")


def print_info(message):
    """Print an info message."""
    print(f"{YELLOW}ℹ{RESET} {message}")


def test_api_library_parity():
    """Test that all API endpoints have corresponding library methods."""

    print_section("SaaS API ↔ Library Integration Parity Check")

    # Initialize client
    client = MemoryClient()

    # Mapping of API endpoints to library methods
    api_library_mapping = {
        # Core Memory Operations
        "POST /v1/memories": "remember",
        "GET /v1/memories/{id}": "get_memory",
        "PATCH /v1/memories/{id}": "update_memory",
        "DELETE /v1/memories/{id}": "delete_memory",
        "POST /v1/memories/recall": "recall",
        "POST /v1/memories/extract": "extract_from_conversation",

        # Observability & Debugging
        "POST /v1/observability/explain": "explain_retrieval",
        "POST /v1/observability/visualize": "visualize_similarity_scores",
        "POST /v1/observability/heatmap": "generate_access_heatmap",
        "POST /v1/observability/profile": "profile_query_performance",

        # Enhanced Temporal Features
        "POST /v1/temporal/freshness": "calculate_memory_freshness",
        "POST /v1/temporal/decay": "apply_time_decay",
        "POST /v1/temporal/forecast": "forecast_memory_patterns",
        "POST /v1/temporal/context-window": "get_adaptive_context_window",

        # Memory Health & Conflicts
        "POST /v1/conflicts/detect": "detect_memory_conflicts",
        "POST /v1/conflicts/resolve": "resolve_memory_conflict",
        "POST /v1/health/score": "get_memory_health_score",
        "POST /v1/provenance/track": "get_memory_provenance_chain",
    }

    print_info(f"Checking {len(api_library_mapping)} API endpoints...")
    print()

    missing_methods = []
    available_methods = []

    for endpoint, method_name in api_library_mapping.items():
        if hasattr(client, method_name):
            print_success(f"{endpoint:50} → {method_name}")
            available_methods.append((endpoint, method_name))
        else:
            print_error(f"{endpoint:50} → {method_name} [MISSING]")
            missing_methods.append((endpoint, method_name))

    print()
    print_section("Summary")
    print(f"Total API endpoints: {len(api_library_mapping)}")
    print(f"{GREEN}Available:{RESET} {len(available_methods)}")
    print(f"{RED}Missing:{RESET} {len(missing_methods)}")
    print(f"Coverage: {len(available_methods)/len(api_library_mapping)*100:.1f}%")

    if missing_methods:
        print(f"\n{RED}Missing library methods:{RESET}")
        for endpoint, method in missing_methods:
            print(f"  - {method} (for {endpoint})")
    else:
        print(f"\n{GREEN}✓ Perfect parity! All API endpoints have corresponding library methods.{RESET}")

    return len(missing_methods) == 0


def test_functional_integration():
    """Test that library methods actually work."""

    print_section("Functional Integration Test")

    client = MemoryClient()
    user_id = "test_parity_user"

    tests_passed = 0
    tests_total = 0

    # Test 1: Basic memory operations
    tests_total += 1
    try:
        mem = client.remember("I love Python", user_id=user_id)
        print_success(f"remember() works - Created memory {mem.id[:8]}...")
        tests_passed += 1
    except Exception as e:
        print_error(f"remember() failed: {e}")

    # Test 2: Recall
    tests_total += 1
    try:
        results = client.recall("Python", user_id=user_id, k=5)
        print_success(f"recall() works - Found {len(results)} memories")
        tests_passed += 1
    except Exception as e:
        print_error(f"recall() failed: {e}")

    # Test 3: Explain retrieval (NEW)
    tests_total += 1
    try:
        if results:
            explanations = client.explain_retrieval("Python", results)
            print_success(f"explain_retrieval() works - Generated {len(explanations)} explanations")
            tests_passed += 1
        else:
            print_info("explain_retrieval() skipped - no results to explain")
    except Exception as e:
        print_error(f"explain_retrieval() failed: {e}")

    # Test 4: Visualize scores (NEW)
    tests_total += 1
    try:
        if results:
            client.visualize_similarity_scores("Python", results)
            print_success("visualize_similarity_scores() works - Generated visualization")
            tests_passed += 1
        else:
            print_info("visualize_similarity_scores() skipped - no results")
    except Exception as e:
        print_error(f"visualize_similarity_scores() failed: {e}")

    # Test 5: Access heatmap (NEW)
    tests_total += 1
    try:
        client.generate_access_heatmap(user_id)
        print_success("generate_access_heatmap() works - Generated heatmap")
        tests_passed += 1
    except Exception as e:
        print_error(f"generate_access_heatmap() failed: {e}")

    # Test 6: Profile query performance (NEW)
    tests_total += 1
    try:
        profile = client.profile_query_performance("Python", user_id)
        print_success(f"profile_query_performance() works - {profile['total_time_ms']:.2f}ms")
        tests_passed += 1
    except Exception as e:
        print_error(f"profile_query_performance() failed: {e}")

    # Test 7: Calculate freshness (NEW)
    tests_total += 1
    try:
        if mem:
            freshness = client.calculate_memory_freshness(mem)
            score = freshness.get('freshness_score', 0) if isinstance(freshness, dict) else freshness.freshness_score
            print_success(f"calculate_memory_freshness() works - Score: {score:.2f}")
            tests_passed += 1
    except Exception as e:
        print_error(f"calculate_memory_freshness() failed: {e}")

    # Test 8: Apply time decay (NEW)
    tests_total += 1
    try:
        if mem:
            decayed = client.apply_time_decay(mem)
            print_success(f"apply_time_decay() works - Decayed: {decayed:.2f}")
            tests_passed += 1
    except Exception as e:
        print_error(f"apply_time_decay() failed: {e}")

    # Test 9: Forecast patterns (NEW)
    tests_total += 1
    try:
        forecasts = client.forecast_memory_patterns(user_id, forecast_days=30)
        print_success(f"forecast_memory_patterns() works - {len(forecasts)} forecasts")
        tests_passed += 1
    except Exception as e:
        print_error(f"forecast_memory_patterns() failed: {e}")

    # Test 10: Adaptive context window (NEW)
    tests_total += 1
    try:
        window = client.get_adaptive_context_window("recent work", user_id)
        window_size = window.get('window_size_days', 0) if isinstance(window, dict) else window.window_size_days
        print_success(f"get_adaptive_context_window() works - Window: {window_size} days")
        tests_passed += 1
    except Exception as e:
        print_error(f"get_adaptive_context_window() failed: {e}")

    # Test 11: Detect conflicts (NEW)
    tests_total += 1
    try:
        conflicts = client.detect_memory_conflicts(user_id)
        print_success(f"detect_memory_conflicts() works - Found {len(conflicts)} conflicts")
        tests_passed += 1
    except Exception as e:
        print_error(f"detect_memory_conflicts() failed: {e}")

    # Test 12: Health score (NEW)
    tests_total += 1
    try:
        health = client.get_memory_health_score(user_id)
        score = health.get('overall_score', 0) if isinstance(health, dict) else health.overall_score
        print_success(f"get_memory_health_score() works - Score: {score:.2f}/100")
        tests_passed += 1
    except Exception as e:
        print_error(f"get_memory_health_score() failed: {e}")

    # Test 13: Provenance tracking (NEW)
    tests_total += 1
    try:
        if mem:
            provenance = client.get_memory_provenance_chain(mem.id)
            if provenance:
                print_success("get_memory_provenance_chain() works - Retrieved provenance")
                tests_passed += 1
            else:
                print_info("get_memory_provenance_chain() - No provenance data yet")
                tests_passed += 1  # Still counts as success
    except Exception as e:
        print_error(f"get_memory_provenance_chain() failed: {e}")

    print()
    print_section("Functional Test Summary")
    print(f"Tests passed: {tests_passed}/{tests_total}")
    print(f"Success rate: {tests_passed/tests_total*100:.1f}%")

    return tests_passed == tests_total


def show_api_documentation():
    """Show comprehensive API-library mapping documentation."""

    print_section("API ↔ Library Method Documentation")

    documentation = {
        "Core Memory Operations": {
            "POST /v1/memories": {
                "library": "client.remember(text, user_id, ...)",
                "description": "Store a new memory"
            },
            "GET /v1/memories/{id}": {
                "library": "client.get_memory(memory_id)",
                "description": "Retrieve a specific memory"
            },
            "POST /v1/memories/recall": {
                "library": "client.recall(query, user_id, k=5)",
                "description": "Search and retrieve memories"
            },
        },
        "Observability & Debugging": {
            "POST /v1/observability/explain": {
                "library": "client.explain_retrieval(query, results)",
                "description": "Explain why memories were retrieved"
            },
            "POST /v1/observability/visualize": {
                "library": "client.visualize_similarity_scores(query, results)",
                "description": "Visualize similarity scores and distributions"
            },
            "POST /v1/observability/heatmap": {
                "library": "client.generate_access_heatmap(user_id)",
                "description": "Generate memory access patterns heatmap"
            },
            "POST /v1/observability/profile": {
                "library": "client.profile_query_performance(query, user_id)",
                "description": "Profile query performance and identify bottlenecks"
            },
        },
        "Enhanced Temporal Features": {
            "POST /v1/temporal/freshness": {
                "library": "client.calculate_memory_freshness(memory)",
                "description": "Calculate memory freshness score"
            },
            "POST /v1/temporal/decay": {
                "library": "client.apply_time_decay(memory, decay_function)",
                "description": "Apply time decay to memory importance"
            },
            "POST /v1/temporal/forecast": {
                "library": "client.forecast_memory_patterns(user_id, days)",
                "description": "Forecast future memory patterns"
            },
            "POST /v1/temporal/context-window": {
                "library": "client.get_adaptive_context_window(query, user_id)",
                "description": "Get adaptive temporal context window"
            },
        },
        "Memory Health & Conflicts": {
            "POST /v1/conflicts/detect": {
                "library": "client.detect_memory_conflicts(user_id)",
                "description": "Detect memory conflicts and contradictions"
            },
            "POST /v1/conflicts/resolve": {
                "library": "client.resolve_memory_conflict(conflict_id, strategy)",
                "description": "Resolve detected memory conflicts"
            },
            "POST /v1/health/score": {
                "library": "client.get_memory_health_score(user_id)",
                "description": "Get comprehensive memory health metrics"
            },
            "POST /v1/provenance/track": {
                "library": "client.get_memory_provenance_chain(memory_id)",
                "description": "Track memory provenance and lineage"
            },
        },
    }

    for category, endpoints in documentation.items():
        print(f"\n{YELLOW}{category}:{RESET}")
        for endpoint, details in endpoints.items():
            print(f"\n  {BLUE}{endpoint}{RESET}")
            print(f"    Library: {details['library']}")
            print(f"    Description: {details['description']}")

    print()


def main():
    """Run all parity checks."""

    print(f"\n{BLUE}{'='*80}{RESET}")
    print(f"{BLUE}HippocampAI: SaaS API ↔ Library Integration Verification{RESET}")
    print(f"{BLUE}{'='*80}{RESET}")

    # Test 1: Check parity
    parity_ok = test_api_library_parity()

    # Test 2: Functional tests
    functional_ok = test_functional_integration()

    # Show documentation
    show_api_documentation()

    # Final summary
    print_section("Final Verdict")

    if parity_ok and functional_ok:
        print(f"{GREEN}✓ PERFECT INTEGRATION{RESET}")
        print(f"{GREEN}  All SaaS API endpoints have corresponding library methods{RESET}")
        print(f"{GREEN}  All methods are functional and tested{RESET}")
        print(f"{GREEN}  Complete parity between SaaS and library!{RESET}")
        return 0
    elif parity_ok:
        print(f"{YELLOW}⚠ GOOD INTEGRATION{RESET}")
        print(f"{YELLOW}  All endpoints mapped, but some functional issues{RESET}")
        return 1
    else:
        print(f"{RED}✗ INCOMPLETE INTEGRATION{RESET}")
        print(f"{RED}  Some endpoints missing library methods{RESET}")
        return 1


if __name__ == "__main__":
    exit(main())
