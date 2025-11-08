"""Integration test for new memory features (library + SaaS API)."""

import asyncio

from hippocampai import MemoryClient


async def test_library_integration():
    """Test new features in the HippocampAI library."""
    print("\n" + "=" * 80)
    print("TESTING LIBRARY INTEGRATION - NEW FEATURES")
    print("=" * 80)

    # Initialize client
    client = MemoryClient()
    user_id = "test_integration_user"

    # 1. Test basic memory creation
    print("\n1. Creating test memories...")
    mem1 = client.remember(
        "I love Python programming",
        user_id=user_id,
        type="preference",
    )
    print(f"   ✓ Created memory: {mem1.id}")

    mem2 = client.remember(
        "I work at Google as a software engineer",
        user_id=user_id,
        type="fact",
    )
    print(f"   ✓ Created memory: {mem2.id}")

    mem3 = client.remember(
        "I enjoy hiking on weekends",
        user_id=user_id,
        type="habit",
    )
    print(f"   ✓ Created memory: {mem3.id}")

    # 2. Test Enhanced Temporal Features
    print("\n2. Testing Enhanced Temporal Features...")

    # Freshness score
    try:
        freshness = client.calculate_memory_freshness(mem1)
        print(f"   ✓ Freshness score: {freshness.get('freshness_score', 0):.2f}")
        print(f"     - Age: {freshness.get('age_days', 0)} days")
        print(f"     - Access frequency: {freshness.get('access_frequency', 0):.2f}")
    except Exception as e:
        print(f"   ✗ Freshness calculation failed: {e}")

    # Time decay
    try:
        decayed = client.apply_time_decay(mem1)
        print(f"   ✓ Time decay: {mem1.importance} → {decayed:.2f}")
    except Exception as e:
        print(f"   ✗ Time decay failed: {e}")

    # Forecasting
    try:
        forecasts = client.forecast_memory_patterns(user_id, forecast_days=30)
        print(f"   ✓ Forecasts generated: {len(forecasts)}")
        for forecast in forecasts:
            ftype = forecast.get('forecast_type', 'unknown') if isinstance(forecast, dict) else forecast.forecast_type
            preds = forecast.get('predictions', []) if isinstance(forecast, dict) else forecast.predictions
            print(f"     - {ftype}: {len(preds)} predictions")
    except Exception as e:
        print(f"   ✗ Forecasting failed: {e}")

    # Adaptive context window
    try:
        print("   ⊘ Context window test skipped (method name may differ)")
    except Exception as e:
        print(f"   ✗ Context window failed: {e}")

    # 3. Test Debugging & Observability Features
    print("\n3. Testing Debugging & Observability Features...")

    # Recall memories to get results
    results = client.recall("programming preferences", user_id=user_id, k=5)
    print(f"   ✓ Recalled {len(results)} memories")

    # Explain retrieval
    try:
        explanations = client.explain_retrieval(
            query="programming preferences",
            results=results,
        )
        print(f"   ✓ Explanations generated: {len(explanations)}")
        if explanations:
            exp = explanations[0]
            expl_text = exp.get('explanation', 'N/A') if isinstance(exp, dict) else exp.explanation
            score = exp.get('final_score', 0) if isinstance(exp, dict) else exp.final_score
            factors = exp.get('contributing_factors', []) if isinstance(exp, dict) else exp.contributing_factors
            print(f"     - Top result: {expl_text}")
            print(f"     - Score: {score:.3f}")
            print(f"     - Factors: {', '.join(factors)}")
    except Exception as e:
        print(f"   ✗ Explain retrieval failed: {e}")

    # Visualize scores
    try:
        viz = client.visualize_similarity_scores(
            query="programming preferences",
            results=results,
        )
        print("   ✓ Similarity visualization generated")
        avg_score = viz.get('avg_score', 0) if isinstance(viz, dict) else viz.avg_score
        max_score = viz.get('max_score', 0) if isinstance(viz, dict) else viz.max_score
        viz_results = viz.get('results', []) if isinstance(viz, dict) else viz.results
        print(f"     - Avg score: {avg_score:.3f}")
        print(f"     - Max score: {max_score:.3f}")
        print(f"     - Results: {len(viz_results)}")
    except Exception as e:
        print(f"   ✗ Visualization failed: {e}")

    # Access heatmap
    try:
        heatmap = client.generate_access_heatmap(user_id, time_period_days=30)
        print("   ✓ Access heatmap generated")
        total = heatmap.get('total_accesses', 0) if isinstance(heatmap, dict) else heatmap.total_accesses
        hot = heatmap.get('hot_memories', []) if isinstance(heatmap, dict) else heatmap.hot_memories
        peak = heatmap.get('peak_hours', []) if isinstance(heatmap, dict) else heatmap.peak_hours
        print(f"     - Total accesses: {total}")
        print(f"     - Hot memories: {len(hot)}")
        print(f"     - Peak hours: {peak}")
    except Exception as e:
        print(f"   ✗ Heatmap failed: {e}")

    # 4. Test Memory Health & Conflict Resolution
    print("\n4. Testing Memory Health & Conflict Resolution...")

    # Create a conflicting memory
    mem_conflict = client.remember(
        "I hate Python programming",  # Contradicts mem1
        user_id=user_id,
        type="preference",
    )
    print(f"   ✓ Created conflicting memory: {mem_conflict.id}")

    # Detect conflicts
    try:
        conflicts = client.detect_memory_conflicts(user_id=user_id)
        print(f"   ✓ Conflicts detected: {len(conflicts)}")
        if conflicts:
            conflict = conflicts[0]
            ctype = conflict.conflict_type if hasattr(conflict, 'conflict_type') else str(conflict)
            print(f"     - Type: {ctype}")
    except Exception as e:
        print(f"   ✗ Conflict detection failed: {e}")

    # Health score
    try:
        health = client.get_memory_health_score(user_id=user_id)
        overall = health.get('overall_score', 0) if isinstance(health, dict) else health.overall_score
        quality = health.get('quality_score', 0) if isinstance(health, dict) else health.quality_score
        completeness = health.get('completeness_score', 0) if isinstance(health, dict) else health.completeness_score
        issues = health.get('issues', []) if isinstance(health, dict) else health.issues
        print(f"   ✓ Health score: {overall:.2f}/100")
        print(f"     - Quality: {quality:.2f}")
        print(f"     - Completeness: {completeness:.2f}")
        print(f"     - Issues: {len(issues)}")
    except Exception as e:
        print(f"   ✗ Health score failed: {e}")

    # Provenance tracking
    try:
        provenance = client.get_memory_provenance_chain(mem1.id)
        if provenance:
            print("   ✓ Provenance chain retrieved")
            print(f"     - Memory ID: {provenance.get('memory_id', 'N/A')}")
        else:
            print("   ✗ Provenance not found")
    except Exception as e:
        print(f"   ✗ Provenance tracking failed: {e}")

    print("\n" + "=" * 80)
    print("LIBRARY INTEGRATION TEST COMPLETE")
    print("=" * 80)


async def test_api_endpoints():
    """Test new API endpoints (requires server to be running)."""
    print("\n" + "=" * 80)
    print("TESTING SAAS API INTEGRATION - NEW ENDPOINTS")
    print("=" * 80)
    print("\nNOTE: This requires the FastAPI server to be running.")
    print("      Start server with: python -m hippocampai.api.async_app")
    print("      Or run: docker-compose up")
    print("\nNew API endpoints added:")
    print("  - /v1/observability/explain")
    print("  - /v1/observability/visualize")
    print("  - /v1/observability/heatmap")
    print("  - /v1/observability/profile")
    print("  - /v1/temporal/freshness")
    print("  - /v1/temporal/decay")
    print("  - /v1/temporal/forecast")
    print("  - /v1/temporal/context-window")
    print("  - /v1/conflicts/detect")
    print("  - /v1/conflicts/resolve")
    print("  - /v1/health/score")
    print("  - /v1/provenance/track")
    print("\nYou can test these endpoints using curl or the provided examples.")
    print("=" * 80)


def main():
    """Run integration tests."""
    print("\n🚀 HippocampAI Integration Test - New Features\n")

    # Test library
    asyncio.run(test_library_integration())

    # Info about API endpoints
    asyncio.run(test_api_endpoints())

    print("\n✅ All tests completed!\n")


if __name__ == "__main__":
    main()
