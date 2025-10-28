"""Comprehensive demo of Advanced Intelligence and Temporal features.

This script demonstrates:
1. Fact Extraction with quality scoring
2. Entity Recognition with canonical naming
3. Relationship Mapping with strength scoring
4. Semantic Clustering with hierarchical analysis
5. Temporal Analytics with peak times and trends
"""

import json
from datetime import datetime, timedelta, timezone

# Import models
from hippocampai.models.memory import Memory, MemoryType

# Import pipeline modules
from hippocampai.pipeline.entity_recognition import EntityRecognizer
from hippocampai.pipeline.fact_extraction import FactExtractionPipeline
from hippocampai.pipeline.relationship_mapping import RelationshipMapper
from hippocampai.pipeline.semantic_clustering import SemanticCategorizer
from hippocampai.pipeline.temporal_analytics import TemporalAnalytics


def demo_fact_extraction():
    """Demonstrate enhanced fact extraction with quality scoring."""
    print("\n" + "=" * 80)
    print("DEMO 1: Fact Extraction with Quality Scoring")
    print("=" * 80)

    extractor = FactExtractionPipeline()

    # Sample texts
    texts = [
        "I work at Google as a Senior Software Engineer in Mountain View, California.",
        "John Smith graduated from Stanford University with a degree in Computer Science in 2015.",
        "I love playing tennis every weekend and enjoying Italian cuisine.",
        "My goal is to learn machine learning and build AI applications next year.",
    ]

    for i, text in enumerate(texts, 1):
        print(f"\n--- Text {i} ---")
        print(f"Input: {text}")

        # Extract facts with quality scoring
        facts = extractor.extract_facts_with_quality(text, source="demo", user_id="demo_user")

        print(f"\nExtracted {len(facts)} facts:")
        for fact in facts:
            print(f"\n  Fact: {fact.fact}")
            print(f"  Category: {fact.category.value}")
            print(f"  Confidence: {fact.confidence:.2f}")
            print(f"  Quality Score: {fact.quality_score:.2f}")
            print(f"  Entities: {', '.join(fact.entities) if fact.entities else 'None'}")
            if fact.temporal:
                print(f"  Temporal: {fact.temporal} ({fact.temporal_type.value})")

            # Show quality metrics
            if "quality_metrics" in fact.metadata:
                metrics = fact.metadata["quality_metrics"]
                print("  Quality Metrics:")
                print(f"    - Specificity: {metrics['specificity']:.2f}")
                print(f"    - Verifiability: {metrics['verifiability']:.2f}")
                print(f"    - Completeness: {metrics['completeness']:.2f}")
                print(f"    - Clarity: {metrics['clarity']:.2f}")


def demo_entity_recognition():
    """Demonstrate enhanced entity recognition with canonical naming."""
    print("\n" + "=" * 80)
    print("DEMO 2: Entity Recognition with Canonical Naming")
    print("=" * 80)

    recognizer = EntityRecognizer()

    # Sample text with various entity types
    text = """
    John Smith works at Google Inc. in San Francisco. He can be reached at john.smith@gmail.com
    or at +1-555-123-4567. John is skilled in Python, TensorFlow, and Docker. He has a Ph.D.
    from MIT and is AWS Certified. Visit his website at https://johnsmith.dev
    """

    print(f"\nInput text:\n{text}")

    # Extract entities
    entities = recognizer.extract_entities(text)

    print(f"\nExtracted {len(entities)} entities:")
    for entity in entities:
        print(f"\n  Text: {entity.text}")
        print(f"  Type: {entity.type.value}")
        print(f"  Canonical Name: {entity.canonical_name}")
        print(f"  Entity ID: {entity.entity_id}")
        print(f"  Confidence: {entity.confidence:.2f}")

    # Show statistics
    stats = recognizer.get_entity_statistics()
    print("\n--- Entity Statistics ---")
    print(f"Total entities: {stats['total_entities']}")
    print(f"By type: {json.dumps(stats['by_type'], indent=2)}")
    print("Top mentioned:")
    for entity in stats["top_mentioned"][:5]:
        print(
            f"  - {entity['canonical_name']} ({entity['type']}): {entity['mention_count']} mentions"
        )


def demo_relationship_mapping():
    """Demonstrate relationship mapping with strength scoring."""
    print("\n" + "=" * 80)
    print("DEMO 3: Relationship Mapping with Strength Scoring")
    print("=" * 80)

    entity_recognizer = EntityRecognizer()
    mapper = RelationshipMapper()

    # Sample texts describing relationships
    texts = [
        "Alice Johnson works at Microsoft in Seattle.",
        "Bob Smith lives in New York City and works at Amazon.",
        "Alice Johnson graduated from Harvard University.",
        "Microsoft is located in Seattle, Washington.",
        "Bob Smith studied Computer Science at MIT.",
    ]

    print("\nProcessing relationship data...")
    for text in texts:
        print(f"  - {text}")

        # Extract entities and relationships
        entities = entity_recognizer.extract_entities(text)
        relationships = entity_recognizer.extract_relationships(text, entities)

        # Add to mapper
        for rel in relationships:
            mapper.add_relationship_from_entity_relationship(rel)

    # Analyze network
    print("\n--- Relationship Network Analysis ---")
    network = mapper.analyze_network()

    print(f"Total entities: {len(network.entities)}")
    print(f"Total relationships: {len(network.relationships)}")
    print(f"Network density: {network.network_density:.2f}")
    print(f"Number of clusters: {len(network.clusters)}")

    print("\nTop 5 central entities:")
    for entity_id, centrality in network.central_entities[:5]:
        print(f"  - {entity_id}: {centrality:.2f}")

    print("\nRelationship strength distribution:")
    for rel in network.relationships[:5]:
        print(
            f"  - {rel.from_entity_id} --[{rel.relation_type.value}]--> {rel.to_entity_id}"
        )
        print(f"    Strength: {rel.strength_score:.2f} ({rel.strength_level.value})")
        print(f"    Co-occurrences: {rel.co_occurrence_count}")

    # Export visualization data
    viz_data = mapper.export_for_visualization()
    print("\nVisualization data exported:")
    print(f"  Nodes: {len(viz_data['nodes'])}")
    print(f"  Edges: {len(viz_data['edges'])}")


def demo_semantic_clustering():
    """Demonstrate semantic clustering with quality metrics."""
    print("\n" + "=" * 80)
    print("DEMO 4: Semantic Clustering with Hierarchical Analysis")
    print("=" * 80)

    categorizer = SemanticCategorizer()

    # Create sample memories
    memories = [
        # Work-related
        Memory(
            id="1",
            text="Had a productive meeting with the engineering team about the new API design",
            user_id="demo_user",
            type=MemoryType.EVENT,
            created_at=datetime.now(timezone.utc) - timedelta(hours=5),
        ),
        Memory(
            id="2",
            text="Completed the quarterly project review and set goals for next quarter",
            user_id="demo_user",
            type=MemoryType.GOAL,
            created_at=datetime.now(timezone.utc) - timedelta(hours=10),
        ),
        # Health-related
        Memory(
            id="3",
            text="Went for a 5-mile run in the morning, feeling energized",
            user_id="demo_user",
            type=MemoryType.EVENT,
            created_at=datetime.now(timezone.utc) - timedelta(hours=12),
        ),
        Memory(
            id="4",
            text="Started a new healthy eating plan with more vegetables and lean protein",
            user_id="demo_user",
            type=MemoryType.HABIT,
            created_at=datetime.now(timezone.utc) - timedelta(hours=15),
        ),
        # Learning-related
        Memory(
            id="5",
            text="Completed two chapters of the machine learning course, really enjoying it",
            user_id="demo_user",
            type=MemoryType.EVENT,
            created_at=datetime.now(timezone.utc) - timedelta(hours=20),
        ),
        Memory(
            id="6",
            text="Want to master deep learning and build a computer vision project",
            user_id="demo_user",
            type=MemoryType.GOAL,
            created_at=datetime.now(timezone.utc) - timedelta(hours=24),
        ),
    ]

    print(f"\nClustering {len(memories)} memories...")

    # Standard clustering
    print("\n--- Standard Clustering ---")
    clusters = categorizer.cluster_memories(memories, max_clusters=5)

    for i, cluster in enumerate(clusters, 1):
        print(f"\nCluster {i}: {cluster.topic}")
        print(f"  Size: {len(cluster.memories)} memories")
        print(f"  Common tags: {', '.join(cluster.tags) if cluster.tags else 'None'}")

        # Compute quality metrics
        metrics = categorizer.compute_cluster_quality_metrics(cluster)
        print("  Quality Metrics:")
        print(f"    - Cohesion: {metrics['cohesion']:.2f}")
        print(f"    - Diversity: {metrics['diversity']:.2f}")
        print(f"    - Temporal density: {metrics['temporal_density']:.2f}")

    # Hierarchical clustering
    print("\n--- Hierarchical Clustering ---")
    hierarchical_result = categorizer.hierarchical_cluster_memories(memories, min_cluster_size=2)

    for i, cluster_data in enumerate(hierarchical_result["clusters"], 1):
        print(f"\nHierarchical Cluster {i}: {cluster_data['topic']}")
        print(f"  Size: {cluster_data['size']} memories")
        print(f"  Cohesion: {cluster_data['cohesion']:.2f}")

    # Optimize cluster count
    optimal_k = categorizer.optimize_cluster_count(memories)
    print(f"\nOptimal number of clusters: {optimal_k}")


def demo_temporal_analytics():
    """Demonstrate temporal analytics with peak times and trends."""
    print("\n" + "=" * 80)
    print("DEMO 5: Temporal Analytics with Peak Times and Trends")
    print("=" * 80)

    analytics = TemporalAnalytics()

    # Create sample memories with temporal patterns
    now = datetime.now(timezone.utc)
    memories = []

    # Create memories with patterns: morning work, evening exercise
    for day in range(30):
        # Morning work memories (around 9 AM)
        memories.append(
            Memory(
                id=f"work_{day}_1",
                text=f"Daily standup meeting on day {day}",
                user_id="demo_user",
                type=MemoryType.EVENT,
                created_at=now - timedelta(days=day, hours=15),  # 9 AM
                importance=7.0,
            )
        )

        # Afternoon work memories (around 2 PM)
        if day % 2 == 0:  # Every other day
            memories.append(
                Memory(
                    id=f"work_{day}_2",
                    text=f"Code review session on day {day}",
                    user_id="demo_user",
                    type=MemoryType.EVENT,
                    created_at=now - timedelta(days=day, hours=10),  # 2 PM
                    importance=6.0,
                )
            )

        # Evening exercise memories (around 6 PM)
        if day % 3 == 0:  # Every third day
            memories.append(
                Memory(
                    id=f"exercise_{day}",
                    text=f"Evening workout on day {day}",
                    user_id="demo_user",
                    type=MemoryType.HABIT,
                    created_at=now - timedelta(days=day, hours=6),  # 6 PM
                    importance=8.0,
                )
            )

    print(f"\nAnalyzing {len(memories)} memories over 30 days...")

    # Peak activity analysis
    print("\n--- Peak Activity Analysis ---")
    peak_analysis = analytics.analyze_peak_activity(memories, timezone_offset=0)

    print(f"Peak hour: {peak_analysis.peak_hour}:00")
    print(f"Peak day: {peak_analysis.peak_day.value}")
    print(f"Peak time period: {peak_analysis.peak_time_period.value}")

    print("\nHourly distribution (top 5 hours):")
    sorted_hours = sorted(
        peak_analysis.hourly_distribution.items(), key=lambda x: x[1], reverse=True
    )
    for hour, count in sorted_hours[:5]:
        print(f"  {hour:02d}:00 - {count} memories")

    # Temporal pattern detection
    print("\n--- Temporal Pattern Detection ---")
    patterns = analytics.detect_temporal_patterns(memories, min_occurrences=3)

    print(f"Detected {len(patterns)} patterns:")
    for pattern in patterns:
        print(f"\n  Pattern: {pattern.description}")
        print(f"  Type: {pattern.pattern_type}")
        print(f"  Frequency: {pattern.frequency:.2f} times per period")
        print(f"  Confidence: {pattern.confidence:.2f}")
        print(f"  Regularity: {pattern.regularity_score:.2f}")
        print(f"  Occurrences: {len(pattern.occurrences)}")
        if pattern.next_predicted:
            print(f"  Next predicted: {pattern.next_predicted.strftime('%Y-%m-%d %H:%M')}")

    # Trend analysis
    print("\n--- Trend Analysis ---")
    activity_trend = analytics.analyze_trends(memories, time_window_days=30, metric="activity")

    print("Activity Trend:")
    print(f"  Direction: {activity_trend.direction.value}")
    print(f"  Strength: {activity_trend.strength:.2f}")
    print(f"  Change rate: {activity_trend.change_rate:.2f} per day")
    print(f"  Current value: {activity_trend.current_value:.2f}")
    if activity_trend.forecast:
        print(f"  Forecast (next period): {activity_trend.forecast:.2f}")

    importance_trend = analytics.analyze_trends(
        memories, time_window_days=30, metric="importance"
    )

    print("\nImportance Trend:")
    print(f"  Direction: {importance_trend.direction.value}")
    print(f"  Strength: {importance_trend.strength:.2f}")
    print(f"  Current average importance: {importance_trend.current_value:.2f}")

    # Temporal clustering
    print("\n--- Temporal Clustering ---")
    temporal_clusters = analytics.cluster_by_time(memories, max_gap_hours=12)

    print(f"Found {len(temporal_clusters)} temporal clusters:")
    for i, cluster in enumerate(temporal_clusters[:5], 1):
        print(f"\n  Cluster {i}:")
        print(f"    Duration: {cluster.duration_hours:.1f} hours")
        print(f"    Memories: {len(cluster.memories)}")
        print(f"    Density: {cluster.density:.2f} memories/hour")
        print(
            f"    Time range: {cluster.start_time.strftime('%Y-%m-%d %H:%M')} to {cluster.end_time.strftime('%Y-%m-%d %H:%M')}"
        )
        if cluster.dominant_type:
            print(f"    Dominant type: {cluster.dominant_type.value}")


def main():
    """Run all demonstrations."""
    print("\n")
    print("=" * 80)
    print("HIPPOCAMPAI - ADVANCED INTELLIGENCE FEATURES DEMO")
    print("=" * 80)

    # Run all demos
    demo_fact_extraction()
    demo_entity_recognition()
    demo_relationship_mapping()
    demo_semantic_clustering()
    demo_temporal_analytics()

    print("\n" + "=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)
    print("\nAll Advanced Intelligence features have been demonstrated successfully!")
    print("\nFor API usage, start the FastAPI server and use the following endpoints:")
    print("  - POST /v1/intelligence/facts:extract")
    print("  - POST /v1/intelligence/entities:extract")
    print("  - POST /v1/intelligence/relationships:analyze")
    print("  - POST /v1/intelligence/clustering:analyze")
    print("  - POST /v1/intelligence/temporal:analyze")
    print("\nSee intelligence_routes.py for complete API documentation.")


if __name__ == "__main__":
    main()
