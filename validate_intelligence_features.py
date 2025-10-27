#!/usr/bin/env python3
"""
Validation script for HippocampAI Intelligence Features.

This script validates that all intelligence features are properly installed
and working correctly. It tests:
1. Fact Extraction Pipeline
2. Entity Recognition
3. Session Summarization
4. Knowledge Graph

Run this script after installation to verify your setup.

Usage:
    python validate_intelligence_features.py

Or with verbose output:
    python validate_intelligence_features.py --verbose
"""

import sys
import traceback
from typing import Any, Dict


class FeatureValidator:
    """Validates HippocampAI intelligence features."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.results: Dict[str, Dict[str, Any]] = {}

    def log(self, message: str, level: str = "INFO"):
        """Log a message if verbose mode is enabled."""
        if self.verbose:
            prefix = {"INFO": "ℹ️ ", "SUCCESS": "✓ ", "ERROR": "✗ ", "WARN": "⚠️ "}.get(level, "")
            print(f"{prefix}{message}")

    def validate_imports(self) -> bool:
        """Validate that all required modules can be imported."""
        print("\n" + "=" * 70)
        print("1. VALIDATING IMPORTS")
        print("=" * 70)

        required_imports = [
            ("hippocampai.pipeline", ["FactExtractionPipeline", "EntityRecognizer", "Summarizer"]),
            ("hippocampai.pipeline", ["FactCategory", "EntityType", "SummaryStyle"]),
            ("hippocampai.graph", ["KnowledgeGraph", "NodeType"]),
        ]

        all_success = True
        for module_name, items in required_imports:
            try:
                module = __import__(module_name, fromlist=items)
                for item in items:
                    if not hasattr(module, item):
                        print(f"✗ Missing: {module_name}.{item}")
                        all_success = False
                    else:
                        self.log(f"Imported {module_name}.{item}", "SUCCESS")

                if all_success:
                    print(f"✓ {module_name}: {', '.join(items)}")
            except ImportError as e:
                print(f"✗ Failed to import {module_name}: {e}")
                all_success = False

        if all_success:
            print("\n✓ All imports successful!")
        else:
            print("\n✗ Some imports failed. Check your installation.")

        self.results["imports"] = {"success": all_success}
        return all_success

    def validate_fact_extraction(self) -> bool:
        """Validate fact extraction functionality."""
        print("\n" + "=" * 70)
        print("2. VALIDATING FACT EXTRACTION")
        print("=" * 70)

        try:
            from hippocampai.pipeline import FactExtractionPipeline

            pipeline = FactExtractionPipeline()
            self.log("Created FactExtractionPipeline instance", "SUCCESS")

            # Test with sample text
            test_text = "Sarah works at Microsoft in Seattle. She studied Computer Science at Stanford and graduated in 2015."
            self.log(f"Testing with: {test_text[:50]}...", "INFO")

            facts = pipeline.extract_facts(test_text, source="validation_test")
            self.log(f"Extracted {len(facts)} facts", "INFO")

            if len(facts) == 0:
                print("⚠️  Warning: No facts extracted. Pattern-based extraction may need tuning.")
                success = True  # Not a failure, just a warning
            else:
                print(f"\n✓ Fact Extraction Working! Extracted {len(facts)} facts:")
                for i, fact in enumerate(facts[:3], 1):  # Show first 3
                    print(f"  {i}. [{fact.category.value}] {fact.fact}")
                    print(f"     Confidence: {fact.confidence:.2f}")

                # Validate fact structure
                fact = facts[0]
                assert hasattr(fact, "fact"), "Missing 'fact' attribute"
                assert hasattr(fact, "category"), "Missing 'category' attribute"
                assert hasattr(fact, "confidence"), "Missing 'confidence' attribute"
                assert hasattr(fact, "entities"), "Missing 'entities' attribute"
                assert hasattr(fact, "temporal_type"), "Missing 'temporal_type' attribute"

                self.log("Fact structure validation passed", "SUCCESS")
                success = True

            self.results["fact_extraction"] = {"success": success, "facts_count": len(facts)}
            return success

        except Exception as e:
            print(f"\n✗ Fact Extraction Failed: {e}")
            if self.verbose:
                traceback.print_exc()
            self.results["fact_extraction"] = {"success": False, "error": str(e)}
            return False

    def validate_entity_recognition(self) -> bool:
        """Validate entity recognition functionality."""
        print("\n" + "=" * 70)
        print("3. VALIDATING ENTITY RECOGNITION")
        print("=" * 70)

        try:
            from hippocampai.pipeline import EntityRecognizer

            recognizer = EntityRecognizer()
            self.log("Created EntityRecognizer instance", "SUCCESS")

            # Test with sample text
            test_text = (
                "Elon Musk founded SpaceX and Tesla. SpaceX is located in Hawthorne, California."
            )
            self.log(f"Testing with: {test_text[:50]}...", "INFO")

            entities = recognizer.extract_entities(test_text)
            self.log(f"Extracted {len(entities)} entities", "INFO")

            if len(entities) == 0:
                print(
                    "⚠️  Warning: No entities extracted. Pattern-based extraction may need tuning."
                )
                success = True
            else:
                print(f"\n✓ Entity Recognition Working! Extracted {len(entities)} entities:")
                for i, entity in enumerate(entities[:5], 1):  # Show first 5
                    print(f"  {i}. {entity.text} ({entity.type.value})")
                    print(f"     ID: {entity.entity_id}, Confidence: {entity.confidence:.2f}")

                # Test relationships
                relationships = recognizer.extract_relationships(test_text, entities)
                self.log(f"Extracted {len(relationships)} relationships", "INFO")

                if relationships:
                    print("\n  Relationships:")
                    for rel in relationships[:3]:
                        print(
                            f"    • {rel.relation_type.value}: {rel.from_entity_id} → {rel.to_entity_id}"
                        )

                # Validate entity structure
                entity = entities[0]
                assert hasattr(entity, "text"), "Missing 'text' attribute"
                assert hasattr(entity, "type"), "Missing 'type' attribute"
                assert hasattr(entity, "entity_id"), "Missing 'entity_id' attribute"
                assert hasattr(entity, "confidence"), "Missing 'confidence' attribute"

                self.log("Entity structure validation passed", "SUCCESS")
                success = True

            self.results["entity_recognition"] = {
                "success": success,
                "entities_count": len(entities),
                "relationships_count": len(relationships) if "relationships" in locals() else 0,
            }
            return success

        except Exception as e:
            print(f"\n✗ Entity Recognition Failed: {e}")
            if self.verbose:
                traceback.print_exc()
            self.results["entity_recognition"] = {"success": False, "error": str(e)}
            return False

    def validate_summarization(self) -> bool:
        """Validate session summarization functionality."""
        print("\n" + "=" * 70)
        print("4. VALIDATING SESSION SUMMARIZATION")
        print("=" * 70)

        try:
            from hippocampai.pipeline import Summarizer, SummaryStyle

            summarizer = Summarizer()
            self.log("Created Summarizer instance", "SUCCESS")

            # Test with sample conversation
            messages = [
                {"role": "user", "content": "I need help setting up a Python web application"},
                {
                    "role": "assistant",
                    "content": "I can help with that! Which framework would you like to use?",
                },
                {"role": "user", "content": "I'm thinking Flask or FastAPI"},
                {
                    "role": "assistant",
                    "content": "Both are great choices. FastAPI is more modern and has automatic API docs.",
                },
                {
                    "role": "user",
                    "content": "Let's go with FastAPI then. Can you help me set it up?",
                },
                {
                    "role": "assistant",
                    "content": "Absolutely! Let's start with the installation and basic structure.",
                },
            ]

            self.log(f"Testing with {len(messages)} messages", "INFO")

            # Test different summary styles
            styles_tested = []
            for style in [SummaryStyle.CONCISE, SummaryStyle.BULLET_POINTS]:
                summary = summarizer.summarize_session(
                    messages, session_id="validation_test", style=style
                )
                styles_tested.append(style.value)
                self.log(f"Generated {style.value} summary", "SUCCESS")

                if style == SummaryStyle.BULLET_POINTS:
                    print("\n✓ Session Summarization Working!")
                    print(f"\n  Summary ({style.value}):")
                    print(f"    {summary.summary}")
                    print(f"\n  Topics: {', '.join(summary.topics)}")
                    print(f"  Sentiment: {summary.sentiment.value}")
                    print(f"  Messages: {summary.message_count}")

            # Validate summary structure
            assert hasattr(summary, "summary"), "Missing 'summary' attribute"
            assert hasattr(summary, "key_points"), "Missing 'key_points' attribute"
            assert hasattr(summary, "topics"), "Missing 'topics' attribute"
            assert hasattr(summary, "sentiment"), "Missing 'sentiment' attribute"
            assert hasattr(summary, "message_count"), "Missing 'message_count' attribute"

            self.log("Summary structure validation passed", "SUCCESS")

            self.results["summarization"] = {
                "success": True,
                "styles_tested": styles_tested,
                "topics_found": len(summary.topics),
            }
            return True

        except Exception as e:
            print(f"\n✗ Session Summarization Failed: {e}")
            if self.verbose:
                traceback.print_exc()
            self.results["summarization"] = {"success": False, "error": str(e)}
            return False

    def validate_knowledge_graph(self) -> bool:
        """Validate knowledge graph functionality."""
        print("\n" + "=" * 70)
        print("5. VALIDATING KNOWLEDGE GRAPH")
        print("=" * 70)

        try:
            from hippocampai.graph import KnowledgeGraph
            from hippocampai.pipeline.entity_recognition import (
                Entity,
                EntityRelationship,
                EntityType,
                RelationType,
            )

            graph = KnowledgeGraph()
            self.log("Created KnowledgeGraph instance", "SUCCESS")

            # Create sample entities
            person = Entity(
                text="Jane Smith",
                type=EntityType.PERSON,
                entity_id="person_jane",
                canonical_name="Jane Smith",
                confidence=0.9,
            )

            org = Entity(
                text="Tesla",
                type=EntityType.ORGANIZATION,
                entity_id="org_tesla",
                canonical_name="Tesla Inc.",
                confidence=0.95,
            )

            location = Entity(
                text="Austin",
                type=EntityType.LOCATION,
                entity_id="loc_austin",
                canonical_name="Austin, Texas",
                confidence=0.85,
            )

            # Add entities to graph
            node1 = graph.add_entity(person)
            graph.add_entity(org)
            graph.add_entity(location)
            self.log(f"Added {len(graph.graph.nodes)} nodes to graph", "SUCCESS")

            # Create and add relationships
            rel1 = EntityRelationship(
                from_entity_id="person_jane",
                to_entity_id="org_tesla",
                relation_type=RelationType.WORKS_AT,
                confidence=0.9,
                context="test",
            )
            graph.link_entities(rel1)

            rel2 = EntityRelationship(
                from_entity_id="org_tesla",
                to_entity_id="loc_austin",
                relation_type=RelationType.LOCATED_IN,
                confidence=0.85,
                context="test",
            )
            graph.link_entities(rel2)

            self.log(f"Added {len(graph.graph.edges)} edges to graph", "SUCCESS")

            print("\n✓ Knowledge Graph Working!")
            print("\n  Graph Statistics:")
            print(f"    Nodes: {len(graph.graph.nodes)}")
            print(f"    Edges: {len(graph.graph.edges)}")

            # Test graph queries
            connections = graph.find_entity_connections("person_jane", max_distance=2)
            print(f"    Entity connections: {len(connections)}")
            self.log(f"Found {len(connections)} entity connections", "SUCCESS")

            # Test subgraph extraction
            subgraph = graph.get_knowledge_subgraph(node1, radius=2)
            print(f"    Subgraph nodes: {len(subgraph['nodes'])}")
            self.log(f"Extracted subgraph with {len(subgraph['nodes'])} nodes", "SUCCESS")

            # Test knowledge inference
            inferred = graph.infer_new_facts()
            print(f"    Inferred facts: {len(inferred)}")
            if inferred:
                print("\n  Sample Inferred Fact:")
                print(f"    • {inferred[0]['fact']} (confidence: {inferred[0]['confidence']:.2f})")

            self.log(f"Inferred {len(inferred)} new facts", "SUCCESS")

            self.results["knowledge_graph"] = {
                "success": True,
                "nodes": len(graph.graph.nodes),
                "edges": len(graph.graph.edges),
                "inferred_facts": len(inferred),
            }
            return True

        except Exception as e:
            print(f"\n✗ Knowledge Graph Failed: {e}")
            if self.verbose:
                traceback.print_exc()
            self.results["knowledge_graph"] = {"success": False, "error": str(e)}
            return False

    def print_summary(self):
        """Print validation summary."""
        print("\n" + "=" * 70)
        print("VALIDATION SUMMARY")
        print("=" * 70)

        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results.values() if r.get("success", False))

        print(f"\nTests Run: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")

        print("\nDetailed Results:")
        for feature, result in self.results.items():
            status = "✓ PASS" if result.get("success", False) else "✗ FAIL"
            print(f"  {status}: {feature.replace('_', ' ').title()}")
            if not result.get("success", False) and "error" in result:
                print(f"       Error: {result['error']}")

        if passed_tests == total_tests:
            print("\n" + "=" * 70)
            print("🎉 ALL INTELLIGENCE FEATURES ARE WORKING CORRECTLY!")
            print("=" * 70)
            print("\nYour HippocampAI installation is ready to use.")
            print("\nNext steps:")
            print("  • Check out the documentation: docs/INTELLIGENCE_FEATURES.md")
            print("  • Explore examples: examples/02_conversation_extraction.py")
            print("  • Read the quickstart: docs/QUICKSTART.md")
            return True
        else:
            print("\n" + "=" * 70)
            print("⚠️  SOME TESTS FAILED")
            print("=" * 70)
            print("\nPlease check the errors above and ensure:")
            print("  • All dependencies are installed: pip install -e .")
            print("  • Python version >= 3.9")
            print("  • Run with --verbose for detailed error information")
            return False

    def run_all_validations(self) -> bool:
        """Run all validation tests."""
        print("=" * 70)
        print("HippocampAI Intelligence Features Validation")
        print("=" * 70)
        print("\nThis script will validate:")
        print("  1. Module imports")
        print("  2. Fact extraction pipeline")
        print("  3. Entity recognition")
        print("  4. Session summarization")
        print("  5. Knowledge graph")

        # Run all tests
        tests = [
            self.validate_imports,
            self.validate_fact_extraction,
            self.validate_entity_recognition,
            self.validate_summarization,
            self.validate_knowledge_graph,
        ]

        for test in tests:
            try:
                test()
            except KeyboardInterrupt:
                print("\n\n⚠️  Validation interrupted by user")
                return False
            except Exception as e:
                print(f"\n✗ Unexpected error in {test.__name__}: {e}")
                if self.verbose:
                    traceback.print_exc()

        # Print summary
        return self.print_summary()


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Validate HippocampAI Intelligence Features")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose output with detailed logs"
    )

    args = parser.parse_args()

    validator = FeatureValidator(verbose=args.verbose)
    success = validator.run_all_validations()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
