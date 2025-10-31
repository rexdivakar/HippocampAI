"""Test intelligence features integration."""

import pytest

from hippocampai.client import MemoryClient
from hippocampai.pipeline import (
    EntityType,
    FactCategory,
    SummaryStyle,
)


class TestIntelligenceIntegration:
    """Test that intelligence features are properly integrated into MemoryClient."""

    @pytest.fixture
    def client(self):
        """Create a test client."""
        return MemoryClient.from_preset("development")

    def test_fact_extraction(self, client):
        """Test fact extraction from text."""
        text = "John works at Google in San Francisco. He studied Computer Science at MIT."

        # Extract facts
        facts = client.extract_facts(text, source="test")

        # Should extract employment and education facts
        assert isinstance(facts, list)
        assert len(facts) > 0

        # Check fact structure
        for fact in facts:
            assert hasattr(fact, "fact")
            assert hasattr(fact, "category")
            assert hasattr(fact, "confidence")
            assert isinstance(fact.category, FactCategory)

    def test_fact_extraction_from_conversation(self, client):
        """Test fact extraction from conversation."""
        conversation = """
        User: I'm a software engineer at Tesla
        Assistant: That's great! How long have you been there?
        User: About 2 years now
        """

        facts = client.extract_facts_from_conversation(conversation, "test_user")

        assert isinstance(facts, list)
        # Pattern-based should still find some facts even without LLM
        # LLM enhances extraction but is not required

    def test_entity_recognition(self, client):
        """Test entity extraction from text."""
        text = "Elon Musk founded SpaceX in California in 2002"

        # Extract entities
        entities = client.extract_entities(text)

        assert isinstance(entities, list)
        assert len(entities) > 0

        # Check entity structure
        for entity in entities:
            assert hasattr(entity, "text")
            assert hasattr(entity, "type")
            assert hasattr(entity, "entity_id")
            assert isinstance(entity.type, EntityType)

        # Should have person, organization, and location
        entity_types = {e.type for e in entities}
        assert EntityType.PERSON in entity_types or EntityType.ORGANIZATION in entity_types

    def test_relationship_extraction(self, client):
        """Test relationship extraction."""
        text = "Steve Jobs worked at Apple and lived in California"

        relationships = client.extract_relationships(text)

        assert isinstance(relationships, list)
        # Pattern-based extraction might find work and location relationships
        if len(relationships) > 0:
            for rel in relationships:
                assert hasattr(rel, "from_entity_id")
                assert hasattr(rel, "to_entity_id")
                assert hasattr(rel, "relation_type")
                assert hasattr(rel, "confidence")

    def test_entity_profile_and_search(self, client):
        """Test entity profile creation and search."""
        text = "Tim Cook is the CEO of Apple"

        # Extract and add entities
        entities = client.extract_entities(text)

        if len(entities) > 0:
            # Add first entity to graph
            entity = entities[0]
            node_id = client.add_entity_to_graph(entity)

            assert isinstance(node_id, str)
            assert node_id.startswith("entity_")

            # Get profile
            profile = client.get_entity_profile(entity.entity_id)
            assert profile is not None
            assert profile.entity_id == entity.entity_id

            # Search for entity
            results = client.search_entities(entity.text.split()[0].lower())
            assert isinstance(results, list)

    def test_conversation_summarization(self, client):
        """Test conversation summarization."""
        messages = [
            {"role": "user", "content": "I need help with Python programming"},
            {
                "role": "assistant",
                "content": "I'd be happy to help! What specifically do you need?",
            },
            {"role": "user", "content": "I want to learn about data structures"},
            {"role": "assistant", "content": "Great! Let's start with lists and dictionaries"},
        ]

        # Generate summary
        summary = client.summarize_conversation(
            messages, session_id="test_session", style=SummaryStyle.CONCISE
        )

        assert summary is not None
        assert hasattr(summary, "summary")
        assert hasattr(summary, "key_points")
        assert hasattr(summary, "topics")
        assert hasattr(summary, "sentiment")
        assert isinstance(summary.summary, str)
        # Summary might be empty without LLM, but key_points should be extracted
        assert len(summary.key_points) > 0 or len(summary.topics) > 0

    def test_bullet_point_summary(self, client):
        """Test bullet point summary style."""
        messages = [
            {"role": "user", "content": "What are the main features of Python?"},
            {
                "role": "assistant",
                "content": "Python has many features: it's easy to learn, has great libraries, and supports multiple paradigms",
            },
        ]

        summary = client.summarize_conversation(messages, style=SummaryStyle.BULLET_POINTS)

        # Verify summary object is returned with correct style
        assert summary.style == SummaryStyle.BULLET_POINTS
        # Key points should be extracted even without LLM
        assert isinstance(summary.key_points, list)

    def test_rolling_summary(self, client):
        """Test rolling summary creation."""
        messages = [{"role": "user", "content": f"Message {i}"} for i in range(20)]

        summary = client.create_rolling_summary(messages, window_size=5)

        # Verify method returns a string (might be empty without LLM)
        assert isinstance(summary, str)

    def test_conversation_insights(self, client):
        """Test insight extraction from conversation."""
        messages = [
            {"role": "user", "content": "I decided to learn machine learning"},
            {"role": "assistant", "content": "That's a great decision!"},
            {"role": "user", "content": "I realized I need to improve my math skills first"},
        ]

        insights = client.extract_conversation_insights(messages, "test_user")

        assert isinstance(insights, dict)
        assert "topics" in insights
        assert "sentiment" in insights
        assert "key_decisions" in insights
        assert "learning_points" in insights

    def test_knowledge_graph_operations(self, client):
        """Test knowledge graph operations."""
        # Extract entities and facts
        text = "Marie Curie was a physicist who worked in France"
        entities = client.extract_entities(text)
        facts = client.extract_facts(text)

        if len(entities) > 0 and len(facts) > 0:
            # Add to graph
            entity = entities[0]
            fact = facts[0]

            entity_node = client.add_entity_to_graph(entity)
            fact_node = client.add_fact_to_graph(fact)

            assert isinstance(entity_node, str)
            assert isinstance(fact_node, str)

            # Create a test memory and link
            memory = client.remember(text, "test_user")

            # Link memory to entity
            success = client.link_memory_to_entity(memory.id, entity.entity_id)
            assert success is True

            # Get entity memories
            memory_ids = client.get_entity_memories(entity.entity_id)
            assert memory.id in memory_ids

    def test_entity_timeline(self, client):
        """Test entity timeline creation."""
        text = "Albert Einstein was born in Germany"
        entities = client.extract_entities(text)

        if len(entities) > 0:
            entity = entities[0]
            client.add_entity_to_graph(entity)

            timeline = client.get_entity_timeline(entity.entity_id)
            assert isinstance(timeline, list)

    def test_knowledge_subgraph(self, client):
        """Test knowledge subgraph extraction."""
        text = "Newton discovered gravity"
        entities = client.extract_entities(text)

        if len(entities) > 0:
            entity = entities[0]
            node_id = client.add_entity_to_graph(entity)

            subgraph = client.get_knowledge_subgraph(node_id, radius=1)

            assert isinstance(subgraph, dict)
            assert "nodes" in subgraph
            assert "edges" in subgraph
            assert isinstance(subgraph["nodes"], list)
            assert isinstance(subgraph["edges"], list)

    def test_enrich_memory_with_intelligence(self, client):
        """Test memory enrichment with all intelligence features."""
        text = "Richard Feynman was a physicist at Caltech who won the Nobel Prize"
        memory = client.remember(text, "test_user")

        # Enrich memory
        enrichment = client.enrich_memory_with_intelligence(memory, add_to_graph=True)

        assert isinstance(enrichment, dict)
        assert "facts" in enrichment
        assert "entities" in enrichment
        assert "relationships" in enrichment
        assert "graph_updated" in enrichment
        assert enrichment["graph_updated"] is True

        # Verify facts were extracted
        assert isinstance(enrichment["facts"], list)

        # Verify entities were extracted
        assert isinstance(enrichment["entities"], list)

        # Verify relationships were extracted
        assert isinstance(enrichment["relationships"], list)

    def test_knowledge_inference(self, client):
        """Test knowledge inference from graph patterns."""
        # Create some related entities and facts
        text1 = "Alice works at Google"
        text2 = "Google is located in California"

        mem1 = client.remember(text1, "test_user")
        mem2 = client.remember(text2, "test_user")

        # Enrich both memories
        client.enrich_memory_with_intelligence(mem1, add_to_graph=True)
        client.enrich_memory_with_intelligence(mem2, add_to_graph=True)

        # Try to infer new facts
        inferred = client.infer_knowledge(user_id="test_user")

        # Inference might find that Alice is likely in California
        assert isinstance(inferred, list)
        # Inference rules are basic, so results might be empty
        # Just verify it doesn't crash

    def test_client_has_intelligence_modules(self, client):
        """Test that client has all intelligence module attributes."""
        assert hasattr(client, "fact_extractor")
        assert hasattr(client, "entity_recognizer")
        assert hasattr(client, "summarizer")
        assert hasattr(client, "graph")

        # Verify graph is KnowledgeGraph, not just MemoryGraph
        from hippocampai.graph import KnowledgeGraph

        assert isinstance(client.graph, KnowledgeGraph)

    def test_intelligence_methods_exist(self, client):
        """Test that all intelligence methods exist on client."""
        # Fact extraction
        assert hasattr(client, "extract_facts")
        assert hasattr(client, "extract_facts_from_conversation")

        # Entity recognition
        assert hasattr(client, "extract_entities")
        assert hasattr(client, "extract_relationships")
        assert hasattr(client, "get_entity_profile")
        assert hasattr(client, "search_entities")

        # Summarization
        assert hasattr(client, "summarize_conversation")
        assert hasattr(client, "create_rolling_summary")
        assert hasattr(client, "extract_conversation_insights")

        # Knowledge graph
        assert hasattr(client, "add_entity_to_graph")
        assert hasattr(client, "add_fact_to_graph")
        assert hasattr(client, "link_memory_to_entity")
        assert hasattr(client, "link_memory_to_fact")
        assert hasattr(client, "get_entity_memories")
        assert hasattr(client, "get_entity_facts")
        assert hasattr(client, "get_entity_connections")
        assert hasattr(client, "get_knowledge_subgraph")
        assert hasattr(client, "get_entity_timeline")
        assert hasattr(client, "infer_knowledge")
        assert hasattr(client, "enrich_memory_with_intelligence")
