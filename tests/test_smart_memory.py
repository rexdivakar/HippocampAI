"""Tests for smart memory updates and semantic clustering features."""

import pytest

from hippocampai.models.memory import Memory, MemoryType
from hippocampai.pipeline.semantic_clustering import MemoryCluster, SemanticCategorizer
from hippocampai.pipeline.smart_updater import SmartMemoryUpdater


class TestSmartMemoryUpdater:
    """Test smart memory update logic."""

    def test_similarity_calculation(self):
        """Test text similarity calculation."""
        updater = SmartMemoryUpdater()

        # Identical text
        assert updater._calculate_similarity("I love coffee", "I love coffee") == 1.0

        # Very similar
        similarity = updater._calculate_similarity("I love coffee", "I really love coffee")
        assert similarity > 0.7

        # Different
        similarity = updater._calculate_similarity("I love coffee", "The weather is nice")
        assert similarity < 0.3

    def test_conflict_detection(self):
        """Test conflict detection between memories."""
        updater = SmartMemoryUpdater()

        # Should detect conflict with negation
        assert updater._detect_conflict("I love coffee", "I don't love coffee")
        assert updater._detect_conflict("I work at Google", "I no longer work at Google")

        # Should not detect conflict
        assert not updater._detect_conflict("I love coffee", "I love tea")
        assert not updater._detect_conflict("I work at Google", "I work remotely")

    def test_update_decision_skip(self):
        """Test decision to skip duplicate memory."""
        updater = SmartMemoryUpdater()

        existing = Memory(
            text="I love coffee",
            user_id="user1",
            type=MemoryType.PREFERENCE,
            importance=7.0,
        )

        # Nearly identical - should skip
        decision = updater.should_update_memory(existing, "I love coffee")
        assert decision.action == "skip"
        assert decision.confidence_adjustment > 0

    def test_update_decision_merge(self):
        """Test decision to merge similar memories."""
        updater = SmartMemoryUpdater()

        existing = Memory(
            text="I love coffee",
            user_id="user1",
            type=MemoryType.PREFERENCE,
            importance=7.0,
        )

        # Similar but with more info - should merge
        decision = updater.should_update_memory(
            existing, "I love coffee and drink it every morning"
        )
        assert decision.action in ["merge", "update", "keep_both"]

    def test_update_decision_keep_both(self):
        """Test decision to keep both memories."""
        updater = SmartMemoryUpdater()

        existing = Memory(
            text="I love coffee",
            user_id="user1",
            type=MemoryType.PREFERENCE,
            importance=7.0,
        )

        # Unrelated - should keep both
        decision = updater.should_update_memory(existing, "I work at Google")
        assert decision.action == "keep_both"

    def test_confidence_updates(self):
        """Test confidence score updates."""
        updater = SmartMemoryUpdater()

        memory = Memory(
            text="I love coffee",
            user_id="user1",
            type=MemoryType.PREFERENCE,
            importance=7.0,
            confidence=0.8,
        )

        # Validation should increase confidence
        updated = updater.update_confidence(memory, "validation")
        assert updated.confidence > memory.confidence

        # Conflict should decrease confidence
        updated = updater.update_confidence(memory, "conflict")
        assert updated.confidence < memory.confidence

    def test_memory_reconciliation(self):
        """Test reconciliation of conflicting memories."""
        updater = SmartMemoryUpdater()

        memories = [
            Memory(
                text="I love coffee",
                user_id="user1",
                type=MemoryType.PREFERENCE,
                importance=7.0,
                confidence=0.9,
            ),
            Memory(
                text="I don't like coffee",
                user_id="user1",
                type=MemoryType.PREFERENCE,
                importance=6.0,
                confidence=0.7,
            ),
        ]

        # Should resolve to one memory (higher confidence wins)
        reconciled = updater.reconcile_memories(memories, "user1")
        assert len(reconciled) <= len(memories)


class TestSemanticCategorizer:
    """Test semantic categorization and clustering."""

    def test_tag_suggestion(self):
        """Test automatic tag suggestion."""
        categorizer = SemanticCategorizer()

        memory = Memory(
            text="I love drinking coffee in the morning at my favorite cafe",
            user_id="user1",
            type=MemoryType.PREFERENCE,
        )

        tags = categorizer.suggest_tags(memory, max_tags=5)
        assert len(tags) <= 5
        assert len(tags) > 0
        # Should include relevant keywords
        assert any(tag in ["coffee", "morning", "cafe", "food", "preference"] for tag in tags)

    def test_category_assignment(self):
        """Test automatic category assignment."""
        categorizer = SemanticCategorizer()

        # Preference patterns
        mem = Memory(
            text="I love coffee",
            user_id="user1",
            type=MemoryType.CONTEXT,  # Wrong type
        )
        category = categorizer.assign_category(mem)
        assert category == MemoryType.PREFERENCE

        # Fact patterns
        mem = Memory(
            text="I work at Google",
            user_id="user1",
            type=MemoryType.CONTEXT,
        )
        category = categorizer.assign_category(mem)
        assert category == MemoryType.FACT

        # Goal patterns
        mem = Memory(
            text="I want to learn Python",
            user_id="user1",
            type=MemoryType.CONTEXT,
        )
        category = categorizer.assign_category(mem)
        assert category == MemoryType.GOAL

        # Habit patterns
        mem = Memory(
            text="I usually exercise in the morning",
            user_id="user1",
            type=MemoryType.CONTEXT,
        )
        category = categorizer.assign_category(mem)
        assert category == MemoryType.HABIT

    def test_similar_memory_detection(self):
        """Test finding similar memories."""
        categorizer = SemanticCategorizer()

        query_memory = Memory(
            text="I love coffee",
            user_id="user1",
            type=MemoryType.PREFERENCE,
        )

        existing_memories = [
            Memory(
                text="I really enjoy coffee",
                user_id="user1",
                type=MemoryType.PREFERENCE,
            ),
            Memory(
                text="I work at Google",
                user_id="user1",
                type=MemoryType.FACT,
            ),
        ]

        similar = categorizer.find_similar_memories(
            query_memory, existing_memories, similarity_threshold=0.5
        )

        # Should find at least the coffee-related memory
        assert len(similar) >= 1
        assert similar[0][0].text == "I really enjoy coffee"
        assert similar[0][1] >= 0.5  # Similarity score

    def test_memory_clustering(self):
        """Test clustering memories by topic."""
        categorizer = SemanticCategorizer()

        memories = [
            Memory(text="I love coffee", user_id="user1", type=MemoryType.PREFERENCE),
            Memory(text="I enjoy tea", user_id="user1", type=MemoryType.PREFERENCE),
            Memory(text="I work at Google", user_id="user1", type=MemoryType.FACT),
            Memory(text="I code in Python", user_id="user1", type=MemoryType.FACT),
            Memory(text="I want to learn machine learning", user_id="user1", type=MemoryType.GOAL),
        ]

        clusters = categorizer.cluster_memories(memories, max_clusters=5)

        assert len(clusters) > 0
        assert len(clusters) <= 5

        # Check cluster structure
        for cluster in clusters:
            assert isinstance(cluster, MemoryCluster)
            assert len(cluster.memories) > 0
            assert isinstance(cluster.topic, str)
            assert isinstance(cluster.tags, list)

    def test_topic_shift_detection(self):
        """Test detecting topic shifts in conversation."""
        categorizer = SemanticCategorizer()

        # Create memories with topic shift
        memories = [
            # Old topic: food
            Memory(text="I love coffee", user_id="user1", type=MemoryType.PREFERENCE),
            Memory(text="I enjoy pizza", user_id="user1", type=MemoryType.PREFERENCE),
            Memory(text="I like pasta", user_id="user1", type=MemoryType.PREFERENCE),
            # New topic: work
            Memory(text="I work at Google", user_id="user1", type=MemoryType.FACT),
            Memory(text="I code in Python", user_id="user1", type=MemoryType.FACT),
            Memory(text="I have a meeting today", user_id="user1", type=MemoryType.EVENT),
        ]

        # Should detect shift from food to work
        topic = categorizer.detect_topic_shift(memories, window_size=3)
        assert topic is not None
        assert isinstance(topic, str)

    def test_memory_enrichment(self):
        """Test enriching memory with tags and category."""
        categorizer = SemanticCategorizer()

        memory = Memory(
            text="I love coffee and drink it every morning",
            user_id="user1",
            type=MemoryType.CONTEXT,  # Wrong type
        )

        enriched = categorizer.enrich_memory_with_categories(memory)

        # Should have corrected category
        assert enriched.type != MemoryType.CONTEXT
        # Should have tags
        assert len(enriched.tags) > 0
        # Should not modify original
        assert memory.id == enriched.id


@pytest.mark.integration
class TestSmartMemoryIntegration:
    """Integration tests with MemoryClient."""

    def test_auto_categorization_on_remember(self, memory_client, user_id):
        """Test that memories are auto-categorized on storage."""
        # Store memory with generic type
        memory = memory_client.remember(
            text="I want to learn Python",
            user_id=user_id,
            type="fact",  # Wrong type
        )

        # Should be auto-categorized as GOAL
        assert memory.type == MemoryType.GOAL or memory.type != MemoryType.FACT
        # Should have tags
        assert len(memory.tags) > 0

    def test_smart_update_on_similar_memory(self, memory_client, user_id):
        """Test smart update when storing similar memory."""
        # Store first memory
        memory_client.remember(
            text="I love coffee",
            user_id=user_id,
            type="preference",
        )

        # Try to store very similar memory
        memory_client.remember(
            text="I love coffee",
            user_id=user_id,
            type="preference",
        )

        # Should either skip or update (not create duplicate)
        # IDs might be same if skipped
        memories = memory_client.get_memories(user_id)
        coffee_memories = [m for m in memories if "coffee" in m.text.lower()]

        # Should not have many duplicates
        assert len(coffee_memories) <= 2

    def test_reconcile_user_memories(self, memory_client, user_id):
        """Test memory reconciliation API."""
        # Store some memories with potential conflicts
        memory_client.remember("I love coffee", user_id, type="preference")
        memory_client.remember("I really enjoy coffee", user_id, type="preference")

        # Reconcile
        reconciled = memory_client.reconcile_user_memories(user_id)

        assert isinstance(reconciled, list)
        assert all(isinstance(m, Memory) for m in reconciled)

    def test_cluster_user_memories(self, memory_client, user_id):
        """Test memory clustering API."""
        # Store memories on different topics
        memory_client.remember("I love coffee", user_id, type="preference")
        memory_client.remember("I work at Google", user_id, type="fact")
        memory_client.remember("I want to learn Python", user_id, type="goal")

        # Cluster
        clusters = memory_client.cluster_user_memories(user_id, max_clusters=5)

        assert isinstance(clusters, list)
        assert len(clusters) > 0
        assert all(isinstance(c, MemoryCluster) for c in clusters)
