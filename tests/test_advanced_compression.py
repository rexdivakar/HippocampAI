"""Tests for advanced compression techniques."""

from datetime import datetime, timedelta, timezone

import pytest

from hippocampai.models.memory import Memory, MemoryType
from hippocampai.pipeline.advanced_compression import (
    AdvancedCompressor,
    SemanticType,
)


@pytest.fixture
def compressor():
    """Create advanced compressor instance."""
    return AdvancedCompressor(llm=None, target_ratio=32.0)


@pytest.fixture
def sample_memories():
    """Create sample memories for compression."""
    now = datetime.now(timezone.utc)
    memories = []

    # Technical knowledge
    memories.append(
        Memory(
            text="Python is a high-level programming language known for its simplicity and readability. It was created by Guido van Rossum and first released in 1991.",
            user_id="test_user",
            type=MemoryType.FACT,
            importance=8.0,
            created_at=now - timedelta(days=10),
        )
    )

    memories.append(
        Memory(
            text="Python supports multiple programming paradigms including object-oriented, functional, and procedural programming styles.",
            user_id="test_user",
            type=MemoryType.FACT,
            importance=7.5,
            created_at=now - timedelta(days=9),
        )
    )

    memories.append(
        Memory(
            text="I use Python daily for data analysis and machine learning projects because it has excellent libraries like NumPy and pandas.",
            user_id="test_user",
            type=MemoryType.PREFERENCE,
            importance=7.0,
            created_at=now - timedelta(days=8),
        )
    )

    # Work-related
    memories.append(
        Memory(
            text="Our team meets every Monday morning at 9 AM to discuss the weekly sprint planning and project updates.",
            user_id="test_user",
            type=MemoryType.EVENT,
            importance=6.0,
            created_at=now - timedelta(days=7),
        )
    )

    memories.append(
        Memory(
            text="The Monday meetings usually last about one hour and involve the entire development team including the product manager.",
            user_id="test_user",
            type=MemoryType.EVENT,
            importance=5.5,
            created_at=now - timedelta(days=6),
        )
    )

    return memories


def test_compress_with_rcc(compressor, sample_memories):
    """Test RCC-style compression."""
    target_tokens = 50
    compressed = compressor.compress_with_rcc(sample_memories, target_tokens)

    assert compressed is not None
    assert compressed.compression_method == "RCC"
    assert compressed.compressed_tokens <= target_tokens * 1.5  # Allow some tolerance
    assert compressed.compression_ratio < 1.0  # Should compress
    assert compressed.quality_score > 0.3  # Reasonable quality
    assert len(compressed.key_entities) > 0
    assert len(compressed.key_facts) > 0


def test_compress_with_rcc_high_compression(compressor, sample_memories):
    """Test aggressive RCC compression."""
    target_tokens = 20  # Very aggressive
    compressed = compressor.compress_with_rcc(sample_memories, target_tokens)

    assert compressed.compressed_tokens <= target_tokens * 2  # Some tolerance
    assert compressed.compression_ratio < 0.3  # High compression


def test_extract_entities(compressor, sample_memories):
    """Test entity extraction."""
    entities = compressor._extract_entities(sample_memories)

    assert len(entities) > 0
    # Should extract proper nouns
    assert any("Python" in e for e in entities)
    assert any("Guido" in e or "Rossum" in e for e in entities)


def test_extract_facts(compressor, sample_memories):
    """Test fact extraction."""
    facts = compressor._extract_facts(sample_memories)

    assert len(facts) > 0
    # Facts should be sentences
    for fact in facts:
        assert len(fact) > 10


def test_extract_relationships(compressor, sample_memories):
    """Test relationship extraction."""
    relationships = compressor._extract_relationships(sample_memories)

    # Should find some relationships
    assert isinstance(relationships, list)


def test_deduplicate_facts(compressor):
    """Test fact deduplication."""
    facts = [
        "Python is a programming language",
        "Python is a programming language",  # Exact duplicate
        "A programming language is Python",  # Similar but different word order
        "Java is a programming language",  # Different fact
    ]

    unique = compressor._deduplicate_facts(facts)

    # Should remove exact duplicates
    assert len(unique) < len(facts)
    assert len(unique) >= 2  # At least 2 unique facts


def test_rank_entities(compressor):
    """Test entity ranking."""
    entities = ["Python", "Java", "Python", "Python", "JavaScript", "Python"]

    ranked = compressor._rank_entities(entities)

    # Most common should be first
    assert ranked[0] == "Python"
    assert len(ranked) == 3  # Three unique entities


def test_prune_tokens_aggressive(compressor):
    """Test aggressive token pruning."""
    text = "I really think that this is a very good example of the basic concept"

    pruned, metrics = compressor.prune_tokens(text, target_reduction=0.5, preserve_semantics=False)

    # Should remove approximately 50% of tokens
    assert metrics.compression_ratio < 0.6
    # Should remove stopwords
    assert "the" not in pruned.lower() or "a" not in pruned.lower()


def test_prune_tokens_semantic_preservation(compressor):
    """Test token pruning with semantic preservation."""
    text = "Machine learning algorithms require large datasets for training effective models"

    pruned, metrics = compressor.prune_tokens(text, target_reduction=0.3, preserve_semantics=True)

    # Should preserve important words
    assert "machine" in pruned.lower() or "learning" in pruned.lower()
    assert "algorithms" in pruned.lower() or "datasets" in pruned.lower()
    # Should have good semantic similarity
    assert metrics.information_retention > 0.5


def test_prune_tokens_metrics(compressor):
    """Test compression metrics from token pruning."""
    text = "The quick brown fox jumps over the lazy dog in the park"

    pruned, metrics = compressor.prune_tokens(text, target_reduction=0.5)

    assert 0.0 < metrics.compression_ratio < 1.0
    assert 0.0 <= metrics.information_retention <= 1.0
    assert 0.0 <= metrics.semantic_density <= 1.0
    assert 0.0 <= metrics.entity_preservation <= 1.0
    assert 0.0 <= metrics.readability_score <= 1.0


def test_semantic_token_pruning(compressor):
    """Test semantic token pruning algorithm."""
    words = ["The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"]

    pruned = compressor._semantic_token_pruning(words, target_reduction=0.4)

    # Should keep important words (longer, capitalized, not stopwords)
    assert len(pruned) < len(words)
    # Should prefer to keep "quick", "brown", "fox" over "the", "over"
    assert "fox" in pruned or "quick" in pruned or "brown" in pruned


def test_convert_episodic_to_semantic(compressor, sample_memories):
    """Test episodic to semantic conversion."""
    # Filter to get python-related memories
    python_memories = [m for m in sample_memories if "Python" in m.text or "python" in m.text]

    semantic_memories = compressor.convert_episodic_to_semantic(python_memories)

    assert isinstance(semantic_memories, list)
    # Should extract at least some semantic knowledge
    if semantic_memories:
        for sem in semantic_memories:
            assert sem.semantic_type in SemanticType
            assert 1 <= sem.abstraction_level <= 5
            assert 0.0 <= sem.confidence <= 1.0
            assert len(sem.source_memory_ids) > 0


def test_group_by_topic(compressor, sample_memories):
    """Test topic-based grouping."""
    groups = compressor._group_by_topic(sample_memories)

    assert isinstance(groups, dict)
    assert len(groups) > 0
    # Each group should have at least one memory
    for topic, memories in groups.items():
        assert len(memories) > 0


def test_infer_topic(compressor):
    """Test topic inference."""
    text1 = "Python programming is great for data science"
    topic1 = compressor._infer_topic(text1)
    assert isinstance(topic1, str)
    assert len(topic1) > 0

    text2 = "The meeting was scheduled for Monday"
    topic2 = compressor._infer_topic(text2)
    assert isinstance(topic2, str)


def test_classify_semantic_type(compressor, sample_memories):
    """Test semantic type classification."""
    # Facts should classify as FACT
    fact_memories = [m for m in sample_memories if m.type == MemoryType.FACT]
    if fact_memories:
        semantic_type = compressor._classify_semantic_type(fact_memories)
        assert semantic_type == SemanticType.FACT

    # Preferences might classify as CONCEPT
    pref_memories = [m for m in sample_memories if m.type == MemoryType.PREFERENCE]
    if pref_memories:
        semantic_type = compressor._classify_semantic_type(pref_memories)
        assert semantic_type in SemanticType


def test_calculate_semantic_confidence(compressor, sample_memories):
    """Test semantic confidence calculation."""
    confidence = compressor._calculate_semantic_confidence(sample_memories)

    assert 0.0 <= confidence <= 1.0
    # More memories should increase confidence
    single_memory_conf = compressor._calculate_semantic_confidence([sample_memories[0]])
    many_memory_conf = compressor._calculate_semantic_confidence(sample_memories)
    assert many_memory_conf >= single_memory_conf


def test_determine_abstraction_level(compressor):
    """Test abstraction level determination."""
    # Concrete text with simple words
    concrete = "I went to the store"
    level1 = compressor._determine_abstraction_level(concrete)
    assert 1 <= level1 <= 3

    # Abstract text with complex words
    abstract = "The epistemological implications of phenomenological interpretation"
    level2 = compressor._determine_abstraction_level(abstract)
    assert 3 <= level2 <= 5


def test_evaluate_compression(compressor):
    """Test compression evaluation."""
    original = "The quick brown fox jumps over the lazy dog in the park on a sunny day"
    compressed = "Quick brown fox jumps over lazy dog"

    metrics = compressor.evaluate_compression(original, compressed)

    assert 0.0 < metrics.compression_ratio < 1.0
    assert 0.0 <= metrics.information_retention <= 1.0
    assert 0.0 <= metrics.semantic_density <= 1.0
    assert 0.0 <= metrics.entity_preservation <= 1.0
    assert 0.0 <= metrics.readability_score <= 1.0


def test_evaluate_compression_with_ground_truth(compressor):
    """Test compression evaluation with ground truth facts."""
    original = (
        "Python is a programming language. It was created by Guido. It is used for data science."
    )
    compressed = "Python: programming language by Guido for data science"

    ground_truth = ["Python", "programming language", "Guido", "data science"]

    metrics = compressor.evaluate_compression(original, compressed, ground_truth)

    # Should preserve key facts
    assert metrics.fact_preservation > 0.5


def test_estimate_semantic_similarity(compressor):
    """Test semantic similarity estimation."""
    text1 = "The cat sat on the mat"
    text2 = "cat sat mat"  # Same key words

    similarity = compressor._estimate_semantic_similarity(text1, text2)

    # Should be high similarity
    assert similarity > 0.5


def test_calculate_entity_preservation(compressor):
    """Test entity preservation calculation."""
    original = "Python and Java are programming languages created by Guido and James"
    compressed = "Python Java programming"

    preservation = compressor._calculate_entity_preservation(original, compressed)

    # Should preserve some entities
    assert preservation > 0.4


def test_calculate_readability(compressor):
    """Test readability calculation."""
    readable = "The cat sat on the mat. It was happy."
    score1 = compressor._calculate_readability(readable)

    unreadable = "x y z"
    score2 = compressor._calculate_readability(unreadable)

    assert 0.0 <= score1 <= 1.0
    assert 0.0 <= score2 <= 1.0


def test_compression_ratio_calculation(compressor, sample_memories):
    """Test that compression ratio is calculated correctly."""
    target_tokens = 30
    compressed = compressor.compress_with_rcc(sample_memories, target_tokens)

    expected_ratio = compressed.compressed_tokens / compressed.original_tokens
    assert abs(compressed.compression_ratio - expected_ratio) < 0.01


def test_quality_score_range(compressor, sample_memories):
    """Test that quality score is in valid range."""
    compressed = compressor.compress_with_rcc(sample_memories, 50)

    assert 0.0 <= compressed.quality_score <= 1.0


def test_semantic_density_calculation(compressor, sample_memories):
    """Test semantic density calculation."""
    compressed = compressor.compress_with_rcc(sample_memories, 50)

    # Semantic density should be fact_count / token_count
    assert compressed.semantic_density >= 0.0
    # Higher compression should maintain or increase density
    assert compressed.semantic_density > 0.0


def test_empty_memory_list(compressor):
    """Test handling of empty memory list."""
    with pytest.raises(ValueError):
        compressor.compress_with_rcc([], 50)


def test_single_memory_compression(compressor, sample_memories):
    """Test compression of single memory."""
    compressed = compressor.compress_with_rcc([sample_memories[0]], 20)

    assert compressed is not None
    assert len(compressed.original_memory_ids) == 1
    assert compressed.compressed_tokens < compressed.original_tokens


def test_stopwords_filtering(compressor):
    """Test that stopwords are properly filtered."""
    text = "the quick brown fox jumps over the lazy dog"
    pruned, metrics = compressor.prune_tokens(text, target_reduction=0.5, preserve_semantics=False)

    # Should remove many common stopwords
    stopword_count = sum(1 for word in ["the", "over"] if word in pruned.lower())
    assert stopword_count < 2  # Should remove most stopwords


def test_metadata_preservation(compressor, sample_memories):
    """Test that metadata is preserved in compressed memory."""
    compressed = compressor.compress_with_rcc(sample_memories, 50)

    assert "num_source_memories" in compressed.metadata
    assert compressed.metadata["num_source_memories"] == len(sample_memories)
    assert "entity_count" in compressed.metadata
    assert "fact_count" in compressed.metadata


def test_key_information_extraction(compressor, sample_memories):
    """Test that key entities and facts are extracted."""
    compressed = compressor.compress_with_rcc(sample_memories, 50)

    # Should extract key entities
    assert len(compressed.key_entities) > 0
    # Python should be recognized as a key entity
    assert any("Python" in entity for entity in compressed.key_entities)

    # Should extract key facts
    assert len(compressed.key_facts) > 0


def test_different_target_ratios(compressor, sample_memories):
    """Test compression with different target ratios."""
    # Low compression
    compressed_low = compressor.compress_with_rcc(sample_memories, 100)

    # High compression
    compressed_high = compressor.compress_with_rcc(sample_memories, 20)

    # Higher compression should result in smaller output
    assert compressed_high.compressed_tokens < compressed_low.compressed_tokens
    # But quality might be lower
    assert compressed_low.quality_score >= compressed_high.quality_score - 0.2
