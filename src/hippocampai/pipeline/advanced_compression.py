"""Advanced memory compression techniques.

This module provides state-of-the-art compression methods:
- Recurrent Context Compression (RCC-style) for 32x+ compression
- Token pruning with semantic preservation
- Episodic to semantic memory conversion
- Sparse attention patterns for efficient retrieval
- Quality metrics for compression validation
"""

import logging
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

from hippocampai.adapters.llm_base import BaseLLM
from hippocampai.models.memory import Memory, MemoryType

logger = logging.getLogger(__name__)


class CompressionQuality(str, Enum):
    """Compression quality levels."""

    LOSSLESS = "lossless"  # No information loss
    HIGH = "high"  # Minimal loss, 95%+ fidelity
    MEDIUM = "medium"  # Moderate loss, 80%+ fidelity
    LOW = "low"  # Significant loss, 60%+ fidelity
    AGGRESSIVE = "aggressive"  # Maximum compression, 40%+ fidelity


class SemanticType(str, Enum):
    """Types of semantic knowledge."""

    FACT = "fact"  # Factual knowledge
    CONCEPT = "concept"  # Conceptual understanding
    SKILL = "skill"  # Procedural knowledge
    RELATIONSHIP = "relationship"  # Relational knowledge
    PRINCIPLE = "principle"  # Abstract principles


class CompressedMemory(BaseModel):
    """Highly compressed memory representation."""

    id: str
    original_memory_ids: list[str]
    compressed_text: str
    compression_method: str
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float  # compressed/original (lower is better)
    quality_score: float = Field(ge=0.0, le=1.0)  # 0-1, higher is better
    semantic_density: float = Field(ge=0.0, le=1.0)  # Information per token
    key_entities: list[str] = Field(default_factory=list)
    key_facts: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SemanticMemory(BaseModel):
    """Semantic memory extracted from episodic memories."""

    id: str
    semantic_type: SemanticType
    content: str
    abstraction_level: int = Field(ge=1, le=5)  # 1=concrete, 5=highly abstract
    source_memory_ids: list[str]
    confidence: float = Field(ge=0.0, le=1.0)
    supporting_evidence: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CompressionMetrics(BaseModel):
    """Quality metrics for compressed memory."""

    compression_ratio: float
    information_retention: float  # Estimated % of info retained
    semantic_density: float  # Info per token
    entity_preservation: float  # % of entities preserved
    fact_preservation: float  # % of facts preserved
    readability_score: float  # 0-1, higher is better
    reconstruction_error: Optional[float] = None  # If ground truth available


class AdvancedCompressor:
    """Advanced compression engine with multiple techniques."""

    def __init__(
        self,
        llm: Optional[BaseLLM] = None,
        target_ratio: float = 32.0,  # Target compression ratio
        min_quality: float = 0.6,  # Minimum quality threshold
    ):
        """Initialize advanced compressor.

        Args:
            llm: Language model for compression
            target_ratio: Target compression ratio (higher = more compression)
            min_quality: Minimum acceptable quality (0-1)
        """
        self.llm = llm
        self.target_ratio = target_ratio
        self.min_quality = min_quality

        # Stopwords for token pruning
        self.stopwords = self._build_stopwords()
        self.filler_patterns = self._build_filler_patterns()

    def _build_stopwords(self) -> set[str]:
        """Build comprehensive stopword list."""
        return {
            # Articles
            "a",
            "an",
            "the",
            # Common verbs
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "am",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            # Conjunctions
            "and",
            "or",
            "but",
            "so",
            "yet",
            "nor",
            # Prepositions
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "from",
            "as",
            "into",
            "through",
            "during",
            # Modals
            "will",
            "would",
            "should",
            "could",
            "can",
            "may",
            "might",
            "must",
            # Pronouns (selective - keep important ones)
            "it",
            "this",
            "that",
            "these",
            "those",
            # Intensifiers
            "very",
            "really",
            "quite",
            "just",
            "only",
            "even",
            "almost",
        }

    def _build_filler_patterns(self) -> list[str]:
        """Build patterns for filler phrases."""
        return [
            r"\bI think\b",
            r"\bI believe\b",
            r"\bIn my opinion\b",
            r"\bkind of\b",
            r"\bsort of\b",
            r"\byou know\b",
            r"\bI mean\b",
            r"\bbasically\b",
            r"\bactually\b",
            r"\bliterally\b",
        ]

    def compress_with_rcc(self, memories: list[Memory], target_tokens: int) -> CompressedMemory:
        """Compress using Recurrent Context Compression style algorithm.

        Implements a multi-pass compression approach inspired by RCC:
        1. Extract key information (entities, facts, relationships)
        2. Remove redundancy across memories
        3. Create hierarchical summary
        4. Iteratively compress until target reached

        Args:
            memories: List of memories to compress
            target_tokens: Target token count

        Returns:
            Highly compressed memory
        """
        if not memories:
            raise ValueError("Cannot compress empty memory list")

        # Calculate original tokens
        original_text = " ".join([m.text for m in memories])
        original_tokens = Memory.estimate_tokens(original_text)

        # Pass 1: Extract key information
        entities = self._extract_entities(memories)
        facts = self._extract_facts(memories)
        relationships = self._extract_relationships(memories)

        # Pass 2: Remove redundancy
        unique_facts = self._deduplicate_facts(facts)
        core_entities = self._rank_entities(entities)[:20]  # Top 20 entities

        # Pass 3: Create compressed representation
        if self.llm:
            compressed_text = self._compress_with_llm_rcc(
                memories, unique_facts, core_entities, target_tokens
            )
        else:
            compressed_text = self._compress_heuristic_rcc(
                unique_facts, core_entities, relationships, target_tokens
            )

        compressed_tokens = Memory.estimate_tokens(compressed_text)

        # Calculate quality metrics
        quality_score = self._calculate_quality(original_text, compressed_text, entities, facts)

        return CompressedMemory(
            id=memories[0].id if len(memories) == 1 else f"compressed_{len(memories)}",
            original_memory_ids=[m.id for m in memories],
            compressed_text=compressed_text,
            compression_method="RCC",
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=compressed_tokens / original_tokens if original_tokens > 0 else 1.0,
            quality_score=quality_score,
            semantic_density=len(unique_facts) / compressed_tokens
            if compressed_tokens > 0
            else 0.0,
            key_entities=core_entities,
            key_facts=unique_facts,
            metadata={
                "num_source_memories": len(memories),
                "entity_count": len(entities),
                "fact_count": len(facts),
                "relationship_count": len(relationships),
            },
        )

    def _extract_entities(self, memories: list[Memory]) -> list[str]:
        """Extract entities from memories."""
        entities = []
        for memory in memories:
            # Simple entity extraction via capitalized words and known patterns
            words = memory.text.split()
            for word in words:
                # Capitalized words (likely proper nouns)
                if word and word[0].isupper() and len(word) > 2:
                    cleaned = re.sub(r"[^\w\s]", "", word)
                    if cleaned and cleaned not in self.stopwords:
                        entities.append(cleaned)

        # Return unique entities with counts
        entity_counts = Counter(entities)
        return list(entity_counts.keys())

    def _extract_facts(self, memories: list[Memory]) -> list[str]:
        """Extract factual statements from memories."""
        facts = []
        for memory in memories:
            # Split into sentences
            sentences = re.split(r"[.!?]+", memory.text)
            for sentence in sentences:
                sentence = sentence.strip()
                # Keep sentences that contain factual indicators
                if len(sentence) > 20 and any(
                    indicator in sentence.lower()
                    for indicator in [
                        "is",
                        "are",
                        "was",
                        "were",
                        "has",
                        "have",
                        "can",
                        "will",
                    ]
                ):
                    facts.append(sentence)

        return facts

    def _extract_relationships(self, memories: list[Memory]) -> list[str]:
        """Extract relationships between entities."""
        relationships = []
        relationship_patterns = [
            r"(\w+)\s+(?:is|are|was|were)\s+(?:a|an|the)?\s*(\w+)",
            r"(\w+)\s+(?:has|have)\s+(\w+)",
            r"(\w+)\s+(?:uses|use|used)\s+(\w+)",
            r"(\w+)\s+(?:works|work|worked)\s+(?:with|at|for)\s+(\w+)",
        ]

        for memory in memories:
            for pattern in relationship_patterns:
                matches = re.finditer(pattern, memory.text, re.IGNORECASE)
                for match in matches:
                    relationships.append(f"{match.group(1)} → {match.group(2)}")

        return list(set(relationships))

    def _deduplicate_facts(self, facts: list[str]) -> list[str]:
        """Remove duplicate or highly similar facts."""
        unique = []
        seen = set()

        for fact in facts:
            # Normalize fact for comparison
            normalized = " ".join(sorted(fact.lower().split()))
            if normalized not in seen:
                seen.add(normalized)
                unique.append(fact)

        return unique

    def _rank_entities(self, entities: list[str]) -> list[str]:
        """Rank entities by importance."""
        # Count frequency
        entity_counts = Counter(entities)
        # Sort by frequency (most common first)
        return [entity for entity, count in entity_counts.most_common()]

    def _compress_with_llm_rcc(
        self,
        memories: list[Memory],
        facts: list[str],
        entities: list[str],
        target_tokens: int,
    ) -> str:
        """Compress using LLM with RCC-style prompting."""
        if self.llm is None:
            return self._compress_heuristic_rcc(facts, entities, [], target_tokens)

        # Prepare context
        memory_text = " ".join([m.text for m in memories[:10]])  # Limit context
        facts_text = " | ".join(facts[:20])  # Top 20 facts
        entities_text = ", ".join(entities[:15])  # Top 15 entities

        prompt = f"""Compress these memories into exactly {target_tokens} tokens or less. Use extreme compression while preserving all critical information.

Key Entities: {entities_text}
Key Facts: {facts_text}

Original Memories (sample):
{memory_text[:1000]}

Create an ultra-compressed summary that:
1. Preserves all key facts and entities
2. Uses minimal tokens (target: {target_tokens})
3. Maintains semantic meaning
4. Uses abbreviations where appropriate

Compressed version:"""

        try:
            response = self.llm.generate(prompt, max_tokens=target_tokens * 2, temperature=0.1)
            return (
                response.strip()
                if response
                else self._compress_heuristic_rcc(facts, entities, [], target_tokens)
            )
        except Exception as e:
            logger.warning(f"LLM compression failed: {e}")
            return self._compress_heuristic_rcc(facts, entities, [], target_tokens)

    def _compress_heuristic_rcc(
        self,
        facts: list[str],
        entities: list[str],
        relationships: list[str],
        target_tokens: int,
    ) -> str:
        """Heuristic compression using extracted information."""
        # Build compressed representation
        parts = []

        # Add top entities
        if entities:
            parts.append(f"Entities: {', '.join(entities[:10])}")

        # Add top facts (compressed)
        if facts:
            compressed_facts = []
            for fact in facts[:15]:  # Limit to 15 facts
                # Remove stopwords from fact
                words = fact.split()
                important_words = [
                    w for w in words if w.lower() not in self.stopwords and len(w) > 2
                ]
                compressed_facts.append(" ".join(important_words[:8]))  # Max 8 words per fact

            parts.append(f"Facts: {' | '.join(compressed_facts)}")

        # Add relationships
        if relationships:
            parts.append(f"Relations: {' ; '.join(relationships[:10])}")

        # Combine and truncate to target
        compressed = ". ".join(parts)

        # Ensure we're under target tokens
        current_tokens = Memory.estimate_tokens(compressed)
        if current_tokens > target_tokens:
            # Truncate by removing less important parts
            words = compressed.split()
            target_words = int(len(words) * (target_tokens / current_tokens))
            compressed = " ".join(words[:target_words])

        return compressed

    def _calculate_quality(
        self, original: str, compressed: str, entities: list[str], facts: list[str]
    ) -> float:
        """Calculate compression quality score."""
        # Check entity preservation
        original.lower()
        compressed_lower = compressed.lower()

        entities_preserved = (
            sum(1 for e in entities if e.lower() in compressed_lower) / len(entities)
            if entities
            else 1.0
        )

        # Check fact preservation (keywords)
        facts_keywords = set()
        for fact in facts:
            words = [
                w.lower() for w in fact.split() if w.lower() not in self.stopwords and len(w) > 3
            ]
            facts_keywords.update(words[:5])  # Top 5 keywords per fact

        keywords_preserved = (
            sum(1 for kw in facts_keywords if kw in compressed_lower) / len(facts_keywords)
            if facts_keywords
            else 1.0
        )

        # Combine metrics
        quality = (entities_preserved * 0.6) + (keywords_preserved * 0.4)

        return min(1.0, quality)

    def prune_tokens(
        self, text: str, target_reduction: float = 0.5, preserve_semantics: bool = True
    ) -> tuple[str, CompressionMetrics]:
        """Advanced token pruning with semantic preservation.

        Args:
            text: Text to prune
            target_reduction: Target reduction ratio (0.5 = remove 50% of tokens)
            preserve_semantics: Whether to preserve semantic meaning

        Returns:
            Tuple of (pruned_text, metrics)
        """
        original_tokens = Memory.estimate_tokens(text)
        words = text.split()

        if not preserve_semantics:
            # Aggressive pruning - just remove stopwords
            pruned_words = [w for w in words if w.lower() not in self.stopwords]
        else:
            # Semantic-preserving pruning
            pruned_words = self._semantic_token_pruning(words, target_reduction)

        pruned_text = " ".join(pruned_words)
        pruned_tokens = Memory.estimate_tokens(pruned_text)

        # Calculate metrics
        semantic_similarity = self._estimate_semantic_similarity(text, pruned_text)

        metrics = CompressionMetrics(
            compression_ratio=pruned_tokens / original_tokens if original_tokens > 0 else 1.0,
            information_retention=semantic_similarity,
            semantic_density=len(set(pruned_words)) / pruned_tokens if pruned_tokens > 0 else 0.0,
            entity_preservation=self._calculate_entity_preservation(text, pruned_text),
            fact_preservation=semantic_similarity,  # Approximation
            readability_score=self._calculate_readability(pruned_text),
        )

        return pruned_text, metrics

    def _semantic_token_pruning(self, words: list[str], target_reduction: float) -> list[str]:
        """Prune tokens while preserving semantic meaning."""
        # Score each word by importance
        word_scores = []

        for i, word in enumerate(words):
            score = 0.0

            # Length bonus (longer words often more meaningful)
            score += min(len(word) / 20.0, 1.0)

            # Capitalization bonus (proper nouns)
            if word and word[0].isupper():
                score += 0.3

            # Position bonus (earlier words often more important)
            position_score = 1.0 - (i / len(words))
            score += position_score * 0.2

            # Stopword penalty
            if word.lower() in self.stopwords:
                score -= 0.5

            # Number bonus (numbers often important)
            if any(c.isdigit() for c in word):
                score += 0.4

            word_scores.append((word, score))

        # Sort by score and keep top (1 - target_reduction)
        word_scores.sort(key=lambda x: x[1], reverse=True)
        keep_count = int(len(words) * (1 - target_reduction))

        # Keep words in original order
        kept_words_set = set(ws[0] for ws in word_scores[:keep_count])
        pruned = [w for w in words if w in kept_words_set]

        return pruned

    def _estimate_semantic_similarity(self, original: str, compressed: str) -> float:
        """Estimate semantic similarity (0-1)."""
        # Simple keyword overlap method
        original_words = set(w.lower() for w in original.split() if w.lower() not in self.stopwords)
        compressed_words = set(
            w.lower() for w in compressed.split() if w.lower() not in self.stopwords
        )

        if not original_words:
            return 1.0

        overlap = len(original_words & compressed_words)
        similarity = overlap / len(original_words)

        return min(1.0, similarity)

    def _calculate_entity_preservation(self, original: str, compressed: str) -> float:
        """Calculate what % of entities are preserved."""
        original_entities = [w for w in original.split() if w and w[0].isupper() and len(w) > 2]
        if not original_entities:
            return 1.0

        preserved = sum(1 for e in original_entities if e in compressed)
        return preserved / len(original_entities)

    def _calculate_readability(self, text: str) -> float:
        """Calculate readability score (0-1)."""
        words = text.split()
        if not words:
            return 0.0

        # Average word length (target 4-6 chars)
        avg_word_length = sum(len(w) for w in words) / len(words)
        length_score = 1.0 - abs(avg_word_length - 5.0) / 10.0

        # Sentence count (prefer complete sentences)
        sentences = re.split(r"[.!?]+", text)
        sentence_score = min(len(sentences) / 5.0, 1.0)  # Target 5+ sentences

        return length_score * 0.6 + sentence_score * 0.4

    def convert_episodic_to_semantic(
        self, memories: list[Memory], min_confidence: float = 0.7
    ) -> list[SemanticMemory]:
        """Convert episodic memories to semantic knowledge.

        Args:
            memories: Episodic memories to convert
            min_confidence: Minimum confidence threshold

        Returns:
            List of semantic memories
        """
        semantic_memories = []

        # Group memories by topic/theme
        memory_groups = self._group_by_topic(memories)

        for topic, group_memories in memory_groups.items():
            # Extract semantic knowledge from each group
            semantic = self._extract_semantic_knowledge(group_memories, topic)

            if semantic and semantic.confidence >= min_confidence:
                semantic_memories.append(semantic)

        return semantic_memories

    def _group_by_topic(self, memories: list[Memory]) -> dict[str, list[Memory]]:
        """Group memories by topic/theme."""
        groups = defaultdict(list)

        for memory in memories:
            # Simple topic extraction via keywords
            topic = self._infer_topic(memory.text)
            groups[topic].append(memory)

        return dict(groups)

    def _infer_topic(self, text: str) -> str:
        """Infer topic from text."""
        # Extract key nouns as topics
        words = text.lower().split()
        nouns = [w for w in words if len(w) > 4 and w not in self.stopwords]

        if nouns:
            # Use most common noun as topic
            noun_counts = Counter(nouns)
            return noun_counts.most_common(1)[0][0]

        return "general"

    def _extract_semantic_knowledge(
        self, memories: list[Memory], topic: str
    ) -> Optional[SemanticMemory]:
        """Extract semantic knowledge from memory group."""
        if not memories:
            return None

        # Determine semantic type
        semantic_type = self._classify_semantic_type(memories)

        # Extract abstract knowledge
        if self.llm:
            content = self._extract_semantic_llm(memories, topic, semantic_type)
        else:
            content = self._extract_semantic_heuristic(memories, topic)

        # Calculate confidence based on consistency
        confidence = self._calculate_semantic_confidence(memories)

        # Extract supporting evidence
        evidence = [m.text for m in memories[:3]]  # Top 3 as evidence

        return SemanticMemory(
            id=f"semantic_{topic}_{len(memories)}",
            semantic_type=semantic_type,
            content=content,
            abstraction_level=self._determine_abstraction_level(content),
            source_memory_ids=[m.id for m in memories],
            confidence=confidence,
            supporting_evidence=evidence,
            metadata={"topic": topic, "source_count": len(memories)},
        )

    def _classify_semantic_type(self, memories: list[Memory]) -> SemanticType:
        """Classify type of semantic knowledge."""
        # Simple classification based on memory types
        types = [m.type for m in memories]
        type_counts = Counter(types)
        most_common_type = type_counts.most_common(1)[0][0]

        type_mapping = {
            MemoryType.FACT: SemanticType.FACT,
            MemoryType.PREFERENCE: SemanticType.CONCEPT,
            MemoryType.GOAL: SemanticType.PRINCIPLE,
            MemoryType.HABIT: SemanticType.SKILL,
            MemoryType.EVENT: SemanticType.FACT,
            MemoryType.CONTEXT: SemanticType.RELATIONSHIP,
        }

        return type_mapping.get(most_common_type, SemanticType.CONCEPT)

    def _extract_semantic_llm(
        self, memories: list[Memory], topic: str, semantic_type: SemanticType
    ) -> str:
        """Extract semantic knowledge using LLM."""
        if self.llm is None:
            return self._extract_semantic_heuristic(memories, topic)

        memory_texts = "\n".join([f"- {m.text}" for m in memories[:10]])

        prompt = f"""Extract abstract {semantic_type.value} knowledge about "{topic}" from these specific memories.

Memories:
{memory_texts}

Provide a single, abstract statement that captures the general {semantic_type.value} across all these specific instances. Be concise and general.

Abstract {semantic_type.value}:"""

        try:
            response = self.llm.generate(prompt, max_tokens=100, temperature=0.3)
            return (
                response.strip() if response else self._extract_semantic_heuristic(memories, topic)
            )
        except Exception as e:
            logger.warning(f"Semantic extraction failed: {e}")
            return self._extract_semantic_heuristic(memories, topic)

    def _extract_semantic_heuristic(self, memories: list[Memory], topic: str) -> str:
        """Heuristic semantic extraction."""
        # Find common patterns
        all_words = []
        for memory in memories:
            words = [
                w.lower()
                for w in memory.text.split()
                if w.lower() not in self.stopwords and len(w) > 3
            ]
            all_words.extend(words)

        # Most common words represent the semantic knowledge
        word_counts = Counter(all_words)
        top_words = [word for word, count in word_counts.most_common(10)]

        return f"Knowledge about {topic}: {', '.join(top_words)}"

    def _calculate_semantic_confidence(self, memories: list[Memory]) -> float:
        """Calculate confidence in semantic extraction."""
        # Based on number of supporting memories and their confidence
        if not memories:
            return 0.0

        avg_confidence = sum(m.confidence for m in memories) / len(memories)
        count_factor = min(len(memories) / 10.0, 1.0)  # More memories = higher confidence

        result: float = (avg_confidence * 0.7) + (count_factor * 0.3)
        return result

    def _determine_abstraction_level(self, content: str) -> int:
        """Determine abstraction level (1-5)."""
        # Simple heuristic based on word complexity and generality
        words = content.split()
        avg_word_length = sum(len(w) for w in words) / len(words) if words else 0

        # Longer words often indicate higher abstraction
        if avg_word_length > 8:
            return 5
        elif avg_word_length > 6:
            return 4
        elif avg_word_length > 5:
            return 3
        elif avg_word_length > 4:
            return 2
        else:
            return 1

    def evaluate_compression(
        self, original: str, compressed: str, ground_truth_facts: Optional[list[str]] = None
    ) -> CompressionMetrics:
        """Comprehensive compression quality evaluation.

        Args:
            original: Original text
            compressed: Compressed text
            ground_truth_facts: Optional list of facts that must be preserved

        Returns:
            Detailed compression metrics
        """
        original_tokens = Memory.estimate_tokens(original)
        compressed_tokens = Memory.estimate_tokens(compressed)

        # Calculate basic metrics
        compression_ratio = compressed_tokens / original_tokens if original_tokens > 0 else 1.0
        semantic_similarity = self._estimate_semantic_similarity(original, compressed)
        entity_preservation = self._calculate_entity_preservation(original, compressed)

        # Calculate fact preservation if ground truth provided
        fact_preservation = 1.0
        if ground_truth_facts:
            preserved = sum(1 for fact in ground_truth_facts if fact.lower() in compressed.lower())
            fact_preservation = preserved / len(ground_truth_facts)

        # Calculate information density
        unique_words = len(set(compressed.lower().split()))
        semantic_density = unique_words / compressed_tokens if compressed_tokens > 0 else 0.0

        # Calculate readability
        readability = self._calculate_readability(compressed)

        return CompressionMetrics(
            compression_ratio=compression_ratio,
            information_retention=semantic_similarity,
            semantic_density=semantic_density,
            entity_preservation=entity_preservation,
            fact_preservation=fact_preservation,
            readability_score=readability,
        )
