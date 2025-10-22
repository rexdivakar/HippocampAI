"""Semantic clustering and auto-categorization for memories.

This module provides:
- Automatic memory clustering by topics
- Dynamic tag suggestion
- Category auto-assignment
- Similar memory detection
- Topic modeling and evolution
"""

from collections import Counter, defaultdict
from typing import Dict, List, Optional, Set, Tuple
import logging
import re

from hippocampai.models.memory import Memory, MemoryType

logger = logging.getLogger(__name__)


class MemoryCluster:
    """Represents a cluster of semantically similar memories."""

    def __init__(self, topic: str, memories: List[Memory]):
        self.topic = topic
        self.memories = memories
        self.tags = self._extract_common_tags()

    def _extract_common_tags(self) -> List[str]:
        """Extract tags common across memories in cluster."""
        all_tags = []
        for mem in self.memories:
            all_tags.extend(mem.tags)

        # Return tags that appear in multiple memories
        tag_counts = Counter(all_tags)
        return [tag for tag, count in tag_counts.items() if count >= 2 or count / len(self.memories) > 0.5]

    def add_memory(self, memory: Memory):
        """Add memory to cluster."""
        self.memories.append(memory)
        self.tags = self._extract_common_tags()


class SemanticCategorizer:
    """Handles semantic clustering and auto-categorization of memories."""

    def __init__(self, llm=None):
        """Initialize categorizer.

        Args:
            llm: Language model for category assignment (optional)
        """
        self.llm = llm

        # Common topic keywords for basic categorization
        self.topic_keywords = {
            "work": ["work", "job", "career", "office", "colleague", "project", "meeting", "deadline"],
            "personal": ["family", "friend", "hobby", "home", "personal", "relationship"],
            "health": ["health", "exercise", "fitness", "diet", "medical", "doctor", "wellness"],
            "food": ["food", "restaurant", "cuisine", "meal", "cooking", "eating", "drink"],
            "travel": ["travel", "trip", "vacation", "hotel", "flight", "destination", "visit"],
            "shopping": ["buy", "purchase", "shopping", "store", "product", "price", "order"],
            "entertainment": ["movie", "music", "game", "show", "watch", "listen", "play"],
            "learning": ["learn", "study", "course", "book", "education", "skill", "knowledge"],
            "technology": ["software", "app", "computer", "phone", "tech", "digital", "code"],
            "finance": ["money", "bank", "payment", "budget", "invest", "financial", "cost"],
        }

    def suggest_tags(self, memory: Memory, max_tags: int = 5) -> List[str]:
        """Suggest tags for a memory based on content analysis.

        Args:
            memory: Memory to tag
            max_tags: Maximum number of tags to suggest

        Returns:
            List of suggested tags
        """
        text_lower = memory.text.lower()
        suggested = set()

        # Extract keywords (nouns, important words)
        words = re.findall(r'\b[a-z]{4,}\b', text_lower)
        word_counts = Counter(words)

        # Add most common meaningful words
        for word, count in word_counts.most_common(3):
            if word not in {'that', 'this', 'with', 'from', 'have', 'been', 'were', 'have'}:
                suggested.add(word)

        # Add topic-based tags
        for topic, keywords in self.topic_keywords.items():
            if any(kw in text_lower for kw in keywords):
                suggested.add(topic)

        # Add type-based tag
        suggested.add(memory.type.value)

        # LLM-based tag suggestion if available
        if self.llm and len(suggested) < max_tags:
            llm_tags = self._llm_suggest_tags(memory.text)
            suggested.update(llm_tags)

        return list(suggested)[:max_tags]

    def _llm_suggest_tags(self, text: str, max_tags: int = 3) -> List[str]:
        """Use LLM to suggest tags."""
        prompt = f"""Generate {max_tags} relevant tags (single words or short phrases) for this text.
Return only the tags, comma-separated, no explanations.

Text: {text}

Tags:"""

        try:
            response = self.llm.generate(prompt, max_tokens=30)
            # Parse comma-separated tags
            tags = [tag.strip().lower() for tag in response.split(',')]
            return [tag for tag in tags if tag and len(tag) < 20][:max_tags]
        except Exception as e:
            logger.warning(f"LLM tag suggestion failed: {e}")
            return []

    def assign_category(self, memory: Memory) -> MemoryType:
        """Auto-assign or refine memory category.

        Args:
            memory: Memory to categorize

        Returns:
            Suggested MemoryType
        """
        text_lower = memory.text.lower()

        # Pattern-based category detection
        category_patterns = {
            MemoryType.PREFERENCE: [
                r'\b(love|like|prefer|enjoy|favorite|hate|dislike)\b',
                r'\b(i want|i need|i wish)\b',
            ],
            MemoryType.FACT: [
                r'\b(is|are|was|were|has|have|works at|lives in)\b',
                r'\b(name is|age is|from|born in)\b',
            ],
            MemoryType.GOAL: [
                r'\b(want to|plan to|goal|aim|trying to|working towards)\b',
                r'\b(will|going to|planning)\b',
            ],
            MemoryType.HABIT: [
                r'\b(always|usually|often|regularly|every day|daily|weekly)\b',
                r'\b(routine|habit|practice)\b',
            ],
            MemoryType.EVENT: [
                r'\b(happened|occurred|went to|attended|visited)\b',
                r'\b(yesterday|last week|last month|on)\b',
            ],
        }

        # Score each category
        scores = defaultdict(int)
        for category, patterns in category_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    scores[category] += 1

        # Use LLM if available and scores are unclear
        if self.llm and (not scores or max(scores.values()) < 2):
            return self._llm_assign_category(memory.text)

        # Return highest scoring category, or default to current type
        if scores:
            return max(scores.items(), key=lambda x: x[1])[0]

        return memory.type

    def _llm_assign_category(self, text: str) -> MemoryType:
        """Use LLM to assign category."""
        categories = ", ".join([t.value for t in MemoryType])
        prompt = f"""Categorize this memory into ONE of these types: {categories}

Memory: {text}

Category (respond with only the category name):"""

        try:
            response = self.llm.generate(prompt, max_tokens=10).strip().lower()

            # Try to match response to MemoryType
            for mem_type in MemoryType:
                if mem_type.value in response:
                    return mem_type

        except Exception as e:
            logger.warning(f"LLM category assignment failed: {e}")

        return MemoryType.FACT  # Default fallback

    def find_similar_memories(
        self,
        memory: Memory,
        existing_memories: List[Memory],
        similarity_threshold: float = 0.7
    ) -> List[Tuple[Memory, float]]:
        """Find memories similar to the given memory.

        Args:
            memory: Query memory
            existing_memories: Pool of existing memories
            similarity_threshold: Minimum similarity score

        Returns:
            List of (memory, similarity_score) tuples
        """
        similar = []

        query_tokens = set(re.findall(r'\w+', memory.text.lower()))

        for existing in existing_memories:
            if existing.id == memory.id:
                continue

            # Calculate token overlap
            existing_tokens = set(re.findall(r'\w+', existing.text.lower()))
            if not query_tokens or not existing_tokens:
                continue

            intersection = query_tokens.intersection(existing_tokens)
            union = query_tokens.union(existing_tokens)
            similarity = len(intersection) / len(union)

            if similarity >= similarity_threshold:
                similar.append((existing, similarity))

        # Sort by similarity
        similar.sort(key=lambda x: x[1], reverse=True)
        return similar

    def cluster_memories(
        self,
        memories: List[Memory],
        max_clusters: int = 10
    ) -> List[MemoryCluster]:
        """Cluster memories by semantic similarity.

        Args:
            memories: List of memories to cluster
            max_clusters: Maximum number of clusters

        Returns:
            List of MemoryCluster objects
        """
        if not memories:
            return []

        # Simple topic-based clustering using keywords
        clusters: Dict[str, List[Memory]] = defaultdict(list)

        for memory in memories:
            # Assign to topic based on content
            topic = self._identify_topic(memory.text)
            clusters[topic].append(memory)

        # Convert to MemoryCluster objects
        cluster_objects = [
            MemoryCluster(topic, mems)
            for topic, mems in clusters.items()
        ]

        # Limit to max_clusters (merge smallest if needed)
        if len(cluster_objects) > max_clusters:
            # Keep largest clusters
            cluster_objects.sort(key=lambda c: len(c.memories), reverse=True)
            cluster_objects = cluster_objects[:max_clusters]

        return cluster_objects

    def _identify_topic(self, text: str) -> str:
        """Identify primary topic of text."""
        text_lower = text.lower()

        # Check topic keywords
        topic_scores = defaultdict(int)
        for topic, keywords in self.topic_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    topic_scores[topic] += 1

        if topic_scores:
            return max(topic_scores.items(), key=lambda x: x[1])[0]

        # Extract most common noun as topic
        words = re.findall(r'\b[a-z]{4,}\b', text_lower)
        if words:
            return Counter(words).most_common(1)[0][0]

        return "general"

    def evolve_topics(
        self,
        old_clusters: List[MemoryCluster],
        new_memories: List[Memory]
    ) -> List[MemoryCluster]:
        """Evolve topic clusters with new memories.

        Args:
            old_clusters: Existing clusters
            new_memories: New memories to incorporate

        Returns:
            Updated list of clusters
        """
        # Assign new memories to existing clusters or create new ones
        unassigned = []

        for memory in new_memories:
            memory_topic = self._identify_topic(memory.text)
            assigned = False

            # Try to assign to existing cluster
            for cluster in old_clusters:
                if cluster.topic == memory_topic:
                    cluster.add_memory(memory)
                    assigned = True
                    break

            if not assigned:
                unassigned.append(memory)

        # Create new clusters for unassigned memories
        if unassigned:
            new_clusters = self.cluster_memories(unassigned, max_clusters=5)
            old_clusters.extend(new_clusters)

        return old_clusters

    def enrich_memory_with_categories(self, memory: Memory) -> Memory:
        """Enrich memory with auto-suggested tags and category.

        Args:
            memory: Memory to enrich

        Returns:
            Enriched memory with tags and refined category
        """
        enriched = memory.model_copy(deep=True)

        # Auto-assign category if not set or is generic
        if not enriched.type or enriched.type == MemoryType.CONTEXT:
            enriched.type = self.assign_category(memory)

        # Add suggested tags if none exist
        if not enriched.tags:
            enriched.tags = self.suggest_tags(memory)
        else:
            # Add additional suggested tags
            suggested = self.suggest_tags(memory)
            existing_tags = set(enriched.tags)
            new_tags = [tag for tag in suggested if tag not in existing_tags]
            enriched.tags.extend(new_tags[:3])  # Add up to 3 new tags

        return enriched

    def detect_topic_shift(
        self,
        recent_memories: List[Memory],
        window_size: int = 10
    ) -> Optional[str]:
        """Detect if there's been a shift in conversation topics.

        Args:
            recent_memories: Recent memories in chronological order
            window_size: Number of recent memories to analyze

        Returns:
            New dominant topic if shift detected, None otherwise
        """
        if len(recent_memories) < window_size:
            return None

        # Get topics for recent window
        recent = recent_memories[-window_size:]
        older = recent_memories[-window_size*2:-window_size] if len(recent_memories) >= window_size*2 else []

        recent_topics = [self._identify_topic(m.text) for m in recent]
        older_topics = [self._identify_topic(m.text) for m in older] if older else []

        # Count topic frequencies
        recent_topic_counts = Counter(recent_topics)
        older_topic_counts = Counter(older_topics) if older_topics else Counter()

        # Get dominant topics
        recent_dominant = recent_topic_counts.most_common(1)[0][0] if recent_topic_counts else None
        older_dominant = older_topic_counts.most_common(1)[0][0] if older_topic_counts else None

        # Detect shift
        if recent_dominant and recent_dominant != older_dominant:
            # Check if shift is significant (>50% of recent memories)
            if recent_topic_counts[recent_dominant] / len(recent) > 0.5:
                return recent_dominant

        return None
