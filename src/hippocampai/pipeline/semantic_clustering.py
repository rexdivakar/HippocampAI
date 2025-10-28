"""Semantic clustering and auto-categorization for memories.

This module provides:
- Automatic memory clustering by topics
- Dynamic tag suggestion
- Category auto-assignment
- Similar memory detection
- Topic modeling and evolution
"""

import logging
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

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
        return [
            tag
            for tag, count in tag_counts.items()
            if count >= 2 or count / len(self.memories) > 0.5
        ]

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
            "work": [
                "work",
                "job",
                "career",
                "office",
                "colleague",
                "project",
                "meeting",
                "deadline",
            ],
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
        words = re.findall(r"\b[a-z]{4,}\b", text_lower)
        word_counts = Counter(words)

        # Add most common meaningful words
        for word, count in word_counts.most_common(3):
            if word not in {"that", "this", "with", "from", "have", "been", "were", "have"}:
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
            tags = [tag.strip().lower() for tag in response.split(",")]
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
        # Note: Order matters - more specific patterns first
        category_patterns = {
            MemoryType.GOAL: [
                r"\b(want to|plan to|planning to|goal|aim|trying to|working towards)\b",
                r"\b(will|going to)\b",
            ],
            MemoryType.PREFERENCE: [
                r"\b(love|like|prefer|enjoy|favorite|hate|dislike)\b",
                r"\b(i want|i need|i wish)(?! to)\b",  # Exclude "want to" which is a goal
            ],
            MemoryType.FACT: [
                r"\b(is|are|was|were|has|have)\b",
                r"\b(work at|works at|working at|live in|lives in|living in)\b",
                r"\b(name is|age is|from|born in)\b",
            ],
            MemoryType.HABIT: [
                r"\b(always|usually|often|regularly|every day|daily|weekly)\b",
                r"\b(routine|habit|practice)\b",
            ],
            MemoryType.EVENT: [
                r"\b(happened|occurred|went to|attended|visited)\b",
                r"\b(yesterday|last week|last month|on)\b",
            ],
        }

        # Score each category
        scores = defaultdict(int)
        for category, patterns in category_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    scores[category] += 1

        # Use LLM if available and scores are unclear (but only if LLM is working)
        if self.llm and (not scores or max(scores.values()) < 2):
            llm_result = self._llm_assign_category(memory.text, fallback=None)
            if llm_result is not None:
                return llm_result
            # If LLM fails, continue with pattern-based scores below

        # Return highest scoring category, or default to current type
        if scores:
            return max(scores.items(), key=lambda x: x[1])[0]

        return memory.type

    def _llm_assign_category(
        self, text: str, fallback: Optional[MemoryType] = MemoryType.FACT
    ) -> Optional[MemoryType]:
        """Use LLM to assign category.

        Args:
            text: Text to categorize
            fallback: Fallback type if LLM fails (None to return None on failure)

        Returns:
            MemoryType if successful, fallback value if failed
        """
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

        return fallback

    def find_similar_memories(
        self, memory: Memory, existing_memories: List[Memory], similarity_threshold: float = 0.7
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

        # Semantic keyword groups for better matching
        synonym_groups = [
            {"love", "like", "enjoy", "adore", "prefer", "fond"},
            {"hate", "dislike", "detest", "loathe"},
            {"work", "job", "career", "employment", "position"},
            {"want", "need", "desire", "wish", "goal"},
            {"always", "usually", "often", "regularly", "frequently"},
        ]

        query_text = memory.text.lower()
        query_tokens = set(re.findall(r"\w+", query_text))

        # Remove common stop words
        stop_words = {
            "i",
            "a",
            "the",
            "is",
            "am",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "at",
            "on",
            "in",
            "to",
            "for",
            "of",
            "and",
            "or",
            "but",
            "not",
            "really",
            "very",
            "quite",
            "just",
            "so",
            "too",
            "also",
        }
        query_tokens = query_tokens - stop_words

        for existing in existing_memories:
            if existing.id == memory.id:
                continue

            existing_text = existing.text.lower()
            existing_tokens = set(re.findall(r"\w+", existing_text))
            existing_tokens = existing_tokens - stop_words

            if not query_tokens or not existing_tokens:
                continue

            # Calculate base token overlap (Jaccard)
            intersection = query_tokens.intersection(existing_tokens)
            union = query_tokens.union(existing_tokens)
            jaccard_sim = len(intersection) / len(union) if union else 0

            # Calculate semantic similarity using synonym groups
            semantic_boost = 0
            for group in synonym_groups:
                query_has = any(token in group for token in query_tokens)
                existing_has = any(token in group for token in existing_tokens)
                if query_has and existing_has:
                    semantic_boost += 0.20  # Boost for semantic match

            # Combine scores
            similarity = min(jaccard_sim + semantic_boost, 1.0)

            # Also check for substring matches (one contains the other)
            if query_text in existing_text or existing_text in query_text:
                similarity = max(similarity, 0.8)

            if similarity >= similarity_threshold:
                similar.append((existing, similarity))

        # Sort by similarity
        similar.sort(key=lambda x: x[1], reverse=True)
        return similar

    def cluster_memories(
        self, memories: List[Memory], max_clusters: int = 10
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
        cluster_objects = [MemoryCluster(topic, mems) for topic, mems in clusters.items()]

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
        words = re.findall(r"\b[a-z]{4,}\b", text_lower)
        if words:
            return Counter(words).most_common(1)[0][0]

        return "general"

    def evolve_topics(
        self, old_clusters: List[MemoryCluster], new_memories: List[Memory]
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

        # Always verify/correct category using pattern-based detection
        suggested_type = self.assign_category(memory)

        # Apply the suggested type if it's different from current type
        # Pattern-based detection is more accurate than user-provided type
        if suggested_type != enriched.type:
            logger.debug(
                f"Auto-correcting memory type: {enriched.type} -> {suggested_type} for text '{memory.text[:50]}'"
            )
            enriched.type = suggested_type

        # Add suggested tags if none exist
        # Use enriched memory so tags reflect the corrected type
        if not enriched.tags:
            enriched.tags = self.suggest_tags(enriched)
        else:
            # Add additional suggested tags
            suggested = self.suggest_tags(enriched)
            existing_tags = set(enriched.tags)
            new_tags = [tag for tag in suggested if tag not in existing_tags]
            enriched.tags.extend(new_tags[:3])  # Add up to 3 new tags

        return enriched

    def detect_topic_shift(
        self, recent_memories: List[Memory], window_size: int = 10
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
        older = (
            recent_memories[-window_size * 2 : -window_size]
            if len(recent_memories) >= window_size * 2
            else []
        )

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

    def hierarchical_cluster_memories(
        self, memories: List[Memory], min_cluster_size: int = 2
    ) -> Dict[str, Any]:
        """Perform hierarchical clustering on memories.

        Args:
            memories: List of memories to cluster
            min_cluster_size: Minimum memories per cluster

        Returns:
            Hierarchical clustering result with tree structure
        """
        if not memories:
            return {"clusters": [], "hierarchy": {}}

        # Build similarity matrix
        similarity_matrix = self._build_similarity_matrix(memories)

        # Perform hierarchical clustering using simple agglomerative approach
        clusters = [[i] for i in range(len(memories))]  # Start with singleton clusters
        cluster_similarities = []

        while len(clusters) > 1:
            # Find most similar pair of clusters
            max_sim = -1
            merge_i, merge_j = -1, -1

            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    # Compute average similarity between clusters (average linkage)
                    sim = self._compute_cluster_similarity(
                        clusters[i], clusters[j], similarity_matrix
                    )
                    if sim > max_sim:
                        max_sim = sim
                        merge_i, merge_j = i, j

            # Stop if no good merges left
            if max_sim < 0.3:  # Similarity threshold
                break

            # Merge clusters
            merged = clusters[merge_i] + clusters[merge_j]
            cluster_similarities.append((merged, max_sim))

            # Remove old clusters and add merged
            clusters = [c for idx, c in enumerate(clusters) if idx not in [merge_i, merge_j]]
            clusters.append(merged)

        # Build result structure
        result_clusters = []
        for cluster_indices in clusters:
            if len(cluster_indices) >= min_cluster_size:
                cluster_memories = [memories[i] for i in cluster_indices]
                topic = self._identify_topic(" ".join([m.text for m in cluster_memories]))

                result_clusters.append(
                    {
                        "topic": topic,
                        "memories": cluster_memories,
                        "size": len(cluster_memories),
                        "cohesion": self._compute_cohesion(cluster_indices, similarity_matrix),
                    }
                )

        return {"clusters": result_clusters, "hierarchy": cluster_similarities}

    def _build_similarity_matrix(self, memories: List[Memory]) -> List[List[float]]:
        """Build pairwise similarity matrix for memories."""
        n = len(memories)
        matrix = [[0.0] * n for _ in range(n)]

        for i in range(n):
            for j in range(i + 1, n):
                similar = self.find_similar_memories(
                    memories[i], [memories[j]], similarity_threshold=0.0
                )
                sim = similar[0][1] if similar else 0.0
                matrix[i][j] = sim
                matrix[j][i] = sim

        return matrix

    def _compute_cluster_similarity(
        self, cluster1: List[int], cluster2: List[int], similarity_matrix: List[List[float]]
    ) -> float:
        """Compute similarity between two clusters (average linkage)."""
        similarities = []
        for i in cluster1:
            for j in cluster2:
                similarities.append(similarity_matrix[i][j])

        return sum(similarities) / len(similarities) if similarities else 0.0

    def _compute_cohesion(self, cluster: List[int], similarity_matrix: List[List[float]]) -> float:
        """Compute cohesion score for a cluster."""
        if len(cluster) <= 1:
            return 1.0

        similarities = []
        for i in range(len(cluster)):
            for j in range(i + 1, len(cluster)):
                similarities.append(similarity_matrix[cluster[i]][cluster[j]])

        return sum(similarities) / len(similarities) if similarities else 0.0

    def compute_cluster_quality_metrics(self, cluster: MemoryCluster) -> Dict[str, float]:
        """Compute quality metrics for a memory cluster.

        Args:
            cluster: Memory cluster to evaluate

        Returns:
            Dictionary of quality metrics
        """
        memories = cluster.memories

        if not memories:
            return {
                "cohesion": 0.0,
                "separation": 0.0,
                "silhouette": 0.0,
                "diversity": 0.0,
                "temporal_density": 0.0,
            }

        # Cohesion: average pairwise similarity within cluster
        if len(memories) > 1:
            similarities = []
            for i in range(len(memories)):
                similar = self.find_similar_memories(
                    memories[i], memories[i + 1 :], similarity_threshold=0.0
                )
                similarities.extend([sim for _, sim in similar])
            cohesion = sum(similarities) / len(similarities) if similarities else 1.0
        else:
            cohesion = 1.0

        # Diversity: variety of memory types and tags
        types = set(m.type for m in memories)
        all_tags = set()
        for m in memories:
            all_tags.update(m.tags)
        diversity = (len(types) / len(MemoryType)) * 0.5 + (min(len(all_tags), 10) / 10) * 0.5

        # Temporal density: how closely memories are clustered in time
        if len(memories) > 1:
            timestamps = [m.created_at.timestamp() for m in memories]
            time_span = max(timestamps) - min(timestamps)
            # Normalize to 0-1 (1 = all within 1 day, 0 = spread over 1 year)
            temporal_density = max(0.0, 1.0 - (time_span / (365 * 24 * 3600)))
        else:
            temporal_density = 1.0

        return {
            "cohesion": cohesion,
            "diversity": diversity,
            "temporal_density": temporal_density,
            "size": len(memories),
            "tag_count": len(all_tags),
        }

    def optimize_cluster_count(
        self, memories: List[Memory], min_k: int = 2, max_k: int = 15
    ) -> int:
        """Determine optimal number of clusters using elbow method.

        Args:
            memories: Memories to cluster
            min_k: Minimum number of clusters
            max_k: Maximum number of clusters

        Returns:
            Optimal number of clusters
        """
        if len(memories) < min_k:
            return len(memories)

        # Try different cluster counts and compute average cohesion
        cohesion_scores = []

        for k in range(min_k, min(max_k + 1, len(memories) + 1)):
            clusters = self.cluster_memories(memories, max_clusters=k)

            # Compute average cohesion
            total_cohesion = 0
            for cluster in clusters:
                metrics = self.compute_cluster_quality_metrics(cluster)
                total_cohesion += metrics["cohesion"] * len(cluster.memories)

            avg_cohesion = total_cohesion / len(memories) if memories else 0
            cohesion_scores.append((k, avg_cohesion))

        # Find elbow point (where adding more clusters doesn't help much)
        if len(cohesion_scores) < 2:
            return min_k

        # Look for biggest improvement drop
        improvements = []
        for i in range(1, len(cohesion_scores)):
            improvement = cohesion_scores[i][1] - cohesion_scores[i - 1][1]
            improvements.append((cohesion_scores[i][0], improvement))

        # Find where improvement drops below threshold
        for k, improvement in improvements:
            if improvement < 0.05:  # Diminishing returns threshold
                return k - 1

        # If no clear elbow, return middle value
        return (min_k + max_k) // 2
