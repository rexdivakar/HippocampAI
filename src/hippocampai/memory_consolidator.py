"""Memory consolidation service for merging similar memories."""

import json
import logging
import time
from typing import Any, Dict, List, Optional, Set

import anthropic

from hippocampai.embedding_service import EmbeddingService
from hippocampai.memory_retriever import MemoryRetriever
from hippocampai.memory_updater import MemoryUpdater

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MemoryConsolidator:
    """Service for consolidating similar memories to reduce redundancy."""

    # Prompt template for memory consolidation
    CONSOLIDATION_PROMPT = """You are helping consolidate similar memories into a single comprehensive memory.

Similar Memories to Consolidate:
{memories_list}

Your task:
Create ONE consolidated memory that:
1. Captures ALL important information from all memories
2. Removes redundancy and repetition
3. Maintains factual accuracy
4. Is clear, concise, and well-structured
5. Preserves any unique details from each memory

Also identify which original memories should be archived after consolidation.

Return ONLY a valid JSON object:
{{
  "consolidated_text": "The single consolidated memory text",
  "consolidated_importance": importance_score_1_to_10,
  "memories_to_archive": ["memory_id_1", "memory_id_2", ...],
  "reasoning": "Brief explanation of consolidation decisions"
}}

Do not include any other text."""

    def __init__(
        self,
        retriever: MemoryRetriever,
        updater: MemoryUpdater,
        embedding_service: EmbeddingService,
        api_key: Optional[str] = None,
        model: str = "claude-3-5-sonnet-20241022",
    ):
        """
        Initialize the memory consolidator.

        Args:
            retriever: MemoryRetriever instance
            updater: MemoryUpdater instance
            embedding_service: EmbeddingService instance
            api_key: Anthropic API key
            model: Claude model to use
        """
        self.retriever = retriever
        self.updater = updater
        self.embeddings = embedding_service

        import os

        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Anthropic API key must be provided or set in ANTHROPIC_API_KEY environment variable"
            )

        self.model = model
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self._last_request_time = 0
        self._min_request_interval = 0.1

        logger.info("MemoryConsolidator initialized")

    def _rate_limit(self) -> None:
        """Apply basic rate limiting."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()

    def find_similar_clusters(
        self,
        user_id: str,
        similarity_threshold: float = 0.85,
        min_cluster_size: int = 2,
        max_memories: int = 100,
    ) -> List[List[Dict[str, Any]]]:
        """
        Find clusters of similar memories.

        Args:
            user_id: User identifier
            similarity_threshold: Minimum similarity for clustering
            min_cluster_size: Minimum memories in a cluster
            max_memories: Maximum memories to analyze

        Returns:
            List of memory clusters (each cluster is a list of memories)
        """
        try:
            # Get all user memories
            all_memories = self.retriever.get_memories_by_filter(
                filters={"user_id": user_id}, limit=max_memories
            )

            # Filter out archived/outdated memories
            active_memories = [
                m
                for m in all_memories
                if not m["metadata"].get("outdated", False)
                and not m["metadata"].get("archived", False)
            ]

            if len(active_memories) < min_cluster_size:
                logger.info(f"Not enough memories to cluster ({len(active_memories)})")
                return []

            logger.info(f"Analyzing {len(active_memories)} memories for clustering")

            # Build similarity graph
            clusters: List[Set[str]] = []
            processed: Set[str] = set()

            for i, memory in enumerate(active_memories):
                memory_id = memory["memory_id"]

                if memory_id in processed:
                    continue

                # Search for similar memories
                similar = self.retriever.search_memories(
                    query=memory["text"], limit=20, filters={"user_id": user_id}
                )

                # Filter by similarity threshold
                cluster_ids = {memory_id}
                for sim_mem in similar:
                    sim_id = sim_mem["memory_id"]
                    sim_score = sim_mem.get("similarity_score", 0)

                    if sim_id != memory_id and sim_score >= similarity_threshold:
                        if not sim_mem["metadata"].get("outdated", False):
                            cluster_ids.add(sim_id)

                # Only keep clusters of sufficient size
                if len(cluster_ids) >= min_cluster_size:
                    # Check if this cluster overlaps with existing ones
                    merged = False
                    for existing_cluster in clusters:
                        if len(cluster_ids & existing_cluster) > 0:
                            existing_cluster.update(cluster_ids)
                            merged = True
                            break

                    if not merged:
                        clusters.append(cluster_ids)

                    processed.update(cluster_ids)

            # Convert ID sets to memory objects
            memory_lookup = {m["memory_id"]: m for m in active_memories}
            memory_clusters = []

            for cluster_ids in clusters:
                cluster_memories = [
                    memory_lookup[mid] for mid in cluster_ids if mid in memory_lookup
                ]
                if len(cluster_memories) >= min_cluster_size:
                    memory_clusters.append(cluster_memories)

            logger.info(f"Found {len(memory_clusters)} memory clusters")

            return memory_clusters

        except Exception as e:
            logger.error(f"Failed to find similar clusters: {e}")
            raise RuntimeError("Cluster finding failed") from e

    def consolidate_cluster(
        self, cluster: List[Dict[str, Any]], max_retries: int = 2
    ) -> Optional[Dict[str, Any]]:
        """
        Consolidate a cluster of similar memories using AI.

        Args:
            cluster: List of similar memories to consolidate
            max_retries: Maximum retry attempts

        Returns:
            Consolidation result dictionary or None if failed
        """
        if len(cluster) < 2:
            logger.warning("Cluster too small for consolidation")
            return None

        try:
            # Format memories for prompt
            memories_list = []
            for i, mem in enumerate(cluster, 1):
                memories_list.append(
                    f"{i}. [ID: {mem['memory_id']}] {mem['text']} "
                    f"(Importance: {mem['metadata'].get('importance', 5)}, "
                    f"Type: {mem['metadata'].get('memory_type', 'fact')})"
                )

            memories_text = "\n".join(memories_list)

            # Generate consolidation using Claude
            for attempt in range(max_retries + 1):
                try:
                    self._rate_limit()

                    prompt = self.CONSOLIDATION_PROMPT.format(memories_list=memories_text)

                    logger.debug(f"Consolidating cluster of {len(cluster)} memories...")
                    message = self.client.messages.create(
                        model=self.model,
                        max_tokens=1024,
                        temperature=0.0,
                        messages=[{"role": "user", "content": prompt}],
                    )

                    # Parse response
                    response_text = message.content[0].text.strip()
                    start_idx = response_text.find("{")
                    end_idx = response_text.rfind("}") + 1

                    if start_idx == -1 or end_idx == 0:
                        raise ValueError("No JSON found in response")

                    json_text = response_text[start_idx:end_idx]
                    result = json.loads(json_text)

                    # Validate result
                    required_fields = [
                        "consolidated_text",
                        "consolidated_importance",
                        "memories_to_archive",
                    ]
                    for field in required_fields:
                        if field not in result:
                            raise ValueError(f"Missing field: {field}")

                    logger.info(
                        f"Consolidation successful: {len(result['memories_to_archive'])} to archive"
                    )
                    return result

                except anthropic.RateLimitError as e:
                    if attempt < max_retries:
                        time.sleep(2**attempt)
                    else:
                        logger.error(f"Rate limit exceeded: {e}")
                        return None

                except anthropic.APIError as e:
                    if attempt < max_retries:
                        time.sleep(1)
                    else:
                        logger.error(f"Claude API error: {e}")
                        return None

                except (ValueError, json.JSONDecodeError) as e:
                    if attempt < max_retries:
                        time.sleep(1)
                    else:
                        logger.error(f"Failed to parse consolidation result: {e}")
                        return None

            return None

        except Exception as e:
            logger.error(f"Cluster consolidation failed: {e}")
            return None

    def apply_consolidation(
        self, cluster: List[Dict[str, Any]], consolidation_result: Dict[str, Any]
    ) -> Optional[str]:
        """
        Apply consolidation result by merging and archiving memories.

        Args:
            cluster: Original cluster of memories
            consolidation_result: Result from consolidate_cluster

        Returns:
            Memory ID of consolidated memory, or None if failed
        """
        try:
            # Get all memory IDs from cluster
            cluster_ids = [m["memory_id"] for m in cluster]

            # Use first memory as base for merge
            primary_id = cluster_ids[0]  # noqa: F841

            # Merge memories
            merged_id = self.updater.merge_memories(
                memory_ids=cluster_ids,
                merged_text=consolidation_result["consolidated_text"],
                merged_importance=consolidation_result["consolidated_importance"],
                reason=f"Consolidated cluster: {consolidation_result.get('reasoning', 'Similar memories merged')}",
            )

            # Archive the memories that should be archived
            for mem_id in consolidation_result.get("memories_to_archive", []):
                if mem_id != merged_id:  # Don't archive the primary memory
                    try:
                        # Mark as archived instead of deleting
                        self.updater.mark_memory_outdated(
                            memory_id=mem_id,
                            reason=f"Archived during consolidation into {merged_id}",
                        )
                    except Exception as e:
                        logger.warning(f"Failed to archive memory {mem_id}: {e}")

            logger.info(
                f"Consolidation applied: merged to {merged_id}, "
                f"archived {len(consolidation_result['memories_to_archive'])} memories"
            )

            return merged_id

        except Exception as e:
            logger.error(f"Failed to apply consolidation: {e}")
            return None

    def consolidate_memories(
        self,
        user_id: str,
        similarity_threshold: float = 0.85,
        max_clusters: Optional[int] = None,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """
        Find and consolidate similar memories for a user.

        Args:
            user_id: User identifier
            similarity_threshold: Similarity threshold for clustering
            max_clusters: Maximum number of clusters to consolidate (None = all)
            dry_run: If True, only analyze without applying changes

        Returns:
            Dictionary with consolidation results and statistics
        """
        try:
            # Find similar clusters
            clusters = self.find_similar_clusters(
                user_id=user_id, similarity_threshold=similarity_threshold
            )

            if not clusters:
                return {
                    "clusters_found": 0,
                    "clusters_consolidated": 0,
                    "memories_consolidated": 0,
                    "memories_archived": 0,
                }

            # Limit number of clusters to process
            if max_clusters:
                clusters = clusters[:max_clusters]

            logger.info(f"Processing {len(clusters)} clusters (dry_run={dry_run})")

            consolidated_count = 0
            total_memories = 0
            total_archived = 0
            consolidation_results = []

            for i, cluster in enumerate(clusters, 1):
                logger.info(f"Processing cluster {i}/{len(clusters)} ({len(cluster)} memories)")

                # Generate consolidation
                result = self.consolidate_cluster(cluster)

                if not result:
                    logger.warning(f"Cluster {i} consolidation failed")
                    continue

                total_memories += len(cluster)

                if not dry_run:
                    # Apply consolidation
                    merged_id = self.apply_consolidation(cluster, result)

                    if merged_id:
                        consolidated_count += 1
                        total_archived += len(result.get("memories_to_archive", []))

                consolidation_results.append(
                    {
                        "cluster_size": len(cluster),
                        "consolidated_text": result["consolidated_text"],
                        "importance": result["consolidated_importance"],
                        "archived_count": len(result.get("memories_to_archive", [])),
                        "reasoning": result.get("reasoning", ""),
                    }
                )

            summary = {
                "clusters_found": len(clusters),
                "clusters_consolidated": consolidated_count,
                "memories_analyzed": total_memories,
                "memories_archived": total_archived,
                "dry_run": dry_run,
                "results": consolidation_results,
            }

            logger.info(
                f"Consolidation complete: {consolidated_count} clusters, "
                f"{total_archived} memories archived"
            )

            return summary

        except Exception as e:
            logger.error(f"Memory consolidation failed: {e}")
            raise RuntimeError("Memory consolidation failed") from e

    def schedule_consolidation(self, user_id: str, frequency_days: int = 7) -> Dict[str, Any]:
        """
        Check if consolidation is needed based on frequency.

        Args:
            user_id: User identifier
            frequency_days: Days between consolidations

        Returns:
            Dictionary with schedule information and recommendation
        """
        # In production, this would check last consolidation date from database
        # For now, we'll just return a recommendation

        try:
            # Get memory count
            all_memories = self.retriever.get_memories_by_filter(
                filters={"user_id": user_id}, limit=200
            )

            active_count = len(
                [
                    m
                    for m in all_memories
                    if not m["metadata"].get("outdated", False)
                    and not m["metadata"].get("archived", False)
                ]
            )

            # Simple heuristic: consolidate if > 50 active memories
            should_consolidate = active_count > 50

            return {
                "user_id": user_id,
                "frequency_days": frequency_days,
                "active_memories": active_count,
                "should_consolidate": should_consolidate,
                "recommendation": "Run consolidation" if should_consolidate else "No action needed",
            }

        except Exception as e:
            logger.error(f"Failed to check consolidation schedule: {e}")
            return {"error": str(e), "should_consolidate": False}
