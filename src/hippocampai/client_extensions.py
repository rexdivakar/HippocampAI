"""Extended features for MemoryClient - Graph, Versioning, Batch ops, etc.

This module extends the MemoryClient with additional capabilities:
- Graph indexing for memory relationships
- Version control and audit trails
- Batch operations
- Memory access tracking
- Advanced filters
- Snapshot management
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

from hippocampai.graph import MemoryGraph, RelationType
from hippocampai.models.memory import Memory
from hippocampai.storage import MemoryKVStore
from hippocampai.telemetry import OperationType
from hippocampai.utils.context_injection import ContextInjector, inject_context
from hippocampai.versioning import AuditEntry, ChangeType, MemoryVersionControl

logger = logging.getLogger(__name__)


class MemoryClientExtensions:
    """
    Mixin class providing extended features for MemoryClient.

    This class is designed to be mixed into MemoryClient to add advanced features
    without bloating the main client file.
    """

    def __init_extensions__(self, enable_graph: bool = True, enable_versioning: bool = True):
        """
        Initialize extended features.

        Args:
            enable_graph: Enable graph indexing
            enable_versioning: Enable version control and audit trails
        """
        # Graph index
        self.graph = MemoryGraph() if enable_graph else None

        # Version control & audit trail
        self.version_control = MemoryVersionControl() if enable_versioning else None

        # KV store for fast lookups
        self.kv_store = MemoryKVStore(cache_ttl=300)  # 5 min cache

        # Context injector
        self.context_injector = ContextInjector()

        logger.info("MemoryClient extensions initialized")

    # === BATCH OPERATIONS ===

    def add_memories(
        self,
        memories: List[Dict[str, Any]],
        user_id: str,
        session_id: Optional[str] = None,
    ) -> List[Memory]:
        """
        Batch add multiple memories at once.

        Args:
            memories: List of memory dictionaries with keys: text, type, importance, tags, ttl_days
            user_id: User ID
            session_id: Optional session ID

        Returns:
            List of created Memory objects
        """
        trace_id = self.telemetry.start_trace(
            operation=OperationType.REMEMBER,
            user_id=user_id,
            session_id=session_id,
            batch_size=len(memories),
        )

        created_memories = []

        try:
            for mem_data in memories:
                memory = self.remember(
                    text=mem_data.get("text", ""),
                    user_id=user_id,
                    session_id=session_id,
                    type=mem_data.get("type", "fact"),
                    importance=mem_data.get("importance"),
                    tags=mem_data.get("tags", []),
                    ttl_days=mem_data.get("ttl_days"),
                )
                created_memories.append(memory)

            self.telemetry.end_trace(
                trace_id, status="success", result={"created": len(created_memories)}
            )
            return created_memories

        except Exception as e:
            self.telemetry.end_trace(trace_id, status="error", result={"error": str(e)})
            raise

    def delete_memories(self, memory_ids: List[str], user_id: Optional[str] = None) -> int:
        """
        Batch delete multiple memories.

        Args:
            memory_ids: List of memory IDs to delete
            user_id: Optional user ID for authorization

        Returns:
            Number of memories successfully deleted
        """
        trace_id = self.telemetry.start_trace(
            operation=OperationType.DELETE,
            user_id=user_id or "system",
            batch_size=len(memory_ids),
        )

        deleted_count = 0

        try:
            for memory_id in memory_ids:
                if self.delete_memory(memory_id, user_id):
                    deleted_count += 1

            self.telemetry.end_trace(
                trace_id, status="success", result={"deleted": deleted_count}
            )
            return deleted_count

        except Exception as e:
            self.telemetry.end_trace(trace_id, status="error", result={"error": str(e)})
            raise

    # === GRAPH OPERATIONS ===

    def add_relationship(
        self,
        source_id: str,
        target_id: str,
        relation_type: RelationType,
        weight: float = 1.0,
    ) -> bool:
        """
        Add a relationship between two memories.

        Args:
            source_id: Source memory ID
            target_id: Target memory ID
            relation_type: Type of relationship
            weight: Relationship weight/strength

        Returns:
            True if successful
        """
        if not self.graph:
            logger.warning("Graph indexing not enabled")
            return False

        # Add audit entry
        if self.version_control:
            self.version_control.add_audit_entry(
                memory_id=source_id,
                change_type=ChangeType.RELATIONSHIP_ADDED,
                metadata={"target_id": target_id, "relation_type": relation_type.value},
            )

        return self.graph.add_relationship(source_id, target_id, relation_type, weight)

    def get_related_memories(
        self,
        memory_id: str,
        relation_types: Optional[List[RelationType]] = None,
        max_depth: int = 1,
    ) -> List[Tuple[str, str, float]]:
        """
        Get memories related to a given memory.

        Args:
            memory_id: Source memory ID
            relation_types: Filter by specific relation types
            max_depth: How many hops to traverse

        Returns:
            List of (memory_id, relation_type, weight) tuples
        """
        if not self.graph:
            return []

        return self.graph.get_related_memories(memory_id, relation_types, max_depth)

    def get_memory_clusters(self, user_id: str) -> List[Set[str]]:
        """
        Find clusters of related memories.

        Args:
            user_id: User ID to filter by

        Returns:
            List of memory ID sets (clusters)
        """
        if not self.graph:
            return []

        return self.graph.get_clusters(user_id)

    def export_graph_to_json(
        self,
        file_path: str,
        user_id: Optional[str] = None,
        indent: int = 2,
    ) -> str:
        """
        Export memory graph to a JSON file for persistence.

        Args:
            file_path: Path where the JSON file will be saved
            user_id: Optional user ID to export only a specific user's graph
            indent: JSON indentation level (default: 2)

        Returns:
            The file path where the graph was saved

        Example:
            >>> client.export_graph_to_json("memory_graph.json")
            >>> client.export_graph_to_json("alice_graph.json", user_id="alice")
        """
        if not self.graph:
            logger.warning("Graph indexing not enabled")
            return ""

        return self.graph.export_to_json(file_path, user_id, indent)

    def import_graph_from_json(
        self,
        file_path: str,
        merge: bool = True,
    ) -> Dict[str, Any]:
        """
        Import memory graph from a JSON file.

        Args:
            file_path: Path to the JSON file to import
            merge: If True, merge with existing graph; if False, replace existing graph

        Returns:
            Dictionary with import statistics including:
            - file_path: Path that was imported
            - nodes_before/after: Node counts before and after import
            - edges_before/after: Edge counts before and after import
            - nodes_imported: Number of nodes in the imported file
            - edges_imported: Number of edges in the imported file
            - merged: Whether the graph was merged or replaced

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file contains invalid graph format

        Example:
            >>> stats = client.import_graph_from_json("memory_graph.json")
            >>> print(f"Imported {stats['nodes_imported']} nodes")
            >>> stats = client.import_graph_from_json("backup.json", merge=False)
        """
        if not self.graph:
            logger.warning("Graph indexing not enabled")
            return {}

        return self.graph.import_from_json(file_path, merge)

    # === VERSION CONTROL ===

    def get_memory_history(self, memory_id: str) -> List:
        """
        Get version history for a memory.

        Args:
            memory_id: Memory ID

        Returns:
            List of MemoryVersion objects
        """
        if not self.version_control:
            logger.warning("Version control not enabled")
            return []

        return self.version_control.get_version_history(memory_id)

    def rollback_memory(self, memory_id: str, version_number: int) -> Optional[Memory]:
        """
        Rollback memory to a previous version.

        Args:
            memory_id: Memory ID
            version_number: Version to rollback to

        Returns:
            Updated Memory object or None
        """
        if not self.version_control:
            logger.warning("Version control not enabled")
            return None

        data = self.version_control.rollback(memory_id, version_number)
        if data:
            # Update the memory
            return self.update_memory(
                memory_id=memory_id,
                text=data.get("text"),
                importance=data.get("importance"),
                tags=data.get("tags"),
            )

        return None

    # === AUDIT TRAIL ===

    def get_audit_trail(
        self,
        memory_id: Optional[str] = None,
        user_id: Optional[str] = None,
        change_type: Optional[ChangeType] = None,
        limit: int = 100,
    ) -> List[AuditEntry]:
        """
        Get audit trail entries.

        Args:
            memory_id: Filter by memory ID
            user_id: Filter by user ID
            change_type: Filter by change type
            limit: Maximum entries to return

        Returns:
            List of AuditEntry objects
        """
        if not self.version_control:
            return []

        return self.version_control.get_audit_trail(memory_id, user_id, change_type, limit)

    # === MEMORY ACCESS TRACKING ===

    def track_memory_access(self, memory_id: str, user_id: str):
        """
        Track that a memory was accessed (updates access_count).

        Args:
            memory_id: Memory ID
            user_id: User who accessed it
        """
        # Find and update the memory
        for coll in [self.config.collection_facts, self.config.collection_prefs]:
            memory_data = self.qdrant.get(coll, memory_id)
            if memory_data:
                payload = memory_data["payload"]
                payload["access_count"] = payload.get("access_count", 0) + 1
                payload["last_accessed_at"] = datetime.now(timezone.utc).isoformat()

                self.qdrant.update(coll, memory_id, payload)

                # Add audit entry
                if self.version_control:
                    self.version_control.add_audit_entry(
                        memory_id=memory_id,
                        change_type=ChangeType.ACCESSED,
                        user_id=user_id,
                    )
                break

    # === SNAPSHOT MANAGEMENT ===

    def create_snapshot(self, collection: str = "facts") -> str:
        """
        Create a snapshot of a memory collection.

        Args:
            collection: "facts" or "prefs"

        Returns:
            Snapshot name/ID
        """
        coll_name = (
            self.config.collection_facts
            if collection == "facts"
            else self.config.collection_prefs
        )

        snapshot_name = self.qdrant.create_snapshot(coll_name)
        logger.info(f"Created snapshot: {snapshot_name} for collection {coll_name}")
        return snapshot_name

    # === CONTEXT INJECTION ===

    def inject_context(
        self,
        prompt: str,
        query: str,
        user_id: str,
        k: int = 5,
        template: str = "default",
    ) -> str:
        """
        Inject relevant memories into a prompt.

        Args:
            prompt: Original prompt/query
            query: Query to find relevant memories
            user_id: User ID
            k: Number of memories to inject
            template: Formatting template

        Returns:
            Prompt with injected context
        """
        # Retrieve relevant memories
        results = self.recall(query, user_id, k=k)

        # Track access
        for result in results:
            self.track_memory_access(result.memory.id, user_id)

        # Inject into prompt
        self.context_injector.template = template
        return self.context_injector.inject_retrieval_results(prompt, results)

    # === ADVANCED FILTERS ===

    def get_memories_advanced(
        self,
        user_id: str,
        filters: Optional[Dict[str, Any]] = None,
        sort_by: str = "created_at",  # created_at, importance, access_count
        sort_order: str = "desc",
        limit: int = 100,
    ) -> List[Memory]:
        """
        Get memories with advanced filtering and sorting.

        Args:
            user_id: User ID
            filters: Advanced filters (supports all get_memories filters plus metadata filters)
            sort_by: Field to sort by
            sort_order: "asc" or "desc"
            limit: Maximum results

        Returns:
            List of Memory objects
        """
        # Get base memories
        memories = self.get_memories(user_id, filters, limit=limit * 2)  # Get more for sorting

        # Apply metadata filters if provided
        if filters and "metadata" in filters:
            metadata_filters = filters["metadata"]
            filtered_memories = []

            for mem in memories:
                match = True
                for key, value in metadata_filters.items():
                    if key not in mem.metadata or mem.metadata[key] != value:
                        match = False
                        break
                if match:
                    filtered_memories.append(mem)

            memories = filtered_memories

        # Sort
        reverse = sort_order == "desc"
        if sort_by == "importance":
            memories.sort(key=lambda m: m.importance, reverse=reverse)
        elif sort_by == "access_count":
            memories.sort(key=lambda m: m.access_count, reverse=reverse)
        else:  # created_at
            memories.sort(key=lambda m: m.created_at, reverse=reverse)

        return memories[:limit]
