"""LlamaIndex integration for HippocampAI.

Provides LlamaIndex-compatible retriever and memory store.

Example:
    >>> from llama_index import VectorStoreIndex
    >>> from hippocampai import MemoryClient
    >>> from hippocampai.integrations.llamaindex import HippocampRetriever
    >>>
    >>> client = MemoryClient()
    >>> retriever = HippocampRetriever(client, user_id="alice")
    >>> # Use with query engine
    >>> response = retriever.retrieve("What do I like?")
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Optional

if TYPE_CHECKING:
    from hippocampai.client import MemoryClient

LLAMAINDEX_AVAILABLE = False

# Check if llama_index is available
try:
    from llama_index.core.retrievers import BaseRetriever
    from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

    LLAMAINDEX_AVAILABLE = True
except ImportError:
    try:
        # Try older import path
        from llama_index.retrievers import BaseRetriever  # type: ignore[no-redef]
        from llama_index.schema import NodeWithScore, QueryBundle, TextNode  # type: ignore[no-redef]

        LLAMAINDEX_AVAILABLE = True
    except ImportError:

        class BaseRetriever:  # type: ignore[no-redef]
            """Stub base retriever when llama_index is not installed."""

            pass

        class NodeWithScore:  # type: ignore[no-redef]
            """Stub for NodeWithScore when llama_index is not installed."""

            def __init__(self, node: Any = None, score: float = 0.0) -> None:
                self.node = node
                self.score = score

        class TextNode:  # type: ignore[no-redef]
            """Stub for TextNode when llama_index is not installed."""

            def __init__(
                self,
                text: str = "",
                id_: Optional[str] = None,
                metadata: Optional[dict[str, Any]] = None,
            ) -> None:
                self.text = text
                self.id_ = id_
                self.metadata = metadata or {}

        class QueryBundle:  # type: ignore[no-redef]
            """Stub for QueryBundle when llama_index is not installed."""

            def __init__(self, query_str: str = "") -> None:
                self.query_str = query_str


class HippocampRetriever(BaseRetriever):
    """LlamaIndex retriever backed by HippocampAI.

    Retrieves relevant memories as LlamaIndex nodes.

    Example:
        >>> retriever = HippocampRetriever(client, user_id="alice", k=5)
        >>> nodes = retriever.retrieve("coffee preferences")
        >>> for node in nodes:
        ...     print(node.text, node.score)
    """

    def __init__(
        self,
        client: MemoryClient,
        user_id: str,
        session_id: Optional[str] = None,
        k: int = 5,
        score_threshold: float = 0.0,
        filter_types: Optional[List[str]] = None,
        **kwargs: Any,
    ):
        if not LLAMAINDEX_AVAILABLE:
            raise ImportError(
                "LlamaIndex is required for this integration. Install with: pip install llama-index"
            )

        super().__init__(**kwargs)
        self.client = client
        self.user_id = user_id
        self.session_id = session_id
        self.k = k
        self.score_threshold = score_threshold
        self.filter_types = filter_types

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes for a query."""
        query = query_bundle.query_str

        results = self.client.recall(
            query=query,
            user_id=self.user_id,
            session_id=self.session_id,
            k=self.k,
        )

        nodes = []
        for result in results:
            # Apply score threshold
            if result.score < self.score_threshold:
                continue

            # Apply type filter
            if self.filter_types:
                mem_type = result.memory.type
                if hasattr(mem_type, "value"):
                    mem_type = mem_type.value
                if mem_type not in self.filter_types:
                    continue

            node = TextNode(
                text=result.memory.text,
                id_=result.memory.id,
                metadata={
                    "memory_id": result.memory.id,
                    "type": str(result.memory.type),
                    "importance": result.memory.importance,
                    "created_at": result.memory.created_at.isoformat(),
                    "tags": result.memory.tags,
                    "user_id": result.memory.user_id,
                },
            )
            nodes.append(NodeWithScore(node=node, score=result.score))

        return nodes

    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Async retrieve (falls back to sync)."""
        return self._retrieve(query_bundle)


class HippocampMemoryStore:
    """LlamaIndex-compatible memory store.

    Can be used as a document store for LlamaIndex indices.

    Example:
        >>> store = HippocampMemoryStore(client, user_id="alice")
        >>> store.add_documents([doc1, doc2])
        >>> docs = store.get_all_documents()
    """

    def __init__(
        self,
        client: MemoryClient,
        user_id: str,
        namespace: Optional[str] = None,
    ):
        if not LLAMAINDEX_AVAILABLE:
            raise ImportError(
                "LlamaIndex is required for this integration. Install with: pip install llama-index"
            )

        self.client = client
        self.user_id = user_id
        self.namespace = namespace

    def add_documents(
        self,
        documents: List[Any],
        memory_type: str = "fact",
    ) -> List[str]:
        """Add documents as memories.

        Args:
            documents: LlamaIndex documents or nodes
            memory_type: Memory type for all documents

        Returns:
            List of memory IDs
        """
        memory_ids = []
        for doc in documents:
            text = doc.text if hasattr(doc, "text") else str(doc)
            metadata = doc.metadata if hasattr(doc, "metadata") else {}

            memory = self.client.remember(
                text=text,
                user_id=self.user_id,
                type=memory_type,
                tags=metadata.get("tags", []),
            )
            memory_ids.append(memory.id)

        return memory_ids

    def get_all_documents(self, limit: int = 1000) -> List[TextNode]:
        """Get all documents as LlamaIndex nodes."""
        memories = self.client.get_memories(self.user_id, limit=limit)

        nodes = []
        for memory in memories:
            node = TextNode(
                text=memory.text,
                id_=memory.id,
                metadata={
                    "type": str(memory.type),
                    "importance": memory.importance,
                    "created_at": memory.created_at.isoformat(),
                    "tags": memory.tags,
                },
            )
            nodes.append(node)

        return nodes

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document by ID."""
        try:
            self.client.delete_memory(doc_id, self.user_id)
            return True
        except Exception:
            return False
