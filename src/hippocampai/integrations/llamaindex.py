"""LlamaIndex integration for HippocampAI.

Provides LlamaIndex-compatible retriever and memory store.

Requires: ``pip install llama-index-core``

Example:
    >>> from hippocampai import MemoryClient
    >>> from hippocampai.integrations.llamaindex import HippocampRetriever
    >>>
    >>> client = MemoryClient()
    >>> retriever = HippocampRetriever(client, user_id="alice")
    >>> response = retriever.retrieve("What do I like?")
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Optional

from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

if TYPE_CHECKING:
    from hippocampai.client import MemoryClient


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
            if result.score < self.score_threshold:
                continue

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
