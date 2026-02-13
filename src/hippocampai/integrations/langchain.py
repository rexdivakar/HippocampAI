"""LangChain integration for HippocampAI.

Provides LangChain-compatible memory and retriever components.

Requires: ``pip install langchain-core langchain-classic``

Example:
    >>> from hippocampai import MemoryClient
    >>> from hippocampai.integrations.langchain import HippocampMemory
    >>>
    >>> client = MemoryClient()
    >>> memory = HippocampMemory(client, user_id="alice")
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

LANGCHAIN_AVAILABLE = False
try:
    from langchain_classic.memory.chat_memory import BaseChatMemory
    from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
    from langchain_core.documents import Document
    from langchain_core.messages import AIMessage, HumanMessage
    from langchain_core.retrievers import BaseRetriever

    LANGCHAIN_AVAILABLE = True
except ImportError:
    BaseChatMemory = object  # type: ignore[assignment,misc]
    BaseRetriever = object  # type: ignore[assignment,misc]
    CallbackManagerForRetrieverRun = Any  # type: ignore[assignment,misc]
    Document = Any  # type: ignore[assignment,misc]
    AIMessage = Any  # type: ignore[assignment,misc]
    HumanMessage = Any  # type: ignore[assignment,misc]

if TYPE_CHECKING:
    from hippocampai.client import MemoryClient


class HippocampMemory(BaseChatMemory):
    """LangChain memory backed by HippocampAI.

    Stores conversation history as memories and retrieves relevant
    context for each interaction.

    Example:
        >>> memory = HippocampMemory(client, user_id="alice")
        >>> memory.save_context(
        ...     {"input": "What's my favorite color?"},
        ...     {"output": "You mentioned you like blue."}
        ... )
    """

    client: Any  # MemoryClient
    user_id: str
    session_id: Optional[str] = None
    memory_key: str = "history"
    input_key: str = "input"
    output_key: str = "output"
    return_messages: bool = True
    k: int = 5

    def __init__(
        self,
        client: MemoryClient,
        user_id: str,
        session_id: Optional[str] = None,
        memory_key: str = "history",
        k: int = 5,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.client = client
        self.user_id = user_id
        self.session_id = session_id
        self.memory_key = memory_key
        self.k = k

    @property
    def memory_variables(self) -> List[str]:
        """Return memory variables."""
        return [self.memory_key]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Load relevant memories for the current input."""
        query = inputs.get(self.input_key, "")

        if not query:
            return {self.memory_key: [] if self.return_messages else ""}

        results = self.client.recall(
            query=query,
            user_id=self.user_id,
            session_id=self.session_id,
            k=self.k,
        )

        if self.return_messages:
            messages = []
            for result in results:
                msg_type = result.memory.metadata.get("message_type", "human")
                if msg_type == "ai":
                    messages.append(AIMessage(content=result.memory.text))
                else:
                    messages.append(HumanMessage(content=result.memory.text))
            return {self.memory_key: messages}
        else:
            history = "\n".join([r.memory.text for r in results])
            return {self.memory_key: history}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save conversation turn to memory."""
        input_text = inputs.get(self.input_key, "")
        output_text = outputs.get(self.output_key, "")

        if input_text:
            self.client.remember(
                text=input_text,
                user_id=self.user_id,
                session_id=self.session_id,
                type="context",
            )

        if output_text:
            self.client.remember(
                text=output_text,
                user_id=self.user_id,
                session_id=self.session_id,
                type="context",
            )

    def clear(self) -> None:
        """Clear memory (not implemented - memories are persistent)."""


class HippocampRetriever(BaseRetriever):
    """LangChain retriever backed by HippocampAI.

    Retrieves relevant memories as LangChain Documents.

    Example:
        >>> retriever = HippocampRetriever(client, user_id="alice")
        >>> docs = retriever.get_relevant_documents("coffee preferences")
    """

    client: Any  # MemoryClient
    user_id: str
    session_id: Optional[str] = None
    k: int = 5
    score_threshold: float = 0.0
    filter_types: Optional[List[str]] = None

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

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> List[Document]:
        """Retrieve relevant documents."""
        results = self.client.recall(
            query=query,
            user_id=self.user_id,
            session_id=self.session_id,
            k=self.k,
        )

        documents = []
        for result in results:
            if result.score < self.score_threshold:
                continue

            if self.filter_types:
                mem_type = result.memory.type
                if hasattr(mem_type, "value"):
                    mem_type = mem_type.value
                if mem_type not in self.filter_types:
                    continue

            doc = Document(
                page_content=result.memory.text,
                metadata={
                    "memory_id": result.memory.id,
                    "type": str(result.memory.type),
                    "importance": result.memory.importance,
                    "score": result.score,
                    "created_at": result.memory.created_at.isoformat(),
                    "tags": result.memory.tags,
                },
            )
            documents.append(doc)

        return documents

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> List[Document]:
        """Async retrieve (falls back to sync)."""
        return self._get_relevant_documents(query, run_manager=run_manager)
