"""LangChain integration for HippocampAI.

Provides LangChain-compatible memory and retriever components.

Example:
    >>> from langchain.chains import ConversationChain
    >>> from hippocampai import MemoryClient
    >>> from hippocampai.integrations.langchain import HippocampMemory
    >>>
    >>> client = MemoryClient()
    >>> memory = HippocampMemory(client, user_id="alice")
    >>> chain = ConversationChain(llm=llm, memory=memory)
    >>> response = chain.predict(input="Hello!")
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from hippocampai.client import MemoryClient

LANGCHAIN_AVAILABLE = False

# Check if langchain is available
try:
    from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
    from langchain_core.documents import Document
    from langchain_core.messages import AIMessage, HumanMessage
    from langchain_core.retrievers import BaseRetriever

    from langchain.memory.chat_memory import BaseChatMemory

    LANGCHAIN_AVAILABLE = True
except ImportError:
    pass


if not LANGCHAIN_AVAILABLE:

    class BaseChatMemory:
        """Stub for BaseChatMemory when langchain is not installed."""

        def __init__(self, **kwargs: Any) -> None:
            pass

    class BaseRetriever:
        """Stub for BaseRetriever when langchain is not installed."""

        pass

    class Document:
        """Stub for Document when langchain is not installed."""

        def __init__(
            self, page_content: str = "", metadata: Optional[Dict[str, Any]] = None
        ) -> None:
            self.page_content = page_content
            self.metadata = metadata or {}

    class AIMessage:
        """Stub for AIMessage when langchain is not installed."""

        def __init__(self, content: str = "") -> None:
            self.content = content

    class HumanMessage:
        """Stub for HumanMessage when langchain is not installed."""

        def __init__(self, content: str = "") -> None:
            self.content = content

    class CallbackManagerForRetrieverRun:
        """Stub for CallbackManagerForRetrieverRun when langchain is not installed."""

        pass


class HippocampMemory(BaseChatMemory):  # type: ignore[misc]
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
    k: int = 5  # Number of memories to retrieve

    def __init__(
        self,
        client: MemoryClient,
        user_id: str,
        session_id: Optional[str] = None,
        memory_key: str = "history",
        k: int = 5,
        **kwargs: Any,
    ):
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "LangChain is required for this integration. Install with: pip install langchain"
            )

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

        # Retrieve relevant memories
        results = self.client.recall(
            query=query,
            user_id=self.user_id,
            session_id=self.session_id,
            k=self.k,
        )

        if self.return_messages:
            messages = []
            for result in results:
                # Determine message type from metadata
                msg_type = result.memory.metadata.get("message_type", "human")
                if msg_type == "ai":
                    messages.append(AIMessage(content=result.memory.text))
                else:
                    messages.append(HumanMessage(content=result.memory.text))
            return {self.memory_key: messages}
        else:
            # Return as string
            history = "\n".join([r.memory.text for r in results])
            return {self.memory_key: history}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save conversation turn to memory."""
        input_text = inputs.get(self.input_key, "")
        output_text = outputs.get(self.output_key, "")

        # Save human message
        if input_text:
            self.client.remember(
                text=input_text,
                user_id=self.user_id,
                session_id=self.session_id,
                type="context",
            )

        # Save AI response
        if output_text:
            self.client.remember(
                text=output_text,
                user_id=self.user_id,
                session_id=self.session_id,
                type="context",
            )

    def clear(self) -> None:
        """Clear memory (not implemented - memories are persistent)."""
        pass


class HippocampRetriever(BaseRetriever):  # type: ignore[misc]
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
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "LangChain is required for this integration. Install with: pip install langchain"
            )

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
