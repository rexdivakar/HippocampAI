"""Framework integrations for HippocampAI.

Official adapters for popular AI frameworks:
- LangChain: Memory and retriever components
- LlamaIndex: Memory store and retriever

Example:
    >>> # LangChain integration
    >>> from hippocampai.integrations.langchain import HippocampMemory
    >>> memory = HippocampMemory(client, user_id="alice")
    >>> chain = ConversationChain(memory=memory)
    >>>
    >>> # LlamaIndex integration
    >>> from hippocampai.integrations.llamaindex import HippocampRetriever
    >>> retriever = HippocampRetriever(client, user_id="alice")
    >>> query_engine = index.as_query_engine(retriever=retriever)
"""

# Lazy imports to avoid requiring these frameworks
def get_langchain_memory(*args, **kwargs):
    """Get LangChain memory component."""
    from hippocampai.integrations.langchain import HippocampMemory
    return HippocampMemory(*args, **kwargs)


def get_langchain_retriever(*args, **kwargs):
    """Get LangChain retriever component."""
    from hippocampai.integrations.langchain import HippocampRetriever
    return HippocampRetriever(*args, **kwargs)


def get_llamaindex_retriever(*args, **kwargs):
    """Get LlamaIndex retriever component."""
    from hippocampai.integrations.llamaindex import HippocampRetriever
    return HippocampRetriever(*args, **kwargs)


__all__ = [
    "get_langchain_memory",
    "get_langchain_retriever",
    "get_llamaindex_retriever",
]
