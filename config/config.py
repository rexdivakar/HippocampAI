"""Configuration settings for Qdrant connection."""

QDRANT_HOST = "192.168.1.120"
QDRANT_PORT = 6334
VECTOR_SIZE = 384

COLLECTIONS = {
    "personal_facts": "Store personal facts and preferences",
    "conversation_history": "Store conversation context and history",
    "knowledge_base": "Store general knowledge and information",
}
