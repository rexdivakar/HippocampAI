"""Session manager for conversation tracking and summarization."""

import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from hippocampai.adapters.llm_base import BaseLLM
from hippocampai.embed.embedder import Embedder
from hippocampai.models.memory import Memory
from hippocampai.models.session import (
    Entity,
    Session,
    SessionFact,
    SessionSearchResult,
    SessionStatus,
)
from hippocampai.vector.qdrant_store import QdrantStore

logger = logging.getLogger(__name__)


class SessionManager:
    """Manages sessions with summarization, fact extraction, and search."""

    # Prompts for LLM operations
    SUMMARIZE_PROMPT = """Summarize the following conversation session in 2-3 sentences. Focus on key topics discussed and main outcomes.

Session Messages:
{messages}

Summary:"""

    EXTRACT_FACTS_PROMPT = """Extract key facts from this conversation session. Return a JSON list of facts with confidence scores.

Session Messages:
{messages}

Return format:
[
  {{"fact": "User prefers Python for ML", "confidence": 0.95}},
  {{"fact": "User is working on NLP project", "confidence": 0.85}}
]

Facts:"""

    EXTRACT_ENTITIES_PROMPT = """Extract named entities from this conversation. Return a JSON list of entities.

Conversation:
{messages}

Entity types: person, organization, location, product, technology, event

Return format:
[
  {{"name": "Python", "type": "technology"}},
  {{"name": "TensorFlow", "type": "technology"}}
]

Entities:"""

    BOUNDARY_DETECTION_PROMPT = """Analyze if there's a topic change or session boundary in the following messages.
Return a JSON object with 'boundary_detected' (boolean) and 'reason' (string).

Last 3 messages from current session:
{previous_messages}

New message:
{new_message}

Return format:
{{"boundary_detected": false, "reason": "Continuing same topic about ML models"}}

Analysis:"""

    def __init__(
        self,
        qdrant_store: QdrantStore,
        embedder: Embedder,
        llm: Optional[BaseLLM] = None,
        collection_name: str = "hippocampai_sessions",
        auto_summarize_threshold: int = 10,  # messages
        inactivity_threshold_minutes: int = 30,
    ):
        """Initialize session manager.

        Args:
            qdrant_store: Vector store for session storage
            embedder: Embedder for session similarity search
            llm: Optional LLM for summarization and extraction
            collection_name: Qdrant collection for sessions
            auto_summarize_threshold: Auto-summarize after N messages
            inactivity_threshold_minutes: Consider session inactive after N minutes
        """
        self.qdrant = qdrant_store
        self.embedder = embedder
        self.llm = llm
        self.collection_name = collection_name
        self.auto_summarize_threshold = auto_summarize_threshold
        self.inactivity_threshold = timedelta(minutes=inactivity_threshold_minutes)

        # In-memory session cache
        self.active_sessions: Dict[str, Session] = {}

        # Initialize collection
        self._initialize_collection()

    def _initialize_collection(self):
        """Initialize Qdrant collection for sessions."""
        try:
            self.qdrant.client.get_collection(self.collection_name)
            logger.info(f"Session collection '{self.collection_name}' already exists")
        except Exception:
            # Collection doesn't exist, create it
            from qdrant_client.models import Distance, VectorParams

            self.qdrant.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedder.dimension, distance=Distance.COSINE
                ),
            )
            logger.info(f"Created session collection '{self.collection_name}'")

    def create_session(
        self,
        user_id: str,
        title: Optional[str] = None,
        parent_session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ) -> Session:
        """Create a new session.

        Args:
            user_id: User ID
            title: Optional session title
            parent_session_id: Optional parent session ID for hierarchical sessions
            metadata: Optional metadata
            tags: Optional tags

        Returns:
            Created Session object
        """
        session = Session(
            user_id=user_id,
            title=title or f"Session {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')}",
            parent_session_id=parent_session_id,
            metadata=metadata or {},
            tags=tags or [],
        )

        # Add to parent's children if specified
        if parent_session_id and parent_session_id in self.active_sessions:
            self.active_sessions[parent_session_id].add_child_session(session.id)

        # Cache in memory
        self.active_sessions[session.id] = session

        # Persist to Qdrant
        self._save_session(session)

        logger.info(f"Created session {session.id} for user {user_id}")
        return session

    def get_session(self, session_id: str) -> Optional[Session]:
        """Get session by ID.

        Args:
            session_id: Session ID

        Returns:
            Session object or None if not found
        """
        # Check cache first
        if session_id in self.active_sessions:
            return self.active_sessions[session_id]

        # Load from Qdrant
        try:
            result = self.qdrant.get(self.collection_name, session_id)
            if result:
                session = Session(**result["payload"])
                self.active_sessions[session_id] = session
                return session
        except Exception as e:
            logger.error(f"Failed to load session {session_id}: {e}")

        return None

    def update_session(
        self,
        session_id: str,
        title: Optional[str] = None,
        summary: Optional[str] = None,
        status: Optional[SessionStatus] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ) -> Optional[Session]:
        """Update session fields.

        Args:
            session_id: Session ID
            title: Optional new title
            summary: Optional summary
            status: Optional status
            metadata: Optional metadata to merge
            tags: Optional tags

        Returns:
            Updated Session or None if not found
        """
        session = self.get_session(session_id)
        if not session:
            return None

        if title:
            session.title = title
        if summary:
            session.summary = summary
        if status:
            session.status = status
        if metadata:
            session.metadata.update(metadata)
        if tags:
            session.tags = tags

        session.update_activity()
        self._save_session(session)

        logger.info(f"Updated session {session_id}")
        return session

    def track_message(
        self,
        session_id: str,
        memory: Memory,
        auto_extract: bool = True,
    ) -> Session:
        """Track a message/memory in the session.

        Args:
            session_id: Session ID
            memory: Memory object from the conversation
            auto_extract: Whether to auto-extract entities and facts

        Returns:
            Updated Session object
        """
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        # Update tracking
        session.update_activity()
        session.memory_count += 1
        session.total_tokens += memory.token_count
        session.total_characters += memory.text_length

        # Update average importance
        if session.memory_count > 0:
            session.avg_importance = (
                session.avg_importance * (session.memory_count - 1) + memory.importance
            ) / session.memory_count

        # Auto-extract entities and facts if LLM available
        if auto_extract and self.llm and session.message_count <= self.auto_summarize_threshold:
            self._extract_and_update(session, memory.text)

        # Auto-summarize if threshold reached
        if session.message_count >= self.auto_summarize_threshold and not session.summary:
            self.summarize_session(session_id)

        self._save_session(session)
        return session

    def summarize_session(self, session_id: str, force: bool = False) -> Optional[str]:
        """Generate summary for a session.

        Args:
            session_id: Session ID
            force: Force re-summarization even if summary exists

        Returns:
            Generated summary or None if LLM not available
        """
        session = self.get_session(session_id)
        if not session:
            return None

        if session.summary and not force:
            return session.summary

        if not self.llm:
            logger.warning("LLM not available for summarization")
            return None

        # Get session memories
        memories = self._get_session_memories(session_id)
        if not memories:
            return None

        # Build message context
        messages = "\n".join([f"- {m.text}" for m in memories[:50]])  # Limit to 50 messages
        prompt = self.SUMMARIZE_PROMPT.format(messages=messages)

        try:
            summary = self.llm.generate(prompt, max_tokens=200, temperature=0.3)
            if summary:
                session.summary = summary.strip()
                self._save_session(session)
                logger.info(f"Generated summary for session {session_id}")
                return session.summary
        except Exception as e:
            logger.error(f"Failed to generate summary for session {session_id}: {e}")

        return None

    def extract_session_facts(
        self, session_id: str, force: bool = False
    ) -> List[SessionFact]:
        """Extract key facts from session.

        Args:
            session_id: Session ID
            force: Force re-extraction even if facts exist

        Returns:
            List of extracted SessionFact objects
        """
        session = self.get_session(session_id)
        if not session:
            return []

        if session.facts and not force:
            return session.facts

        if not self.llm:
            logger.warning("LLM not available for fact extraction")
            return []

        # Get session memories
        memories = self._get_session_memories(session_id)
        if not memories:
            return []

        # Build message context
        messages = "\n".join([f"- {m.text}" for m in memories[:50]])
        prompt = self.EXTRACT_FACTS_PROMPT.format(messages=messages)

        try:
            response = self.llm.generate(prompt, max_tokens=500, temperature=0.2)
            if response:
                # Parse JSON response
                import json

                facts_data = json.loads(response.strip())
                session.facts = [
                    SessionFact(
                        fact=f["fact"],
                        confidence=f.get("confidence", 0.9),
                        sources=[m.id for m in memories],
                    )
                    for f in facts_data
                ]
                self._save_session(session)
                logger.info(f"Extracted {len(session.facts)} facts from session {session_id}")
                return session.facts
        except Exception as e:
            logger.error(f"Failed to extract facts from session {session_id}: {e}")

        return []

    def extract_session_entities(self, session_id: str, force: bool = False) -> Dict[str, Entity]:
        """Extract entities from session.

        Args:
            session_id: Session ID
            force: Force re-extraction

        Returns:
            Dictionary of entity_name -> Entity
        """
        session = self.get_session(session_id)
        if not session:
            return {}

        if session.entities and not force:
            return session.entities

        if not self.llm:
            # Fallback to simple regex-based extraction
            return self._extract_entities_simple(session_id)

        # Get session memories
        memories = self._get_session_memories(session_id)
        if not memories:
            return {}

        # Build message context
        messages = "\n".join([f"- {m.text}" for m in memories[:50]])
        prompt = self.EXTRACT_ENTITIES_PROMPT.format(messages=messages)

        try:
            response = self.llm.generate(prompt, max_tokens=500, temperature=0.2)
            if response:
                # Parse JSON response
                import json

                entities_data = json.loads(response.strip())
                for entity_data in entities_data:
                    session.add_entity(
                        name=entity_data["name"],
                        entity_type=entity_data.get("type", "unknown"),
                    )
                self._save_session(session)
                logger.info(
                    f"Extracted {len(session.entities)} entities from session {session_id}"
                )
                return session.entities
        except Exception as e:
            logger.error(f"Failed to extract entities from session {session_id}: {e}")

        return {}

    def _extract_entities_simple(self, session_id: str) -> Dict[str, Entity]:
        """Simple regex-based entity extraction fallback."""
        session = self.get_session(session_id)
        if not session:
            return {}

        memories = self._get_session_memories(session_id)
        if not memories:
            return {}

        # Simple patterns for common technologies and capitalized words
        tech_pattern = r'\b(Python|JavaScript|TypeScript|Java|Go|Rust|TensorFlow|PyTorch|React|Vue|Angular|Docker|Kubernetes|AWS|Azure|GCP)\b'

        for memory in memories:
            # Extract technology mentions
            for match in re.finditer(tech_pattern, memory.text):
                session.add_entity(match.group(0), "technology")

            # Extract capitalized words (potential names/products)
            for match in re.finditer(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', memory.text):
                if len(match.group(0).split()) <= 3:  # Limit to 3-word phrases
                    session.add_entity(match.group(0), "unknown")

        self._save_session(session)
        return session.entities

    def _extract_and_update(self, session: Session, text: str):
        """Quick extraction from single message (lighter than full session extraction)."""
        # Simple entity extraction
        tech_pattern = r'\b(Python|JavaScript|TypeScript|Java|Go|Rust|TensorFlow|PyTorch|React|Vue|Angular|Docker|Kubernetes|AWS|Azure|GCP)\b'
        for match in re.finditer(tech_pattern, text):
            session.add_entity(match.group(0), "technology")

    def detect_session_boundary(
        self,
        current_session_id: str,
        new_message: str,
        threshold: float = 0.7,
    ) -> Tuple[bool, str]:
        """Detect if new message should start a new session.

        Args:
            current_session_id: Current session ID
            new_message: New message text
            threshold: Confidence threshold for boundary detection

        Returns:
            Tuple of (boundary_detected: bool, reason: str)
        """
        session = self.get_session(current_session_id)
        if not session:
            return False, "Session not found"

        # Check inactivity
        time_since_last = datetime.now(timezone.utc) - session.last_activity_at
        if time_since_last > self.inactivity_threshold:
            return True, f"Inactivity for {time_since_last.total_seconds() / 60:.1f} minutes"

        # If LLM available, use it for semantic boundary detection
        if self.llm:
            memories = self._get_session_memories(current_session_id)
            if len(memories) >= 3:
                # Get last 3 messages
                recent_messages = "\n".join([f"- {m.text}" for m in memories[-3:]])
                prompt = self.BOUNDARY_DETECTION_PROMPT.format(
                    previous_messages=recent_messages, new_message=new_message
                )

                try:
                    response = self.llm.generate(prompt, max_tokens=200, temperature=0.1)
                    if response:
                        import json

                        result = json.loads(response.strip())
                        if result.get("boundary_detected", False):
                            return True, result.get("reason", "Topic change detected")
                except Exception as e:
                    logger.error(f"Boundary detection failed: {e}")

        # Fallback: Simple heuristic based on topic keywords
        if self._detect_topic_change_simple(session, new_message):
            return True, "Significant topic change detected"

        return False, "Continue current session"

    def _detect_topic_change_simple(self, session: Session, new_message: str) -> bool:
        """Simple topic change detection using entity overlap."""
        # Extract entities from new message
        new_entities = set()
        tech_pattern = r'\b(Python|JavaScript|TypeScript|Java|Go|Rust|TensorFlow|PyTorch|React|Vue|Angular|Docker|Kubernetes|AWS|Azure|GCP)\b'
        for match in re.finditer(tech_pattern, new_message):
            new_entities.add(match.group(0))

        if not new_entities or not session.entities:
            return False

        # Calculate overlap with existing entities
        existing_entities = set(session.entities.keys())
        overlap = len(new_entities.intersection(existing_entities)) / len(new_entities)

        # If less than 30% overlap, consider it a topic change
        return overlap < 0.3

    def search_sessions(
        self,
        query: str,
        user_id: Optional[str] = None,
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SessionSearchResult]:
        """Search sessions by semantic similarity.

        Args:
            query: Search query
            user_id: Optional user ID filter
            k: Number of results
            filters: Optional filters (status, tags, etc.)

        Returns:
            List of SessionSearchResult objects
        """
        # Embed query
        query_vector = self.embedder.encode_single(query)

        # Build filters
        qdrant_filters = {}
        if user_id:
            qdrant_filters["user_id"] = user_id
        if filters:
            if "status" in filters:
                qdrant_filters["status"] = filters["status"]
            if "tags" in filters:
                qdrant_filters["tags"] = filters["tags"]

        # Search
        try:
            results = self.qdrant.search(
                collection_name=self.collection_name,
                vector=query_vector,
                limit=k,
                filters=qdrant_filters if qdrant_filters else None,
            )

            search_results = []
            for r in results:
                session = Session(**r["payload"])
                search_results.append(
                    SessionSearchResult(
                        session=session,
                        score=r["score"],
                        breakdown={"similarity": r["score"]},
                    )
                )

            logger.info(f"Found {len(search_results)} sessions for query: {query}")
            return search_results

        except Exception as e:
            logger.error(f"Session search failed: {e}")
            return []

    def get_user_sessions(
        self,
        user_id: str,
        status: Optional[SessionStatus] = None,
        limit: int = 50,
    ) -> List[Session]:
        """Get sessions for a user.

        Args:
            user_id: User ID
            status: Optional status filter
            limit: Maximum results

        Returns:
            List of Session objects
        """
        filters = {"user_id": user_id}
        if status:
            filters["status"] = status.value

        try:
            results = self.qdrant.scroll(
                collection_name=self.collection_name, filters=filters, limit=limit
            )

            sessions = [Session(**r["payload"]) for r in results]
            sessions.sort(key=lambda s: s.started_at, reverse=True)

            logger.info(f"Found {len(sessions)} sessions for user {user_id}")
            return sessions

        except Exception as e:
            logger.error(f"Failed to get user sessions: {e}")
            return []

    def get_child_sessions(self, parent_session_id: str) -> List[Session]:
        """Get child sessions of a parent session.

        Args:
            parent_session_id: Parent session ID

        Returns:
            List of child Session objects
        """
        parent = self.get_session(parent_session_id)
        if not parent:
            return []

        child_sessions = []
        for child_id in parent.child_session_ids:
            child = self.get_session(child_id)
            if child:
                child_sessions.append(child)

        return child_sessions

    def complete_session(self, session_id: str, generate_summary: bool = True) -> Optional[Session]:
        """Complete a session and optionally generate final summary.

        Args:
            session_id: Session ID
            generate_summary: Whether to generate summary

        Returns:
            Completed Session or None
        """
        session = self.get_session(session_id)
        if not session:
            return None

        session.complete()

        # Generate summary if requested
        if generate_summary and not session.summary:
            self.summarize_session(session_id)

        # Extract final facts and entities
        if not session.facts:
            self.extract_session_facts(session_id)
        if not session.entities:
            self.extract_session_entities(session_id)

        self._save_session(session)

        # Remove from active cache
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]

        logger.info(f"Completed session {session_id}")
        return session

    def _save_session(self, session: Session):
        """Save session to Qdrant."""
        # Create embedding from session content
        text_for_embedding = f"{session.title} {session.summary or ''} {' '.join(session.tags)}"
        if session.facts:
            text_for_embedding += " " + " ".join([f.fact for f in session.facts[:10]])

        vector = self.embedder.encode_single(text_for_embedding)

        # Save to Qdrant
        self.qdrant.upsert(
            collection_name=self.collection_name,
            id=session.id,
            vector=vector,
            payload=session.model_dump(mode="json"),
        )

    def _get_session_memories(self, session_id: str) -> List[Memory]:
        """Get all memories for a session.

        Note: This method queries Qdrant directly. For full integration,
        use client.get_session_memories() which goes through the MemoryClient.
        """
        try:
            memories = []
            # Query both collections
            for collection in [self.qdrant.collection_facts, self.qdrant.collection_prefs]:
                results = self.qdrant.scroll(
                    collection_name=collection,
                    filters={"session_id": session_id},
                    limit=1000,
                )
                for data in results:
                    memories.append(Memory(**data["payload"]))

            # Sort by created_at
            memories.sort(key=lambda m: m.created_at)
            return memories
        except Exception as e:
            logger.error(f"Failed to get session memories: {e}")
            return []

    def delete_session(self, session_id: str) -> bool:
        """Delete a session.

        Args:
            session_id: Session ID

        Returns:
            True if deleted, False otherwise
        """
        try:
            self.qdrant.delete(collection_name=self.collection_name, ids=[session_id])

            # Remove from cache
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]

            logger.info(f"Deleted session {session_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete session {session_id}: {e}")
            return False

    def get_session_statistics(self, session_id: str) -> Dict[str, Any]:
        """Get statistics for a session.

        Args:
            session_id: Session ID

        Returns:
            Dictionary with session statistics
        """
        session = self.get_session(session_id)
        if not session:
            return {}

        return {
            "id": session.id,
            "user_id": session.user_id,
            "status": session.status.value,
            "duration_seconds": session.duration_seconds(),
            "message_count": session.message_count,
            "memory_count": session.memory_count,
            "entity_count": len(session.entities),
            "fact_count": len(session.facts),
            "avg_importance": session.avg_importance,
            "total_tokens": session.total_tokens,
            "total_characters": session.total_characters,
            "top_entities": [
                {"name": e.name, "type": e.type, "mentions": e.mentions}
                for e in session.get_top_entities(5)
            ],
            "child_sessions": len(session.child_session_ids),
            "has_parent": session.parent_session_id is not None,
        }
