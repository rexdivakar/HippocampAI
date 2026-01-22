"""Authentication and session management routes."""

import logging
import os
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from hippocampai.vector.qdrant_store import QdrantStore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])


# ============================================
# REQUEST/RESPONSE MODELS
# ============================================


class SignupRequest(BaseModel):
    """Request to create a new session."""

    username: Optional[str] = None  # Optional display name
    metadata: Optional[dict] = None  # Additional user metadata


class SignupResponse(BaseModel):
    """Response with new session credentials."""

    user_id: str
    session_id: str
    created_at: str
    message: str


class ValidateSessionRequest(BaseModel):
    """Request to validate a session."""

    unique_id: str  # Can be user_id, session_id, or any unique identifier
    api_key: Optional[str] = None


class ValidateSessionResponse(BaseModel):
    """Response from session validation."""

    valid: bool
    user_id: str
    message: str


# ============================================
# SESSION VALIDATION
# ============================================


def _get_user_id_from_session(unique_id: str) -> Optional[str]:
    """
    Find the actual user_id from a session ID or user ID.

    Priority order:
    1. If unique_id starts with "user_", check if it exists as a user_id directly
    2. Look for the signup record (has metadata.session_id matching the unique_id)
    3. Fall back to any record with matching session_id

    Returns the user_id from the found record, or None if not found.
    """
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        client = QdrantClient(url=qdrant_url)

        collections_to_check = [
            "hippocampai_facts",
            "hippocampai_prefs",
            "personal_facts",
            "hippocampai_sessions",
        ]

        # Strategy 1: If input looks like a user_id (starts with "user_"), check directly
        if unique_id.startswith("user_"):
            user_id_filter = Filter(
                must=[FieldCondition(key="user_id", match=MatchValue(value=unique_id))]
            )
            for collection_name in collections_to_check:
                try:
                    if not client.collection_exists(collection_name):
                        continue
                    results, _ = client.scroll(
                        collection_name=collection_name,
                        scroll_filter=user_id_filter,
                        limit=1,
                        with_payload=True,
                    )
                    if results and len(results) > 0:
                        logger.info(f"Found user_id '{unique_id}' directly in {collection_name}")
                        return unique_id
                except Exception as e:
                    logger.debug(f"Could not check {collection_name}: {e}")
                    continue

        # Strategy 2: Look for the signup record (metadata.session_id matches)
        # This is the authoritative record that links session_id to user_id
        signup_filter = Filter(
            must=[FieldCondition(key="metadata.session_id", match=MatchValue(value=unique_id))]
        )
        for collection_name in collections_to_check:
            try:
                if not client.collection_exists(collection_name):
                    continue
                results, _ = client.scroll(
                    collection_name=collection_name,
                    scroll_filter=signup_filter,
                    limit=1,
                    with_payload=True,
                )
                if results and len(results) > 0:
                    payload = results[0].payload
                    user_id = payload.get("user_id")
                    if user_id:
                        logger.info(
                            f"Found user_id '{user_id}' via signup record for session '{unique_id}' in {collection_name}"
                        )
                        return user_id
            except Exception as e:
                logger.debug(f"Could not check {collection_name} for signup: {e}")
                continue

        # Strategy 3: Check if unique_id is used as user_id anywhere
        user_id_filter = Filter(
            must=[FieldCondition(key="user_id", match=MatchValue(value=unique_id))]
        )
        for collection_name in collections_to_check:
            try:
                if not client.collection_exists(collection_name):
                    continue
                results, _ = client.scroll(
                    collection_name=collection_name,
                    scroll_filter=user_id_filter,
                    limit=1,
                    with_payload=True,
                )
                if results and len(results) > 0:
                    logger.info(
                        f"Found user_id '{unique_id}' as direct user_id in {collection_name}"
                    )
                    return unique_id
            except Exception as e:
                logger.debug(f"Could not check {collection_name}: {e}")
                continue

        logger.warning(f"No user_id found for ID: {unique_id}")
        return None

    except Exception as e:
        logger.exception(f"Failed to get user_id for {unique_id}: {e}")
        return None


def _check_session_exists(unique_id: str) -> bool:
    """
    Check if a unique ID exists in Qdrant.

    Searches for the ID in multiple fields across all collections.
    Returns True if the ID is found anywhere in the database.
    """
    return _get_user_id_from_session(unique_id) is not None


# ============================================
# ENDPOINTS
# ============================================


@router.post("/signup", response_model=SignupResponse)
async def signup(request: SignupRequest) -> SignupResponse:
    """
    Create a new user session.

    Generates a unique user_id that can be used for authentication.
    No password required - session ID is the credential.
    """
    # Generate unique session ID
    session_id = str(uuid4())
    user_id = f"user_{session_id[:8]}"

    # Create a welcome memory to initialize the session
    try:
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        store = QdrantStore(url=qdrant_url)

        # Get embedder to generate vector for the welcome memory
        from hippocampai.embed.embedder import get_embedder

        embedder = get_embedder()

        # Create welcome message
        welcome_text = f"Welcome to HippocampAI! Your session was created on {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}."

        # Generate embedding vector
        vector = embedder.encode_single(welcome_text)

        # Create memory ID
        memory_id = str(uuid4())

        # Build payload
        payload = {
            "id": memory_id,
            "user_id": user_id,
            "session_id": session_id,
            "text": welcome_text,
            "type": "fact",
            "importance": 5.0,
            "tags": ["welcome", "session"],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "access_count": 0,
            "metadata": {
                "session_id": session_id,
                "username": request.username or "Anonymous",
                "signup_date": datetime.now(timezone.utc).isoformat(),
                **(request.metadata or {}),
            },
        }

        # Store in Qdrant
        store.upsert(
            collection_name=store.collection_facts,
            id=memory_id,
            vector=vector,
            payload=payload,
        )

        logger.info(f"Created new session: user_id={user_id}, session_id={session_id}")

        return SignupResponse(
            user_id=user_id,
            session_id=session_id,
            created_at=datetime.now(timezone.utc).isoformat(),
            message="Session created successfully! Save your user_id and session_id to access your account.",
        )

    except Exception as e:
        logger.exception(f"Failed to create session: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create session: {str(e)}",
        )


@router.post("/validate", response_model=ValidateSessionResponse)
async def validate_session(request: ValidateSessionRequest) -> ValidateSessionResponse:
    """
    Validate a session using unique ID.

    Checks if the unique_id exists anywhere in Qdrant (user_id, session_id, or metadata).
    Returns the actual user_id from Qdrant so the UI can use it for subsequent API calls.
    """
    unique_id = request.unique_id.strip()

    if not unique_id:
        raise HTTPException(status_code=400, detail="unique_id is required")

    # Get the actual user_id from Qdrant
    actual_user_id = _get_user_id_from_session(unique_id)

    if not actual_user_id:
        logger.warning(f"Invalid session attempt: {unique_id}")
        return ValidateSessionResponse(
            valid=False,
            user_id=unique_id,
            message="Invalid session. ID not found in system.",
        )

    # Optional API key validation
    api_key_required = os.getenv("API_KEY_REQUIRED", "false").lower() == "true"
    if api_key_required and request.api_key:
        valid_keys = os.getenv("VALID_API_KEYS", "").split(",")
        if request.api_key not in valid_keys:
            return ValidateSessionResponse(
                valid=False,
                user_id=actual_user_id,
                message="Invalid API key.",
            )

    logger.info(f"Session validated: {unique_id} -> user_id: {actual_user_id}")
    return ValidateSessionResponse(
        valid=True,
        user_id=actual_user_id,  # Return the actual user_id from Qdrant
        message="Session valid.",
    )


@router.get("/session/{user_id}/exists")
async def check_session_exists(user_id: str) -> dict:
    """
    Quick check if a session exists.

    Used for login form validation.
    """
    exists = _check_session_exists(user_id)
    return {
        "exists": exists,
        "user_id": user_id,
    }
