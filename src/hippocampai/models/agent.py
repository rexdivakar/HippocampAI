"""Multi-agent models for agent-specific memory management."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4

from pydantic import BaseModel, Field


class MemoryVisibility(str, Enum):
    """Memory visibility levels for multi-agent systems."""

    PRIVATE = "private"  # Only accessible by owning agent
    SHARED = "shared"  # Accessible by agents with permission
    PUBLIC = "public"  # Accessible by all agents of the user


class AgentRole(str, Enum):
    """Agent role types."""

    ASSISTANT = "assistant"  # General assistant
    SPECIALIST = "specialist"  # Domain-specific agent
    COORDINATOR = "coordinator"  # Coordinates other agents
    OBSERVER = "observer"  # Read-only agent


class PermissionType(str, Enum):
    """Permission types for agent memory access."""

    READ = "read"
    WRITE = "write"
    SHARE = "share"
    DELETE = "delete"


class Agent(BaseModel):
    """Represents an AI agent with its own memory space."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    user_id: str
    role: AgentRole = AgentRole.ASSISTANT
    description: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    is_active: bool = True

    # Memory statistics
    total_memories: int = 0
    private_memories: int = 0
    shared_memories: int = 0


class Run(BaseModel):
    """Represents a single execution run of an agent.

    Organizes memories in a User → Agent → Run hierarchy.
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    agent_id: str
    user_id: str
    name: Optional[str] = None
    status: str = "active"  # active, completed, failed
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Run statistics
    memories_created: int = 0
    memories_accessed: int = 0


class AgentPermission(BaseModel):
    """Permission for an agent to access another agent's memories."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    granter_agent_id: str  # Agent granting permission
    grantee_agent_id: str  # Agent receiving permission
    permissions: set[PermissionType] = Field(default_factory=lambda: {PermissionType.READ})
    memory_filters: Optional[dict[str, Any]] = None  # Optional filters (type, tags, etc.)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None
    is_active: bool = True

    def has_permission(self, permission: PermissionType) -> bool:
        """Check if permission is granted."""
        return self.is_active and permission in self.permissions

    def is_expired(self) -> bool:
        """Check if permission has expired."""
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) > self.expires_at


class MemoryTransfer(BaseModel):
    """Record of memory transfer between agents."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    memory_id: str
    source_agent_id: str
    target_agent_id: str
    transfer_type: str  # copy, move, share
    transferred_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentMemoryStats(BaseModel):
    """Statistics for agent memory usage."""

    agent_id: str
    total_memories: int
    memories_by_visibility: dict[str, int]
    memories_by_type: dict[str, int]
    memories_by_run: dict[str, int]
    shared_with_agents: list[str]
    can_access_from_agents: list[str]
