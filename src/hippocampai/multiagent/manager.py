"""Multi-agent manager for agent-specific memory spaces and permissions."""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set
from collections import defaultdict

from hippocampai.models.agent import (
    Agent,
    AgentPermission,
    AgentRole,
    MemoryTransfer,
    MemoryVisibility,
    PermissionType,
    Run,
    AgentMemoryStats,
)
from hippocampai.models.memory import Memory

logger = logging.getLogger(__name__)


class MultiAgentManager:
    """Manages multi-agent memory spaces, permissions, and transfers."""

    def __init__(self):
        """Initialize multi-agent manager."""
        self.agents: Dict[str, Agent] = {}
        self.runs: Dict[str, Run] = {}
        self.permissions: Dict[str, AgentPermission] = {}
        self.transfers: List[MemoryTransfer] = []

    # === AGENT MANAGEMENT ===

    def create_agent(
        self,
        name: str,
        user_id: str,
        role: AgentRole = AgentRole.ASSISTANT,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Agent:
        """Create a new agent.

        Args:
            name: Agent name
            user_id: User ID owning the agent
            role: Agent role
            description: Optional description
            metadata: Optional metadata

        Returns:
            Created Agent object
        """
        agent = Agent(
            name=name,
            user_id=user_id,
            role=role,
            description=description,
            metadata=metadata or {},
        )

        self.agents[agent.id] = agent
        logger.info(f"Created agent: {agent.name} ({agent.id}) for user {user_id}")
        return agent

    def get_agent(self, agent_id: str) -> Optional[Agent]:
        """Get agent by ID."""
        return self.agents.get(agent_id)

    def list_agents(self, user_id: Optional[str] = None) -> List[Agent]:
        """List all agents, optionally filtered by user."""
        agents = list(self.agents.values())
        if user_id:
            agents = [a for a in agents if a.user_id == user_id]
        return agents

    def update_agent(
        self,
        agent_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        role: Optional[AgentRole] = None,
        metadata: Optional[Dict[str, Any]] = None,
        is_active: Optional[bool] = None,
    ) -> Optional[Agent]:
        """Update agent properties."""
        agent = self.agents.get(agent_id)
        if not agent:
            return None

        if name is not None:
            agent.name = name
        if description is not None:
            agent.description = description
        if role is not None:
            agent.role = role
        if metadata is not None:
            agent.metadata.update(metadata)
        if is_active is not None:
            agent.is_active = is_active

        agent.updated_at = datetime.now(timezone.utc)
        return agent

    def delete_agent(self, agent_id: str) -> bool:
        """Delete an agent."""
        if agent_id in self.agents:
            del self.agents[agent_id]
            # Clean up associated runs
            self.runs = {k: v for k, v in self.runs.items() if v.agent_id != agent_id}
            # Clean up permissions
            self.permissions = {
                k: v
                for k, v in self.permissions.items()
                if v.granter_agent_id != agent_id and v.grantee_agent_id != agent_id
            }
            logger.info(f"Deleted agent: {agent_id}")
            return True
        return False

    # === RUN MANAGEMENT ===

    def create_run(
        self,
        agent_id: str,
        user_id: str,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Run:
        """Create a new run for an agent.

        Args:
            agent_id: Agent ID
            user_id: User ID
            name: Optional run name
            metadata: Optional metadata

        Returns:
            Created Run object
        """
        run = Run(
            agent_id=agent_id,
            user_id=user_id,
            name=name,
            metadata=metadata or {},
        )

        self.runs[run.id] = run
        logger.info(f"Created run: {run.id} for agent {agent_id}")
        return run

    def get_run(self, run_id: str) -> Optional[Run]:
        """Get run by ID."""
        return self.runs.get(run_id)

    def list_runs(
        self, agent_id: Optional[str] = None, user_id: Optional[str] = None
    ) -> List[Run]:
        """List runs, optionally filtered by agent or user."""
        runs = list(self.runs.values())
        if agent_id:
            runs = [r for r in runs if r.agent_id == agent_id]
        if user_id:
            runs = [r for r in runs if r.user_id == user_id]
        return runs

    def complete_run(self, run_id: str, status: str = "completed") -> Optional[Run]:
        """Mark a run as completed."""
        run = self.runs.get(run_id)
        if not run:
            return None

        run.status = status
        run.completed_at = datetime.now(timezone.utc)
        return run

    # === PERMISSION MANAGEMENT ===

    def grant_permission(
        self,
        granter_agent_id: str,
        grantee_agent_id: str,
        permissions: Set[PermissionType],
        memory_filters: Optional[Dict[str, Any]] = None,
        expires_at: Optional[datetime] = None,
    ) -> AgentPermission:
        """Grant permission for one agent to access another's memories.

        Args:
            granter_agent_id: Agent granting permission
            grantee_agent_id: Agent receiving permission
            permissions: Set of permissions to grant
            memory_filters: Optional filters (type, tags, etc.)
            expires_at: Optional expiration datetime

        Returns:
            Created AgentPermission object
        """
        permission = AgentPermission(
            granter_agent_id=granter_agent_id,
            grantee_agent_id=grantee_agent_id,
            permissions=permissions,
            memory_filters=memory_filters,
            expires_at=expires_at,
        )

        self.permissions[permission.id] = permission
        logger.info(
            f"Granted permissions {permissions} from {granter_agent_id} to {grantee_agent_id}"
        )
        return permission

    def revoke_permission(self, permission_id: str) -> bool:
        """Revoke a permission."""
        if permission_id in self.permissions:
            self.permissions[permission_id].is_active = False
            logger.info(f"Revoked permission: {permission_id}")
            return True
        return False

    def check_permission(
        self, agent_id: str, target_agent_id: str, permission: PermissionType
    ) -> bool:
        """Check if an agent has permission to access another agent's memories.

        Args:
            agent_id: Agent requesting access
            target_agent_id: Agent whose memories are being accessed
            permission: Permission type to check

        Returns:
            True if permission is granted
        """
        # Agent always has full access to its own memories
        if agent_id == target_agent_id:
            return True

        # Check for explicit permissions
        for perm in self.permissions.values():
            if (
                perm.granter_agent_id == target_agent_id
                and perm.grantee_agent_id == agent_id
                and perm.is_active
                and not perm.is_expired()
                and perm.has_permission(permission)
            ):
                return True

        return False

    def list_permissions(
        self,
        granter_agent_id: Optional[str] = None,
        grantee_agent_id: Optional[str] = None,
    ) -> List[AgentPermission]:
        """List permissions, optionally filtered."""
        perms = list(self.permissions.values())

        if granter_agent_id:
            perms = [p for p in perms if p.granter_agent_id == granter_agent_id]
        if grantee_agent_id:
            perms = [p for p in perms if p.grantee_agent_id == grantee_agent_id]

        return perms

    # === MEMORY VISIBILITY & FILTERING ===

    def can_access_memory(
        self, agent_id: str, memory: Memory, permission: PermissionType = PermissionType.READ
    ) -> bool:
        """Check if an agent can access a specific memory.

        Args:
            agent_id: Agent requesting access
            memory: Memory to access
            permission: Permission type needed

        Returns:
            True if agent can access the memory
        """
        # Check if memory has agent_id field
        if not hasattr(memory, "agent_id") or not memory.agent_id:
            # Legacy memory without agent_id - accessible by default
            return True

        # Agent owns the memory
        if memory.agent_id == agent_id:
            return True

        # Check visibility
        visibility = getattr(memory, "visibility", MemoryVisibility.PRIVATE)

        if visibility == MemoryVisibility.PUBLIC:
            # Public memories accessible by all agents of same user
            agent = self.agents.get(agent_id)
            if agent and agent.user_id == memory.user_id:
                return True

        elif visibility == MemoryVisibility.SHARED:
            # Check explicit permissions
            return self.check_permission(agent_id, memory.agent_id, permission)

        # Private - only owner can access
        return False

    def filter_accessible_memories(
        self, agent_id: str, memories: List[Memory], permission: PermissionType = PermissionType.READ
    ) -> List[Memory]:
        """Filter memories to only those accessible by agent."""
        return [m for m in memories if self.can_access_memory(agent_id, m, permission)]

    # === MEMORY TRANSFER ===

    def transfer_memory(
        self,
        memory: Memory,
        source_agent_id: str,
        target_agent_id: str,
        transfer_type: str = "copy",
    ) -> Optional[MemoryTransfer]:
        """Transfer a memory from one agent to another.

        Args:
            memory: Memory to transfer
            source_agent_id: Source agent ID
            target_agent_id: Target agent ID
            transfer_type: "copy", "move", or "share"

        Returns:
            MemoryTransfer record or None if not allowed
        """
        # Verify source agent owns the memory
        if hasattr(memory, "agent_id") and memory.agent_id != source_agent_id:
            logger.warning(f"Agent {source_agent_id} doesn't own memory {memory.id}")
            return None

        # Check permission for share
        if not self.check_permission(source_agent_id, target_agent_id, PermissionType.SHARE):
            logger.warning(
                f"Agent {source_agent_id} doesn't have SHARE permission for {target_agent_id}"
            )
            return None

        # Create transfer record
        transfer = MemoryTransfer(
            memory_id=memory.id,
            source_agent_id=source_agent_id,
            target_agent_id=target_agent_id,
            transfer_type=transfer_type,
        )

        self.transfers.append(transfer)
        logger.info(
            f"Transferred memory {memory.id} from {source_agent_id} to {target_agent_id} ({transfer_type})"
        )
        return transfer

    def get_transfer_history(
        self, agent_id: Optional[str] = None, memory_id: Optional[str] = None
    ) -> List[MemoryTransfer]:
        """Get memory transfer history."""
        transfers = self.transfers

        if agent_id:
            transfers = [
                t
                for t in transfers
                if t.source_agent_id == agent_id or t.target_agent_id == agent_id
            ]
        if memory_id:
            transfers = [t for t in transfers if t.memory_id == memory_id]

        return transfers

    # === STATISTICS ===

    def get_agent_stats(self, agent_id: str, memories: List[Memory]) -> AgentMemoryStats:
        """Get memory statistics for an agent.

        Args:
            agent_id: Agent ID
            memories: All memories to analyze

        Returns:
            AgentMemoryStats object
        """
        # Filter to agent's memories
        agent_memories = [
            m for m in memories if hasattr(m, "agent_id") and m.agent_id == agent_id
        ]

        # Count by visibility
        by_visibility = defaultdict(int)
        for mem in agent_memories:
            visibility = getattr(mem, "visibility", MemoryVisibility.PRIVATE)
            by_visibility[visibility.value] += 1

        # Count by type
        by_type = defaultdict(int)
        for mem in agent_memories:
            by_type[mem.type.value] += 1

        # Count by run
        by_run = defaultdict(int)
        for mem in agent_memories:
            run_id = getattr(mem, "run_id", None)
            if run_id:
                by_run[run_id] += 1

        # Find shared relationships
        shared_with = set()
        can_access_from = set()

        for perm in self.permissions.values():
            if perm.is_active and not perm.is_expired():
                if perm.granter_agent_id == agent_id:
                    shared_with.add(perm.grantee_agent_id)
                if perm.grantee_agent_id == agent_id:
                    can_access_from.add(perm.granter_agent_id)

        return AgentMemoryStats(
            agent_id=agent_id,
            total_memories=len(agent_memories),
            memories_by_visibility=dict(by_visibility),
            memories_by_type=dict(by_type),
            memories_by_run=dict(by_run),
            shared_with_agents=list(shared_with),
            can_access_from_agents=list(can_access_from),
        )

    def update_agent_memory_counts(self, agent_id: str, memories: List[Memory]):
        """Update agent's memory counts."""
        agent = self.agents.get(agent_id)
        if not agent:
            return

        agent_memories = [
            m for m in memories if hasattr(m, "agent_id") and m.agent_id == agent_id
        ]

        agent.total_memories = len(agent_memories)
        agent.private_memories = sum(
            1
            for m in agent_memories
            if getattr(m, "visibility", MemoryVisibility.PRIVATE) == MemoryVisibility.PRIVATE
        )
        agent.shared_memories = sum(
            1
            for m in agent_memories
            if getattr(m, "visibility", MemoryVisibility.PRIVATE) != MemoryVisibility.PRIVATE
        )
