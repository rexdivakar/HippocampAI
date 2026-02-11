"""Tests for multi-agent memory management."""

import pytest

from hippocampai.models.agent import (
    Agent,
    AgentPermission,
    AgentRole,
    MemoryVisibility,
    PermissionType,
    Run,
)
from hippocampai.models.memory import Memory, MemoryType
from hippocampai.multiagent import MultiAgentManager


class TestAgent:
    """Test Agent model."""

    def test_create_agent(self):
        """Test creating an agent."""
        agent = Agent(
            name="Test Agent",
            user_id="user1",
            role=AgentRole.ASSISTANT,
            description="Test description",
        )

        assert agent.id is not None
        assert agent.name == "Test Agent"
        assert agent.user_id == "user1"
        assert agent.role == AgentRole.ASSISTANT
        assert agent.is_active is True
        assert agent.total_memories == 0

    def test_agent_roles(self):
        """Test different agent roles."""
        roles = [
            AgentRole.ASSISTANT,
            AgentRole.SPECIALIST,
            AgentRole.COORDINATOR,
            AgentRole.OBSERVER,
        ]

        for role in roles:
            agent = Agent(name=f"Agent {role.value}", user_id="user1", role=role)
            assert agent.role == role


class TestRun:
    """Test Run model."""

    def test_create_run(self):
        """Test creating a run."""
        run = Run(
            agent_id="agent1",
            user_id="user1",
            name="Test Run",
            metadata={"purpose": "testing"},
        )

        assert run.id is not None
        assert run.agent_id == "agent1"
        assert run.user_id == "user1"
        assert run.name == "Test Run"
        assert run.status == "active"
        assert run.memories_created == 0


class TestAgentPermission:
    """Test AgentPermission model."""

    def test_create_permission(self):
        """Test creating a permission."""
        perm = AgentPermission(
            granter_agent_id="agent1",
            grantee_agent_id="agent2",
            permissions={PermissionType.READ},
        )

        assert perm.id is not None
        assert perm.granter_agent_id == "agent1"
        assert perm.grantee_agent_id == "agent2"
        assert perm.is_active is True
        assert perm.has_permission(PermissionType.READ)
        assert not perm.has_permission(PermissionType.WRITE)

    def test_multiple_permissions(self):
        """Test permission with multiple types."""
        perm = AgentPermission(
            granter_agent_id="agent1",
            grantee_agent_id="agent2",
            permissions={PermissionType.READ, PermissionType.WRITE},
        )

        assert perm.has_permission(PermissionType.READ)
        assert perm.has_permission(PermissionType.WRITE)
        assert not perm.has_permission(PermissionType.DELETE)

    def test_permission_expiration(self):
        """Test permission expiration."""
        from datetime import datetime, timedelta, timezone

        # Not expired
        future = datetime.now(timezone.utc) + timedelta(days=1)
        perm = AgentPermission(
            granter_agent_id="agent1",
            grantee_agent_id="agent2",
            permissions={PermissionType.READ},
            expires_at=future,
        )
        assert not perm.is_expired()

        # Expired
        past = datetime.now(timezone.utc) - timedelta(days=1)
        perm.expires_at = past
        assert perm.is_expired()


class TestMultiAgentManager:
    """Test MultiAgentManager."""

    def test_create_agent(self):
        """Test creating an agent via manager."""
        manager = MultiAgentManager()

        agent = manager.create_agent(name="Test Agent", user_id="user1", role=AgentRole.ASSISTANT)

        assert agent.id in manager.agents
        assert manager.get_agent(agent.id) == agent

    def test_list_agents(self):
        """Test listing agents."""
        manager = MultiAgentManager()

        manager.create_agent("Agent 1", "user1", AgentRole.ASSISTANT)
        manager.create_agent("Agent 2", "user1", AgentRole.SPECIALIST)
        manager.create_agent("Agent 3", "user2", AgentRole.COORDINATOR)

        # List all
        all_agents = manager.list_agents()
        assert len(all_agents) == 3

        # List for user1
        user1_agents = manager.list_agents(user_id="user1")
        assert len(user1_agents) == 2
        assert all(a.user_id == "user1" for a in user1_agents)

    def test_update_agent(self):
        """Test updating agent properties."""
        manager = MultiAgentManager()

        agent = manager.create_agent("Original Name", "user1", AgentRole.ASSISTANT)

        # Update
        updated = manager.update_agent(
            agent.id,
            name="New Name",
            description="Updated description",
            is_active=False,
        )

        assert updated is not None
        assert updated.name == "New Name"
        assert updated.description == "Updated description"
        assert updated.is_active is False

    def test_delete_agent(self):
        """Test deleting an agent."""
        manager = MultiAgentManager()

        agent = manager.create_agent("Test Agent", "user1", AgentRole.ASSISTANT)
        assert manager.get_agent(agent.id) is not None

        # Delete
        success = manager.delete_agent(agent.id)
        assert success
        assert manager.get_agent(agent.id) is None

    def test_create_run(self):
        """Test creating a run."""
        manager = MultiAgentManager()

        agent = manager.create_agent("Test Agent", "user1", AgentRole.ASSISTANT)
        run = manager.create_run(agent.id, "user1", name="Test Run")

        assert run.id in manager.runs
        assert run.agent_id == agent.id
        assert manager.get_run(run.id) == run

    def test_list_runs(self):
        """Test listing runs."""
        manager = MultiAgentManager()

        agent1 = manager.create_agent("Agent 1", "user1", AgentRole.ASSISTANT)
        agent2 = manager.create_agent("Agent 2", "user1", AgentRole.SPECIALIST)

        manager.create_run(agent1.id, "user1", name="Run 1")
        manager.create_run(agent1.id, "user1", name="Run 2")
        manager.create_run(agent2.id, "user1", name="Run 3")

        # List all
        all_runs = manager.list_runs()
        assert len(all_runs) == 3

        # List for agent1
        agent1_runs = manager.list_runs(agent_id=agent1.id)
        assert len(agent1_runs) == 2

    def test_complete_run(self):
        """Test completing a run."""
        manager = MultiAgentManager()

        agent = manager.create_agent("Test Agent", "user1", AgentRole.ASSISTANT)
        run = manager.create_run(agent.id, "user1", name="Test Run")

        assert run.status == "active"

        # Complete
        completed = manager.complete_run(run.id, status="completed")
        assert completed is not None
        assert completed.status == "completed"
        assert completed.completed_at is not None

    def test_grant_permission(self):
        """Test granting permissions."""
        manager = MultiAgentManager()

        agent1 = manager.create_agent("Agent 1", "user1", AgentRole.ASSISTANT)
        agent2 = manager.create_agent("Agent 2", "user1", AgentRole.SPECIALIST)

        perm = manager.grant_permission(
            agent1.id, agent2.id, {PermissionType.READ, PermissionType.WRITE}
        )

        assert perm.id in manager.permissions
        assert perm.granter_agent_id == agent1.id
        assert perm.grantee_agent_id == agent2.id

    def test_check_permission(self):
        """Test checking permissions."""
        manager = MultiAgentManager()

        agent1 = manager.create_agent("Agent 1", "user1", AgentRole.ASSISTANT)
        agent2 = manager.create_agent("Agent 2", "user1", AgentRole.SPECIALIST)

        # No permission initially
        assert not manager.check_permission(agent2.id, agent1.id, PermissionType.READ)

        # Grant permission
        manager.grant_permission(agent1.id, agent2.id, {PermissionType.READ})

        # Now has permission
        assert manager.check_permission(agent2.id, agent1.id, PermissionType.READ)
        assert not manager.check_permission(agent2.id, agent1.id, PermissionType.WRITE)

        # Agent always has permission to own memories
        assert manager.check_permission(agent1.id, agent1.id, PermissionType.READ)
        assert manager.check_permission(agent1.id, agent1.id, PermissionType.DELETE)

    def test_revoke_permission(self):
        """Test revoking permissions."""
        manager = MultiAgentManager()

        agent1 = manager.create_agent("Agent 1", "user1", AgentRole.ASSISTANT)
        agent2 = manager.create_agent("Agent 2", "user1", AgentRole.SPECIALIST)

        perm = manager.grant_permission(agent1.id, agent2.id, {PermissionType.READ})

        assert manager.check_permission(agent2.id, agent1.id, PermissionType.READ)

        # Revoke
        success = manager.revoke_permission(perm.id)
        assert success
        assert not perm.is_active

    def test_can_access_memory(self):
        """Test memory access control."""
        manager = MultiAgentManager()

        agent1 = manager.create_agent("Agent 1", "user1", AgentRole.ASSISTANT)
        agent2 = manager.create_agent("Agent 2", "user1", AgentRole.SPECIALIST)

        # Private memory
        private_mem = Memory(
            text="Private data",
            user_id="user1",
            type=MemoryType.FACT,
            agent_id=agent1.id,
            visibility=MemoryVisibility.PRIVATE.value,
        )

        # Owner can access
        assert manager.can_access_memory(agent1.id, private_mem, PermissionType.READ)

        # Other agent cannot access private memory
        assert not manager.can_access_memory(agent2.id, private_mem, PermissionType.READ)

        # Public memory
        public_mem = Memory(
            text="Public data",
            user_id="user1",
            type=MemoryType.FACT,
            agent_id=agent1.id,
            visibility=MemoryVisibility.PUBLIC.value,
        )

        # Any agent of same user can access public memory
        assert manager.can_access_memory(agent2.id, public_mem, PermissionType.READ)

        # Shared memory with permission
        shared_mem = Memory(
            text="Shared data",
            user_id="user1",
            type=MemoryType.FACT,
            agent_id=agent1.id,
            visibility=MemoryVisibility.SHARED.value,
        )

        # Without permission, cannot access
        assert not manager.can_access_memory(agent2.id, shared_mem, PermissionType.READ)

        # Grant permission
        manager.grant_permission(agent1.id, agent2.id, {PermissionType.READ})

        # Now can access
        assert manager.can_access_memory(agent2.id, shared_mem, PermissionType.READ)

    def test_filter_accessible_memories(self):
        """Test filtering memories by accessibility."""
        manager = MultiAgentManager()

        agent1 = manager.create_agent("Agent 1", "user1", AgentRole.ASSISTANT)
        agent2 = manager.create_agent("Agent 2", "user1", AgentRole.SPECIALIST)

        memories = [
            Memory(
                text="Private",
                user_id="user1",
                type=MemoryType.FACT,
                agent_id=agent1.id,
                visibility=MemoryVisibility.PRIVATE.value,
            ),
            Memory(
                text="Public",
                user_id="user1",
                type=MemoryType.FACT,
                agent_id=agent1.id,
                visibility=MemoryVisibility.PUBLIC.value,
            ),
            Memory(
                text="Shared",
                user_id="user1",
                type=MemoryType.FACT,
                agent_id=agent1.id,
                visibility=MemoryVisibility.SHARED.value,
            ),
        ]

        # Agent2 should only see public memory (no permissions granted)
        accessible = manager.filter_accessible_memories(agent2.id, memories, PermissionType.READ)
        assert len(accessible) == 1
        assert accessible[0].text == "Public"

        # Grant permission to shared memories
        manager.grant_permission(agent1.id, agent2.id, {PermissionType.READ})

        # Now should see public + shared
        accessible = manager.filter_accessible_memories(agent2.id, memories, PermissionType.READ)
        assert len(accessible) == 2

    def test_memory_transfer(self):
        """Test transferring memory between agents."""
        manager = MultiAgentManager()

        agent1 = manager.create_agent("Agent 1", "user1", AgentRole.ASSISTANT)
        agent2 = manager.create_agent("Agent 2", "user1", AgentRole.SPECIALIST)

        memory = Memory(
            text="Test data",
            user_id="user1",
            type=MemoryType.FACT,
            agent_id=agent1.id,
        )

        # Grant permission
        manager.grant_permission(agent1.id, agent2.id, {PermissionType.SHARE})

        # Transfer
        transfer = manager.transfer_memory(memory, agent1.id, agent2.id, "copy")

        assert transfer is not None
        assert transfer.memory_id == memory.id
        assert transfer.source_agent_id == agent1.id
        assert transfer.target_agent_id == agent2.id
        assert transfer.transfer_type == "copy"

    def test_get_agent_stats(self):
        """Test getting agent statistics."""
        manager = MultiAgentManager()

        agent1 = manager.create_agent("Agent 1", "user1", AgentRole.ASSISTANT)
        agent2 = manager.create_agent("Agent 2", "user1", AgentRole.SPECIALIST)

        memories = [
            Memory(
                text=f"Memory {i}",
                user_id="user1",
                type=MemoryType.FACT,
                agent_id=agent1.id,
                visibility=MemoryVisibility.PRIVATE.value
                if i % 2 == 0
                else MemoryVisibility.PUBLIC.value,
            )
            for i in range(5)
        ]

        # Grant some permissions
        manager.grant_permission(agent1.id, agent2.id, {PermissionType.READ})

        stats = manager.get_agent_stats(agent1.id, memories)

        assert stats.agent_id == agent1.id
        assert stats.total_memories == 5
        assert "private" in stats.memories_by_visibility
        assert "public" in stats.memories_by_visibility
        assert agent2.id in stats.shared_with_agents


@pytest.mark.integration
class TestMultiAgentIntegration:
    """Integration tests with MemoryClient."""

    def test_create_agent_with_client(self, memory_client, user_id):
        """Test creating agents via MemoryClient."""
        agent = memory_client.create_agent(
            name="Test Agent", user_id=user_id, role=AgentRole.ASSISTANT
        )

        assert agent.id is not None
        assert agent.name == "Test Agent"
        assert agent.user_id == user_id

    def test_agent_specific_memories(self, memory_client, user_id):
        """Test storing and retrieving agent-specific memories."""
        agent = memory_client.create_agent("Test Agent", user_id, AgentRole.ASSISTANT)

        # Store memories for agent
        mem = memory_client.remember(
            text="Agent-specific memory",
            user_id=user_id,
            agent_id=agent.id,
            visibility=MemoryVisibility.PRIVATE.value,
        )

        assert mem.agent_id == agent.id

        # Retrieve agent's memories
        memories = memory_client.get_agent_memories(agent.id)
        assert len(memories) >= 1
        assert any(m.id == mem.id for m in memories)

    def test_permission_based_access(self, memory_client, user_id):
        """Test permission-based memory access."""
        agent1 = memory_client.create_agent("Agent 1", user_id, AgentRole.ASSISTANT)
        agent2 = memory_client.create_agent("Agent 2", user_id, AgentRole.SPECIALIST)

        # Agent1 stores shared memory
        memory_client.remember(
            text="Shared data",
            user_id=user_id,
            agent_id=agent1.id,
            visibility=MemoryVisibility.SHARED.value,
        )

        # Agent2 cannot access without permission
        accessible = memory_client.get_agent_memories(agent1.id, requesting_agent_id=agent2.id)
        assert len(accessible) == 0

        # Grant permission
        memory_client.grant_agent_permission(agent1.id, agent2.id, {PermissionType.READ})

        # Now agent2 can access
        accessible = memory_client.get_agent_memories(agent1.id, requesting_agent_id=agent2.id)
        assert len(accessible) >= 1
