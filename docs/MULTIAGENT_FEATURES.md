# Multi-Agent Memory Management

HippocampAI now supports multi-agent systems with isolated memory spaces, permission-based sharing, and agent-to-agent memory transfer - similar to Mem0's multi-agent architecture.

## Features Overview

### 1. Agent-Specific Memory Spaces

Each agent has its own isolated memory namespace, preventing memory conflicts in multi-agent systems.

```python
from hippocampai import EnhancedMemoryClient, AgentRole

client = EnhancedMemoryClient(provider="groq")

# Create agents with different roles
research_agent = client.create_agent(
    name="Research Assistant",
    user_id="alice",
    role=AgentRole.SPECIALIST
)

writing_agent = client.create_agent(
    name="Writing Assistant",
    user_id="alice",
    role=AgentRole.ASSISTANT
)

# Each agent stores memories in its own space
client.remember(
    text="Python was created in 1991",
    user_id="alice",
    agent_id=research_agent.id
)

client.remember(
    text="Blog outline: Introduction to Python",
    user_id="alice",
    agent_id=writing_agent.id
)
```

### 2. Memory Visibility Levels

Control who can access memories with three visibility levels:

- **PRIVATE**: Only the owning agent can access
- **SHARED**: Accessible with explicit permission
- **PUBLIC**: All agents of the same user can access

```python
from hippocampai import MemoryVisibility

# Private memory (default)
private = client.remember(
    text="Confidential research notes",
    user_id="alice",
    agent_id=research_agent.id,
    visibility=MemoryVisibility.PRIVATE.value
)

# Shared memory
shared = client.remember(
    text="Research summary for team",
    user_id="alice",
    agent_id=research_agent.id,
    visibility=MemoryVisibility.SHARED.value
)

# Public memory
public = client.remember(
    text="Project deadline: Friday",
    user_id="alice",
    agent_id=coordinator.id,
    visibility=MemoryVisibility.PUBLIC.value
)
```

### 3. Agent Permissions

Grant fine-grained permissions for agents to access each other's memories.

**Permission Types:**
- `READ`: View memories
- `WRITE`: Modify memories
- `SHARE`: Transfer memories
- `DELETE`: Remove memories

```python
from hippocampai import PermissionType

# Grant read permission
permission = client.grant_agent_permission(
    granter_agent_id=research_agent.id,
    grantee_agent_id=writing_agent.id,
    permissions={PermissionType.READ, PermissionType.SHARE}
)

# Check permission
can_read = client.check_agent_permission(
    writing_agent.id,
    research_agent.id,
    PermissionType.READ
)

# Revoke permission
client.revoke_agent_permission(permission.id)
```

### 4. Agent-to-Agent Memory Transfer

Transfer memories between agents with three modes:

- **copy**: Duplicate memory to target agent
- **move**: Transfer ownership to target agent
- **share**: Grant access without duplication

```python
# Copy memory to another agent
transfer = client.transfer_memory(
    memory_id=memory.id,
    source_agent_id=research_agent.id,
    target_agent_id=writing_agent.id,
    transfer_type="copy"
)

# Move (transfer ownership)
client.transfer_memory(
    memory_id=memory.id,
    source_agent_id=agent1.id,
    target_agent_id=agent2.id,
    transfer_type="move"
)
```

### 5. User-Agent-Run Hierarchy

Organize memories in a three-level hierarchy for complex multi-agent workflows.

```python
# Create a run for specific task
run = client.create_run(
    agent_id=research_agent.id,
    user_id="alice",
    name="Python Research Session"
)

# Store memories in the run
memory = client.remember(
    text="Python 3.12 features documented",
    user_id="alice",
    agent_id=research_agent.id,
    run_id=run.id
)

# Complete the run
client.complete_run(run.id, status="completed")

# List runs for an agent
runs = client.list_runs(agent_id=research_agent.id)
```

## Agent Roles

Different roles for different agent types:

| Role | Description | Use Case |
|------|-------------|----------|
| **ASSISTANT** | General-purpose helper | Generic AI assistants |
| **SPECIALIST** | Domain expert | Research, coding, analysis agents |
| **COORDINATOR** | Manages other agents | Orchestration, workflow management |
| **OBSERVER** | Read-only monitoring | Logging, auditing agents |

```python
specialist = client.create_agent(
    "Domain Expert",
    "alice",
    role=AgentRole.SPECIALIST,
    description="Expert in machine learning"
)

coordinator = client.create_agent(
    "Project Manager",
    "alice",
    role=AgentRole.COORDINATOR,
    description="Coordinates all project agents"
)
```

## API Reference

### Agent Management

#### `create_agent(name, user_id, role, description=None, metadata=None) -> Agent`

Create a new agent with its own memory space.

**Args:**
- `name` (str): Agent name
- `user_id` (str): User ID owning the agent
- `role` (AgentRole): Agent role (default: ASSISTANT)
- `description` (str, optional): Description
- `metadata` (dict, optional): Additional metadata

**Returns:**
- `Agent`: Created agent object

**Example:**
```python
agent = client.create_agent(
    "Research Bot",
    "alice",
    role=AgentRole.SPECIALIST,
    description="Research automation agent",
    metadata={"version": "1.0"}
)
```

#### `get_agent(agent_id) -> Optional[Agent]`

Get agent by ID.

#### `list_agents(user_id=None) -> List[Agent]`

List all agents, optionally filtered by user.

#### `update_agent(agent_id, **kwargs) -> Optional[Agent]`

Update agent properties (name, description, role, metadata, is_active).

#### `delete_agent(agent_id) -> bool`

Delete an agent and all associated data.

### Memory Operations with Agents

#### `remember(..., agent_id=None, run_id=None, visibility=None) -> Memory`

Store a memory for a specific agent.

**New Parameters:**
- `agent_id` (str, optional): Agent ID
- `run_id` (str, optional): Run ID for organization
- `visibility` (str, optional): "private", "shared", or "public"

**Example:**
```python
memory = client.remember(
    text="Important finding",
    user_id="alice",
    agent_id=agent.id,
    run_id=run.id,
    visibility=MemoryVisibility.SHARED.value
)
```

#### `get_agent_memories(agent_id, requesting_agent_id=None, filters=None, limit=100) -> List[Memory]`

Get memories for an agent, respecting permissions.

**Args:**
- `agent_id` (str): Agent whose memories to retrieve
- `requesting_agent_id` (str, optional): Agent requesting access (for permission filtering)
- `filters` (dict, optional): Additional filters
- `limit` (int): Maximum memories

**Returns:**
- `List[Memory]`: Accessible memories

**Example:**
```python
# Get own memories
own_memories = client.get_agent_memories(agent1.id)

# Get another agent's memories (filtered by permissions)
accessible = client.get_agent_memories(
    agent_id=agent1.id,
    requesting_agent_id=agent2.id
)
```

### Run Management

#### `create_run(agent_id, user_id, name=None, metadata=None) -> Run`

Create a new run for organizing memories.

#### `get_run(run_id) -> Optional[Run]`

Get run by ID.

#### `list_runs(agent_id=None, user_id=None) -> List[Run]`

List runs, optionally filtered.

#### `complete_run(run_id, status="completed") -> Optional[Run]`

Mark a run as completed.

### Permission Management

#### `grant_agent_permission(granter_agent_id, grantee_agent_id, permissions, memory_filters=None, expires_at=None) -> AgentPermission`

Grant permission for one agent to access another's memories.

**Args:**
- `granter_agent_id` (str): Agent granting permission
- `grantee_agent_id` (str): Agent receiving permission
- `permissions` (Set[PermissionType]): Permissions to grant
- `memory_filters` (dict, optional): Filter which memories to share
- `expires_at` (datetime, optional): Expiration time

#### `revoke_agent_permission(permission_id) -> bool`

Revoke a permission.

#### `check_agent_permission(agent_id, target_agent_id, permission) -> bool`

Check if an agent has a specific permission.

#### `list_agent_permissions(granter_agent_id=None, grantee_agent_id=None) -> List[AgentPermission]`

List permissions, optionally filtered.

### Memory Transfer

#### `transfer_memory(memory_id, source_agent_id, target_agent_id, transfer_type="copy") -> Optional[MemoryTransfer]`

Transfer a memory between agents.

**Args:**
- `memory_id` (str): Memory to transfer
- `source_agent_id` (str): Source agent
- `target_agent_id` (str): Target agent
- `transfer_type` (str): "copy", "move", or "share"

**Returns:**
- `MemoryTransfer`: Transfer record

### Statistics

#### `get_agent_stats(agent_id) -> AgentMemoryStats`

Get memory statistics for an agent.

**Returns:**
- `AgentMemoryStats` with:
  - `total_memories`: Total memory count
  - `memories_by_visibility`: Count by visibility level
  - `memories_by_type`: Count by memory type
  - `memories_by_run`: Count by run
  - `shared_with_agents`: List of agents with access
  - `can_access_from_agents`: List of agents this agent can access

## Use Cases

### 1. Multi-Agent Research System

```python
# Create specialized agents
research_agent = client.create_agent("Researcher", "alice", AgentRole.SPECIALIST)
summarizer_agent = client.create_agent("Summarizer", "alice", AgentRole.ASSISTANT)
writer_agent = client.create_agent("Writer", "alice", AgentRole.ASSISTANT)

# Research phase
run1 = client.create_run(research_agent.id, "alice", "Research Phase")
client.remember(
    "Finding: Python adoption increased 27% in 2023",
    "alice",
    agent_id=research_agent.id,
    run_id=run1.id,
    visibility=MemoryVisibility.SHARED.value
)
client.complete_run(run1.id)

# Grant access to downstream agents
client.grant_agent_permission(
    research_agent.id,
    summarizer_agent.id,
    {PermissionType.READ}
)

# Summarizer retrieves research
research_data = client.get_agent_memories(
    research_agent.id,
    requesting_agent_id=summarizer_agent.id
)

# Create summary
client.remember(
    "Summary: Python remains top language for AI/ML",
    "alice",
    agent_id=summarizer_agent.id,
    visibility=MemoryVisibility.SHARED.value
)
```

### 2. Collaborative Writing Agents

```python
# Agents with different roles
outliner = client.create_agent("Outliner", "bob", AgentRole.ASSISTANT)
writer = client.create_agent("Writer", "bob", AgentRole.ASSISTANT)
editor = client.create_agent("Editor", "bob", AgentRole.SPECIALIST)

# Outliner creates structure
outline = client.remember(
    "Blog structure: Intro, 3 main points, Conclusion",
    "bob",
    agent_id=outliner.id,
    visibility=MemoryVisibility.PUBLIC.value  # All agents can see
)

# Writer drafts content (references outline via public visibility)
draft = client.remember(
    "Draft: [content based on outline]",
    "bob",
    agent_id=writer.id,
    visibility=MemoryVisibility.SHARED.value
)

# Grant editor access
client.grant_agent_permission(
    writer.id,
    editor.id,
    {PermissionType.READ, PermissionType.WRITE}
)

# Editor reviews and creates feedback
feedback = client.remember(
    "Edits needed: Strengthen conclusion",
    "bob",
    agent_id=editor.id,
    visibility=MemoryVisibility.SHARED.value
)
```

### 3. Agent Coordination System

```python
# Coordinator manages multiple specialized agents
coordinator = client.create_agent("Coordinator", "eve", AgentRole.COORDINATOR)
agents = [
    client.create_agent(f"Worker {i}", "eve", AgentRole.SPECIALIST)
    for i in range(3)
]

# Coordinator tracks overall progress
client.remember(
    "Project status: Phase 1 complete, Phase 2 in progress",
    "eve",
    agent_id=coordinator.id,
    visibility=MemoryVisibility.PUBLIC.value
)

# Coordinator can access all agent memories
for agent in agents:
    client.grant_agent_permission(
        agent.id,
        coordinator.id,
        {PermissionType.READ}
    )

    # Check agent progress
    memories = client.get_agent_memories(
        agent.id,
        requesting_agent_id=coordinator.id
    )
    print(f"{agent.name}: {len(memories)} tasks completed")
```

## Data Models

### Agent

```python
class Agent:
    id: str
    name: str
    user_id: str
    role: AgentRole  # ASSISTANT, SPECIALIST, COORDINATOR, OBSERVER
    description: Optional[str]
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    is_active: bool
    total_memories: int
    private_memories: int
    shared_memories: int
```

### Run

```python
class Run:
    id: str
    agent_id: str
    user_id: str
    name: Optional[str]
    status: str  # "active", "completed", "failed"
    started_at: datetime
    completed_at: Optional[datetime]
    metadata: Dict[str, Any]
    memories_created: int
    memories_accessed: int
```

### AgentPermission

```python
class AgentPermission:
    id: str
    granter_agent_id: str
    grantee_agent_id: str
    permissions: Set[PermissionType]  # READ, WRITE, SHARE, DELETE
    memory_filters: Optional[Dict[str, Any]]
    created_at: datetime
    expires_at: Optional[datetime]
    is_active: bool
```

### MemoryVisibility

```python
class MemoryVisibility(Enum):
    PRIVATE = "private"  # Only owning agent
    SHARED = "shared"    # With explicit permission
    PUBLIC = "public"    # All agents of user
```

## Memory Model Extensions

The `Memory` model now includes:

```python
class Memory:
    # ... existing fields ...
    agent_id: Optional[str]  # Owning agent
    run_id: Optional[str]    # Associated run
    visibility: Optional[str]  # Visibility level
```

**Backward Compatibility:** These fields are optional. Existing memories without agent_id work as before.

## Performance Considerations

### Memory Overhead

- Agent metadata: ~1KB per agent
- Permission records: ~500 bytes per permission
- Run metadata: ~800 bytes per run

### Query Performance

- Agent memory filtering: O(n) where n = user's memories
- Permission checks: O(p) where p = permissions count
- Recommend: Cache frequently accessed permissions

### Scaling Guidelines

| Agents per User | Performance | Notes |
|----------------|-------------|-------|
| 1-10 | Excellent | No overhead |
| 10-100 | Good | Minimal impact |
| 100-1000 | Fair | Consider permission caching |
| 1000+ | Optimize | Use agent hierarchies, batch operations |

## Best Practices

### 1. Use Appropriate Visibility Levels

```python
# Private: Sensitive agent-specific data
client.remember(
    "API key: xyz123",
    user_id,
    agent_id=agent.id,
    visibility=MemoryVisibility.PRIVATE.value
)

# Shared: Data for specific agents
client.remember(
    "Analysis results",
    user_id,
    agent_id=analyst.id,
    visibility=MemoryVisibility.SHARED.value
)

# Public: Common reference data
client.remember(
    "Project guidelines",
    user_id,
    agent_id=coordinator.id,
    visibility=MemoryVisibility.PUBLIC.value
)
```

### 2. Organize with Runs

```python
# Group related tasks
run = client.create_run(agent.id, user_id, "Data Collection Phase")

for source in data_sources:
    client.remember(
        f"Data from {source}",
        user_id,
        agent_id=agent.id,
        run_id=run.id
    )

client.complete_run(run.id)
```

### 3. Grant Minimal Permissions

```python
# Only grant necessary permissions
client.grant_agent_permission(
    source.id,
    target.id,
    {PermissionType.READ},  # Read-only
    memory_filters={"type": "fact"},  # Only facts
    expires_at=datetime.now() + timedelta(hours=1)  # Time-limited
)
```

### 4. Clean Up Inactive Agents

```python
# Deactivate instead of delete to preserve history
client.update_agent(agent.id, is_active=False)

# Later, delete if truly no longer needed
client.delete_agent(agent.id)
```

## Migration Guide

### For Existing Code

Multi-agent features are **fully backward compatible**:

```python
# Old code (still works)
memory = client.remember("Data", "user1")

# New code (with agent support)
memory = client.remember(
    "Data",
    "user1",
    agent_id=agent.id,
    visibility=MemoryVisibility.PRIVATE.value
)
```

### Adding Multi-Agent to Existing App

```python
# 1. Create agents for existing functionality
main_agent = client.create_agent("Main Assistant", user_id, AgentRole.ASSISTANT)

# 2. Update memory storage to include agent_id
memory = client.remember(
    text,
    user_id,
    agent_id=main_agent.id  # Add this
)

# 3. Create specialized agents as needed
specialist = client.create_agent("Specialist", user_id, AgentRole.SPECIALIST)

# 4. Set up permissions
client.grant_agent_permission(
    specialist.id,
    main_agent.id,
    {PermissionType.READ}
)
```

## Testing

Run the test suite:

```bash
# Run multi-agent tests
pytest tests/test_multiagent.py -v

# Run integration tests
pytest tests/test_multiagent.py::TestMultiAgentIntegration -v
```

## Demo

Run the comprehensive demo:

```bash
export GROQ_API_KEY='your-key-here'
python examples/12_multiagent_demo.py
```

## Comparison with Mem0

| Feature | HippocampAI | Mem0 |
|---------|-------------|------|
| Agent-Specific Spaces | ✅ Full isolation | ✅ Yes |
| Visibility Levels | ✅ Private/Shared/Public | ✅ Similar |
| Permission System | ✅ READ/WRITE/SHARE/DELETE | ⚠️ Basic |
| Memory Transfer | ✅ Copy/Move/Share | ✅ Yes |
| User-Agent-Run Hierarchy | ✅ Yes | ✅ Yes |
| Agent Roles | ✅ 4 predefined roles | ❌ No |
| Permission Expiration | ✅ Time-based expiry | ❌ No |
| Memory Filters in Permissions | ✅ Type/tag filtering | ⚠️ Limited |
| Agent Statistics | ✅ Comprehensive | ⚠️ Basic |
| Backward Compatible | ✅ Yes | ✅ Yes |

## Future Enhancements

Planned improvements:
- [ ] Agent-to-agent messaging system
- [ ] Agent groups/teams
- [ ] Delegation chains
- [ ] Agent capability declarations
- [ ] Advanced permission rules (time-based, quota-based)
- [ ] Agent performance analytics

## References

- Multi-Agent Manager: `src/hippocampai/multiagent/manager.py`
- Agent Models: `src/hippocampai/models/agent.py`
- Memory Extensions: `src/hippocampai/models/memory.py:40-43`
- Integration: `src/hippocampai/client.py:1789-2045`
- Tests: `tests/test_multiagent.py`
- Demo: `examples/12_multiagent_demo.py`

---

**Questions or Issues?** Please open an issue on GitHub!
