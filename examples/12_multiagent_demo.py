"""Demo: Multi-Agent Memory Management.

This example demonstrates:
- Agent-specific memory spaces
- Shared vs private memories
- Agent-to-agent memory transfer
- User-Agent-Run hierarchy
- Agent memory permissions
"""

import os
from hippocampai import EnhancedMemoryClient, AgentRole, PermissionType, MemoryVisibility

# Get API key from environment
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    print("ERROR: Please set GROQ_API_KEY environment variable")
    print("export GROQ_API_KEY='your-api-key-here'")
    exit(1)

# Initialize client
print("=" * 70)
print("Multi-Agent Memory Management Demo")
print("=" * 70)

client = EnhancedMemoryClient(provider="groq")
user_id = "alice"

print("\n1. CREATING AGENTS")
print("-" * 70)

# Create multiple agents for different purposes
research_agent = client.create_agent(
    name="Research Assistant",
    user_id=user_id,
    role=AgentRole.SPECIALIST,
    description="Specialized in research and information gathering"
)
print(f"✓ Created {research_agent.name} (role: {research_agent.role.value})")

writing_agent = client.create_agent(
    name="Writing Assistant",
    user_id=user_id,
    role=AgentRole.ASSISTANT,
    description="Helps with writing and content creation"
)
print(f"✓ Created {writing_agent.name} (role: {writing_agent.role.value})")

coordinator = client.create_agent(
    name="Project Coordinator",
    user_id=user_id,
    role=AgentRole.COORDINATOR,
    description="Coordinates between different agents"
)
print(f"✓ Created {coordinator.name} (role: {coordinator.role.value})")

print("\n2. AGENT-SPECIFIC MEMORY SPACES")
print("-" * 70)

# Each agent stores memories in its own space
print("\n→ Research Agent gathering information...")
research_memories = [
    "Python was created by Guido van Rossum in 1991",
    "Machine learning is a subset of artificial intelligence",
    "Neural networks are inspired by biological neurons",
]

for text in research_memories:
    mem = client.remember(
        text=text,
        user_id=user_id,
        agent_id=research_agent.id,
        visibility=MemoryVisibility.PRIVATE.value
    )
    print(f"  [{research_agent.name}] {text[:50]}...")

print("\n→ Writing Agent creating content...")
writing_memories = [
    "Blog post outline: Introduction to AI",
    "Target audience: Software developers new to AI",
    "Writing style: Technical but accessible",
]

for text in writing_memories:
    mem = client.remember(
        text=text,
        user_id=user_id,
        agent_id=writing_agent.id,
        visibility=MemoryVisibility.PRIVATE.value
    )
    print(f"  [{writing_agent.name}] {text[:50]}...")

print("\n3. USER-AGENT-RUN HIERARCHY")
print("-" * 70)

# Create runs to organize memories within an agent
run1 = client.create_run(
    agent_id=research_agent.id,
    user_id=user_id,
    name="Python Research Session"
)
print(f"\n✓ Created run: {run1.name}")

# Store memories in the run
print("  Storing memories in run...")
for i, text in enumerate(["Python 3.12 released", "Type hints improved"], 1):
    mem = client.remember(
        text=text,
        user_id=user_id,
        agent_id=research_agent.id,
        run_id=run1.id,
        visibility=MemoryVisibility.PRIVATE.value
    )
    print(f"    {i}. {text}")

client.complete_run(run1.id)
print(f"  ✓ Run completed")

print("\n4. MEMORY VISIBILITY LEVELS")
print("-" * 70)

# Private: Only accessible by owning agent
private_mem = client.remember(
    text="Research notes: Draft findings (confidential)",
    user_id=user_id,
    agent_id=research_agent.id,
    visibility=MemoryVisibility.PRIVATE.value
)
print(f"✓ PRIVATE memory: Only {research_agent.name} can access")

# Shared: Accessible with explicit permission
shared_mem = client.remember(
    text="Research summary: Key findings about Python",
    user_id=user_id,
    agent_id=research_agent.id,
    visibility=MemoryVisibility.SHARED.value
)
print(f"✓ SHARED memory: Requires permission to access")

# Public: Accessible by all agents of same user
public_mem = client.remember(
    text="Project deadline: Submit blog post by Friday",
    user_id=user_id,
    agent_id=coordinator.id,
    visibility=MemoryVisibility.PUBLIC.value
)
print(f"✓ PUBLIC memory: All agents can access")

print("\n5. GRANTING PERMISSIONS")
print("-" * 70)

# Grant writing agent permission to read research agent's shared memories
permission = client.grant_agent_permission(
    granter_agent_id=research_agent.id,
    grantee_agent_id=writing_agent.id,
    permissions={PermissionType.READ}
)
print(f"✓ Granted READ permission: {research_agent.name} → {writing_agent.name}")

# Check permission
has_permission = client.check_agent_permission(
    writing_agent.id,
    research_agent.id,
    PermissionType.READ
)
print(f"  Can {writing_agent.name} read {research_agent.name}'s memories? {has_permission}")

print("\n6. ACCESSING AGENT MEMORIES")
print("-" * 70)

# Research agent accesses own memories
print(f"\n→ {research_agent.name}'s own memories:")
own_memories = client.get_agent_memories(research_agent.id)
print(f"  Total: {len(own_memories)} memories")
for mem in own_memories[:3]:
    print(f"    - {mem.text[:60]}...")

# Writing agent tries to access research agent's memories (filtered by permissions)
print(f"\n→ {writing_agent.name} accessing {research_agent.name}'s memories:")
accessible = client.get_agent_memories(
    agent_id=research_agent.id,
    requesting_agent_id=writing_agent.id
)
print(f"  Accessible: {len(accessible)} memories (filtered by permissions)")
for mem in accessible:
    print(f"    - {mem.text[:60]}... [{mem.visibility}]")

print("\n7. MEMORY TRANSFER")
print("-" * 70)

# Copy a memory from research agent to writing agent
print(f"\n→ Copying memory from {research_agent.name} to {writing_agent.name}...")
transfer = client.transfer_memory(
    memory_id=shared_mem.id,
    source_agent_id=research_agent.id,
    target_agent_id=writing_agent.id,
    transfer_type="copy"
)

if transfer:
    print(f"  ✓ Memory transferred successfully")
    print(f"    Transfer type: {transfer.transfer_type}")
    print(f"    Source: {transfer.source_agent_id[:8]}...")
    print(f"    Target: {transfer.target_agent_id[:8]}...")
else:
    print(f"  ✗ Transfer failed (permission denied)")

print("\n8. AGENT STATISTICS")
print("-" * 70)

# Get statistics for each agent
for agent in [research_agent, writing_agent, coordinator]:
    stats = client.get_agent_stats(agent.id)
    if stats:
        print(f"\n{agent.name}:")
        print(f"  Total memories: {stats.total_memories}")
        print(f"  By visibility: {stats.memories_by_visibility}")
        print(f"  Shared with: {len(stats.shared_with_agents)} agent(s)")
        print(f"  Can access from: {len(stats.can_access_from_agents)} agent(s)")

print("\n9. LISTING AGENTS AND RUNS")
print("-" * 70)

# List all agents
all_agents = client.list_agents(user_id=user_id)
print(f"\nTotal agents for user '{user_id}': {len(all_agents)}")
for agent in all_agents:
    print(f"  - {agent.name} ({agent.role.value})")

# List runs for research agent
runs = client.list_runs(agent_id=research_agent.id)
print(f"\nRuns for {research_agent.name}: {len(runs)}")
for run in runs:
    print(f"  - {run.name or run.id[:8]} (status: {run.status})")

print("\n10. USE CASE: COLLABORATIVE AGENTS")
print("-" * 70)

print("\nScenario: Research agent gathers info, writing agent creates content")
print()

# Research agent gathers information
print(f"1. {research_agent.name} gathers information...")
info = client.remember(
    text="Key insight: Python's simplicity makes it ideal for beginners",
    user_id=user_id,
    agent_id=research_agent.id,
    visibility=MemoryVisibility.SHARED.value
)
print(f"   ✓ Stored as SHARED memory")

# Grant permission if not already granted
print(f"\n2. Grant access to {writing_agent.name}...")
print(f"   ✓ Permission already granted")

# Writing agent accesses the information
print(f"\n3. {writing_agent.name} retrieves shared information...")
research_data = client.get_agent_memories(
    agent_id=research_agent.id,
    requesting_agent_id=writing_agent.id
)
print(f"   ✓ Retrieved {len(research_data)} shared memories")

# Writing agent creates content based on research
print(f"\n4. {writing_agent.name} creates content...")
content = client.remember(
    text="Blog draft: Why Python is perfect for beginners - based on research findings",
    user_id=user_id,
    agent_id=writing_agent.id,
    visibility=MemoryVisibility.PRIVATE.value,
    tags=["blog", "draft", "python"]
)
print(f"   ✓ Draft created")

# Coordinator tracks progress
print(f"\n5. {coordinator.name} tracks progress...")
progress = client.remember(
    text="Project status: Research complete, writing in progress",
    user_id=user_id,
    agent_id=coordinator.id,
    visibility=MemoryVisibility.PUBLIC.value
)
print(f"   ✓ Status updated (PUBLIC - visible to all agents)")

print("\n" + "=" * 70)
print("Multi-Agent Demo Completed!")
print("=" * 70)
print("\nKey Takeaways:")
print("  • Each agent has its own isolated memory space")
print("  • Memories can be PRIVATE, SHARED, or PUBLIC")
print("  • Permissions control access to shared memories")
print("  • Memories can be transferred between agents")
print("  • User → Agent → Run hierarchy organizes memories")
print("  • Perfect for multi-agent AI systems!")
