"""Demo: Multi-agent collaboration with shared memory spaces."""

import time
from hippocampai.client import MemoryClient
from hippocampai.multiagent.collaboration import CollaborationManager
from hippocampai.models.agent import PermissionType
from hippocampai.models.collaboration import CollaborationEventType

def main():
    """Demonstrate multi-agent collaboration features."""

    # Create collaboration manager
    collab_manager = CollaborationManager()

    # Create multiple agents
    print("=" * 80)
    print("1. Creating Agents")
    print("=" * 80)

    # Agent 1: Research assistant
    research_client = MemoryClient(user_id="user_collaborative_demo")
    research_agent = research_client.create_agent(
        name="Research Assistant",
        role="assistant",
        description="Handles research and information gathering"
    )
    print(f"✓ Created Research Agent: {research_agent.id}")

    # Agent 2: Writing assistant
    writing_agent = research_client.create_agent(
        name="Writing Assistant",
        role="assistant",
        description="Handles writing and content creation"
    )
    print(f"✓ Created Writing Agent: {writing_agent.id}")

    # Agent 3: Analytics assistant
    analytics_agent = research_client.create_agent(
        name="Analytics Assistant",
        role="specialist",
        description="Analyzes patterns and generates insights"
    )
    print(f"✓ Created Analytics Agent: {analytics_agent.id}")

    # 2. Create shared memory space
    print("\n" + "=" * 80)
    print("2. Creating Shared Memory Space")
    print("=" * 80)

    space = collab_manager.create_space(
        name="Research Project: AI Ethics",
        owner_agent_id=research_agent.id,
        description="Collaborative space for AI ethics research",
        tags=["research", "ai", "ethics"]
    )
    print(f"✓ Created space: {space.name} (ID: {space.id})")
    print(f"  Owner: {research_agent.name}")

    # 3. Add collaborators with permissions
    print("\n" + "=" * 80)
    print("3. Adding Collaborators with Permissions")
    print("=" * 80)

    # Add writing agent with READ and WRITE permissions
    collab_manager.add_collaborator(
        space_id=space.id,
        agent_id=writing_agent.id,
        permissions=[PermissionType.READ, PermissionType.WRITE],
        inviter_id=research_agent.id
    )
    print(f"✓ Added {writing_agent.name} with READ, WRITE permissions")

    # Add analytics agent with READ permission only
    collab_manager.add_collaborator(
        space_id=space.id,
        agent_id=analytics_agent.id,
        permissions=[PermissionType.READ],
        inviter_id=research_agent.id
    )
    print(f"✓ Added {analytics_agent.name} with READ permission")

    # 4. Check notifications
    print("\n" + "=" * 80)
    print("4. Checking Notifications")
    print("=" * 80)

    writing_notifications = collab_manager.get_notifications(writing_agent.id)
    print(f"✓ {writing_agent.name} has {len(writing_notifications)} notification(s)")
    for notif in writing_notifications:
        print(f"  - {notif.title}: {notif.message}")

    # 5. Collaborative memory creation
    print("\n" + "=" * 80)
    print("5. Collaborative Memory Creation")
    print("=" * 80)

    # Research agent adds research findings
    research_memory = research_client.remember(
        "AI bias can emerge from training data that reflects historical inequalities",
        type="fact",
        importance=8.0,
        tags=["ai_bias", "ethics"],
        agent_id=research_agent.id
    )
    print(f"✓ {research_agent.name} created memory: {research_memory.id}")

    # Add to shared space
    collab_manager.add_memory_to_space(
        space_id=space.id,
        memory_id=research_memory.id,
        agent_id=research_agent.id
    )
    print(f"  Added to shared space")

    # Writing agent adds writing draft
    writing_memory = research_client.remember(
        "Draft outline for AI ethics paper focusing on bias mitigation strategies",
        type="context",
        importance=7.0,
        tags=["writing", "outline"],
        agent_id=writing_agent.id
    )
    print(f"✓ {writing_agent.name} created memory: {writing_memory.id}")

    collab_manager.add_memory_to_space(
        space_id=space.id,
        memory_id=writing_memory.id,
        agent_id=writing_agent.id
    )
    print(f"  Added to shared space")

    # 6. Check space events
    print("\n" + "=" * 80)
    print("6. Collaboration Activity Log")
    print("=" * 80)

    events = collab_manager.get_space_events(space.id, limit=10)
    print(f"✓ Found {len(events)} events in space:")
    for event in events[:5]:
        print(f"  [{event.timestamp.strftime('%H:%M:%S')}] {event.event_type.value} by {event.agent_id[:8]}...")
        if event.data:
            print(f"    Data: {event.data}")

    # 7. Update permissions
    print("\n" + "=" * 80)
    print("7. Updating Permissions")
    print("=" * 80)

    # Grant analytics agent WRITE permission
    collab_manager.update_permissions(
        space_id=space.id,
        agent_id=analytics_agent.id,
        permissions=[PermissionType.READ, PermissionType.WRITE],
        updater_id=research_agent.id
    )
    print(f"✓ Updated {analytics_agent.name} permissions to READ, WRITE")

    # Check notification
    analytics_notifications = collab_manager.get_notifications(analytics_agent.id)
    print(f"✓ {analytics_agent.name} has {len(analytics_notifications)} notification(s)")

    # 8. List all spaces for an agent
    print("\n" + "=" * 80)
    print("8. Listing Spaces by Agent")
    print("=" * 80)

    research_spaces = collab_manager.list_spaces(agent_id=research_agent.id)
    print(f"✓ {research_agent.name} participates in {len(research_spaces)} space(s):")
    for sp in research_spaces:
        print(f"  - {sp.name}")
        print(f"    Collaborators: {len(sp.collaborator_agent_ids)}")
        print(f"    Memories: {len(sp.memory_ids)}")

    # 9. Demonstrate permission checks
    print("\n" + "=" * 80)
    print("9. Permission Verification")
    print("=" * 80)

    can_write = space.has_permission(writing_agent.id, PermissionType.WRITE)
    can_delete = space.has_permission(writing_agent.id, PermissionType.DELETE)
    print(f"✓ {writing_agent.name} can write: {can_write}")
    print(f"✓ {writing_agent.name} can delete: {can_delete}")

    # 10. Conflict simulation
    print("\n" + "=" * 80)
    print("10. Conflict Detection & Resolution")
    print("=" * 80)

    from hippocampai.models.collaboration import ConflictType, ResolutionStrategy

    # Simulate concurrent edit conflict
    conflict = collab_manager.detect_conflict(
        space_id=space.id,
        memory_id=research_memory.id,
        conflicting_versions=[
            {"text": "Version 1 by research agent", "updated_by": research_agent.id},
            {"text": "Version 2 by writing agent", "updated_by": writing_agent.id}
        ],
        conflict_type=ConflictType.CONCURRENT_UPDATE
    )
    print(f"✓ Conflict detected: {conflict.conflict_type.value}")
    print(f"  Conflict ID: {conflict.id}")
    print(f"  Versions: {len(conflict.conflicting_versions)}")

    # Resolve conflict (latest wins)
    collab_manager.resolve_conflict(
        conflict_id=conflict.id,
        resolved_version={"text": "Merged version by owner", "updated_by": research_agent.id},
        resolved_by=research_agent.id,
        strategy=ResolutionStrategy.LATEST_WINS
    )
    print(f"✓ Conflict resolved using {ResolutionStrategy.LATEST_WINS.value}")

    # 11. Real-time notifications
    print("\n" + "=" * 80)
    print("11. Notification Management")
    print("=" * 80)

    # Get unread notifications
    unread_notifications = collab_manager.get_notifications(
        writing_agent.id,
        unread_only=True
    )
    print(f"✓ {writing_agent.name} has {len(unread_notifications)} unread notification(s)")

    # Mark as read
    if unread_notifications:
        for notif in unread_notifications:
            collab_manager.mark_notification_read(writing_agent.id, notif.id)
        print(f"✓ Marked all notifications as read")

    # 12. Cleanup
    print("\n" + "=" * 80)
    print("12. Summary")
    print("=" * 80)

    print(f"✓ Collaboration demo completed successfully!")
    print(f"  - Created {len(collab_manager.spaces)} shared space(s)")
    print(f"  - {len(space.collaborator_agent_ids)} collaborator(s)")
    print(f"  - {len(space.memory_ids)} shared memor(ies)")
    print(f"  - {len(events)} collaboration event(s)")
    print(f"  - {len(collab_manager.conflicts)} conflict(s) detected & resolved")

    print("\n💡 Key Features Demonstrated:")
    print("  1. Shared memory spaces for team collaboration")
    print("  2. Fine-grained permission control")
    print("  3. Real-time event tracking")
    print("  4. Conflict detection and resolution")
    print("  5. Notification system")
    print("  6. Multi-agent coordination")


if __name__ == "__main__":
    main()
