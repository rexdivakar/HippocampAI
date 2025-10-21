"""Demonstration of memory graph persistence with JSON export/import.

This example shows how to:
- Build a memory graph with relationships
- Export the graph to a JSON file
- Import the graph from a JSON file
- Use user-specific exports
- Handle graph merging and replacement modes

Run this with:
    python examples/09_graph_persistence_demo.py
"""

from hippocampai import MemoryClient, RelationType


def print_section(title: str):
    """Print a section header."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def main():
    """Run the graph persistence demonstration."""

    # === 1. CREATE CLIENT AND BUILD GRAPH ===
    print_section("1. Create Client and Build Memory Graph")

    client = MemoryClient(enable_telemetry=False)
    print("✓ Client initialized")

    # Create memories for alice
    print("\nCreating memories for Alice...")
    m1 = client.remember("I love Python programming", user_id="alice", type="preference")
    m2 = client.remember("Python is great for data science", user_id="alice", type="fact")
    m3 = client.remember("I use Jupyter notebooks daily", user_id="alice", type="preference")
    m4 = client.remember("Machine learning is fascinating", user_id="alice", type="preference")
    m5 = client.remember("Pandas is my favorite data library", user_id="alice", type="preference")

    # Add memories to graph
    for memory in [m1, m2, m3, m4, m5]:
        client.graph.add_memory(memory.id, "alice", {"text": memory.text})

    # Create relationships
    print("\nCreating relationships...")
    client.add_relationship(m1.id, m2.id, RelationType.RELATED_TO, weight=0.9)
    client.add_relationship(m2.id, m4.id, RelationType.LEADS_TO, weight=0.8)
    client.add_relationship(m1.id, m3.id, RelationType.SUPPORTS, weight=0.7)
    client.add_relationship(m4.id, m5.id, RelationType.RELATED_TO, weight=0.85)
    client.add_relationship(m3.id, m5.id, RelationType.SIMILAR_TO, weight=0.75)

    # Create memories for bob
    print("\nCreating memories for Bob...")
    b1 = client.remember("I prefer Java for backend dev", user_id="bob", type="preference")
    b2 = client.remember("Spring Boot is excellent", user_id="bob", type="fact")

    # Add bob's memories to graph
    for memory in [b1, b2]:
        client.graph.add_memory(memory.id, "bob", {"text": memory.text})

    client.add_relationship(b1.id, b2.id, RelationType.RELATED_TO, weight=0.95)

    # Display graph stats
    stats = client.graph.get_graph_stats()
    print("\n✓ Graph created:")
    print(f"  Nodes: {stats['num_nodes']}")
    print(f"  Edges: {stats['num_edges']}")
    print(f"  Density: {stats['density']:.3f}")
    print(f"  Clusters: {stats['num_clusters']}")

    # === 2. EXPORT FULL GRAPH ===
    print_section("2. Export Full Graph to JSON")

    full_graph_path = "full_memory_graph.json"
    result_path = client.export_graph_to_json(full_graph_path)
    print(f"✓ Exported full graph to: {result_path}")

    # Read and display the file
    import json

    with open(full_graph_path, "r") as f:
        graph_data = json.load(f)
    print(f"  Nodes in file: {len(graph_data['nodes'])}")
    print(f"  Edges in file: {len(graph_data['links'])}")

    # === 3. EXPORT USER-SPECIFIC GRAPH ===
    print_section("3. Export User-Specific Graph")

    alice_graph_path = "alice_memory_graph.json"
    client.export_graph_to_json(alice_graph_path, user_id="alice")
    print(f"✓ Exported Alice's graph to: {alice_graph_path}")

    with open(alice_graph_path, "r") as f:
        alice_data = json.load(f)
    print(f"  Alice's nodes: {len(alice_data['nodes'])}")
    print(f"  Alice's edges: {len(alice_data['links'])}")

    bob_graph_path = "bob_memory_graph.json"
    client.export_graph_to_json(bob_graph_path, user_id="bob")
    print(f"✓ Exported Bob's graph to: {bob_graph_path}")

    with open(bob_graph_path, "r") as f:
        bob_data = json.load(f)
    print(f"  Bob's nodes: {len(bob_data['nodes'])}")
    print(f"  Bob's edges: {len(bob_data['links'])}")

    # === 4. CLEAR AND RESTORE GRAPH ===
    print_section("4. Clear and Restore Graph from JSON")

    # Clear the graph
    print("Clearing graph...")
    client.graph.graph.clear()
    client.graph._memory_index.clear()
    client.graph._user_graphs.clear()

    stats_after_clear = client.graph.get_graph_stats()
    print(f"  Nodes after clear: {stats_after_clear['num_nodes']}")
    print(f"  Edges after clear: {stats_after_clear['num_edges']}")

    # Import the full graph back
    print(f"\nImporting from {full_graph_path}...")
    import_stats = client.import_graph_from_json(full_graph_path)

    print("✓ Import completed:")
    print(f"  Nodes before: {import_stats['nodes_before']}")
    print(f"  Nodes after: {import_stats['nodes_after']}")
    print(f"  Edges before: {import_stats['edges_before']}")
    print(f"  Edges after: {import_stats['edges_after']}")
    print(f"  Merged: {import_stats['merged']}")

    # Verify relationships still exist
    related = client.get_related_memories(m1.id)
    print(f"\n  Verified: {m1.id[:8]}... has {len(related)} relationships")

    # === 5. MERGE MODE DEMONSTRATION ===
    print_section("5. Merge Mode - Adding More Data")

    # Create a new client with fresh memories
    print("Creating new memories for Charlie...")
    c1 = client.remember("I enjoy Rust programming", user_id="charlie", type="preference")
    c2 = client.remember("Rust is memory-safe", user_id="charlie", type="fact")

    client.graph.add_memory(c1.id, "charlie", {"text": c1.text})
    client.graph.add_memory(c2.id, "charlie", {"text": c2.text})
    client.add_relationship(c1.id, c2.id, RelationType.RELATED_TO)

    # Export charlie's graph
    charlie_graph_path = "charlie_memory_graph.json"
    client.export_graph_to_json(charlie_graph_path, user_id="charlie")
    print(f"✓ Exported Charlie's graph to: {charlie_graph_path}")

    # Clear graph
    client.graph.graph.clear()
    client.graph._memory_index.clear()
    client.graph._user_graphs.clear()

    # Import full graph
    print("\nImporting full graph (Alice + Bob)...")
    client.import_graph_from_json(full_graph_path)

    # Merge Charlie's graph
    print("Merging Charlie's graph (merge=True)...")
    merge_stats = client.import_graph_from_json(charlie_graph_path, merge=True)

    print("✓ Merge completed:")
    print(f"  Nodes before merge: {merge_stats['nodes_before']}")
    print(f"  Nodes after merge: {merge_stats['nodes_after']}")
    print("  Total users: Alice, Bob, Charlie")

    # === 6. REPLACE MODE DEMONSTRATION ===
    print_section("6. Replace Mode - Complete Graph Replacement")

    print("Importing Bob's graph with merge=False (replace)...")
    replace_stats = client.import_graph_from_json(bob_graph_path, merge=False)

    print("✓ Replace completed:")
    print(f"  Nodes before: {replace_stats['nodes_before']}")
    print(f"  Nodes after: {replace_stats['nodes_after']}")
    print("  Graph now contains only Bob's memories")

    # === 7. GET GRAPH STATISTICS ===
    print_section("7. Final Graph Statistics")

    # Restore full graph for final stats
    client.import_graph_from_json(full_graph_path, merge=False)

    alice_stats = client.graph.get_graph_stats(user_id="alice")
    bob_stats = client.graph.get_graph_stats(user_id="bob")

    print("Alice's graph:")
    print(f"  Nodes: {alice_stats['num_nodes']}")
    print(f"  Edges: {alice_stats['num_edges']}")
    print(f"  Avg degree: {alice_stats['avg_degree']:.2f}")

    print("\nBob's graph:")
    print(f"  Nodes: {bob_stats['num_nodes']}")
    print(f"  Edges: {bob_stats['num_edges']}")
    print(f"  Avg degree: {bob_stats['avg_degree']:.2f}")

    # === 8. CLEANUP ===
    print_section("8. Cleanup")

    import os

    print("Removing exported files...")
    for file in [full_graph_path, alice_graph_path, bob_graph_path, charlie_graph_path]:
        if os.path.exists(file):
            os.remove(file)
            print(f"  ✓ Removed {file}")

    # Delete test memories
    memories = []
    for user in ["alice", "bob", "charlie"]:
        memories.extend(client.get_memories(user_id=user))

    memory_ids = [m.id for m in memories]
    if memory_ids:
        for user in ["alice", "bob", "charlie"]:
            user_mems = [m.id for m in memories if m.user_id == user]
            if user_mems:
                deleted = client.delete_memories(user_mems, user_id=user)
                print(f"  ✓ Deleted {deleted} memories for {user}")

    # === USAGE NOTES ===
    print_section("Usage Notes")

    print("""
Graph Persistence Features:

1. Export to JSON:
   # Export full graph
   client.export_graph_to_json("graph.json")

   # Export user-specific graph
   client.export_graph_to_json("alice_graph.json", user_id="alice")

   # Custom indentation
   client.export_graph_to_json("graph.json", indent=4)

2. Import from JSON:
   # Import and merge with existing graph
   stats = client.import_graph_from_json("graph.json")

   # Replace existing graph
   stats = client.import_graph_from_json("graph.json", merge=False)

3. Import Statistics:
   Returns a dictionary with:
   - file_path: Path that was imported
   - nodes_before/after: Node counts
   - edges_before/after: Edge counts
   - nodes_imported: Nodes in the file
   - edges_imported: Edges in the file
   - merged: Whether it was merged or replaced

4. Use Cases:
   - Backup and restore memory graphs
   - Transfer graphs between environments
   - Share user-specific memory graphs
   - Version control for memory states
   - Debugging and analysis
   - Graph migrations

5. Best Practices:
   - Export graphs regularly for backup
   - Use user-specific exports for privacy
   - Test imports with merge=True first
   - Keep export files in version control
   - Use descriptive filenames with timestamps
""")

    print("\n" + "=" * 60)
    print("  Graph persistence demonstration completed!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
