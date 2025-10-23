"""Demo of Intelligence Features: Fact Extraction, Entity Recognition, and Summarization.

This example demonstrates:
1. Fact extraction from conversations
2. Entity recognition and tracking
3. Relationship extraction
4. Session summarization
5. Knowledge graph building and querying
6. Knowledge inference

Run with: python examples/11_intelligence_features_demo.py
"""

from hippocampai import MemoryClient
from hippocampai.pipeline import SummaryStyle, EntityType, FactCategory


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def demo_fact_extraction(client):
    """Demonstrate fact extraction capabilities."""
    print_section("1. Fact Extraction")

    # Example text with various facts
    text = """
    John Smith works at Google as a Senior Software Engineer in San Francisco.
    He studied Computer Science at MIT and graduated in 2015.
    He's proficient in Python, Java, and Go.
    He wants to learn machine learning and plans to take an online course.
    """

    print("Input text:")
    print(text.strip())

    # Extract facts
    facts = client.extract_facts(text, source="profile")

    print(f"\nExtracted {len(facts)} facts:\n")

    # Group by category
    facts_by_category = {}
    for fact in facts:
        category = fact.category.value
        if category not in facts_by_category:
            facts_by_category[category] = []
        facts_by_category[category].append(fact)

    for category, category_facts in sorted(facts_by_category.items()):
        print(f"\n{category.upper()}:")
        for fact in category_facts:
            print(f"  • {fact.fact}")
            print(f"    Confidence: {fact.confidence:.2f}")
            if fact.entities:
                print(f"    Entities: {', '.join(fact.entities)}")
            if fact.temporal:
                print(f"    Temporal: {fact.temporal} ({fact.temporal_type.value})")


def demo_conversation_fact_extraction(client):
    """Demonstrate fact extraction from conversations."""
    print_section("2. Conversation Fact Extraction")

    conversation = """
    User: I just started working at SpaceX as a software engineer
    Assistant: Congratulations! That's exciting. What will you be working on?
    User: I'll be working on the Starship navigation systems. I studied aerospace engineering at MIT.
    Assistant: That's a perfect background for the role!
    User: Yes, I'm really excited. I'll be based in the Boca Chica facility in Texas.
    """

    print("Conversation:")
    print(conversation.strip())

    # Extract facts from conversation
    facts = client.extract_facts_from_conversation(conversation, user_id="demo_user")

    print(f"\nExtracted {len(facts)} facts from conversation:\n")

    for fact in facts:
        print(f"[{fact.category.value}] {fact.fact}")
        if fact.temporal_type.value != "present":
            print(f"  Temporal: {fact.temporal_type.value}")


def demo_entity_recognition(client):
    """Demonstrate entity recognition."""
    print_section("3. Entity Recognition")

    text = "Elon Musk founded SpaceX in 2002 in Hawthorne, California. Tesla is also based in California."

    print("Input text:")
    print(text)

    # Extract entities
    entities = client.extract_entities(text)

    print(f"\nExtracted {len(entities)} entities:\n")

    for entity in entities:
        print(f"{entity.type.value.upper()}: {entity.text}")
        print(f"  ID: {entity.entity_id}")
        print(f"  Confidence: {entity.confidence:.2f}")
        print()


def demo_entity_tracking(client):
    """Demonstrate entity profile tracking."""
    print_section("4. Entity Profile Tracking")

    # Mention an entity multiple times
    texts = [
        "Tim Cook is the CEO of Apple",
        "Tim Cook announced new products today",
        "Apple's CEO Tim Cook spoke at the conference",
    ]

    print("Processing multiple mentions of Tim Cook:")
    for i, text in enumerate(texts, 1):
        print(f"\n{i}. {text}")
        entities = client.extract_entities(text)

        # Add to graph
        for entity in entities:
            if "tim" in entity.text.lower() and entity.type == EntityType.PERSON:
                client.add_entity_to_graph(entity)

    # Get profile
    results = client.search_entities("tim cook", entity_type=EntityType.PERSON)

    if results:
        profile = results[0]
        print(f"\n--- Entity Profile ---")
        print(f"Canonical name: {profile.canonical_name}")
        print(f"Type: {profile.type.value}")
        print(f"Mention count: {profile.mention_count}")
        print(f"Aliases: {profile.aliases}")
        print(f"First seen: {profile.first_seen}")
        print(f"Last seen: {profile.last_seen}")


def demo_relationship_extraction(client):
    """Demonstrate relationship extraction."""
    print_section("5. Relationship Extraction")

    text = "Steve Jobs founded Apple. Apple is located in Cupertino. Jobs studied at Reed College."

    print("Input text:")
    print(text)

    # Extract relationships
    relationships = client.extract_relationships(text)

    print(f"\nExtracted {len(relationships)} relationships:\n")

    for rel in relationships:
        print(f"{rel.relation_type.value}:")
        print(f"  From: {rel.from_entity_id}")
        print(f"  To: {rel.to_entity_id}")
        print(f"  Confidence: {rel.confidence:.2f}")
        print(f"  Context: {rel.context}")
        print()


def demo_summarization(client):
    """Demonstrate conversation summarization."""
    print_section("6. Conversation Summarization")

    messages = [
        {"role": "user", "content": "I need help planning my career transition"},
        {
            "role": "assistant",
            "content": "I'd be happy to help! What field are you currently in and what are you considering?",
        },
        {
            "role": "user",
            "content": "I'm a mechanical engineer but I want to move into software engineering",
        },
        {
            "role": "assistant",
            "content": "That's a great transition! Many mechanical engineers have successfully made that switch.",
        },
        {
            "role": "user",
            "content": "I need to learn Python and data structures. Should I take an online course?",
        },
        {
            "role": "assistant",
            "content": "Yes, I'd recommend starting with Python fundamentals, then moving to data structures and algorithms.",
        },
        {
            "role": "user",
            "content": "I decided to enroll in the Stanford online CS course and plan to complete it in 6 months",
        },
    ]

    print("Conversation (7 messages):\n")
    for msg in messages:
        print(f"{msg['role'].upper()}: {msg['content']}")

    # Generate different summary styles
    print("\n--- Summaries ---\n")

    for style in [SummaryStyle.CONCISE, SummaryStyle.BULLET_POINTS, SummaryStyle.EXECUTIVE]:
        summary = client.summarize_conversation(messages, session_id="demo_session", style=style)

        print(f"\n{style.value.upper()} STYLE:")
        print(summary.summary if summary.summary else "• " + "\n• ".join(summary.key_points))

    # Generate detailed analysis
    summary = client.summarize_conversation(
        messages, session_id="demo_session", style=SummaryStyle.DETAILED
    )

    print("\n--- Detailed Analysis ---")
    print(f"\nTopics: {', '.join(summary.topics)}")
    print(f"Sentiment: {summary.sentiment.value}")
    print(f"Questions asked: {summary.questions_asked}")
    print(f"Questions answered: {summary.questions_answered}")

    if summary.action_items:
        print(f"\nAction Items:")
        for item in summary.action_items:
            print(f"  □ {item}")

    if summary.key_points:
        print(f"\nKey Points:")
        for point in summary.key_points[:3]:
            print(f"  • {point}")


def demo_conversation_insights(client):
    """Demonstrate insight extraction."""
    print_section("7. Conversation Insights")

    messages = [
        {"role": "user", "content": "I decided to quit my job and start my own company"},
        {"role": "assistant", "content": "That's a big decision! What made you decide to do that?"},
        {
            "role": "user",
            "content": "I realized that I want more control over my work and I learned that I'm good at building products",
        },
        {"role": "assistant", "content": "Those are great reasons. Do you have a business plan?"},
        {
            "role": "user",
            "content": "Yes, I plan to launch a SaaS product for small businesses in the next 3 months",
        },
    ]

    print("Conversation about career decision:\n")
    for msg in messages:
        print(f"{msg['role'].upper()}: {msg['content']}")

    # Extract insights
    insights = client.extract_conversation_insights(messages, user_id="demo_user")

    print("\n--- Extracted Insights ---")

    if insights["key_decisions"]:
        print("\nKey Decisions:")
        for decision in insights["key_decisions"]:
            print(f"  • {decision}")

    if insights["learning_points"]:
        print("\nLearning Points:")
        for learning in insights["learning_points"]:
            print(f"  • {learning}")

    print(f"\nTopics: {', '.join(insights['topics'])}")
    print(f"Sentiment: {insights['sentiment']}")


def demo_knowledge_graph(client):
    """Demonstrate knowledge graph building and querying."""
    print_section("8. Knowledge Graph Operations")

    # Create a rich memory
    text = """Marie Curie was a physicist and chemist who conducted pioneering research on radioactivity.
    She was the first woman to win a Nobel Prize, and the first person to win two Nobel Prizes.
    She worked at the University of Paris in France. She discovered the elements polonium and radium."""

    print("Input text:")
    print(text)

    # Create memory and enrich with intelligence
    memory = client.remember(text, user_id="demo_user", session_id="science_session")

    print("\nEnriching memory with intelligence features...")
    enrichment = client.enrich_memory_with_intelligence(memory, add_to_graph=True)

    print(f"\n--- Enrichment Results ---")
    print(f"Facts extracted: {len(enrichment['facts'])}")
    print(f"Entities extracted: {len(enrichment['entities'])}")
    print(f"Relationships extracted: {len(enrichment['relationships'])}")
    print(f"Graph updated: {enrichment['graph_updated']}")

    # Show extracted facts
    print("\nFacts:")
    for fact in enrichment["facts"][:3]:
        print(f"  • [{fact.category.value}] {fact.fact}")

    # Show extracted entities
    print("\nEntities:")
    for entity in enrichment["entities"][:5]:
        print(f"  • {entity.type.value}: {entity.text}")

    # Query the graph
    if enrichment["entities"]:
        entity = enrichment["entities"][0]

        print(f"\n--- Knowledge Graph Query for '{entity.text}' ---")

        # Get memories
        memory_ids = client.get_entity_memories(entity.entity_id)
        print(f"\nMemories mentioning this entity: {len(memory_ids)}")

        # Get facts
        fact_ids = client.get_entity_facts(entity.entity_id)
        print(f"Facts about this entity: {len(fact_ids)}")

        # Get connections
        connections = client.get_entity_connections(entity.entity_id, max_distance=2)
        print(
            f"Connected entities: {sum(len(entities) for entities in connections.values())}"
        )

        # Get timeline
        timeline = client.get_entity_timeline(entity.entity_id)
        print(f"Timeline events: {len(timeline)}")

        if timeline:
            print("\nTimeline:")
            for event in timeline[:3]:
                print(f"  • {event['type']}: {event.get('text', 'N/A')[:60]}...")


def demo_knowledge_inference(client):
    """Demonstrate knowledge inference."""
    print_section("9. Knowledge Inference")

    # Add related facts
    texts = [
        "Alice Johnson works at Google",
        "Google is headquartered in Mountain View",
        "Mountain View is in California",
    ]

    print("Building knowledge from related facts:\n")
    for text in texts:
        print(f"• {text}")
        memory = client.remember(text, user_id="demo_user")
        client.enrich_memory_with_intelligence(memory, add_to_graph=True)

    print("\nInferring new knowledge from graph patterns...")

    # Attempt knowledge inference
    inferred = client.infer_knowledge(user_id="demo_user")

    if inferred:
        print(f"\nInferred {len(inferred)} new facts:\n")
        for fact in inferred[:3]:
            print(f"• {fact['fact']}")
            print(f"  Confidence: {fact['confidence']:.2f}")
            print(f"  Rule: {fact['rule']}")
            print()
    else:
        print("\nNo new facts inferred (inference requires specific patterns)")


def demo_entity_search(client):
    """Demonstrate entity search capabilities."""
    print_section("10. Entity Search")

    # Add various entities
    texts = [
        "John Smith works at Microsoft",
        "Jane Smith is a professor at Stanford",
        "Apple Inc. is a technology company",
        "Google was founded by Larry Page",
        "Microsoft is based in Redmond, Washington",
    ]

    print("Adding entities to knowledge base:\n")
    for text in texts:
        print(f"• {text}")
        entities = client.extract_entities(text)
        for entity in entities:
            client.add_entity_to_graph(entity)

    # Search examples
    print("\n--- Search Examples ---\n")

    # Search for people named Smith
    results = client.search_entities("smith", entity_type=EntityType.PERSON)
    print(f"People named 'Smith': {len(results)}")
    for profile in results:
        print(f"  • {profile.canonical_name} ({profile.mention_count} mentions)")

    # Search for organizations
    results = client.search_entities("microsoft", entity_type=EntityType.ORGANIZATION)
    print(f"\nOrganizations matching 'microsoft': {len(results)}")
    for profile in results:
        print(f"  • {profile.canonical_name} ({profile.mention_count} mentions)")

    # Search with minimum mentions
    results = client.search_entities("", min_mentions=1)
    print(f"\nAll entities with 1+ mentions: {len(results)}")


def main():
    """Run all intelligence feature demos."""
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "Intelligence Features Demo" + " " * 27 + "║")
    print("╚" + "═" * 68 + "╝")

    # Initialize client
    print("\nInitializing HippocampAI client...")
    client = MemoryClient.from_preset("development")

    # Run all demos
    demo_fact_extraction(client)
    demo_conversation_fact_extraction(client)
    demo_entity_recognition(client)
    demo_entity_tracking(client)
    demo_relationship_extraction(client)
    demo_summarization(client)
    demo_conversation_insights(client)
    demo_knowledge_graph(client)
    demo_entity_search(client)
    demo_knowledge_inference(client)

    print_section("Demo Complete!")
    print(
        "\nAll intelligence features demonstrated successfully!"
    )
    print("\nKey Takeaways:")
    print("  • Fact extraction automatically identifies structured information")
    print("  • Entity recognition tracks people, places, and organizations")
    print("  • Summarization provides insights into conversations")
    print("  • Knowledge graph connects memories, entities, and facts")
    print("  • These features work together to build rich, queryable knowledge\n")


if __name__ == "__main__":
    main()
