# Intelligence Features

HippocampAI includes advanced intelligence features for extracting structured knowledge from conversations and building rich knowledge graphs:

1. **Fact Extraction** - Automatically extract structured facts from text and conversations
2. **Entity Recognition** - Identify and track people, organizations, locations, and more
3. **Session Summarization** - Generate summaries with key points, action items, and sentiment analysis
4. **Knowledge Graph** - Build and query a graph of memories, entities, facts, and their relationships

## Table of Contents

- [Fact Extraction](#fact-extraction)
- [Entity Recognition](#entity-recognition)
- [Session Summarization](#session-summarization)
- [Knowledge Graph](#knowledge-graph)
- [Complete Example](#complete-example)

## Fact Extraction

Extract structured facts from text with automatic categorization and confidence scoring.

### Supported Fact Categories

- `EMPLOYMENT` - Job and employment information
- `OCCUPATION` - Job titles and roles
- `LOCATION` - Where someone lives or is based
- `EDUCATION` - Educational background
- `SKILL` - Technical and professional skills
- `PREFERENCE` - Likes, dislikes, and preferences
- `GOAL` - Objectives and plans
- `HABIT` - Recurring behaviors
- `OPINION` - Viewpoints and beliefs
- `ATTRIBUTE` - Personal attributes
- `POSSESSION` - Things owned
- `OTHER` - Other factual information

### Basic Usage

```python
from hippocampai import MemoryClient

client = MemoryClient.from_preset("local")

# Extract facts from text
text = "John works at Google in San Francisco. He studied Computer Science at MIT."
facts = client.extract_facts(text, source="profile")

for fact in facts:
    print(f"[{fact.category.value}] {fact.fact}")
    print(f"  Confidence: {fact.confidence:.2f}")
    print(f"  Entities: {fact.entities}")
    print(f"  Temporal: {fact.temporal} ({fact.temporal_type.value})")
```

Output:
```
[employment] works at Google
  Confidence: 0.85
  Entities: ['Google']
  Temporal: None (present)

[location] in San Francisco
  Confidence: 0.85
  Entities: ['San Francisco']
  Temporal: None (present)

[education] studied Computer Science at MIT
  Confidence: 0.85
  Entities: ['MIT']
  Temporal: None (past)
```

### Extract Facts from Conversations

```python
conversation = """
User: I'm a software engineer at Tesla
Assistant: That's great! How long have you been there?
User: About 2 years now. I'm working on the autopilot team.
"""

facts = client.extract_facts_from_conversation(conversation, user_id="alice")

for fact in facts:
    print(f"{fact.category.value}: {fact.fact}")
```

## Entity Recognition

Identify and track entities (people, organizations, locations, etc.) across your memories.

### Supported Entity Types

- `PERSON` - People's names
- `ORGANIZATION` - Companies, institutions, groups
- `LOCATION` - Cities, states, countries, places
- `DATE` - Dates and time references
- `TIME` - Time expressions
- `MONEY` - Monetary amounts
- `PRODUCT` - Product names
- `EVENT` - Events and occurrences
- `SKILL` - Programming languages, technologies, skills
- `TOPIC` - Subject areas and topics
- `OTHER` - Other entities

### Basic Usage

```python
# Extract entities
text = "Elon Musk founded SpaceX in California in 2002"
entities = client.extract_entities(text)

for entity in entities:
    print(f"{entity.type.value}: {entity.text} (ID: {entity.entity_id})")
```

### Entity Profiles and Tracking

Entities are automatically tracked across mentions, building a complete profile:

```python
# Extract and add to knowledge graph
entities = client.extract_entities("Tim Cook is the CEO of Apple")

for entity in entities:
    # Add to graph
    node_id = client.add_entity_to_graph(entity)

    # Get profile
    profile = client.get_entity_profile(entity.entity_id)

    print(f"Entity: {profile.canonical_name}")
    print(f"Type: {profile.type.value}")
    print(f"Aliases: {profile.aliases}")
    print(f"First seen: {profile.first_seen}")
    print(f"Mention count: {profile.mention_count}")
```

### Search Entities

```python
# Search for people named "John"
results = client.search_entities("john", entity_type=EntityType.PERSON)

# Search for frequently mentioned entities
results = client.search_entities("google", min_mentions=5)

for profile in results:
    print(f"{profile.canonical_name} - {profile.mention_count} mentions")
```

### Extract Relationships

```python
text = "Steve Jobs worked at Apple and lived in California"
relationships = client.extract_relationships(text)

for rel in relationships:
    print(f"{rel.relation_type.value}: {rel.from_entity_id} -> {rel.to_entity_id}")
    print(f"  Confidence: {rel.confidence}")
    print(f"  Context: {rel.context}")
```

## Session Summarization

Generate comprehensive summaries of conversations with automatic extraction of key points, action items, topics, and sentiment.

### Summary Styles

- `CONCISE` - Brief 2-3 sentence summary
- `DETAILED` - Detailed 1-2 paragraph summary
- `BULLET_POINTS` - Summary as bullet points
- `NARRATIVE` - Story-like narrative summary
- `EXECUTIVE` - Executive summary highlighting decisions and outcomes

### Basic Usage

```python
from hippocampai.pipeline import SummaryStyle

messages = [
    {"role": "user", "content": "I need help with Python programming"},
    {"role": "assistant", "content": "I'd be happy to help! What specifically do you need?"},
    {"role": "user", "content": "I want to learn about data structures"},
    {"role": "assistant", "content": "Great! Let's start with lists and dictionaries"},
]

summary = client.summarize_conversation(
    messages,
    session_id="session_123",
    style=SummaryStyle.BULLET_POINTS
)

print("Summary:", summary.summary)
print("\nKey Points:")
for point in summary.key_points:
    print(f"  • {point}")

print("\nTopics:", summary.topics)
print("Sentiment:", summary.sentiment.value)
print("Action Items:", summary.action_items)
print("Questions asked:", summary.questions_asked)
print("Questions answered:", summary.questions_answered)
```

### Rolling Summaries

Create summaries of recent message windows:

```python
# Get summary of last 10 messages
summary = client.create_rolling_summary(messages, window_size=10)
print(summary)
```

### Extract Insights

Extract decisions, learning points, and patterns from conversations:

```python
insights = client.extract_conversation_insights(messages, user_id="alice")

print("Topics:", insights["topics"])
print("Sentiment:", insights["sentiment"])
print("\nKey Decisions:")
for decision in insights["key_decisions"]:
    print(f"  • {decision}")

print("\nLearning Points:")
for learning in insights["learning_points"]:
    print(f"  • {learning}")
```

## Knowledge Graph

Build a rich knowledge graph connecting memories, entities, facts, and topics.

### Graph Structure

The knowledge graph contains four types of nodes:

1. **Memory nodes** - Your stored memories
2. **Entity nodes** - Recognized entities (people, places, organizations)
3. **Fact nodes** - Extracted structured facts
4. **Topic nodes** - Topics and themes

These nodes are connected by typed relationships (e.g., `RELATED_TO`, `PART_OF`, `SUPPORTS`).

### Building the Graph

```python
# Create a memory
memory = client.remember(
    "Marie Curie was a physicist who worked in France and won two Nobel Prizes",
    user_id="alice"
)

# Enrich memory with intelligence (automatic extraction + graph building)
enrichment = client.enrich_memory_with_intelligence(memory, add_to_graph=True)

print(f"Extracted {len(enrichment['facts'])} facts")
print(f"Extracted {len(enrichment['entities'])} entities")
print(f"Extracted {len(enrichment['relationships'])} relationships")
```

### Manual Graph Building

For more control, you can manually extract and add to the graph:

```python
# Extract entities
entities = client.extract_entities(memory.text)

# Extract facts
facts = client.extract_facts(memory.text)

# Add to graph
for entity in entities:
    # Add entity node
    entity_node = client.add_entity_to_graph(entity)

    # Link memory to entity
    client.link_memory_to_entity(memory.id, entity.entity_id)

for fact in facts:
    # Add fact node
    fact_id = f"fact_{hash(fact.fact) % 10**10}"
    fact_node = client.add_fact_to_graph(fact, fact_id)

    # Link memory to fact
    client.link_memory_to_fact(memory.id, fact_id)

    # Link facts to entities they mention
    for entity_name in fact.entities:
        for entity in entities:
            if entity.text.lower() == entity_name.lower():
                client.link_fact_to_entity(fact_id, entity.entity_id)
```

### Query the Graph

#### Get Entity Memories

Find all memories that mention a specific entity:

```python
memory_ids = client.get_entity_memories("person_marie_curie")

for memory_id in memory_ids:
    # Fetch and process the memory
    pass
```

#### Get Entity Facts

Get all facts about an entity:

```python
fact_ids = client.get_entity_facts("person_marie_curie")
print(f"Found {len(fact_ids)} facts about Marie Curie")
```

#### Entity Timeline

Get a chronological timeline of facts and memories about an entity:

```python
timeline = client.get_entity_timeline("person_marie_curie")

for event in timeline:
    print(f"{event['timestamp']}: {event['type']}")
    if event['type'] == 'fact':
        print(f"  {event['text']}")
```

#### Find Connected Entities

Discover entities connected to a given entity:

```python
connections = client.get_entity_connections(
    "person_marie_curie",
    max_distance=2
)

for relation_type, connected in connections.items():
    print(f"{relation_type}:")
    for entity_id, distance in connected:
        print(f"  {entity_id} (distance: {distance})")
```

#### Get Subgraphs

Extract a subgraph around a node:

```python
subgraph = client.get_knowledge_subgraph(
    "person_marie_curie",
    radius=2,
    include_types=["entity", "fact"]
)

print(f"Nodes: {len(subgraph['nodes'])}")
print(f"Edges: {len(subgraph['edges'])}")

# Nodes have full data
for node in subgraph['nodes']:
    print(f"  {node['id']} - {node['node_type']}")

# Edges show relationships
for edge in subgraph['edges']:
    print(f"  {edge['source']} -> {edge['target']}")
```

### Knowledge Inference

Infer new facts from existing knowledge graph patterns:

```python
# Add some related information
client.remember("Alice works at Google", "alice")
client.remember("Google is located in California", "alice")

# Enrich both memories
# ... (enrich code)

# Infer new facts
inferred = client.infer_knowledge(user_id="alice")

for fact in inferred:
    print(f"{fact['fact']} (confidence: {fact['confidence']:.2f})")
    print(f"  Rule: {fact['rule']}")
    print(f"  Supporting facts: {fact['supporting_facts']}")
```

## Complete Example

Here's a complete example showcasing all intelligence features:

```python
from hippocampai import MemoryClient
from hippocampai.pipeline import SummaryStyle, EntityType

# Initialize client
client = MemoryClient.from_preset("local")

# Sample conversation
conversation = """
User: I just started working at SpaceX as a software engineer
Assistant: Congratulations! That's exciting. What will you be working on?
User: I'll be working on the Starship navigation systems. I studied aerospace engineering at MIT.
Assistant: That's a perfect background for the role!
User: Yes, I'm really excited. I'll be based in the Boca Chica facility in Texas.
"""

# 1. Extract and store facts
facts = client.extract_facts_from_conversation(conversation, user_id="alice")
print(f"\n=== Extracted {len(facts)} Facts ===")
for fact in facts:
    print(f"[{fact.category.value}] {fact.fact}")

# 2. Extract entities
entities = client.extract_entities(conversation)
print(f"\n=== Extracted {len(entities)} Entities ===")
for entity in entities:
    print(f"{entity.type.value}: {entity.text}")

# 3. Extract relationships
relationships = client.extract_relationships(conversation, entities)
print(f"\n=== Extracted {len(relationships)} Relationships ===")
for rel in relationships:
    print(f"{rel.relation_type.value}: {rel.from_entity_id} -> {rel.to_entity_id}")

# 4. Generate summary
messages = [
    {"role": "user", "content": line.replace("User: ", "")}
    if line.startswith("User: ")
    else {"role": "assistant", "content": line.replace("Assistant: ", "")}
    for line in conversation.strip().split("\n")
    if line.strip()
]

summary = client.summarize_conversation(
    messages,
    session_id="onboarding_chat",
    style=SummaryStyle.DETAILED
)

print(f"\n=== Summary ===")
print(summary.summary)
print(f"\nTopics: {', '.join(summary.topics)}")
print(f"Sentiment: {summary.sentiment.value}")

# 5. Store as memory and enrich with knowledge graph
memory = client.remember(conversation, user_id="alice", session_id="onboarding_chat")
enrichment = client.enrich_memory_with_intelligence(memory, add_to_graph=True)

print(f"\n=== Knowledge Graph Enrichment ===")
print(f"Added {len(enrichment['entities'])} entities to graph")
print(f"Added {len(enrichment['facts'])} facts to graph")
print(f"Added {len(enrichment['relationships'])} relationships to graph")

# 6. Query the knowledge graph
# Find all SpaceX-related entities
spacex_results = client.search_entities("spacex", entity_type=EntityType.ORGANIZATION)

if spacex_results:
    spacex_entity = spacex_results[0]

    # Get all memories mentioning SpaceX
    spacex_memories = client.get_entity_memories(spacex_entity.entity_id)
    print(f"\n=== Memories about SpaceX: {len(spacex_memories)} ===")

    # Get facts about SpaceX
    spacex_facts = client.get_entity_facts(spacex_entity.entity_id)
    print(f"Facts about SpaceX: {len(spacex_facts)}")

    # Get connected entities
    connections = client.get_entity_connections(spacex_entity.entity_id, max_distance=2)
    print(f"Connected entities: {sum(len(v) for v in connections.values())}")

    # Get entity timeline
    timeline = client.get_entity_timeline(spacex_entity.entity_id)
    print(f"Timeline events: {len(timeline)}")

# 7. Infer new knowledge
inferred = client.infer_knowledge(user_id="alice")
if inferred:
    print(f"\n=== Inferred Facts: {len(inferred)} ===")
    for fact in inferred[:3]:
        print(f"• {fact['fact']} (confidence: {fact['confidence']:.2f})")

print("\n=== Complete! ===")
```

## Best Practices

### 1. Enrich Memories Automatically

Set up automatic enrichment for all important memories:

```python
def store_and_enrich(text, user_id, session_id=None):
    """Store memory and automatically enrich with intelligence."""
    memory = client.remember(text, user_id, session_id=session_id)
    enrichment = client.enrich_memory_with_intelligence(memory, add_to_graph=True)
    return memory, enrichment
```

### 2. Batch Processing

For existing conversations, process in batches:

```python
for conversation in historical_conversations:
    facts = client.extract_facts_from_conversation(conversation, user_id)
    entities = client.extract_entities(conversation)

    # Store and enrich
    memory = client.remember(conversation, user_id)
    client.enrich_memory_with_intelligence(memory, add_to_graph=True)
```

### 3. Regular Summarization

Summarize sessions regularly:

```python
# At the end of each session
summary = client.summarize_conversation(
    session_messages,
    session_id=current_session_id,
    style=SummaryStyle.EXECUTIVE
)

# Store summary as metadata
client.update_session(
    current_session_id,
    summary=summary.summary,
    metadata={
        "key_points": summary.key_points,
        "topics": summary.topics,
        "sentiment": summary.sentiment.value,
    }
)
```

### 4. Entity-Centric Retrieval

Use entities to improve retrieval:

```python
# Find entity in user's question
entities = client.extract_entities(user_question)

if entities:
    # Get memories related to those entities
    all_memories = []
    for entity in entities:
        client.add_entity_to_graph(entity)
        memory_ids = client.get_entity_memories(entity.entity_id)
        all_memories.extend(memory_ids)

    # Use for context
    context = gather_memories(all_memories)
```

### 5. Knowledge Graph Exploration

Build interactive exploration tools:

```python
def explore_entity(entity_id, depth=2):
    """Explore an entity's knowledge graph neighborhood."""
    subgraph = client.get_knowledge_subgraph(entity_id, radius=depth)
    connections = client.get_entity_connections(entity_id, max_distance=depth)
    timeline = client.get_entity_timeline(entity_id)

    return {
        "subgraph": subgraph,
        "connections": connections,
        "timeline": timeline,
    }
```

## Performance Considerations

### Pattern-Based vs LLM-Based Extraction

- **Pattern-based**: Fast, works without LLM, good for common patterns
- **LLM-based**: More accurate, handles complex cases, requires LLM

The system automatically uses both when LLM is available, falling back to pattern-based when not.

### Graph Size

The knowledge graph grows with usage. Consider:

- Periodic cleanup of old/unused entities
- Merging duplicate entities
- Pruning low-confidence connections

### Memory vs Computation

- **Extract on write**: Slower writes, faster reads
- **Extract on demand**: Faster writes, slower first access
- **Hybrid**: Extract critical info on write, detailed analysis on demand

## Advanced Topics

### Custom Fact Patterns

Extend the fact extraction with custom patterns:

```python
# Access the fact extractor
fact_extractor = client.fact_extractor

# Add custom pattern
custom_pattern = {
    "pattern": r'\b(vegetarian|vegan|pescatarian)\b',
    "extract": lambda m: f"dietary preference: {m.group(1)}",
    "entity_group": 1
}

fact_extractor.fact_patterns[FactCategory.PREFERENCE].append(custom_pattern)
```

### Custom Entity Types

Add domain-specific entity types:

```python
from hippocampai.pipeline import EntityType

# Extend EntityType enum (requires modification of the enum class)
# Or use OTHER type with custom metadata

entity = Entity(
    text="GPT-4",
    type=EntityType.PRODUCT,
    confidence=0.95,
    entity_id="product_gpt4",
    canonical_name="GPT-4",
    metadata={"category": "AI Model"}
)
```

### Graph Analytics

Analyze your knowledge graph:

```python
# Most connected entities
entity_profiles = client.entity_recognizer.entities.values()
sorted_by_mentions = sorted(
    entity_profiles,
    key=lambda p: p.mention_count,
    reverse=True
)

print("Top 10 Most Mentioned Entities:")
for i, profile in enumerate(sorted_by_mentions[:10], 1):
    print(f"{i}. {profile.canonical_name} - {profile.mention_count} mentions")
```

## See Also

- [API Reference](API_REFERENCE.md) - Complete API documentation
- [Examples](../examples/) - Working code examples
- [Features](FEATURES.md) - Overview of all features
