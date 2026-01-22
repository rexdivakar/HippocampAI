"""LLM prompt templates for memory consolidation."""

import json
from typing import Any

from hippocampai.models import Memory


def build_consolidation_prompt(
    memories: list[Memory],
    user_context: dict[str, Any] | None = None,
    cluster_theme: str | None = None,
) -> str:
    """
    Build the main LLM prompt for memory consolidation.

    This prompt instructs the LLM to act as a "sleep consolidation agent"
    that reviews recent memories and makes decisions about what to keep,
    promote, archive, or synthesize.

    Args:
        memories: List of Memory objects to review
        user_context: Optional context about the user (preferences, goals, etc.)
        cluster_theme: Optional theme/topic of this cluster

    Returns:
        Formatted prompt string
    """
    # Format memories as numbered list
    memory_list = []
    for idx, mem in enumerate(memories, 1):
        memory_list.append(
            f"{idx}. [{mem.type.upper()}] {mem.text[:150]}... "
            f"(importance: {mem.importance:.1f}, "
            f"tags: {', '.join(mem.tags) if mem.tags else 'none'}, "
            f"accessed: {mem.access_count} times, "
            f"id: {mem.id})"
        )

    memories_text = "\n".join(memory_list)

    # Format user context if provided
    context_section = ""
    if user_context:
        context_items = [f"- {k}: {v}" for k, v in user_context.items()]
        context_section = f"""
**User Context:**
{chr(10).join(context_items)}
"""

    # Build the main prompt
    prompt = f"""You are a memory consolidation agent performing a nightly "sleep phase" review.

Your task is to analyze the following {len(memories)} memories from the last 24 hours and make consolidation decisions.

{f"**Cluster Theme:** {cluster_theme}" if cluster_theme else ""}
{context_section}

**Memories to Review:**
{memories_text}

**Your Objectives:**
1. **Promote Important Facts**: Identify memories that contain valuable, long-term knowledge (strategic decisions, key learnings, important relationships, goals). These should have increased importance scores.

2. **Archive Low-Value Memories**: Flag memories that are transient, redundant, or trivial (e.g., "had coffee", routine tasks with no unique insight, duplicate information).

3. **Update/Merge Memories**: If multiple memories contain related information, suggest merging them or updating the text to be more concise and informative.

4. **Synthesize New Memories**: Create higher-level summary or insight memories that capture the essence of related events (e.g., "Productive morning: completed Q4 planning and strategy meeting").

**Decision Criteria:**
- **Delete/Archive** if:
  - Importance < 3.0
  - Never accessed (access_count = 0)
  - Purely transient/routine with no lasting value
  - Duplicate or redundant information

- **Promote** if:
  - Strategic importance (goals, decisions, relationships)
  - Novel insights or learnings
  - Emotional significance
  - Referenced frequently (access_count > 0)

- **Synthesize** when:
  - Multiple related memories form a coherent story
  - A higher-level insight emerges from the pattern
  - Summary would be more useful than individual events

**Output Format (Strict JSON):**
Return ONLY a valid JSON object with this structure:

```json
{{
  "promoted_facts": [
    {{
      "id": "memory-id-here",
      "reason": "Why this is important for long-term storage",
      "new_importance": 8.5
    }}
  ],
  "low_value_memory_ids": ["memory-id-1", "memory-id-2"],
  "updated_memories": [
    {{
      "id": "memory-id-here",
      "new_text": "Updated, more concise text",
      "new_importance": 7.0,
      "merge_from_ids": ["memory-id-x", "memory-id-y"]
    }}
  ],
  "synthetic_memories": [
    {{
      "text": "High-level summary or insight",
      "type": "context",
      "importance": 7.0,
      "tags": ["tag1", "tag2"],
      "source_ids": ["memory-id-1", "memory-id-2"]
    }}
  ],
  "reasoning": "Brief explanation of your consolidation strategy"
}}
```

**Important:**
- Return ONLY valid JSON, no markdown, no extra text
- Use existing memory IDs exactly as shown
- New importance scores must be between 0.0 and 10.0
- Be conservative: when in doubt, keep the memory
- Prioritize user's goals and preferences if provided in context

Begin your analysis now:
"""

    return prompt


def build_simple_review_prompt(memory_texts: list[str]) -> str:
    """
    Simplified prompt for quick batch review without full Memory objects.

    Args:
        memory_texts: List of memory text strings

    Returns:
        Simplified prompt
    """
    memories_text = "\n".join([f"{i + 1}. {text}" for i, text in enumerate(memory_texts)])

    return f"""You are reviewing these {len(memory_texts)} memories for consolidation.
Identify which memories are valuable (keep), which are low-value (archive), and any insights to synthesize.

**Memories:**
{memories_text}

Return JSON with:
- keep_indices: [list of indices to keep, 1-indexed]
- archive_indices: [list of indices to archive, 1-indexed]
- synthesis: "optional high-level insight or summary"

Example:
{{"keep_indices": [1, 3, 5], "archive_indices": [2, 4], "synthesis": "User focused on project planning today"}}

JSON:
"""


def build_cluster_theme_prompt(memories: list[Memory]) -> str:
    """
    Generate a prompt to ask LLM to identify the theme/topic of a memory cluster.

    Args:
        memories: List of memories in the cluster

    Returns:
        Prompt to generate cluster theme
    """
    memory_texts = [f"- {mem.text[:100]}" for mem in memories[:10]]  # Limit to first 10
    memories_sample = "\n".join(memory_texts)

    return f"""Analyze these {len(memories)} related memories and identify a concise theme or topic.

**Memories:**
{memories_sample}

Return a JSON object with:
- theme: A short phrase (2-5 words) describing the main topic
- tags: List of 3-5 relevant tags

Example:
{{"theme": "Morning work routine", "tags": ["work", "morning", "productivity", "planning"]}}

JSON:
"""


def build_synthetic_memory_prompt(
    source_memories: list[Memory],
    synthesis_goal: str = "summary",
) -> str:
    """
    Prompt to generate a synthetic memory from multiple sources.

    Args:
        source_memories: Memories to synthesize
        synthesis_goal: Type of synthesis ("summary", "insight", "pattern")

    Returns:
        Prompt for synthesis
    """
    memory_details = []
    for mem in source_memories:
        memory_details.append(f"- [{mem.type}] {mem.text} (importance: {mem.importance:.1f})")

    memories_text = "\n".join(memory_details)

    synthesis_instructions = {
        "summary": "Create a concise summary that captures the key points",
        "insight": "Extract a deeper insight or lesson learned",
        "pattern": "Identify a recurring pattern or habit",
    }

    instruction = synthesis_instructions.get(synthesis_goal, synthesis_instructions["summary"])

    return f"""Based on these {len(source_memories)} related memories, {instruction}:

**Source Memories:**
{memories_text}

Return JSON with:
- text: The synthesized memory text (1-2 sentences)
- type: Memory type ("context", "insight", or "habit")
- importance: Importance score (0-10)
- tags: Relevant tags (3-5 items)

Example:
{{
  "text": "Productive morning session: completed Q4 planning and strategic roadmap review, identified key priorities",
  "type": "context",
  "importance": 7.5,
  "tags": ["work", "planning", "Q4", "strategy", "productivity"]
}}

JSON:
"""


# System message for all consolidation prompts
CONSOLIDATION_SYSTEM_MESSAGE = """You are a memory consolidation AI that mimics the brain's hippocampal replay during sleep.

Your role is to:
1. Identify valuable long-term memories worth keeping
2. Filter out transient, low-value information
3. Synthesize higher-level insights from related memories
4. Maintain the user's knowledge base in a clean, organized state

Key principles:
- Be conservative: preserve potentially valuable information
- Be concise: consolidate redundant information
- Be insightful: find patterns and higher-level meanings
- Be respectful: honor the user's goals and preferences
- Be precise: always return valid, parseable JSON

You are running during the user's "sleep phase" - a nightly automated process.
"""


# Example user context for testing
EXAMPLE_USER_CONTEXT = {
    "name": "Alex",
    "role": "Product Manager",
    "current_goals": ["Launch Q4 product features", "Improve team collaboration"],
    "preferences": {
        "focus_areas": ["strategic planning", "team dynamics"],
        "interests": ["AI", "productivity tools"],
    },
    "recent_topics": ["Q4 roadmap", "hiring", "sprint planning"],
}


# Example memory data for testing prompts
EXAMPLE_MEMORIES_DATA = [
    {
        "id": "mem-1",
        "text": "Had morning coffee at 9am",
        "type": "event",
        "importance": 2.0,
        "tags": [],
        "access_count": 0,
    },
    {
        "id": "mem-2",
        "text": "Q4 roadmap meeting: discussed AI-powered features, analytics dashboard, mobile app improvements",
        "type": "event",
        "importance": 7.0,
        "tags": ["work", "Q4", "planning"],
        "access_count": 2,
    },
    {
        "id": "mem-3",
        "text": "Decision: prioritize AI features over mobile improvements for Q4",
        "type": "fact",
        "importance": 8.5,
        "tags": ["decision", "Q4", "AI"],
        "access_count": 1,
    },
    {
        "id": "mem-4",
        "text": "Lunch break at 12:30pm",
        "type": "event",
        "importance": 1.0,
        "tags": [],
        "access_count": 0,
    },
    {
        "id": "mem-5",
        "text": "Realized need to hire 2 more engineers for AI team",
        "type": "goal",
        "importance": 7.5,
        "tags": ["hiring", "AI", "team"],
        "access_count": 0,
    },
    {
        "id": "mem-6",
        "text": "Team standup: everyone on track with sprint goals",
        "type": "event",
        "importance": 4.0,
        "tags": ["team", "standup"],
        "access_count": 0,
    },
    {
        "id": "mem-7",
        "text": "Sprint retrospective: team wants better async communication tools",
        "type": "preference",
        "importance": 6.0,
        "tags": ["team", "feedback", "tools"],
        "access_count": 1,
    },
]


def get_example_consolidation_response() -> dict[str, Any]:
    """Example of expected LLM JSON response for testing/documentation."""
    return {
        "promoted_facts": [
            {
                "id": "mem-3",
                "reason": "Strategic product decision with long-term impact on Q4 roadmap",
                "new_importance": 9.0,
            }
        ],
        "low_value_memory_ids": ["mem-1", "mem-4"],
        "updated_memories": [
            {
                "id": "mem-2",
                "new_text": "Q4 roadmap planning: prioritizing AI features (analytics dashboard, intelligent automation) over mobile improvements",
                "new_importance": 8.0,
                "merge_from_ids": ["mem-3"],
            }
        ],
        "synthetic_memories": [
            {
                "text": "Productive Q4 planning day: finalized roadmap prioritizing AI features, identified hiring needs (2 engineers), gathered team feedback on communication tools",
                "type": "context",
                "importance": 7.5,
                "tags": ["work", "Q4", "planning", "AI", "hiring", "team"],
                "source_ids": ["mem-2", "mem-3", "mem-5", "mem-7"],
            }
        ],
        "reasoning": "Consolidated strategic planning activities into coherent narrative. Promoted key product decision. Archived low-value transient events (coffee, lunch). Preserved team feedback and hiring goals.",
    }


if __name__ == "__main__":
    # Demo: generate a sample prompt
    from hippocampai.models import Memory, MemoryType

    # Convert example data to Memory objects
    example_memories = [
        Memory(
            id=m["id"],
            text=m["text"],
            user_id="user-demo",
            type=MemoryType(m["type"]),
            importance=m["importance"],
            tags=m["tags"],
            access_count=m["access_count"],
        )
        for m in EXAMPLE_MEMORIES_DATA
    ]

    # Generate prompt
    prompt = build_consolidation_prompt(
        memories=example_memories,
        user_context=EXAMPLE_USER_CONTEXT,
        cluster_theme="Work and Planning",
    )

    print("=" * 80)
    print("CONSOLIDATION PROMPT EXAMPLE")
    print("=" * 80)
    print(prompt)
    print("\n" + "=" * 80)
    print("EXPECTED JSON RESPONSE")
    print("=" * 80)
    print(json.dumps(get_example_consolidation_response(), indent=2))
