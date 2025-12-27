"""Synthetic data generator for benchmarks.

Generates realistic memory data for benchmarking without
requiring external datasets.
"""

import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Iterator

# Sample data pools for generating realistic content
NAMES = [
    "Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Henry",
    "Ivy", "Jack", "Kate", "Leo", "Maya", "Noah", "Olivia", "Peter",
]

COMPANIES = [
    "Acme Corp", "TechStart", "DataFlow", "CloudNine", "InnovateLabs",
    "FutureTech", "SmartSystems", "GlobalNet", "CyberDyne", "QuantumLeap",
]

LOCATIONS = [
    "New York", "San Francisco", "London", "Tokyo", "Berlin", "Paris",
    "Sydney", "Toronto", "Singapore", "Amsterdam", "Seattle", "Austin",
]

SKILLS = [
    "Python", "JavaScript", "Machine Learning", "Data Science", "DevOps",
    "Cloud Architecture", "Product Management", "UX Design", "SQL",
    "Kubernetes", "React", "Node.js", "Go", "Rust", "TypeScript",
]

TOPICS = [
    "project deadline", "meeting notes", "customer feedback", "bug report",
    "feature request", "team update", "quarterly review", "product launch",
    "technical discussion", "brainstorming session", "code review",
]

PREFERENCES = [
    "prefers dark mode", "likes detailed explanations", "wants concise answers",
    "prefers visual diagrams", "likes code examples", "wants step-by-step guides",
    "prefers morning meetings", "likes async communication", "wants weekly updates",
]


@dataclass
class SyntheticMemory:
    """A synthetic memory for benchmarking."""

    id: str
    user_id: str
    content: str
    memory_type: str
    importance: float
    created_at: datetime
    metadata: dict = field(default_factory=dict)


def generate_fact_content() -> str:
    """Generate a realistic fact content."""
    templates = [
        "{name} works at {company} as a {role}.",
        "{name} is based in {location}.",
        "{name} has expertise in {skill}.",
        "{name} joined {company} in {year}.",
        "{name} manages a team of {count} people.",
        "{name} is working on {topic}.",
        "The {topic} is scheduled for {date}.",
        "{company} is headquartered in {location}.",
        "{name} previously worked at {company}.",
        "{name} holds a degree in {field}.",
    ]

    template = random.choice(templates)
    return template.format(
        name=random.choice(NAMES),
        company=random.choice(COMPANIES),
        location=random.choice(LOCATIONS),
        skill=random.choice(SKILLS),
        role=random.choice(["Engineer", "Manager", "Director", "VP", "Analyst"]),
        year=random.randint(2015, 2024),
        count=random.randint(3, 20),
        topic=random.choice(TOPICS),
        date=f"{random.randint(1, 28)} {random.choice(['Jan', 'Feb', 'Mar', 'Apr'])} 2025",
        field=random.choice(["Computer Science", "Engineering", "Business", "Design"]),
    )


def generate_preference_content() -> str:
    """Generate a realistic preference content."""
    templates = [
        "User {pref}.",
        "The user has indicated they {pref}.",
        "Preference noted: {pref}.",
        "User setting: {pref}.",
    ]

    template = random.choice(templates)
    return template.format(pref=random.choice(PREFERENCES))


def generate_event_content() -> str:
    """Generate a realistic event content."""
    templates = [
        "Meeting with {name} about {topic}.",
        "Discussed {topic} with the team.",
        "Completed {topic} successfully.",
        "Started working on {topic}.",
        "Received feedback on {topic}.",
        "Presented {topic} to stakeholders.",
        "Reviewed {topic} with {name}.",
    ]

    template = random.choice(templates)
    return template.format(
        name=random.choice(NAMES),
        topic=random.choice(TOPICS),
    )


def generate_memory(
    user_id: str,
    memory_type: str | None = None,
    base_time: datetime | None = None,
) -> SyntheticMemory:
    """Generate a single synthetic memory.

    Args:
        user_id: User ID for the memory
        memory_type: Type of memory (random if None)
        base_time: Base time for created_at (now if None)

    Returns:
        Generated SyntheticMemory
    """
    if memory_type is None:
        memory_type = random.choice(["fact", "preference", "event"])

    if base_time is None:
        base_time = datetime.now(timezone.utc)

    # Generate content based on type
    if memory_type == "fact":
        content = generate_fact_content()
    elif memory_type == "preference":
        content = generate_preference_content()
    else:
        content = generate_event_content()

    # Random time offset (up to 90 days in the past)
    time_offset = timedelta(
        days=random.randint(0, 90),
        hours=random.randint(0, 23),
        minutes=random.randint(0, 59),
    )

    return SyntheticMemory(
        id=str(uuid.uuid4()),
        user_id=user_id,
        content=content,
        memory_type=memory_type,
        importance=round(random.uniform(1, 10), 1),
        created_at=base_time - time_offset,
        metadata={
            "source": random.choice(["user_input", "inferred", "imported"]),
            "confidence": round(random.uniform(0.7, 1.0), 2),
        },
    )


def generate_memories(
    count: int,
    user_id: str | None = None,
    num_users: int = 1,
) -> Iterator[SyntheticMemory]:
    """Generate multiple synthetic memories.

    Args:
        count: Number of memories to generate
        user_id: Fixed user ID (generates random if None and num_users > 1)
        num_users: Number of distinct users to distribute memories across

    Yields:
        Generated SyntheticMemory objects
    """
    user_ids = (
        [user_id] if user_id
        else [f"bench_user_{i}" for i in range(num_users)]
    )

    base_time = datetime.now(timezone.utc)

    for i in range(count):
        uid = user_ids[i % len(user_ids)]
        yield generate_memory(uid, base_time=base_time)


def generate_queries(count: int) -> list[str]:
    """Generate realistic search queries.

    Args:
        count: Number of queries to generate

    Returns:
        List of query strings
    """
    query_templates = [
        "What does {name} work on?",
        "Tell me about {name}'s role",
        "Where is {name} located?",
        "What are {name}'s skills?",
        "Information about {company}",
        "Who works at {company}?",
        "Recent meetings about {topic}",
        "User preferences for {pref_topic}",
        "What happened with {topic}?",
        "Details about the {topic}",
        "{name}'s background",
        "Team members in {location}",
    ]

    queries = []
    for _ in range(count):
        template = random.choice(query_templates)
        query = template.format(
            name=random.choice(NAMES),
            company=random.choice(COMPANIES),
            topic=random.choice(TOPICS),
            location=random.choice(LOCATIONS),
            pref_topic=random.choice(["communication", "meetings", "interface"]),
        )
        queries.append(query)

    return queries


def generate_bitemporal_facts(
    count: int,
    user_id: str,
) -> list[dict]:
    """Generate bi-temporal facts for benchmarking.

    Args:
        count: Number of facts to generate
        user_id: User ID for the facts

    Returns:
        List of fact dictionaries
    """
    facts = []
    base_time = datetime.now(timezone.utc)

    for i in range(count):
        # Random validity period
        valid_from = base_time - timedelta(days=random.randint(30, 365))
        valid_to = (
            valid_from + timedelta(days=random.randint(30, 180))
            if random.random() > 0.3 else None
        )

        facts.append({
            "user_id": user_id,
            "subject": random.choice(NAMES),
            "predicate": random.choice([
                "works_at", "lives_in", "has_skill", "manages", "reports_to"
            ]),
            "object_value": random.choice(COMPANIES + LOCATIONS + SKILLS),
            "valid_from": valid_from,
            "valid_to": valid_to,
            "confidence": round(random.uniform(0.7, 1.0), 2),
            "source": random.choice(["user_stated", "inferred", "imported"]),
        })

    return facts
