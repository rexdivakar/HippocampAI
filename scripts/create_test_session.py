#!/usr/bin/env python3
"""Create a test session with synthetic chat data."""

import os
from datetime import datetime, timezone
from uuid import uuid4

# Set up the environment
os.environ.setdefault('QDRANT_URL', 'http://100.113.229.40:6333')

from hippocampai.vector.qdrant_store import QdrantStore
from hippocampai.embed.embedder import get_embedder

# Initialize - use environment variable
qdrant_url = os.getenv('QDRANT_URL', 'http://100.113.229.40:6333')
store = QdrantStore(url=qdrant_url)
embedder = get_embedder()

# The specific session ID requested
session_id = '234ccf7f-7eea-40fe-88bc-1c57dea88bac'
user_id = f'user_{session_id[:8]}'

print(f'Creating session with:')
print(f'  session_id: {session_id}')
print(f'  user_id: {user_id}')

# Create welcome memory
now = datetime.now(timezone.utc)
welcome_text = f'Welcome to HippocampAI! Your session was created on {now.strftime("%Y-%m-%d %H:%M:%S UTC")}.'
vector = embedder.encode_single(welcome_text)
memory_id = str(uuid4())

payload = {
    'id': memory_id,
    'user_id': user_id,
    'session_id': session_id,
    'text': welcome_text,
    'type': 'fact',
    'importance': 5.0,
    'tags': ['welcome', 'session'],
    'created_at': now.isoformat(),
    'updated_at': now.isoformat(),
    'access_count': 0,
    'metadata': {
        'session_id': session_id,
        'username': 'TestUser',
        'signup_date': now.isoformat(),
    },
}

store.upsert(
    collection_name=store.collection_facts,
    id=memory_id,
    vector=vector,
    payload=payload,
)

print(f'✓ Session created successfully!')
print(f'  Memory ID: {memory_id}')

# Now create synthetic chat data
synthetic_conversations = [
    {
        'user': 'My name is Alex and I work as a software engineer at TechCorp.',
        'assistant': 'Nice to meet you, Alex! Being a software engineer at TechCorp sounds exciting. What kind of projects do you work on?',
        'type': 'fact',
        'importance': 8.0,
        'tags': ['identity', 'work', 'introduction'],
    },
    {
        'user': 'I love Python programming and machine learning.',
        'assistant': 'Python is a great choice for ML! Are you working on any specific machine learning projects at TechCorp?',
        'type': 'preference',
        'importance': 7.5,
        'tags': ['preference', 'programming', 'ml'],
    },
    {
        'user': 'I want to learn more about vector databases and RAG systems.',
        'assistant': 'Vector databases are fascinating! They are essential for RAG (Retrieval Augmented Generation) systems. Would you like me to explain how they work?',
        'type': 'goal',
        'importance': 7.0,
        'tags': ['goal', 'learning', 'technology'],
    },
    {
        'user': 'I usually wake up at 6 AM and code for 2 hours before work.',
        'assistant': 'That is an impressive morning routine! Early morning coding sessions can be very productive. What do you typically work on during those hours?',
        'type': 'habit',
        'importance': 6.5,
        'tags': ['habit', 'routine', 'productivity'],
    },
    {
        'user': 'Yesterday I had a meeting with the AI team about our new chatbot project.',
        'assistant': 'That sounds like an exciting project! How did the meeting go? Are you using any specific AI frameworks for the chatbot?',
        'type': 'event',
        'importance': 6.0,
        'tags': ['event', 'meeting', 'project'],
    },
    {
        'user': 'I prefer dark mode in all my applications.',
        'assistant': 'Dark mode is easier on the eyes, especially during those early morning coding sessions! Most modern IDEs and apps support it now.',
        'type': 'preference',
        'importance': 5.5,
        'tags': ['preference', 'ui', 'settings'],
    },
    {
        'user': 'My goal is to build an AI assistant that can remember conversations.',
        'assistant': 'That is exactly what HippocampAI helps with! Persistent memory for AI assistants is a game-changer for user experience.',
        'type': 'goal',
        'importance': 8.5,
        'tags': ['goal', 'ai', 'memory', 'project'],
    },
    {
        'user': 'I drink about 4 cups of coffee every day.',
        'assistant': 'Coffee is a programmer essential! Just make sure to stay hydrated with water too. Do you have a favorite coffee blend?',
        'type': 'habit',
        'importance': 5.0,
        'tags': ['habit', 'coffee', 'lifestyle'],
    },
    {
        'user': 'Last week I deployed my first production ML model.',
        'assistant': 'Congratulations! Deploying your first production ML model is a huge milestone. What kind of model was it?',
        'type': 'event',
        'importance': 8.0,
        'tags': ['event', 'achievement', 'ml', 'deployment'],
    },
    {
        'user': 'I live in San Francisco and enjoy hiking on weekends.',
        'assistant': 'San Francisco has amazing hiking trails! The Bay Area offers great outdoor activities. Do you have a favorite trail?',
        'type': 'fact',
        'importance': 6.5,
        'tags': ['fact', 'location', 'hobby'],
    },
]

print(f'\nCreating {len(synthetic_conversations)} synthetic conversation memories...')

for i, conv in enumerate(synthetic_conversations):
    # Create conversation memory
    conv_text = f"User: {conv['user']}\nAssistant: {conv['assistant']}"
    vector = embedder.encode_single(conv_text)
    memory_id = str(uuid4())
    
    payload = {
        'id': memory_id,
        'user_id': user_id,
        'session_id': session_id,
        'text': conv_text,
        'type': conv['type'],
        'importance': conv['importance'],
        'tags': conv['tags'],
        'created_at': now.isoformat(),
        'updated_at': now.isoformat(),
        'access_count': i,  # Simulate different access counts
        'metadata': {
            'session_id': session_id,
            'user_message': conv['user'],
            'assistant_message': conv['assistant'],
            'timestamp': now.isoformat(),
        },
    }
    
    store.upsert(
        collection_name=store.collection_facts,
        id=memory_id,
        vector=vector,
        payload=payload,
    )
    
    print(f'  ✓ Created memory {i+1}: {conv["type"]} - {conv["user"][:50]}...')

print(f'\n✅ All done! Created session and {len(synthetic_conversations)} memories.')
print(f'\nTo test in UI, use:')
print(f'  Session ID: {session_id}')
print(f'  User ID: {user_id}')
