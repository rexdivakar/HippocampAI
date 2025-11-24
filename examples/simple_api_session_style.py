"""
Example: Using HippocampAI with zep-compatible Session API

This example shows how to manage conversations using the
Session API that's compatible with zep patterns.
"""

from hippocampai import SimpleSession as Session

print("=== HippocampAI Session API (zep-compatible) ===\n")

# Create a session
print("1. Creating session...")
session = Session(session_id="customer_support_123", user_id="customer_456")
print(f"   Session created: {session.session_id}\n")

# Simulate a customer support conversation
print("2. Adding conversation messages...")
session.add_message("user", "Hi, I need help with my order")
print("   User: Hi, I need help with my order")

session.add_message("assistant", "Hello! I'd be happy to help. What's your order number?")
print("   Assistant: Hello! I'd be happy to help. What's your order number?")

session.add_message("user", "It's ORDER-12345")
print("   User: It's ORDER-12345")

session.add_message("assistant", "Thank you! Let me look that up for you...")
print("   Assistant: Thank you! Let me look that up for you...")

session.add_message("user", "I ordered on Monday and still haven't received it")
print("   User: I ordered on Monday and still haven't received it\n")

# Get all messages in the session
print("3. Retrieving conversation history...")
messages = session.get_messages()
print(f"   Total messages in session: {len(messages)}\n")

# Search within the conversation
print("4. Searching conversation...")
results = session.search("order number")
if results:
    print(f"   Found {len(results)} relevant messages:")
    for i, result in enumerate(results[:3], 1):
        print(f"   {i}. {result.memory.text}")
print()

# Get conversation summary
print("5. Getting conversation summary...")
try:
    summary = session.get_summary()
    print(f"   Summary: {summary}\n")
except Exception as e:
    print(f"   Summary not available (requires LLM): {e}\n")

# Alternative: Create a different session
print("6. Creating another session (multi-session support)...")
session2 = Session(session_id="tech_support_789", user_id="customer_456")
session2.add_message("user", "My internet is slow")
session2.add_message("assistant", "Let's troubleshoot that together")

messages2 = session2.get_messages()
print(f"   Session 2 messages: {len(messages2)}\n")

# Sessions are isolated
print("7. Verifying session isolation...")
print(f"   Session 1: {len(session.get_messages())} messages")
print(f"   Session 2: {len(session2.get_messages())} messages")
print("   ✅ Sessions are properly isolated\n")

# Clear a session
print("8. Clearing session 2...")
count = session2.clear()
print(f"   Cleared {count} messages from session 2\n")

print("✅ Session API demo complete!")
print("\n💡 This API is compatible with zep - easy migration!")
