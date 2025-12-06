"""Test the _get_user_id_from_session function directly."""

import sys
import os
sys.path.insert(0, "/Users/rexdivakar/workspace/HippocampAI/src")

# Set environment variable
os.environ["QDRANT_URL"] = "http://100.113.229.40:6333"

from hippocampai.api.auth_routes import _get_user_id_from_session

session_id = "234ccf7f-7eea-40fe-88bc-1c57dea88ba2"

print(f"Testing _get_user_id_from_session with: {session_id}\n")
result = _get_user_id_from_session(session_id)

print(f"Result: {result}")
print(f"\nExpected: 0177aa41-efb9-4b24-b423-ffd78f61521c")
print(f"Match: {result == '0177aa41-efb9-4b24-b423-ffd78f61521c'}")
