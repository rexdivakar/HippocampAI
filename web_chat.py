"""Flask web server for memory-enhanced chat interface.

This provides a REST API for the chat frontend using HippocampAI.

Endpoints:
    POST /api/chat/message - Send message and get response
    GET /api/chat/history - Get conversation history
    GET /api/chat/memories - Get user's stored memories
    GET /api/chat/stats - Get memory statistics
    POST /api/chat/clear - Clear current session
    GET /api/health - Health check

Usage:
    python web_chat.py

    Then open: http://localhost:5000
"""

import logging
import sys
from datetime import datetime
from typing import Dict, List

from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

from hippocampai import MemoryClient
from hippocampai.config import get_config
from hippocampai.models.memory import RetrievalResult

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, template_folder="web", static_folder="web/static")
CORS(app)

# Store chat sessions and conversation history per user
chat_clients: Dict[str, MemoryClient] = {}
conversation_history: Dict[str, List[Dict]] = {}


def get_or_create_client(user_id: str) -> MemoryClient:
    """Get existing memory client or create new one."""
    if user_id not in chat_clients:
        logger.info(f"Creating new memory client for user: {user_id}")
        config = get_config()
        chat_clients[user_id] = MemoryClient(config=config)
        conversation_history[user_id] = []
    return chat_clients[user_id]


@app.route("/")
def index() -> str:
    """Serve the chat interface."""
    return render_template("chat.html")


@app.route("/api/chat/message", methods=["POST"])
def send_message():
    """
    Send a message and get AI response.

    Request JSON:
        {
            "user_id": "alice",
            "message": "Hello!"
        }

    Response JSON:
        {
            "success": true,
            "response": "AI response here",
            "timestamp": "2025-10-05T15:30:00",
            "memories_count": 5
        }
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({"success": False, "error": "No data provided"}), 400

        user_id = data.get("user_id", "default_user")
        message = data.get("message", "").strip()

        if not message:
            return jsonify({"success": False, "error": "Message cannot be empty"}), 400

        # Get or create memory client
        client = get_or_create_client(user_id)
        session_id = f"web_session_{user_id}"

        # Add to conversation history
        if user_id not in conversation_history:
            conversation_history[user_id] = []

        conversation_history[user_id].append(
            {"role": "user", "content": message, "timestamp": datetime.now().isoformat()}
        )

        # Extract memories from conversation
        if len(conversation_history[user_id]) >= 2:
            conv_text = "\n".join([msg["content"] for msg in conversation_history[user_id][-3:]])
            try:
                client.extract_from_conversation(conv_text, user_id, session_id)
            except Exception as e:
                logger.warning(f"Memory extraction failed: {e}")

        # Retrieve relevant memories
        memories: List[RetrievalResult] = client.recall(query=message, user_id=user_id, k=5)

        # Generate response
        if client.llm:
            # Build context from memories
            context = ""
            if memories:
                context = "Relevant memories:\n"
                for mem in memories[:3]:
                    context += f"- {mem.memory.text}\n"
                context += "\n"

            prompt = f"{context}User: {message}\n\nProvide a helpful, conversational response:"
            response = client.llm.generate(prompt, max_tokens=512, temperature=0.7)
        else:
            # Fallback response
            response = "I understand. I've noted that and will remember our conversation."
            if memories:
                response += f" I recall {len(memories)} related memories about you."

        # Add response to history
        conversation_history[user_id].append(
            {"role": "assistant", "content": response, "timestamp": datetime.now().isoformat()}
        )

        return jsonify(
            {
                "success": True,
                "response": response,
                "timestamp": datetime.now().isoformat(),
                "memories_count": len(memories),
            }
        )

    except Exception as e:
        logger.error(f"Error in send_message: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/chat/history", methods=["GET"])
def get_history():
    """
    Get conversation history for current session.

    Query params:
        user_id: User identifier

    Response JSON:
        {
            "success": true,
            "history": [...]
        }
    """
    try:
        user_id = request.args.get("user_id", "default_user")

        history = conversation_history.get(user_id, [])
        return jsonify({"success": True, "history": history})

    except Exception as e:
        logger.error(f"Error in get_history: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/chat/memories", methods=["GET"])
def get_memories():
    """
    Get user's stored memories.

    Query params:
        user_id: User identifier
        limit: Max results (default: 20)

    Response JSON:
        {
            "success": true,
            "memories": [...]
        }
    """
    try:
        user_id = request.args.get("user_id", "default_user")
        limit = int(request.args.get("limit", 20))

        client = get_or_create_client(user_id)

        # Get memories from both collections
        facts = client.qdrant.scroll(
            collection_name=client.config.collection_facts,
            filters={"user_id": user_id},
            limit=limit,
        )
        prefs = client.qdrant.scroll(
            collection_name=client.config.collection_prefs,
            filters={"user_id": user_id},
            limit=limit,
        )

        all_memories = []
        for mem in facts + prefs:
            payload = mem.get("payload", {})
            all_memories.append(
                {
                    "id": mem.get("id"),
                    "text": payload.get("text"),
                    "type": payload.get("type"),
                    "importance": payload.get("importance"),
                    "created_at": payload.get("created_at"),
                }
            )

        return jsonify({"success": True, "memories": all_memories, "count": len(all_memories)})

    except Exception as e:
        logger.error(f"Error in get_memories: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/chat/stats", methods=["GET"])
def get_stats():
    """
    Get memory statistics.

    Query params:
        user_id: User identifier

    Response JSON:
        {
            "success": true,
            "stats": {...}
        }
    """
    try:
        user_id = request.args.get("user_id", "default_user")
        client = get_or_create_client(user_id)

        # Get all memories
        facts = client.qdrant.scroll(
            collection_name=client.config.collection_facts, filters={"user_id": user_id}, limit=1000
        )
        prefs = client.qdrant.scroll(
            collection_name=client.config.collection_prefs, filters={"user_id": user_id}, limit=1000
        )

        all_memories = facts + prefs
        total = len(all_memories)

        # Calculate stats
        importances = [mem.get("payload", {}).get("importance", 5.0) for mem in all_memories]
        avg_importance = sum(importances) / len(importances) if importances else 0

        stats = {
            "total_memories": total,
            "facts_count": len(facts),
            "prefs_count": len(prefs),
            "avg_importance": round(avg_importance, 2),
        }

        return jsonify({"success": True, "stats": stats})

    except Exception as e:
        logger.error(f"Error in get_stats: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/chat/clear", methods=["POST"])
def clear_session():
    """
    Clear current session history.

    Request JSON:
        {
            "user_id": "alice"
        }

    Response JSON:
        {
            "success": true,
            "message": "Session cleared"
        }
    """
    try:
        data = request.get_json()
        user_id = data.get("user_id", "default_user")

        if user_id in conversation_history:
            conversation_history[user_id] = []

        return jsonify({"success": True, "message": "Session cleared"})

    except Exception as e:
        logger.error(f"Error in clear_session: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    try:
        config = get_config()
        return jsonify(
            {
                "status": "healthy",
                "provider": config.llm_provider,
                "qdrant_url": config.qdrant_url,
                "timestamp": datetime.now().isoformat(),
            }
        )
    except Exception as e:
        return jsonify({"status": "unhealthy", "error": str(e)}), 500


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({"success": False, "error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.error(f"Internal error: {error}", exc_info=True)
    return jsonify({"success": False, "error": "Internal server error"}), 500


def main() -> None:
    """Run the web server."""
    # Check configuration
    try:
        config = get_config()
    except Exception as e:
        print(f"ERROR: Failed to load configuration: {e}")
        print("Make sure .env file exists and is configured correctly")
        sys.exit(1)

    # Print startup info
    print("=" * 60)
    print("  HippocampAI Memory-Enhanced Chat Server")
    print("=" * 60)
    print(f"LLM Provider: {config.llm_provider}")
    print(f"Model: {config.llm_model}")
    print(f"Qdrant: {config.qdrant_url}")
    print("\nServer starting at: http://localhost:5000")
    print("\nEndpoints:")
    print("  - Web Interface: http://localhost:5000/")
    print("  - Health Check: http://localhost:5000/api/health")
    print("  - API Docs: See web_chat.py docstrings")
    print("\nPress Ctrl+C to stop")
    print("=" * 60)
    print()

    # Run Flask app
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True,
        use_reloader=False,
    )


if __name__ == "__main__":
    main()
