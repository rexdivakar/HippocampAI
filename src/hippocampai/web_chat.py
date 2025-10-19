"""Flask web server for memory-enhanced chat interface.

This provides a REST API for the chat frontend.

Endpoints:
    POST /api/chat/message - Send message and get response
    GET /api/chat/history - Get conversation history
    GET /api/chat/memories - Get user's stored memories
    GET /api/chat/stats - Get memory statistics
    POST /api/chat/clear - Clear current session
    POST /api/chat/end - End session with summary

Usage:
    python -m hippocampai.web_chat

    Then open: http://localhost:5020
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict

from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

from hippocampai.ai_chat import MemoryEnhancedChat
from hippocampai.settings import get_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Determine template/static directory inside the package
_APP_ROOT = Path(__file__).resolve().parent
_WEB_ROOT = _APP_ROOT / "web"

# Initialize Flask app with package-relative assets
app = Flask(
    __name__,
    template_folder=str(_WEB_ROOT),
    static_folder=str(_WEB_ROOT / "static") if (_WEB_ROOT / "static").exists() else None,
)
CORS(app)  # Enable CORS for API calls

# Store chat instances per user
# In production, use Redis or similar for session storage
chat_sessions: Dict[str, MemoryEnhancedChat] = {}


def get_or_create_chat(user_id: str) -> MemoryEnhancedChat:
    """Get existing chat session or create new one."""
    if user_id not in chat_sessions:
        logger.info(f"Creating new chat session for user: {user_id}")
        chat_sessions[user_id] = MemoryEnhancedChat(
            user_id=user_id, auto_extract_memories=True, auto_consolidate=True
        )
    return chat_sessions[user_id]


@app.route("/")
def index():
    """Serve the chat interface."""
    return render_template("chat.html")


@app.route("/api/chat/message", methods=["POST"])
def send_message():
    """
    Send a message and get AI response.

    Request JSON:
        {
            "user_id": "alice",
            "message": "Hello!",
            "context_type": "personal"  // optional
        }

    Response JSON:
        {
            "success": true,
            "response": "AI response here",
            "timestamp": "2025-10-05T15:30:00"
        }
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({"success": False, "error": "No data provided"}), 400

        user_id = data.get("user_id", "default_user")
        message = data.get("message", "").strip()
        context_type = data.get("context_type")

        if not message:
            return jsonify({"success": False, "error": "Message cannot be empty"}), 400

        # Get or create chat session
        chat = get_or_create_chat(user_id)

        # Send message and get response
        response = chat.send_message(message, context_type=context_type)

        return jsonify(
            {
                "success": True,
                "response": response,
                "timestamp": datetime.now().isoformat(),
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
            "history": [
                {
                    "role": "user",
                    "content": "Hello",
                    "timestamp": "..."
                },
                ...
            ]
        }
    """
    try:
        user_id = request.args.get("user_id", "default_user")

        if user_id not in chat_sessions:
            return jsonify({"success": True, "history": []})

        chat = chat_sessions[user_id]
        history = chat.get_conversation_history()

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
        type: Memory type filter (optional)
        limit: Max results (default: 20)

    Response JSON:
        {
            "success": true,
            "memories": [...]
        }
    """
    try:
        user_id = request.args.get("user_id", "default_user")
        memory_type = request.args.get("type")
        limit = int(request.args.get("limit", 20))

        chat = get_or_create_chat(user_id)
        memories = chat.get_user_memories(memory_type=memory_type, limit=limit)

        return jsonify({"success": True, "memories": memories, "count": len(memories)})

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
            "stats": {
                "total_memories": 42,
                "by_type": {...},
                "by_category": {...},
                ...
            }
        }
    """
    try:
        user_id = request.args.get("user_id", "default_user")

        chat = get_or_create_chat(user_id)
        stats = chat.get_memory_stats()

        return jsonify({"success": True, "stats": stats})

    except Exception as e:
        logger.error(f"Error in get_stats: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/chat/clear", methods=["POST"])
def clear_session():
    """
    Clear current session without saving.

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

        if user_id in chat_sessions:
            chat_sessions[user_id].clear_session()

        return jsonify({"success": True, "message": "Session cleared"})

    except Exception as e:
        logger.error(f"Error in clear_session: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/chat/end", methods=["POST"])
def end_session():
    """
    End session and save summary.

    Request JSON:
        {
            "user_id": "alice"
        }

    Response JSON:
        {
            "success": true,
            "summary_id": "uuid-here",
            "message": "Session ended and summary saved"
        }
    """
    try:
        data = request.get_json()
        user_id = data.get("user_id", "default_user")

        summary_id = None
        if user_id in chat_sessions:
            summary_id = chat_sessions[user_id].end_conversation()
            # Don't delete session, just clear it
            chat_sessions[user_id].clear_session()

        return jsonify(
            {
                "success": True,
                "summary_id": summary_id,
                "message": "Session ended and summary saved" if summary_id else "No active session",
            }
        )

    except Exception as e:
        logger.error(f"Error in end_session: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    try:
        settings = get_settings()
        return jsonify(
            {
                "status": "healthy",
                "provider": settings.llm.provider,
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


def main():
    """Run the web server."""
    # Check API key
    settings = get_settings()
    provider = settings.llm.provider.lower()

    has_api_key = False
    if provider == "anthropic" and settings.llm.anthropic_api_key:
        has_api_key = True
    elif provider == "openai" and settings.llm.openai_api_key:
        has_api_key = True
    elif provider == "groq" and settings.llm.groq_api_key:
        has_api_key = True

    if not has_api_key:
        print(f"ERROR: {provider.upper()}_API_KEY not set in .env file!")
        print("Please add your API key to continue.")
        sys.exit(1)

    # Print startup info
    print("=" * 60)
    print("  HippocampAI Memory-Enhanced Chat Server")
    print("=" * 60)
    print(f"Provider: {provider}")
    print(f"Qdrant: {settings.qdrant.host}:{settings.qdrant.port}")
    print("\nServer starting at: http://localhost:5020")
    print("\nEndpoints:")
    print("  - Web Interface: http://localhost:5020/")
    print("  - API Docs: See hippocampai.web_chat docstrings")
    print("\nPress Ctrl+C to stop")
    print("=" * 60)
    print()

    # Run Flask app
    app.run(
        host="0.0.0.0",
        port=5020,
        debug=True,
        use_reloader=False,  # Disable reloader to prevent duplicate sessions
    )


if __name__ == "__main__":
    main()
