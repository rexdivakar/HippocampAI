"""WebSocket support for real-time updates."""

import logging
from datetime import datetime
from typing import Any, Optional

import socketio

logger = logging.getLogger(__name__)

# Create Socket.IO server
sio = socketio.AsyncServer(
    async_mode="asgi",
    cors_allowed_origins="*",
    logger=True,
    engineio_logger=False,
)

# Store connected clients and their subscriptions
# Format: {sid: {"user_id": str, "spaces": set[str], "agents": set[str]}}
connected_clients: dict[str, dict[str, Any]] = {}


# ============================================================================
# CONNECTION HANDLERS
# ============================================================================

@sio.event
async def connect(sid: str, environ: dict, auth: Optional[dict] = None):
    """Handle client connection."""
    logger.info(f"Client connected: {sid}")

    # Initialize client data
    connected_clients[sid] = {
        "user_id": auth.get("user_id") if auth else None,
        "spaces": set(),
        "agents": set(),
        "connected_at": datetime.now(),
    }

    await sio.emit("connected", {"sid": sid, "timestamp": datetime.now().isoformat()}, room=sid)


@sio.event
async def disconnect(sid: str):
    """Handle client disconnection."""
    logger.info(f"Client disconnected: {sid}")
    if sid in connected_clients:
        del connected_clients[sid]


# ============================================================================
# SUBSCRIPTION HANDLERS
# ============================================================================

@sio.event
async def subscribe_user(sid: str, data: dict):
    """Subscribe to updates for a specific user."""
    user_id = data.get("user_id")
    if not user_id:
        await sio.emit("error", {"message": "user_id required"}, room=sid)
        return

    if sid in connected_clients:
        connected_clients[sid]["user_id"] = user_id
        await sio.enter_room(sid, f"user:{user_id}")
        logger.info(f"Client {sid} subscribed to user {user_id}")
        await sio.emit("subscribed", {"type": "user", "id": user_id}, room=sid)


@sio.event
async def subscribe_space(sid: str, data: dict):
    """Subscribe to updates for a specific collaboration space."""
    space_id = data.get("space_id")
    if not space_id:
        await sio.emit("error", {"message": "space_id required"}, room=sid)
        return

    if sid in connected_clients:
        connected_clients[sid]["spaces"].add(space_id)
        await sio.enter_room(sid, f"space:{space_id}")
        logger.info(f"Client {sid} subscribed to space {space_id}")
        await sio.emit("subscribed", {"type": "space", "id": space_id}, room=sid)


@sio.event
async def subscribe_agent(sid: str, data: dict):
    """Subscribe to updates for a specific agent."""
    agent_id = data.get("agent_id")
    if not agent_id:
        await sio.emit("error", {"message": "agent_id required"}, room=sid)
        return

    if sid in connected_clients:
        connected_clients[sid]["agents"].add(agent_id)
        await sio.enter_room(sid, f"agent:{agent_id}")
        logger.info(f"Client {sid} subscribed to agent {agent_id}")
        await sio.emit("subscribed", {"type": "agent", "id": agent_id}, room=sid)


@sio.event
async def unsubscribe(sid: str, data: dict):
    """Unsubscribe from a room."""
    sub_type = data.get("type")  # "user", "space", or "agent"
    sub_id = data.get("id")

    if not sub_type or not sub_id:
        await sio.emit("error", {"message": "type and id required"}, room=sid)
        return

    room = f"{sub_type}:{sub_id}"
    await sio.leave_room(sid, room)

    # Update client data
    if sid in connected_clients:
        if sub_type == "user":
            connected_clients[sid]["user_id"] = None
        elif sub_type == "space":
            connected_clients[sid]["spaces"].discard(sub_id)
        elif sub_type == "agent":
            connected_clients[sid]["agents"].discard(sub_id)

    logger.info(f"Client {sid} unsubscribed from {room}")
    await sio.emit("unsubscribed", {"type": sub_type, "id": sub_id}, room=sid)


# ============================================================================
# BROADCAST FUNCTIONS
# ============================================================================

async def broadcast_memory_created(user_id: str, memory: dict):
    """Broadcast when a new memory is created."""
    await sio.emit(
        "memory:created",
        {
            "memory": memory,
            "timestamp": datetime.now().isoformat(),
        },
        room=f"user:{user_id}",
    )
    logger.debug(f"Broadcast memory:created to user:{user_id}")


async def broadcast_memory_updated(user_id: str, memory: dict):
    """Broadcast when a memory is updated."""
    await sio.emit(
        "memory:updated",
        {
            "memory": memory,
            "timestamp": datetime.now().isoformat(),
        },
        room=f"user:{user_id}",
    )
    logger.debug(f"Broadcast memory:updated to user:{user_id}")


async def broadcast_memory_deleted(user_id: str, memory_id: str):
    """Broadcast when a memory is deleted."""
    await sio.emit(
        "memory:deleted",
        {
            "memory_id": memory_id,
            "timestamp": datetime.now().isoformat(),
        },
        room=f"user:{user_id}",
    )
    logger.debug(f"Broadcast memory:deleted to user:{user_id}")


async def broadcast_collaboration_event(space_id: str, event: dict):
    """Broadcast collaboration events."""
    await sio.emit(
        "collaboration:event",
        {
            "event": event,
            "timestamp": datetime.now().isoformat(),
        },
        room=f"space:{space_id}",
    )
    logger.debug(f"Broadcast collaboration:event to space:{space_id}")


async def broadcast_agent_notification(agent_id: str, notification: dict):
    """Broadcast notification to an agent."""
    await sio.emit(
        "agent:notification",
        {
            "notification": notification,
            "timestamp": datetime.now().isoformat(),
        },
        room=f"agent:{agent_id}",
    )
    logger.debug(f"Broadcast agent:notification to agent:{agent_id}")


async def broadcast_health_alert(user_id: str, alert: dict):
    """Broadcast health alerts."""
    await sio.emit(
        "health:alert",
        {
            "alert": alert,
            "timestamp": datetime.now().isoformat(),
        },
        room=f"user:{user_id}",
    )
    logger.debug(f"Broadcast health:alert to user:{user_id}")


async def broadcast_pattern_detected(user_id: str, pattern: dict):
    """Broadcast when a new pattern is detected."""
    await sio.emit(
        "prediction:pattern",
        {
            "pattern": pattern,
            "timestamp": datetime.now().isoformat(),
        },
        room=f"user:{user_id}",
    )
    logger.debug(f"Broadcast prediction:pattern to user:{user_id}")


async def broadcast_anomaly_detected(user_id: str, anomaly: dict):
    """Broadcast when an anomaly is detected."""
    await sio.emit(
        "prediction:anomaly",
        {
            "anomaly": anomaly,
            "timestamp": datetime.now().isoformat(),
        },
        room=f"user:{user_id}",
    )
    logger.debug(f"Broadcast prediction:anomaly to user:{user_id}")


async def broadcast_healing_action(user_id: str, action: dict):
    """Broadcast when a healing action is performed."""
    await sio.emit(
        "healing:action",
        {
            "action": action,
            "timestamp": datetime.now().isoformat(),
        },
        room=f"user:{user_id}",
    )
    logger.debug(f"Broadcast healing:action to user:{user_id}")


async def broadcast_health_score_changed(user_id: str, old_score: float, new_score: float):
    """Broadcast when health score changes significantly."""
    await sio.emit(
        "health:score_changed",
        {
            "old_score": old_score,
            "new_score": new_score,
            "change": new_score - old_score,
            "timestamp": datetime.now().isoformat(),
        },
        room=f"user:{user_id}",
    )
    logger.debug(f"Broadcast health:score_changed to user:{user_id}")


# ============================================================================
# STATUS & DIAGNOSTICS
# ============================================================================

@sio.event
async def get_status(sid: str):
    """Get connection status and subscriptions."""
    if sid in connected_clients:
        client = connected_clients[sid]
        await sio.emit(
            "status",
            {
                "connected": True,
                "user_id": client["user_id"],
                "spaces": list(client["spaces"]),
                "agents": list(client["agents"]),
                "connected_at": client["connected_at"].isoformat(),
            },
            room=sid,
        )
    else:
        await sio.emit("status", {"connected": False}, room=sid)


@sio.event
async def ping(sid: str):
    """Ping-pong for connection health check."""
    await sio.emit("pong", {"timestamp": datetime.now().isoformat()}, room=sid)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_connected_clients_count() -> int:
    """Get the number of connected clients."""
    return len(connected_clients)


def get_user_connections(user_id: str) -> list[str]:
    """Get all connection IDs for a specific user."""
    return [
        sid for sid, client in connected_clients.items()
        if client.get("user_id") == user_id
    ]


def get_space_connections(space_id: str) -> list[str]:
    """Get all connection IDs subscribed to a space."""
    return [
        sid for sid, client in connected_clients.items()
        if space_id in client.get("spaces", set())
    ]


def get_agent_connections(agent_id: str) -> list[str]:
    """Get all connection IDs subscribed to an agent."""
    return [
        sid for sid, client in connected_clients.items()
        if agent_id in client.get("agents", set())
    ]
