#!/bin/bash

echo "🔄 Restarting HippocampAI Backend with WebSocket..."
echo ""

# Kill old backend on port 8000
echo "Killing old backend on port 8000..."
lsof -ti:8000 | xargs kill -9 2>/dev/null
sleep 1

# Verify port is free
if lsof -i:8000 > /dev/null 2>&1; then
    echo "❌ Port 8000 still in use. Please manually kill the process:"
    lsof -i:8000
    exit 1
fi

echo "✅ Port 8000 is now free"
echo ""

# Start backend with socket_app
echo "Starting backend with WebSocket support..."
echo "Command: python3 -m uvicorn hippocampai.api.app:socket_app --host 0.0.0.0 --port 8000 --reload"
echo ""

python3 -m uvicorn hippocampai.api.app:socket_app --host 0.0.0.0 --port 8000 --reload
