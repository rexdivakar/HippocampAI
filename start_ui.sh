#!/bin/bash

echo "🚀 Starting HippocampAI UI with WebSocket support..."
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if Qdrant is running
echo "Checking Qdrant..."
if curl -s http://localhost:6333/collections > /dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} Qdrant is running on port 6333"
else
    echo -e "${YELLOW}⚠${NC}  Qdrant not detected on port 6333"
    echo "   You may need to start Qdrant:"
    echo "   docker run -p 6333:6333 qdrant/qdrant"
    echo ""
fi

# Check if python-socketio is installed
echo "Checking python-socketio..."
if python3 -c "import socketio" 2>/dev/null; then
    echo -e "${GREEN}✓${NC} python-socketio is installed"
else
    echo -e "${YELLOW}⚠${NC}  python-socketio not found, installing..."
    pip install python-socketio
fi

echo ""
echo "Starting backend with WebSocket support..."
echo "Command: python3 -m uvicorn hippocampai.api.app:socket_app --host 0.0.0.0 --port 8000 --reload"
echo ""

# Start backend with socket_app (important!)
python3 -m uvicorn hippocampai.api.app:socket_app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!

# Wait for backend to start
echo "Waiting for backend to start..."
sleep 3

# Check if backend is running
if curl -s http://localhost:8000/healthz > /dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} Backend is running on http://localhost:8000"
else
    echo -e "${RED}✗${NC} Backend failed to start"
    kill $BACKEND_PID 2>/dev/null
    exit 1
fi

echo ""
echo "Starting frontend..."
cd frontend

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "Installing frontend dependencies..."
    npm install
fi

npm run dev &
FRONTEND_PID=$!

echo ""
echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✅ HippocampAI UI is running!${NC}"
echo ""
echo "   🔧 Backend:  http://localhost:8000"
echo "   🎨 Frontend: http://localhost:3000"
echo "   📚 API Docs: http://localhost:8000/docs"
echo ""
echo "   The WebSocket indicator should show ${GREEN}Live${NC} (green) in the UI"
echo ""
echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
echo ""
echo "Press Ctrl+C to stop both servers"
echo ""

# Wait for Ctrl+C
trap "echo ''; echo 'Stopping servers...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; echo 'Stopped.'; exit" INT TERM
wait
