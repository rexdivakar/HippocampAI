#!/bin/bash

# Run all HippocampAI examples

set -e

echo "=========================================="
echo "  Running HippocampAI Examples"
echo "=========================================="
echo ""

# Check if Qdrant is running
echo "Checking Qdrant connection..."
if ! curl -s http://localhost:6333/health > /dev/null 2>&1; then
    echo "ERROR: Qdrant is not running at http://localhost:6333"
    echo "Please start Qdrant first:"
    echo "  docker run -p 6333:6333 qdrant/qdrant"
    exit 1
fi
echo "✓ Qdrant is running"
echo ""

# Run examples
examples=(
    "01_basic_usage.py"
    "02_conversation_extraction.py"
    "03_hybrid_retrieval.py"
    "04_custom_configuration.py"
    "05_multi_user.py"
)

for example in "${examples[@]}"; do
    echo "=========================================="
    echo "  Running: $example"
    echo "=========================================="
    echo ""

    python "examples/$example"

    echo ""
    echo "Press Enter to continue to next example (or Ctrl+C to stop)..."
    read
    echo ""
done

echo "=========================================="
echo "  All Examples Completed!"
echo "=========================================="
