#!/bin/bash

# HippocampAI Example Runner Script

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print functions
print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo
}

print_step() {
    echo -e "${GREEN}▶${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

# Check if .env exists
check_env() {
    if [ ! -f .env ]; then
        print_error ".env file not found"
        print_step "Creating from template..."
        cp .env .env
        print_warning "Please edit .env and add your ANTHROPIC_API_KEY"
        exit 1
    fi
}

# Check if API key is set
check_api_key() {
    if ! grep -q "ANTHROPIC_API_KEY=sk-" .env 2>/dev/null; then
        print_warning "ANTHROPIC_API_KEY not configured in .env"
        print_step "Add your API key to .env file"
        return 1
    fi
    return 0
}

# Check if Qdrant is running
check_qdrant() {
    print_step "Checking Qdrant connection..."

    # Get host and port from .env
    QDRANT_HOST=$(grep QDRANT_HOST .env | cut -d '=' -f2)
    QDRANT_PORT=$(grep QDRANT_PORT .env | cut -d '=' -f2)

    if [ -z "$QDRANT_HOST" ]; then
        QDRANT_HOST="localhost"
    fi
    if [ -z "$QDRANT_PORT" ]; then
        QDRANT_PORT="6334"
    fi

    # Try to connect
    if curl -s "http://${QDRANT_HOST}:${QDRANT_PORT}/collections" >/dev/null 2>&1; then
        print_success "Qdrant is running at ${QDRANT_HOST}:${QDRANT_PORT}"
        return 0
    else
        print_error "Cannot connect to Qdrant at ${QDRANT_HOST}:${QDRANT_PORT}"
        print_step "Start Qdrant with: docker run -p 6334:6333 qdrant/qdrant"
        return 1
    fi
}

# Run example
run_example() {
    local example_file=$1
    local example_name=$2

    print_header "$example_name"

    if [ ! -f "$example_file" ]; then
        print_error "Example file not found: $example_file"
        return 1
    fi

    print_step "Running $example_file..."
    echo

    # Set PYTHONPATH to include project root
    PYTHONPATH="$(pwd):$PYTHONPATH" python "$example_file"

    echo
    print_success "Example completed!"
    echo
}

# Show menu
show_menu() {
    print_header "HippocampAI Example Runner"

    echo "Available examples:"
    echo
    echo "  1) Basic Memory Storage"
    echo "  2) Memory Extraction (requires API key)"
    echo "  3) Memory Retrieval & Search"
    echo "  4) Advanced Features (deduplication, updates)"
    echo "  5) Smart Retrieval & Sessions (requires API key)"
    echo "  6) All Examples (in sequence)"
    echo "  7) Setup & Configuration Test"
    echo "  0) Exit"
    echo
}

# Run all examples
run_all() {
    print_header "Running All Examples"

    local has_api_key=0
    check_api_key && has_api_key=1

    run_example "examples/embedding_example.py" "1. Embedding Service"
    sleep 2

    run_example "examples/memory_store_example.py" "2. Memory Storage"
    sleep 2

    run_example "examples/memory_retriever_example.py" "3. Memory Retrieval"
    sleep 2

    if [ $has_api_key -eq 1 ]; then
        run_example "examples/memory_extractor_example.py" "4. Memory Extraction (AI)"
        sleep 2

        run_example "examples/advanced_memory_example.py" "5. Advanced Features (AI)"
        sleep 2

        run_example "examples/smart_retrieval_example.py" "6. Smart Retrieval (AI)"
    else
        print_warning "Skipping AI-powered examples (no API key)"
    fi

    print_success "All examples completed!"
}

# Main script
main() {
    # Check prerequisites
    check_env

    if [ $# -eq 0 ]; then
        # Interactive mode
        while true; do
            show_menu
            read -p "Select example (0-7): " choice

            case $choice in
                1)
                    run_example "examples/memory_store_example.py" "Basic Memory Storage"
                    ;;
                2)
                    if check_api_key; then
                        run_example "examples/memory_extractor_example.py" "Memory Extraction"
                    fi
                    ;;
                3)
                    run_example "examples/memory_retriever_example.py" "Memory Retrieval"
                    ;;
                4)
                    if check_api_key; then
                        run_example "examples/advanced_memory_example.py" "Advanced Features"
                    fi
                    ;;
                5)
                    if check_api_key; then
                        run_example "examples/smart_retrieval_example.py" "Smart Retrieval & Sessions"
                    fi
                    ;;
                6)
                    check_qdrant || exit 1
                    run_all
                    ;;
                7)
                    print_header "Setup & Configuration Test"
                    PYTHONPATH="$(pwd):$PYTHONPATH" python setup.py
                    ;;
                0)
                    print_step "Goodbye!"
                    exit 0
                    ;;
                *)
                    print_error "Invalid choice"
                    ;;
            esac

            read -p "Press Enter to continue..."
        done
    else
        # Command line mode
        case $1 in
            --all)
                check_qdrant || exit 1
                run_all
                ;;
            --setup)
                PYTHONPATH="$(pwd):$PYTHONPATH" python setup.py
                ;;
            --check)
                check_qdrant
                check_api_key
                ;;
            *)
                echo "Usage: $0 [--all|--setup|--check]"
                echo "  --all    Run all examples"
                echo "  --setup  Run setup script"
                echo "  --check  Check prerequisites"
                echo "  (no args) Interactive mode"
                exit 1
                ;;
        esac
    fi
}

# Run main
main "$@"
