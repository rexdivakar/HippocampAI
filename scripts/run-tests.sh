#!/usr/bin/env bash

set -euo pipefail

# Ensure we run from repository root
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="${PYTHONPATH:-}:${REPO_ROOT}/src"

echo "Running unit tests with pytest..."
python -m pytest "$@"
