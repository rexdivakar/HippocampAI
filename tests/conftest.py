"""Test configuration for HippocampAI."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

# Ensure the project root does not shadow third-party packages like qdrant_client.
while str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))

if str(SRC) not in sys.path:
    sys.path.append(str(SRC))
