"""Interactive setup script for HippocampAI.

This script runs the interactive setup routine to initialize the HippocampAI
environment, create .env files, and set up Qdrant collections.

Note: This file was renamed from setup.py to avoid conflicts with pip/setuptools
during package installation. The package build configuration is in pyproject.toml.
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    """Import the setup helper after ensuring ``src`` is importable."""
    project_root = Path(__file__).resolve().parent
    src_dir = project_root / "src"
    if src_dir.exists():
        sys.path.insert(0, str(src_dir))

    from hippocampai.setup_app import run_initial_setup

    return run_initial_setup()


if __name__ == "__main__":
    # Only run the interactive setup when invoked directly, not during packaging
    sys.exit(main())
