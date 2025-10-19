"""Compatibility bootstrap for HippocampAI.

This script allows ``python setup.py`` to run the interactive setup routine
without introducing side effects during packaging (`pip install -e .`, builds,
etc.).  When invoked directly it imports the real implementation from
``hippocampai.setup_app``.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _load_run_initial_setup():
    """Import the setup helper after ensuring ``src`` is importable."""
    project_root = Path(__file__).resolve().parent
    src_dir = project_root / "src"
    if src_dir.exists():
        sys.path.insert(0, str(src_dir))

    from hippocampai.setup_app import run_initial_setup

    return run_initial_setup


def main() -> int:
    run_initial_setup = _load_run_initial_setup()
    return run_initial_setup()


if __name__ == "__main__":
    setup_commands = {"egg_info", "develop", "build", "sdist", "bdist_wheel"}
    if any(arg in setup_commands for arg in sys.argv[1:]):
        # Allow packaging tool invocations to proceed without triggering the setup.
        sys.exit(0)

    sys.exit(main())
