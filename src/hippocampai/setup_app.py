"""Utilities for running the interactive HippocampAI setup routine.

This module exposes `run_initial_setup`, which mirrors the behaviour of the
legacy `setup.py` bootstrap script without executing any side effects at import
time.  The logic here can be invoked explicitly from the CLI or other tooling
without impacting package installation.
"""

from __future__ import annotations

import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Callable


def print_header(text: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {text}")
    print(f"{'=' * 60}\n")


def print_step(step: int, total: int, text: str) -> None:
    print(f"[{step}/{total}] {text}...")


def print_success(text: str) -> None:
    print(f"✓ {text}")


def print_error(text: str) -> None:
    print(f"✗ {text}", file=sys.stderr)


def print_warning(text: str) -> None:
    print(f"⚠ {text}")


def check_python_version() -> bool:
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print_error(f"Python 3.8+ required (current: {version.major}.{version.minor})")
        return False

    print_success(f"Python {version.major}.{version.minor}.{version.micro}")
    return True


def check_dependencies() -> bool:
    required_packages = [
        "qdrant_client",
        "sentence_transformers",
        "anthropic",
        "yaml",
    ]

    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print_success(f"{package} installed")
        except ImportError:
            missing.append(package)
            print_error(f"{package} not installed")

    if missing:
        print_warning(
            f"\nMissing packages: {', '.join(missing)}\nRun: pip install -r requirements.txt"
        )
        return False

    return True


def create_directories() -> bool:
    project_root = Path(__file__).resolve().parents[2]
    directories = [
        project_root / "logs",
        project_root / "data",
        project_root / "backups",
    ]

    for directory in directories:
        try:
            directory.mkdir(parents=True, exist_ok=True)
            print_success(f"Created {directory.relative_to(project_root)}/ directory")
        except Exception as exc:  # noqa: BLE001
            print_error(f"Failed to create {directory}: {exc}")
            return False

    return True


def create_env_file() -> bool:
    project_root = Path(__file__).resolve().parents[2]
    env_example = project_root / ".env.example"
    env_file = project_root / ".env"

    if env_file.exists():
        print_warning(".env file already exists, skipping")
        return True

    if not env_example.exists():
        print_error(".env.example not found")
        return False

    try:
        env_file.write_text(env_example.read_text())
        print_success("Created .env file from template")
        print_warning("Please edit .env and add your API key (e.g. ANTHROPIC_API_KEY)")
        return True
    except Exception as exc:  # noqa: BLE001
        print_error(f"Failed to create .env file: {exc}")
        return False


def validate_configuration() -> bool:
    try:
        from hippocampai.config import get_config

        config = get_config()
        print_success("Configuration validated")
        print(f"  Qdrant URL: {config.qdrant_url}")
        print(f"  Collections: {config.collection_facts}, {config.collection_prefs}")
        print(f"  Embedding model: {config.embed_model}")
        print(f"  LLM provider: {config.llm_provider}")
        return True
    except Exception as exc:  # noqa: BLE001
        print_error(f"Configuration validation failed: {exc}")
        return False


def test_qdrant_connection() -> bool:
    try:
        from qdrant_client import QdrantClient

        from hippocampai.config import get_config

        config = get_config()
        client = QdrantClient(url=config.qdrant_url)
        client.get_collections()
        print_success(f"Connected to Qdrant at {config.qdrant_url}")
        return True
    except Exception as exc:  # noqa: BLE001
        print_error(f"Qdrant connection failed: {exc}")
        print_warning("Make sure Qdrant is running and accessible")
        return False


def initialize_collections() -> bool:
    try:
        from hippocampai.config import get_config
        from hippocampai.vector.qdrant_store import QdrantStore

        config = get_config()
        store = QdrantStore(
            url=config.qdrant_url,
            collection_facts=config.collection_facts,
            collection_prefs=config.collection_prefs,
            dimension=config.embed_dimension,
            hnsw_m=config.hnsw_m,
            ef_construction=config.ef_construction,
            ef_search=config.ef_search,
        )

        collections_response = store.client.get_collections()
        collections: Iterable[str] = []
        if hasattr(collections_response, "collections"):
            collections = [
                getattr(item, "name", str(item)) for item in collections_response.collections
            ]

        print_success("Initialized Qdrant collections")
        for name in collections:
            print(f"  - {name}")
        return True
    except Exception as exc:  # noqa: BLE001
        print_error(f"Failed to initialize collections: {exc}")
        return False


def run_initial_setup(config_path: str | None = None) -> int:
    """Run the full setup routine. Returns an exit code."""
    print_header("HippocampAI Memory Assistant - Setup")

    steps: list[tuple[str, Callable[[], bool]]] = [
        ("Checking Python version", check_python_version),
        ("Checking dependencies", check_dependencies),
        ("Creating directories", create_directories),
        ("Creating .env file", create_env_file),
        ("Validating configuration", validate_configuration),
        ("Testing Qdrant connection", test_qdrant_connection),
        ("Initializing collections", initialize_collections),
    ]

    failed_steps: list[str] = []
    total = len(steps)

    for index, (name, func) in enumerate(steps, start=1):
        print_step(index, total, name)
        try:
            if not func():
                failed_steps.append(name)
        except Exception as exc:  # noqa: BLE001
            print_error(f"Unexpected error: {exc}")
            failed_steps.append(name)

    print_header("Setup Summary")

    if not failed_steps:
        print_success("Setup completed successfully!")
        print("\nNext steps:")
        print("  1. Edit .env and add your API key (e.g. ANTHROPIC_API_KEY)")
        print("  2. Set LLM_PROVIDER in .env if needed (anthropic, openai, groq, ollama)")
        print("  3. Review config.yaml for additional options")
        print("  4. Run examples: ./run_example.sh or python examples/memory_store_example.py\n")
        return 0

    print_error(f"Setup failed ({len(failed_steps)} steps failed)")
    print("\nFailed steps:")
    for step in failed_steps:
        print(f"  - {step}")
    print("\nPlease fix the errors and run setup again.\n")
    return 1


if __name__ == "__main__":
    sys.exit(run_initial_setup())
