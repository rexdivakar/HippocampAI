"""Setup script for initializing HippocampAI memory assistant."""

import sys
import os
from pathlib import Path
import subprocess


def print_header(text: str) -> None:
    """Print a formatted header."""
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}\n")


def print_step(step: int, total: int, text: str) -> None:
    """Print a step indicator."""
    print(f"[{step}/{total}] {text}...")


def print_success(text: str) -> None:
    """Print success message."""
    print(f"✓ {text}")


def print_error(text: str) -> None:
    """Print error message."""
    print(f"✗ {text}", file=sys.stderr)


def print_warning(text: str) -> None:
    """Print warning message."""
    print(f"⚠ {text}")


def check_python_version() -> bool:
    """Check if Python version is 3.8 or higher."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print_error(f"Python 3.8+ required (current: {version.major}.{version.minor})")
        return False
    print_success(f"Python {version.major}.{version.minor}.{version.micro}")
    return True


def check_dependencies() -> bool:
    """Check if required Python packages are installed."""
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
        print_warning(f"\nMissing packages: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")
        return False

    return True


def create_directories() -> bool:
    """Create necessary directories."""
    project_root = Path(__file__).parent
    directories = [
        project_root / "logs",
        project_root / "data",
        project_root / "backups",
    ]

    for directory in directories:
        try:
            directory.mkdir(parents=True, exist_ok=True)
            print_success(f"Created {directory.name}/ directory")
        except Exception as e:
            print_error(f"Failed to create {directory}: {e}")
            return False

    return True


def create_env_file() -> bool:
    """Create .env file from .env if it doesn't exist."""
    project_root = Path(__file__).parent
    env_example = project_root / ".env"
    env_file = project_root / ".env"

    if env_file.exists():
        print_warning(".env file already exists, skipping")
        return True

    if not env_example.exists():
        print_error(".env not found")
        return False

    try:
        # Copy .env to .env
        with open(env_example, 'r') as src:
            content = src.read()

        with open(env_file, 'w') as dst:
            dst.write(content)

        print_success("Created .env file from template")
        print_warning("⚠ Please edit .env and add your ANTHROPIC_API_KEY")
        return True

    except Exception as e:
        print_error(f"Failed to create .env file: {e}")
        return False


def test_qdrant_connection() -> bool:
    """Test connection to Qdrant."""
    try:
        from src.settings import get_settings

        settings = get_settings()

        from qdrant_client import QdrantClient

        client = QdrantClient(
            host=settings.qdrant.host,
            port=settings.qdrant.port
        )

        # Test connection
        client.get_collections()

        print_success(f"Connected to Qdrant at {settings.qdrant.host}:{settings.qdrant.port}")
        return True

    except Exception as e:
        print_error(f"Qdrant connection failed: {e}")
        print_warning(f"Make sure Qdrant is running and accessible")
        return False


def initialize_collections() -> bool:
    """Initialize Qdrant collections."""
    try:
        from src.qdrant_client import QdrantManager
        from src.settings import get_settings

        settings = get_settings()

        manager = QdrantManager(
            host=settings.qdrant.host,
            port=settings.qdrant.port
        )

        manager.create_collections()
        print_success("Initialized Qdrant collections")

        # List collections
        collections = manager.list_collections()
        for coll in collections:
            print(f"  - {coll}")

        return True

    except Exception as e:
        print_error(f"Failed to initialize collections: {e}")
        return False


def validate_configuration() -> bool:
    """Validate configuration files."""
    try:
        from src.settings import get_settings

        settings = get_settings()

        # Print configuration summary
        print_success("Configuration validated")
        print(f"  Qdrant: {settings.qdrant.host}:{settings.qdrant.port}")
        print(f"  Embedding: {settings.embedding.model}")
        print(f"  LLM Provider: {settings.llm.provider}")
        print(f"  Log level: {settings.logging_config.level}")

        # Check API key for the configured provider
        provider = settings.llm.provider.lower()
        has_api_key = False

        if provider == "anthropic" and settings.llm.anthropic_api_key:
            has_api_key = True
        elif provider == "openai" and settings.llm.openai_api_key:
            has_api_key = True
        elif provider == "groq" and settings.llm.groq_api_key:
            has_api_key = True

        if not has_api_key:
            print_warning(f"{provider.upper()}_API_KEY not set - AI features will not work")
            print(f"  Add your API key to .env file")
            print(f"  Or change LLM_PROVIDER to a provider with a configured API key")

        return True

    except Exception as e:
        print_error(f"Configuration validation failed: {e}")
        return False


def main():
    """Run the setup process."""
    print_header("HippocampAI Memory Assistant - Setup")

    steps = [
        ("Checking Python version", check_python_version),
        ("Checking dependencies", check_dependencies),
        ("Creating directories", create_directories),
        ("Creating .env file", create_env_file),
        ("Validating configuration", validate_configuration),
        ("Testing Qdrant connection", test_qdrant_connection),
        ("Initializing collections", initialize_collections),
    ]

    total_steps = len(steps)
    failed_steps = []

    for i, (step_name, step_func) in enumerate(steps, 1):
        print_step(i, total_steps, step_name)

        try:
            if not step_func():
                failed_steps.append(step_name)
        except Exception as e:
            print_error(f"Unexpected error: {e}")
            failed_steps.append(step_name)

    # Summary
    print_header("Setup Summary")

    if not failed_steps:
        print_success("Setup completed successfully!")
        print("\nNext steps:")
        print("  1. Edit .env file and add your API key (ANTHROPIC_API_KEY, OPENAI_API_KEY, or GROQ_API_KEY)")
        print("  2. Set LLM_PROVIDER in .env (options: anthropic, openai, groq)")
        print("  3. Review config.yaml for customization")
        print("  4. Run examples: ./run_example.sh or python examples/memory_store_example.py")
        print("  5. Start using the memory assistant!\n")
        return 0
    else:
        print_error(f"Setup failed ({len(failed_steps)} steps failed)")
        print("\nFailed steps:")
        for step in failed_steps:
            print(f"  - {step}")
        print("\nPlease fix the errors and run setup again.\n")
        return 1


if __name__ == "__main__":
    setup_commands = {"egg_info", "develop", "build", "sdist", "bdist_wheel"}
    if any(arg in setup_commands for arg in sys.argv[1:]):
        # Allow packaging tools to probe without interactive setup.
        sys.exit(0)

    sys.exit(main())
