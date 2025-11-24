# Contributing to HippocampAI

Thank you for your interest in contributing to HippocampAI! We welcome contributions from the community.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Code Style](#code-style)
- [Submitting Changes](#submitting-changes)
- [Reporting Issues](#reporting-issues)
- [Feature Requests](#feature-requests)

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment for all contributors.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:

   ```bash
   git clone https://github.com/YOUR_USERNAME/HippocampAI.git
   cd HippocampAI
   ```

3. **Add upstream remote**:

   ```bash
   git remote add upstream https://github.com/rexdivakar/HippocampAI.git
   ```

## Development Setup

### Prerequisites

- Python 3.9 or higher
- Docker (for Qdrant)
- Git

### Installation

1. **Create a virtual environment**:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install in development mode**:

   ```bash
   # Install with all dependencies
   pip install -e ".[dev,test,all]"
   ```

3. **Start Qdrant**:

   ```bash
   docker run -p 6333:6333 qdrant/qdrant
   ```

4. **Set up environment**:

   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

5. **Install pre-commit hooks** (optional but recommended):

   ```bash
   pre-commit install
   ```

## Making Changes

### Branching Strategy

1. **Create a feature branch** from `main`:

   ```bash
   git checkout -b feature/your-feature-name
   ```

   Use prefixes:
   - `feature/` - New features
   - `fix/` - Bug fixes
   - `docs/` - Documentation changes
   - `refactor/` - Code refactoring
   - `test/` - Test improvements

2. **Keep your branch updated**:

   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

### Coding Guidelines

- Follow PEP 8 style guide
- Use type hints for all functions
- Write docstrings for public APIs
- Keep functions small and focused
- Add tests for new functionality
- Update documentation as needed

### Code Structure

```
src/hippocampai/
├── client.py           # Main MemoryClient API
├── config.py           # Configuration management
├── telemetry.py        # Observability system
├── models/             # Data models (Pydantic)
├── embed/              # Embedding generation
├── vector/             # Vector store integration
├── retrieval/          # Hybrid retrieval
├── pipeline/           # Memory processing
├── api/                # REST API
├── cli/                # Command-line interface
└── utils/              # Utility functions
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=hippocampai --cov-report=html

# Run specific test file
pytest tests/test_retrieval.py

# Run specific test
pytest tests/test_retrieval.py::test_hybrid_retrieval

# Run with verbose output
pytest -v

# Run only unit tests
pytest -m unit

# Skip slow tests
pytest -m "not slow"
```

### Writing Tests

1. **Place tests** in `tests/` directory
2. **Name test files** with `test_*.py` pattern
3. **Name test functions** with `test_*` pattern
4. **Use fixtures** from `conftest.py`
5. **Mark tests** appropriately:

   ```python
   import pytest

   @pytest.mark.unit
   def test_something():
       assert True

   @pytest.mark.integration
   def test_with_qdrant():
       # Integration test
       pass

   @pytest.mark.slow
   def test_slow_operation():
       # Slow test
       pass
   ```

### Test Coverage

- Aim for >80% code coverage
- Focus on critical paths
- Test edge cases and error handling
- Mock external dependencies (Qdrant, LLMs) when appropriate

## Code Style

### Formatting

We use **Black** for code formatting and **Ruff** for linting.

```bash
# Format code
black src/ tests/

# Run linter
ruff check src/ tests/

# Auto-fix linting issues
ruff check --fix src/ tests/
```

### Type Checking

We use **MyPy** for static type checking.

```bash
# Run type checker
mypy src/hippocampai/
```

### Pre-commit Hooks

If you installed pre-commit hooks, they will run automatically on commit:

```bash
# Run manually
pre-commit run --all-files
```

## Submitting Changes

### Pull Request Process

1. **Update your branch**:

   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Run tests and checks**:

   ```bash
   pytest
   black src/ tests/
   ruff check src/ tests/
   mypy src/hippocampai/
   ```

3. **Commit your changes**:

   ```bash
   git add .
   git commit -m "feat: add new feature"
   ```

   Use [Conventional Commits](https://www.conventionalcommits.org/):
   - `feat:` - New feature
   - `fix:` - Bug fix
   - `docs:` - Documentation changes
   - `style:` - Formatting changes
   - `refactor:` - Code refactoring
   - `test:` - Test changes
   - `chore:` - Build/tooling changes

4. **Push to your fork**:

   ```bash
   git push origin feature/your-feature-name
   ```

5. **Open a Pull Request** on GitHub:
   - Use a clear, descriptive title
   - Reference related issues
   - Describe what changed and why
   - Include screenshots/examples if applicable
   - Check "Allow edits from maintainers"

### Pull Request Checklist

- [ ] Code follows project style guidelines
- [ ] Tests pass locally
- [ ] New tests added for new functionality
- [ ] Documentation updated
- [ ] CHANGELOG.md updated (for significant changes)
- [ ] Type hints added
- [ ] No breaking changes (or clearly documented)

## Reporting Issues

### Bug Reports

When reporting bugs, please include:

1. **Description** - Clear description of the bug
2. **Steps to reproduce** - Minimal code to reproduce
3. **Expected behavior** - What should happen
4. **Actual behavior** - What actually happens
5. **Environment**:
   - Python version
   - HippocampAI version
   - OS and version
   - Qdrant version
6. **Logs/Errors** - Full error messages and stack traces

**Template:**

```markdown
### Bug Description
[Clear description]

### Steps to Reproduce
```python
from hippocampai import MemoryClient
client = MemoryClient()
# ... code to reproduce
```

### Expected Behavior

[What should happen]

### Actual Behavior

[What actually happens]

### Environment

- Python: 3.11
- HippocampAI: 0.1.5
- OS: macOS 13.0
- Qdrant: 1.7.0

### Error Messages

```
[Full error output]
```

```

## Feature Requests

We welcome feature requests! Please:

1. **Check existing issues** first
2. **Describe the feature** clearly
3. **Explain the use case** - Why is it needed?
4. **Provide examples** - How would it work?
5. **Consider alternatives** - Other ways to solve it?

**Template:**

```markdown
### Feature Description
[Clear description of the feature]

### Use Case
[Why is this needed? What problem does it solve?]

### Proposed API
```python
# Example of how it might work
client.new_feature(param=value)
```

### Alternatives Considered

[Other approaches you've thought about]

```

## Development Tips

### Running Examples

```bash
# Run individual examples
python examples/01_basic_usage.py

# Run all examples
./run_examples.sh
```

### Testing Telemetry

```bash
# Demo telemetry without Qdrant
python demo_telemetry.py

# Test new features with Qdrant
python test_new_features.py
```

### Debugging

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python your_script.py

# Run with Python debugger
python -m pdb your_script.py
```

### Working with Qdrant

```bash
# Start Qdrant with persistence
docker run -p 6333:6333 -v $(pwd)/qdrant_data:/qdrant/storage qdrant/qdrant

# Check Qdrant status
curl http://localhost:6333/collections

# View Qdrant dashboard
open http://localhost:6333/dashboard
```

## Documentation

### Updating Documentation

- **README.md** - Update for significant features
- **docs/** - Add/update guides as needed
- **Docstrings** - Keep docstrings up to date
- **Examples** - Add examples for new features

### Documentation Style

Use Google-style docstrings:

```python
def function(arg1: str, arg2: int) -> bool:
    """Short description.

    Longer description if needed.

    Args:
        arg1: Description of arg1
        arg2: Description of arg2

    Returns:
        Description of return value

    Raises:
        ValueError: When something goes wrong

    Example:
        >>> function("test", 42)
        True
    """
    pass
```

## Release Process

*(For maintainers)*

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Create git tag: `git tag -a v0.3.0 -m "Release 0.2.5"`
4. Push tag: `git push origin v0.3.0`
5. Build: `python -m build`
6. Upload: `python -m twine upload dist/*`
7. Create GitHub release with changelog

## Getting Help

- **Issues**: [GitHub Issues](https://github.com/rexdivakar/HippocampAI/issues)
- **Discord**: [Join our community](https://discord.gg/pPSNW9J7gB)
- **Email**: <rexdivakar@hotmail.com>

## Recognition

Contributors will be recognized in:

- `README.md` contributors section
- Release notes
- GitHub contributors page

Thank you for contributing to HippocampAI! 🎉
