# Testing HippocampAI Package

This directory contains test scripts to validate your HippocampAI installation from PyPI or TestPyPI.

## Quick Start

### Run Unit Tests Locally

Use the repo helper script to execute the pytest suite with the correct `PYTHONPATH`:

```bash
./scripts/run-tests.sh
# Or pass through any pytest arguments
./scripts/run-tests.sh -k recency
```

### 1. Install from TestPyPI

```bash
# Install from TestPyPI (for testing before production release)
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ hippocampai

# Or install from production PyPI (after publishing)
pip install hippocampai
```

### 2. Run Installation Tests

```bash
# Basic installation and import tests
python test_install.py
```

This validates:
- ✅ Package is installed correctly
- ✅ All modules can be imported
- ✅ Dependencies are present
- ✅ Core classes are available
- ✅ Package metadata is correct

### 3. Run Functional Tests

```bash
# Functional tests without requiring Qdrant
python test_functional.py
```

This validates:
- ✅ Memory creation and validation
- ✅ Configuration loading
- ✅ BM25 scoring
- ✅ Score fusion (RRF)
- ✅ Importance decay
- ✅ Caching
- ✅ Pydantic validation

## Test Scripts

### test_install.py

**Purpose**: Validate package installation and basic imports

**What it tests**:
- Package version and metadata
- Core module imports (MemoryClient, Memory, MemoryType)
- Submodule imports (pipeline, retrieval, adapters, etc.)
- Dependency availability
- Basic class instantiation

**Expected output**:
```
============================================================
  HippocampAI Installation Test Suite
============================================================

Testing package installation and basic functionality...

Running: Basic imports... ✅ PASS: Basic imports
Running: Package version... ✅ PASS: Package version
  Package version: 0.1.0
...
============================================================
  TEST SUMMARY
============================================================

Total Tests: 16
Passed: 16 ✅
Failed: 0 ❌
Success Rate: 100.0%
```

### test_functional.py

**Purpose**: Test actual functionality without external dependencies

**What it tests**:
- Creating Memory objects with different types
- Loading and accessing configuration
- BM25 text scoring
- Reciprocal rank fusion
- Importance decay calculations
- Score combination logic
- Caching functionality
- Pydantic model validation

**Expected output**:
```
============================================================
  HippocampAI Functional Test Suite
============================================================

📝 Testing Memory Creation...
  ✓ Created preference: I prefer dark mode...
  ✓ Created fact: Python is a programming language...
  ✓ Created goal: Learn machine learning...

  Total memories created: 3

⚙️  Testing Configuration...
  • Qdrant URL: http://localhost:6333
  • Embedding Model: BAAI/bge-small-en-v1.5
  • Top K Results: 20
...
```

## Common Test Scenarios

### Scenario 1: Fresh Installation Test

```bash
# Create a clean virtual environment
python -m venv test_env
source test_env/bin/activate  # On Windows: test_env\Scripts\activate

# Install from TestPyPI
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ hippocampai

# Run tests
python test_install.py
python test_functional.py
```

### Scenario 2: Quick Validation

```bash
# Just test that imports work
python -c "from hippocampai import MemoryClient; print('✅ Import successful')"

# Check version
python -c "import hippocampai; print(f'Version: {hippocampai.__version__}')"
```

### Scenario 3: Test Specific Functionality

```python
# test_quick.py
from hippocampai import MemoryClient, Memory, MemoryType
from datetime import datetime

# Create a memory
memory = Memory(
    id="test_1",
    user_id="user_123",
    session_id="session_1",
    text="Test memory",
    type=MemoryType.FACT,
    timestamp=datetime.now(),
    importance=0.8
)

print(f"✅ Created memory: {memory.text}")
print(f"   Type: {memory.type.value}")
print(f"   Importance: {memory.importance}")
```

## Testing with Qdrant

If you have Qdrant running, you can test full functionality:

```python
# test_with_qdrant.py
from hippocampai import MemoryClient

# This will connect to Qdrant
client = MemoryClient(user_id="test_user")

# Store a memory
client.remember(
    text="I prefer Python over JavaScript",
    user_id="test_user",
    session_id="test_session",
    type="preference"
)

# Recall memories
results = client.recall(
    query="What programming language do I prefer?",
    user_id="test_user",
    k=5
)

for result in results:
    print(f"Memory: {result.memory.text}")
    print(f"Score: {result.score}")
```

To run with Qdrant:

```bash
# Start Qdrant (using Docker)
docker run -p 6333:6333 qdrant/qdrant

# Run your test
python test_with_qdrant.py
```

## Troubleshooting

### Import Errors

```bash
# Problem: ModuleNotFoundError: No module named 'hippocampai'
# Solution: Make sure package is installed
pip list | grep hippocampai

# If not listed, reinstall
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ hippocampai
```

### Dependency Errors

```bash
# Problem: Missing dependencies (pydantic, qdrant-client, etc.)
# Solution: Use --extra-index-url to install from both TestPyPI and PyPI
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ hippocampai

# Or install dependencies separately
pip install pydantic qdrant-client sentence-transformers
```

### Version Mismatch

```bash
# Check installed version
pip show hippocampai

# Upgrade to latest
pip install --upgrade -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ hippocampai
```

## Expected Test Results

### Passing Tests

All tests should pass with fresh installation:

- ✅ **test_install.py**: 100% pass rate (16/16 tests)
- ✅ **test_functional.py**: 100% pass rate (8/8 tests)

### Known Limitations

Some features require external services and will be skipped:

- ⚠️ **Qdrant connection**: Requires running Qdrant instance
- ⚠️ **LLM routing**: Requires LLM provider (Ollama/OpenAI)
- ⚠️ **Embedding generation**: May be slow on first run (model download)

These are expected and don't indicate package issues.

## Continuous Testing

### Before Each Release

```bash
# 1. Build the package
python -m build

# 2. Install locally
pip install dist/hippocampai-*.whl

# 3. Run tests
python test_install.py
python test_functional.py

# 4. If all pass, publish to TestPyPI
twine upload --repository testpypi dist/*

# 5. Test from TestPyPI
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ hippocampai
python test_install.py

# 6. If successful, publish to PyPI
```

### Automated Testing

Add these tests to your CI/CD pipeline:

```yaml
# .github/workflows/test-package.yml
name: Test Package

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.9", "3.10", "3.11", "3.12"]

    steps:
    - uses: actions/checkout@v4
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v5
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install package
      run: |
        pip install -e .

    - name: Run installation tests
      run: python test_install.py

    - name: Run functional tests
      run: python test_functional.py
```

## Test Coverage

Current test coverage:

| Component | Coverage |
|-----------|----------|
| Core Models | ✅ 100% |
| Imports | ✅ 100% |
| Configuration | ✅ 100% |
| BM25 Retrieval | ✅ 100% |
| Score Fusion | ✅ 100% |
| Utilities | ✅ 100% |
| Qdrant Operations | ⚠️ Requires service |
| LLM Operations | ⚠️ Requires service |

## Support

If tests fail:

1. Check Python version (3.9+)
2. Verify clean virtual environment
3. Ensure all dependencies installed
4. Check test output for specific errors
5. Open issue at: https://github.com/rexdivakar/HippocampAI/issues

## Quick Reference

```bash
# Install
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ hippocampai

# Test installation
python test_install.py

# Test functionality
python test_functional.py

# Quick import test
python -c "from hippocampai import MemoryClient; print('✅ Success')"

# Check version
python -c "import hippocampai; print(hippocampai.__version__)"
```
