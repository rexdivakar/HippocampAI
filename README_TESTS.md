# Test Scripts for HippocampAI

Quick validation scripts to test your HippocampAI package installation.

## Available Test Scripts

| Script | Purpose | Run Time |
|--------|---------|----------|
| `test_install.py` | Package installation & imports | ~5 seconds |
| `test_functional.py` | Core functionality testing | ~10 seconds |

## Quick Start

### 1. Install Package

```bash
# From TestPyPI (for testing)
pip install -i https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ \
    hippocampai

# Or from PyPI (production)
pip install hippocampai
```

### 2. Run Tests

```bash
# Installation test
python test_install.py

# Functional test  
python test_functional.py
```

## Expected Output

### test_install.py
```
============================================================
  HippocampAI Installation Test Suite
============================================================

Total Tests: 16
Passed: 16 ✅
Failed: 0 ❌
Success Rate: 100.0%
```

### test_functional.py
```
============================================================
  HippocampAI Functional Test Suite
============================================================

Total Tests: 8
Passed: 8 ✅
Failed: 0 ❌
Success Rate: 100.0%
```

## What Gets Tested

### test_install.py
- ✅ Package imports
- ✅ Version & metadata
- ✅ Dependencies
- ✅ All modules load

### test_functional.py
- ✅ Memory creation
- ✅ Configuration
- ✅ BM25 scoring
- ✅ Score fusion
- ✅ Caching

## Full Documentation

See [TESTING.md](TESTING.md) for complete testing guide.

## Quick Validation

```bash
# One-liner test
python -c "from hippocampai import MemoryClient; print('✅ Success')"
```
