# Contributing to HippocampAI

Thanks for helping improve HippocampAI! This guide keeps contributions smooth and reviewable.

## Getting Started

1. **Clone & install** (use a virtual environment):
   ```bash
   git clone https://github.com/rexdivakar/HippocampAI.git
   cd HippocampAI
   python -m venv .venv && source .venv/bin/activate
   pip install -e .[core,dev,test]
   ```
2. **Bootstrap the project** (optional, for running demos):
   ```bash
   python setup_initial.py
   ```
3. **Configure providers** by copying `.env.example` to `.env` and filling in API keys as needed.

## Coding Standards

- Keep imports sorted with `ruff check --select I --fix .`.
- Run `ruff check` and `./scripts/run-tests.sh` before each commit; add targeted tests for new behaviour.
- Use type hints throughout and prefer dataclasses or Pydantic models where possible.
- Log meaningful context (`logger.info/debug`) rather than printing directly.
- All timestamps must be timezone-aware UTC. Use helpers in `hippocampai.utils.time` (`now_utc()`, `isoformat_utc()`, etc.) instead of `datetime.utcnow()`.

## Pull Requests

- Create feature or bugfix branches off `main`.
- Squash commits sensibly; commit messages should describe _why_ a change is needed.
- Provide a concise summary of the change in the PR description, including testing performed.
- Update documentation (`docs/` and `README.md`) when you add or modify features or commands.
- Ensure CI (lint, tests, dependency checks) passes; the `dependency-health` workflow runs `pip-audit` and `liccheck`.

## Reporting Issues

When filing an issue, include:

- What happened vs. what you expected.
- Steps to reproduce (commands, config, snippets).
- Environment details (OS, Python version, optional: vector DB / provider info).

We appreciate your time and contributions—thank you for making HippocampAI better!
