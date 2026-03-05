# Contributing to Smithery

Thanks for your interest in contributing. Here's how to get started.

## Development Setup

```bash
git clone https://github.com/NP-compete/smithery.git
cd smithery
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

## Workflow

1. **Fork** and create a feature branch from `main`.
2. Make your changes. Write tests for new functionality.
3. Run the quality checks locally:

```bash
make lint       # ruff check + format
make typecheck  # mypy
make test       # pytest
```

4. Open a pull request against `main`.

## Code Standards

- Python 3.10+ with type annotations everywhere.
- Format with **ruff format**, lint with **ruff check**.
- Strict mypy (`disallow_untyped_defs = true`).
- Tests go in `tests/` mirroring the source layout.
- 80%+ coverage required. Run `make coverage` to check.

## Commit Messages

Use conventional commits:

```
feat: add multi-step plan generation
fix: correct parameter schema validation
docs: update evaluation metrics table
test: add unit tests for MCPImporter
```

## Pull Requests

- Keep PRs focused. One concern per PR.
- Fill in the PR template completely.
- All CI checks must pass before merge.
- Squash merge is the default merge strategy.

## Reporting Bugs

Use the [bug report template](https://github.com/NP-compete/smithery/issues/new?template=bug_report.yml).

## Requesting Features

Use the [feature request template](https://github.com/NP-compete/smithery/issues/new?template=feature_request.yml).

## Questions

Open a thread in [Discussions](https://github.com/NP-compete/smithery/discussions).
