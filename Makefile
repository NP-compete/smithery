.DEFAULT_GOAL := help
SHELL := /bin/bash

.PHONY: help install lint format typecheck test coverage clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install in editable mode with dev dependencies
	pip install -e ".[dev]"
	pre-commit install

lint: ## Run linter (ruff check)
	ruff check .
	ruff format --check .

format: ## Auto-format code
	ruff format .
	ruff check --fix .

typecheck: ## Run mypy type checker
	mypy smithery

test: ## Run unit tests
	pytest -m "not slow and not gpu" -q

test-all: ## Run all tests including slow and GPU
	pytest -q

coverage: ## Run tests with coverage report
	pytest --cov=smithery --cov-report=term-missing --cov-report=html -m "not slow and not gpu"

clean: ## Remove build artifacts
	rm -rf build/ dist/ *.egg-info .pytest_cache .mypy_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
