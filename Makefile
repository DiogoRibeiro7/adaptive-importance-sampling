# Makefile for Safe-ICE development.
#
# Targets assume the package is installed in the active environment:
#   pip install -e . && pip install --group dev

.PHONY: help install dev-setup test test-all test-slow lint format typecheck \
        check coverage docs docs-serve clean build docker demo examples

PYTHON      := python
PACKAGE     := safe_ice
DOCKER_IMAGE := safe-ice
DOCKER_TAG  := latest

help:
	@echo "Safe-ICE development targets"
	@echo ""
	@echo "  install     Install the package and dev dependencies"
	@echo "  dev-setup   install + install pre-commit hooks"
	@echo ""
	@echo "  test        Run the fast test suite (skips 'slow' tests)"
	@echo "  test-all    Run every test, including slow ones"
	@echo "  coverage    Run tests and write an HTML coverage report"
	@echo ""
	@echo "  lint        Check style with ruff"
	@echo "  format      Reformat with ruff"
	@echo "  typecheck   Run mypy"
	@echo "  check       format + lint + typecheck + test-all"
	@echo ""
	@echo "  docs        Build the Sphinx documentation"
	@echo "  build       Build the sdist and wheel"
	@echo "  clean       Remove build and cache artifacts"

# ---------------------------------------------------------------- environment

install:
	$(PYTHON) -m pip install -e .
	$(PYTHON) -m pip install --group dev

dev-setup: install
	pre-commit install
	@echo "Development environment ready."

# --------------------------------------------------------------------- tests

test:
	pytest

test-all:
	pytest -m ""

test-slow:
	pytest -m slow

coverage:
	pytest -m "" --cov=$(PACKAGE) --cov-report=term-missing --cov-report=html
	@echo "Coverage report written to htmlcov/index.html"

# ------------------------------------------------------------- code quality

lint:
	ruff check .
	ruff format --check .

format:
	ruff check --fix .
	ruff format .

typecheck:
	mypy

check: format lint typecheck test-all
	@echo "All checks passed."

# ---------------------------------------------------------------------- docs

docs:
	$(MAKE) -C docs clean
	$(MAKE) -C docs html
	@echo "Documentation built at docs/build/html/index.html"

docs-serve: docs
	cd docs/build/html && $(PYTHON) -m http.server 8000

# ------------------------------------------------------------------ packaging

clean:
	rm -rf dist/ build/ *.egg-info
	rm -rf .pytest_cache .ruff_cache .mypy_cache .benchmarks
	rm -rf htmlcov/ .coverage coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

build: clean
	$(PYTHON) -m pip install --upgrade build twine
	$(PYTHON) -m build
	$(PYTHON) -m twine check --strict dist/*

# Releases are cut by tagging; see .github/workflows/release.yml.
# Bump the version with: python scripts/pyproject_editor.py bump-version patch

# -------------------------------------------------------------------- docker

docker:
	docker build -t $(DOCKER_IMAGE):$(DOCKER_TAG) .

docker-run: docker
	docker run --rm -it $(DOCKER_IMAGE):$(DOCKER_TAG)

docker-jupyter:
	docker build -f Dockerfile.jupyter -t $(DOCKER_IMAGE)-jupyter:$(DOCKER_TAG) .
	docker run --rm -p 8888:8888 $(DOCKER_IMAGE)-jupyter:$(DOCKER_TAG)

docker-docs:
	docker build -f Dockerfile.docs -t $(DOCKER_IMAGE)-docs:$(DOCKER_TAG) .
	docker run --rm -p 8000:8000 $(DOCKER_IMAGE)-docs:$(DOCKER_TAG)

# ------------------------------------------------------------------- running

demo:
	safe-ice demo

examples:
	$(PYTHON) examples/basic_usage.py
	$(PYTHON) examples/high_dimensional.py
