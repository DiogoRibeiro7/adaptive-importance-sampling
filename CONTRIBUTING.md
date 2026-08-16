# Contributing to Safe-ICE

Thank you for your interest in contributing to Safe-ICE! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Code Style](#code-style)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Reporting Issues](#reporting-issues)
- [Feature Requests](#feature-requests)
- [Questions?](#questions)

## Getting Started

We welcome contributions of all kinds:
- Bug fixes
- New features
- Documentation improvements
- Test coverage improvements
- Performance optimizations
- Example scripts

Before making significant changes, please open an issue to discuss your proposed changes.

## Development Setup

### 1. Fork and Clone

Fork the repository on GitHub, then clone your fork:

```bash
git clone https://github.com/YOUR-USERNAME/adaptive-importance-sampling-ice.git
cd adaptive-importance-sampling-ice
```

### 2. Install the project

Safe-ICE needs Python 3.11 or newer. Development dependencies live in a PEP 735
group, so any modern installer works:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -e .
pip install --group dev          # requires pip >= 25.1
```

Alternatively, with `uv`:

```bash
uv sync --group dev
```

Poetry also works if you prefer it (`poetry install --with dev`); the project
keeps a `poetry.lock` for reproducible environments.

### 3. Install Pre-commit Hooks

We use pre-commit hooks to keep formatting and typing consistent:

```bash
pre-commit install
```

They run ruff and mypy before each commit.

### 4. Verify Installation

```bash
pytest
```

`pytest` skips tests marked `slow`. Run the whole suite with `pytest -m ""`.

## Code Style

We maintain strict code quality standards:

### Formatting and linting
- **ruff** handles both, replacing black, isort and flake8. Line length is 88.
- Run: `ruff format .` then `ruff check --fix .`
- CI runs `ruff check .` and `ruff format --check .`; both must be clean.

### Type Checking
- **mypy** in strict mode. All functions must have type hints.
- Run: `mypy` (the files it checks are configured in `pyproject.toml`).

### Documentation
- All public functions/classes must have docstrings
- Use NumPy-style docstrings
- Include parameter types and return types
- Add usage examples for complex functions

### Example Docstring

```python
def estimate_failure_probability(
    self, initial_params: Optional[vMFNMParameters] = None
) -> Tuple[float, NDArrayF, NDArrayF]:
    """Estimate failure probability using Safe-ICE algorithm.

    Parameters
    ----------
    initial_params : vMFNMParameters, optional
        Initial parameters for the vMFNM distribution.
        If None, uses default initialization.

    Returns
    -------
    pf : float
        Estimated failure probability
    samples : NDArrayF
        Generated samples, shape (n_total_samples, dimension)
    weights : NDArrayF
        Importance weights, shape (n_total_samples,)

    Examples
    --------
    >>> ice = SafeICE(g, dimension=2)
    >>> pf, samples, weights = ice.estimate_failure_probability()
    >>> print(f"Failure probability: {pf:.2e}")
    """
```

## Testing

### Running Tests

Run all tests:
```bash
pytest
```

Run with coverage:
```bash
pytest --cov=safe_ice --cov-report=html
```

Run specific test file:
```bash
pytest tests/test_distributions.py
```

Run with markers:
```bash
pytest              # Skips slow tests by default
pytest -m integration  # Run only integration tests
```

### Writing Tests

- Place unit tests in `tests/`
- Name test files as `test_*.py`
- Use pytest fixtures for common setup
- Aim for >80% code coverage
- Test edge cases and error conditions

Example test:
```python
import pytest
import numpy as np
from safe_ice import SafeICE


def test_safe_ice_dimension():
    """Test SafeICE respects dimension parameter."""

    def g(u):
        return 3.5 - np.linalg.norm(u, axis=-1)

    ice = SafeICE(g, dimension=5)
    assert ice.dimension == 5
```

## Submitting Changes

### 1. Create a Feature Branch

Create a branch from `develop`:

```bash
git checkout develop
git pull origin develop
git checkout -b feature/your-feature-name
```

Branch naming conventions:
- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation updates
- `test/` - Test improvements
- `perf/` - Performance improvements

### 2. Make Your Changes

- Write clear, concise commit messages
- Include tests for new functionality
- Update documentation as needed
- Ensure all tests pass

### 3. Commit Your Changes

Commit messages should follow this format:
```
type: Brief description (max 50 chars)

Longer description if needed. Explain the problem this
commit is solving and why this approach was chosen.

Fixes #123  # Reference issues if applicable
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Test additions/changes
- `perf`: Performance improvements
- `refactor`: Code restructuring
- `style`: Formatting changes
- `chore`: Maintenance tasks

### 4. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a pull request on GitHub:
- Target the `develop` branch
- Provide a clear description
- Reference any related issues
- Ensure CI checks pass

### 5. Pull Request Checklist

Before submitting:
- [ ] Tests pass locally (`pytest`)
- [ ] Code is formatted (`ruff format .`)
- [ ] Linting passes (`ruff check .`)
- [ ] Type checking passes (`mypy`)
- [ ] Documentation is updated
- [ ] CHANGELOG.md is updated (for significant changes)

## Reporting Issues

### Bug Reports

When reporting bugs, please include:
1. Python version and OS
2. Complete error message and traceback
3. Minimal reproducible example
4. Expected vs actual behavior
5. Steps to reproduce

Use this template:
```markdown
**Environment:**
- Python version:
- OS:
- Safe-ICE version:

**Description:**
[Clear description of the bug]

**To Reproduce:**
```python
# Minimal code to reproduce
```

**Expected behavior:**
[What should happen]

**Actual behavior:**
[What actually happens]

**Error message:**
```
[Complete traceback]
```
```

### Security Issues

For security vulnerabilities, please email the maintainers directly instead of opening a public issue.

## Feature Requests

For feature requests, please:
1. Check existing issues/PRs first
2. Provide clear use case
3. Explain why this feature would be useful
4. Consider implementation approach

## Questions?

- Check the documentation first
- Search existing issues
- Open a discussion issue for general questions
- Tag issues appropriately

## Code of Conduct

Please note that this project adheres to a Code of Conduct. By participating, you are expected to:
- Be respectful and inclusive
- Welcome newcomers and help them get started
- Focus on constructive criticism
- Accept feedback gracefully

## Recognition

Contributors will be recognized in:
- The AUTHORS file
- Release notes
- Project documentation

Thank you for contributing to Safe-ICE!