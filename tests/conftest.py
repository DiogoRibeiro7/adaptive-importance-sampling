"""Shared pytest fixtures for the Safe-ICE test suite.

Every test that needs randomness takes the :func:`rng` fixture rather than
touching NumPy's global random state. Each test therefore gets its own
generator seeded identically, which keeps failures reproducible and stops one
test's draws from perturbing another's.
"""

from __future__ import annotations

import numpy as np
import pytest

#: Fixed seed for the per-test generator. Change it to shake out tests that
#: only pass for one particular stream of random numbers.
DEFAULT_SEED = 20240117


@pytest.fixture
def rng() -> np.random.Generator:
    """A fresh, deterministically seeded NumPy random generator."""
    return np.random.default_rng(DEFAULT_SEED)


@pytest.fixture
def seed() -> int:
    """The seed used by :func:`rng`, for passing to ``random_state`` arguments."""
    return DEFAULT_SEED
