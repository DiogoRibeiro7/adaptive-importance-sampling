"""Performance benchmarks for the hot paths of the Safe-ICE algorithm.

These live outside ``tests/`` on purpose. ``testpaths`` in pyproject.toml points
at ``tests``, so a normal ``pytest`` run ignores this file and stays fast; the
performance-regression workflow runs it explicitly with ``--benchmark-only``.

Run locally with:

    pytest benchmarks/ --benchmark-only

The three densities below were each a per-sample Python loop calling scalar
PDFs, which made a 2-D run take about two minutes. They are now vectorised, and
these benchmarks exist so that does not quietly regress.
"""

from __future__ import annotations

import numpy as np
import pytest

from safe_ice import SafeICE
from safe_ice.core.parameters import vMFNMParameters
from safe_ice.distributions._numeric import radii_and_directions
from safe_ice.distributions.mixture import vMFNMDistribution
from safe_ice.optimization.penalized_em import PenalizedEMOptimizer
from safe_ice.problems.benchmarks import BenchmarkProblems

SEED = 20240117


def make_params(rng: np.random.Generator, K: int, d: int) -> vMFNMParameters:
    """Build a well-formed mixture with K components in d dimensions."""
    mu = rng.standard_normal((K, d))
    mu /= np.linalg.norm(mu, axis=1, keepdims=True)
    return vMFNMParameters(
        pi=rng.dirichlet(np.ones(K)),
        m=rng.uniform(1.0, 5.0, size=K),
        Omega=rng.uniform(1.0, 8.0, size=K),
        mu=mu,
        kappa=rng.uniform(0.5, 20.0, size=K),
    )


@pytest.mark.benchmark(group="density")
@pytest.mark.parametrize(("n", "d", "K"), [(1000, 2, 20), (1000, 10, 20)])
def test_mixture_pdf(benchmark, n: int, d: int, K: int) -> None:
    """Mixture density over a full batch of samples."""
    rng = np.random.default_rng(SEED)
    dist = vMFNMDistribution(make_params(rng, K, d))
    X = rng.standard_normal((n, d)) * 2.0

    result = benchmark(dist.pdf, X)
    assert np.all(np.isfinite(result))


@pytest.mark.benchmark(group="density")
def test_heavy_tailed_density(benchmark) -> None:
    """Heavy-tailed safety component of the proposal."""
    rng = np.random.default_rng(SEED)
    d, K, n = 2, 20, 1000
    params = make_params(rng, K, d)
    ice = SafeICE(
        limit_state_function=lambda u: np.sum(u, axis=-1),
        dimension=d,
        N=n,
        random_state=SEED,
    )
    X = rng.standard_normal((n, d)) * 2.0

    result = benchmark(ice._evaluate_heavy_tailed_density, X, params)
    assert np.all(np.isfinite(result))


@pytest.mark.benchmark(group="em")
def test_em_e_step(benchmark) -> None:
    """Posterior responsibilities for a full batch."""
    rng = np.random.default_rng(SEED)
    d, K, n = 2, 20, 1000
    params = make_params(rng, K, d)
    X = rng.standard_normal((n, d)) * 2.0
    radii, directions = radii_and_directions(X)
    weights = rng.uniform(0.0, 2.0, size=n)
    optimizer = PenalizedEMOptimizer()

    result = benchmark(optimizer._e_step, X, radii, directions, params, weights)
    assert result.shape == (n, K)


@pytest.mark.benchmark(group="end-to-end")
def test_safe_ice_run(benchmark) -> None:
    """A short end-to-end run on the four-mode benchmark problem."""
    g = BenchmarkProblems.four_mode_series_system()

    def run() -> float:
        ice = SafeICE(
            limit_state_function=g,
            dimension=2,
            N=500,
            max_iterations=3,
            random_state=SEED,
        )
        pf, _ = ice.run(verbose=False)
        return float(pf)

    # Deterministic, so a single round per sample is enough.
    pf = benchmark.pedantic(run, rounds=3, iterations=1)
    assert np.isfinite(pf)
