"""The proposal components must be probability densities on R^d.

Every component of ``q_safe`` sits in the denominator of the importance
weights, so if one of them integrates to something other than 1 the failure
probability is scaled by that factor no matter how many samples are drawn.
That is what a missing polar Jacobian did: the heavy-tailed component
integrated to 2.7 at d=2 and 114 at d=5, and estimates came out at roughly
half the analytical answer regardless of N.

Each density is written in polar form as a radial density times an angular
one, so recovering a density on R^d needs ``du = r^(d-1) dr dw``. These tests
integrate the components numerically and would fail again if that factor were
dropped.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from safe_ice import OptimizedSafeICE, SafeICE
from safe_ice.core.parameters import vMFNMParameters
from safe_ice.distributions.mixture import vMFNMDistribution


def build_params(rng: np.random.Generator, K: int, d: int) -> vMFNMParameters:
    """A well-formed mixture with K components in d dimensions."""
    mu = rng.standard_normal((K, d))
    mu /= np.linalg.norm(mu, axis=1, keepdims=True)
    return vMFNMParameters(
        pi=rng.dirichlet(np.ones(K)),
        m=rng.uniform(1.0, 4.0, size=K),
        Omega=rng.uniform(1.0, 6.0, size=K),
        mu=mu,
        kappa=rng.uniform(0.5, 8.0, size=K),
    )


def integrate_density(density_fn, d: int, n: int = 200_000, scale: float = 6.0):
    """Estimate the integral of a density over R^d, by importance sampling.

    Uses a wide Gaussian as the reference:
    ``int q(u) du = E_{u~N(0, s^2 I)}[ q(u) / N(u; 0, s^2 I) ]``.

    Returns the estimate and its standard error.
    """
    rng = np.random.default_rng(4242)
    u = rng.standard_normal((n, d)) * scale
    log_reference = stats.multivariate_normal.logpdf(
        u, mean=np.zeros(d), cov=(scale**2) * np.eye(d)
    )
    ratio = np.asarray(density_fn(u), dtype=np.float64) / np.exp(log_reference)
    return float(np.mean(ratio)), float(np.std(ratio) / np.sqrt(n))


# A generous tolerance: this is a Monte Carlo integral of a heavy-tailed
# integrand, so it is noisy. The bug being guarded against was off by 2.7x to
# 114x, which this catches many times over.
TOLERANCE = 0.15


@pytest.mark.parametrize("d", [2, 3])
@pytest.mark.parametrize("K", [1, 3])
def test_vmfnm_mixture_integrates_to_one(d: int, K: int, rng) -> None:
    dist = vMFNMDistribution(build_params(rng, K, d))
    integral, stderr = integrate_density(dist.pdf, d)
    assert integral == pytest.approx(1.0, abs=max(TOLERANCE, 3 * stderr))


@pytest.mark.parametrize("d", [2, 3])
@pytest.mark.parametrize("K", [1, 3])
def test_heavy_tailed_component_integrates_to_one(d: int, K: int, rng, seed) -> None:
    params = build_params(rng, K, d)
    ice = SafeICE(
        limit_state_function=lambda u: np.sum(u, axis=-1),
        dimension=d,
        N=10,
        random_state=seed,
    )
    integral, stderr = integrate_density(
        lambda u: ice._evaluate_heavy_tailed_density(u, params), d
    )
    assert integral == pytest.approx(1.0, abs=max(TOLERANCE, 3 * stderr))


@pytest.mark.parametrize("lambda_val", [0.0, 0.5, 0.95])
def test_safe_mixture_integrates_to_one(lambda_val: float, rng, seed) -> None:
    d, K = 2, 3
    params = build_params(rng, K, d)
    ice = SafeICE(
        limit_state_function=lambda u: np.sum(u, axis=-1),
        dimension=d,
        N=10,
        random_state=seed,
    )
    integral, stderr = integrate_density(
        lambda u: ice._evaluate_safe_mixture_density(u, params, lambda_val), d
    )
    assert integral == pytest.approx(1.0, abs=max(TOLERANCE, 3 * stderr))


@pytest.mark.parametrize("d", [2, 3])
def test_optimized_components_integrate_to_one(d: int, rng, seed) -> None:
    """OptimizedSafeICE has its own density code, with the same requirement."""
    params = build_params(rng, 3, d)
    ice = OptimizedSafeICE(
        limit_state_function=lambda u: np.sum(u, axis=-1),
        dimension=d,
        N=10,
        random_state=seed,
    )

    light, light_se = integrate_density(
        lambda u: ice._evaluate_vmfnm_density_vectorized(u, params, 1.0), d
    )
    assert light == pytest.approx(1.0, abs=max(TOLERANCE, 3 * light_se))

    heavy, heavy_se = integrate_density(
        lambda u: ice._evaluate_heavy_tailed_density_vectorized(u, params), d
    )
    assert heavy == pytest.approx(1.0, abs=max(TOLERANCE, 3 * heavy_se))


class TestLambdaCap:
    """The heavy-tailed component must never be annealed fully away."""

    def test_lambda_never_reaches_one(self, seed) -> None:
        # sigma -> 0 drives the raw cosine schedule to exactly 1.0, which would
        # drop the heavy-tailed component and let a single tail sample carry
        # the whole estimate.
        ice = SafeICE(
            limit_state_function=lambda u: np.sum(u, axis=-1),
            dimension=2,
            N=10,
            random_state=seed,
        )
        for sigma in (1.0, 0.5, 1e-3, 1e-8, 0.0):
            assert ice._cosine_annealing_schedule(sigma, 1.0) <= ice.lambda_max

    def test_lambda_max_is_configurable(self, seed) -> None:
        ice = SafeICE(
            limit_state_function=lambda u: np.sum(u, axis=-1),
            dimension=2,
            N=10,
            lambda_max=0.8,
            random_state=seed,
        )
        assert ice._cosine_annealing_schedule(0.0, 1.0) == pytest.approx(0.8)


class TestKnownAnalyticalAnswers:
    """End-to-end accuracy against problems with closed-form solutions."""

    @pytest.mark.slow
    def test_linear_limit_state_is_unbiased(self) -> None:
        """The estimate should centre on the analytical value, not half of it."""
        a = np.array([1.0, 1.0])
        a0 = 3.0
        analytical = float(stats.norm.cdf(-a0 / np.linalg.norm(a)))

        def limit_state(u: np.ndarray) -> np.ndarray:
            return a0 + np.dot(u.reshape(-1, 2), a)

        estimates = []
        for s in range(6):
            ice = SafeICE(
                limit_state_function=limit_state,
                dimension=2,
                N=2000,
                max_iterations=15,
                random_state=s,
            )
            pf, _ = ice.run(verbose=False)
            estimates.append(pf)

        mean_estimate = float(np.mean(estimates))
        # Before the Jacobian fix this ratio sat at ~0.54 and did not improve
        # with more samples.
        assert 0.75 < mean_estimate / analytical < 1.35
