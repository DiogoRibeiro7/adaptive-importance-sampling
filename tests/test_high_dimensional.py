"""The estimator has to work in high dimensions, which is its whole point.

A fixed ``1e-15`` floor used to be applied to the proposal density before it
was divided into the prior. Probability densities on R^d shrink geometrically
with dimension, so that floor sat far below any real value at d=2 but above
almost all of them by d=30:

    d      median proposal density     fraction hitting the 1e-15 floor
    2      4.5e-02                     0%
    10     1.6e-08                     0.1%
    20     4.3e-15                     11%
    30     7.1e-22                     100%

Clamping inflates the denominator, so the estimate collapses: d=30 returned
about 1e-9 for a true 1.6e-2. It presented as a hard ceiling somewhere between
d=20 and d=30, and looked like importance-weight degeneracy, which is a real
phenomenon and was the wrong diagnosis here.

These tests fix the behaviour at dimensions where any fixed floor of that kind
would bite.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from safe_ice import SafeICE
from safe_ice.core.safe_ice import DENSITY_FLOOR


def sphere_problem(beta: float):
    """P(||u|| > beta) for a standard normal, known exactly as a chi-square tail."""

    def limit_state(u: np.ndarray) -> np.ndarray:
        return beta - np.linalg.norm(u, axis=-1)

    return limit_state


class TestDensityFloor:
    """The floor must only guard division by zero, never distort a density."""

    def test_floor_is_the_smallest_positive_float(self) -> None:
        assert float(np.finfo(np.float64).tiny) == DENSITY_FLOOR

    def test_floor_is_far_below_realistic_high_dimensional_densities(self) -> None:
        """At d=200 a proposal density is around 1e-150; the floor must clear it."""
        assert DENSITY_FLOOR < 1e-300

    @pytest.mark.parametrize("d", [30, 50])
    def test_proposal_densities_fall_below_the_old_floor(self, d: int, seed) -> None:
        """Confirms the premise: real densities here are smaller than 1e-15.

        If this ever fails, the dimensions chosen for these tests no longer
        exercise the bug and should be raised.
        """
        ice = SafeICE(
            limit_state_function=sphere_problem(float(d) ** 0.5 + 2.0),
            dimension=d,
            N=2000,
            random_state=seed,
        )
        params = ice._initialize_vmfnm_parameters()
        samples = ice._generate_safe_mixture_samples(params, 0.95)
        densities = ice._evaluate_safe_mixture_density(samples, params, 0.95)

        assert float(np.median(densities)) < 1e-15
        assert float(np.min(densities)) > 0.0


class TestHighDimensionalAccuracy:
    """End-to-end accuracy against the chi-square tail, which is exact."""

    @pytest.mark.slow
    @pytest.mark.parametrize(("d", "beta"), [(20, 6.0), (30, 7.0), (50, 9.0)])
    def test_sphere_problem(self, d: int, beta: float) -> None:
        expected = float(1 - stats.chi2.cdf(beta**2, df=d))

        estimates = []
        for s in range(4):
            ice = SafeICE(
                limit_state_function=sphere_problem(beta),
                dimension=d,
                N=2000,
                max_iterations=15,
                random_state=s,
            )
            pf, _ = ice.run(verbose=False)
            estimates.append(pf)

        median = float(np.median(estimates))
        # Generous, because this is a stochastic estimator. The bug being
        # guarded against drove d=30 to 1e-9 against a true 1.6e-2, seven
        # orders of magnitude out, so a factor of three catches it easily.
        assert expected / 3 < median < expected * 3, f"estimates: {estimates}"

    @pytest.mark.slow
    def test_estimate_does_not_collapse_with_dimension(self) -> None:
        """Accuracy must not degrade systematically as d grows.

        The failure mode was not noise but collapse: every seed returned a value
        many orders of magnitude too small.
        """
        ratios = []
        for d, beta in ((10, 5.0), (30, 7.0), (50, 9.0)):
            expected = float(1 - stats.chi2.cdf(beta**2, df=d))
            ice = SafeICE(
                limit_state_function=sphere_problem(beta),
                dimension=d,
                N=2000,
                max_iterations=15,
                random_state=0,
            )
            pf, _ = ice.run(verbose=False)
            ratios.append(pf / expected)

        assert all(0.3 < r < 3.0 for r in ratios), f"ratios by dimension: {ratios}"
