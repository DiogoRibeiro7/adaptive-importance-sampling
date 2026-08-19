"""The map from physical units into the space the estimator works in.

Without this there is no way to apply the estimator to measured data at all: it
draws from a standard normal prior, and real inputs are lognormal discharges,
Gumbel wind speeds, skewed strengths. The paper says the same in its
introduction -- a Nataf or Rosenblatt transformation maps the original
distributions to Gaussian ones.

Two properties are checked here that a transform can get wrong while looking
right. The marginals and the correlation of the samples it produces must be the
ones asked for, and a problem stated in physical units must give the same answer
as its analytical solution.

The second of those turned up a trap that has nothing to do with the transform
and everything to do with using it. Safe-ICE smooths the failure indicator as
``Phi(-g/sigma)`` from ``sigma0 = 1``, which assumes ``g`` is of order one. A
resistance minus a load in physical units is not: on the lognormal problem below
the spread of ``g`` is 33, so ``Phi(-g/1)`` is already a hard indicator and the
smoothing does nothing. That turned a true 1.82e-05 into 5.13e-06, with runs
between 0.03x and 0.61x -- wrong, and not obviously so. ``wrap`` therefore
divides by the spread of ``g`` by default, which cannot change the answer
because ``{g <= 0}`` and ``{g/c <= 0}`` are the same set.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from safe_ice import SafeICE, SubsetSimulation
from safe_ice.transforms import MarginalTransform

# Two lognormal resistances. ln R - ln S is normal, so P(R < S) is exact.
RESISTANCE = stats.lognorm(s=0.15, scale=200.0)
LOAD = stats.lognorm(s=0.25, scale=60.0)
EXACT_PF = float(
    stats.norm.cdf(-(np.log(200.0) - np.log(60.0)) / np.sqrt(0.15**2 + 0.25**2))
)


def resistance_minus_load(x: np.ndarray) -> np.ndarray:
    """Failure when the load exceeds the resistance, in physical units."""
    return x[:, 0] - x[:, 1]


class TestRoundTrip:
    """to_physical and to_standard must invert each other exactly."""

    @pytest.mark.parametrize(
        "marginals",
        [
            [stats.lognorm(s=0.3, scale=10.0)],
            [stats.norm(loc=5.0, scale=2.0), stats.gumbel_r(loc=30.0, scale=4.0)],
            [stats.expon(scale=3.0), stats.weibull_min(c=2.0, scale=7.0)],
        ],
    )
    def test_u_to_x_to_u(self, marginals, rng) -> None:
        transform = MarginalTransform(marginals)
        u = rng.standard_normal((2000, len(marginals)))

        assert transform.to_standard(transform.to_physical(u)) == pytest.approx(
            u, abs=1e-9
        )

    def test_round_trip_with_correlation(self, rng) -> None:
        correlation = np.array([[1.0, 0.6], [0.6, 1.0]])
        transform = MarginalTransform(
            [stats.lognorm(s=0.3, scale=10.0), stats.gumbel_r(loc=2.0, scale=1.0)],
            correlation=correlation,
        )
        u = rng.standard_normal((2000, 2))

        assert transform.to_standard(transform.to_physical(u)) == pytest.approx(
            u, abs=1e-9
        )

    def test_a_single_point_is_accepted(self) -> None:
        transform = MarginalTransform([RESISTANCE, LOAD])
        physical = transform.to_physical(np.zeros(2))

        assert physical.shape == (1, 2)
        # u = 0 is the median of each marginal.
        assert physical[0, 0] == pytest.approx(200.0)
        assert physical[0, 1] == pytest.approx(60.0)


class TestMarginalsAreWhatWasAskedFor:
    @pytest.mark.parametrize(
        "marginal",
        [
            stats.lognorm(s=0.4, scale=50.0),
            stats.gumbel_r(loc=30.0, scale=4.0),
            stats.weibull_min(c=1.8, scale=12.0),
        ],
    )
    def test_samples_follow_the_marginal(self, marginal) -> None:
        """A Kolmogorov-Smirnov test against the distribution asked for."""
        samples = MarginalTransform([marginal]).sample(20_000, random_state=0)

        _statistic, p_value = stats.kstest(samples[:, 0], marginal.cdf)
        assert p_value > 0.01, f"KS p-value {p_value:.4f}"


class TestNatafCorrelation:
    """A physical correlation is not the correlation of the underlying normals."""

    @pytest.mark.parametrize("target", [0.3, 0.6, 0.8, -0.5])
    def test_sample_correlation_hits_the_target(self, target: float) -> None:
        correlation = np.array([[1.0, target], [target, 1.0]])
        transform = MarginalTransform(
            [stats.lognorm(s=0.3, scale=200.0), stats.lognorm(s=0.3, scale=80.0)],
            correlation=correlation,
        )
        samples = transform.sample(200_000, random_state=0)

        achieved = float(np.corrcoef(samples.T)[0, 1])
        assert achieved == pytest.approx(target, abs=0.02)

    def test_gaussian_correlation_differs_for_skewed_marginals(self) -> None:
        """The distortion is the reason the Nataf solve is needed at all."""
        target = 0.8
        transform = MarginalTransform(
            [stats.lognorm(s=0.3, scale=200.0), stats.lognorm(s=0.3, scale=80.0)],
            correlation=np.array([[1.0, target], [target, 1.0]]),
        )
        rho_z = float(transform.gaussian_correlation[0, 1])

        assert rho_z != pytest.approx(target, abs=1e-3)
        assert rho_z == pytest.approx(0.806, abs=0.01)

    def test_normal_marginals_need_no_correction(self) -> None:
        """With normal marginals the transform is linear, so rho_z == rho_x."""
        target = 0.7
        transform = MarginalTransform(
            [stats.norm(loc=3.0, scale=2.0), stats.norm(loc=-1.0, scale=5.0)],
            correlation=np.array([[1.0, target], [target, 1.0]]),
        )

        assert float(transform.gaussian_correlation[0, 1]) == pytest.approx(
            target, abs=1e-4
        )

    def test_independent_marginals_skip_the_solve(self) -> None:
        transform = MarginalTransform([RESISTANCE, LOAD])
        assert transform.gaussian_correlation == pytest.approx(np.eye(2))


class TestAgainstAnAnalyticalAnswer:
    """A physical problem whose answer is known in closed form."""

    def test_crude_monte_carlo_in_physical_space(self, rng) -> None:
        """First check the problem itself, before involving an estimator."""
        transform = MarginalTransform([RESISTANCE, LOAD])
        samples = transform.sample(2_000_000, random_state=1)
        observed = float((resistance_minus_load(samples) <= 0).mean())

        assert observed == pytest.approx(EXACT_PF, rel=0.25)

    @pytest.mark.slow
    @pytest.mark.parametrize("estimator", [SafeICE, SubsetSimulation])
    def test_estimators_reproduce_the_closed_form(self, estimator) -> None:
        transform = MarginalTransform([RESISTANCE, LOAD])
        wrapped = transform.wrap(resistance_minus_load)

        estimates = [
            estimator(limit_state_function=wrapped, dimension=2, random_state=seed).run(
                verbose=False
            )[0]
            for seed in range(6)
        ]
        median = float(np.median(estimates))

        assert EXACT_PF / 2 < median < EXACT_PF * 2, (
            f"{estimator.__name__}: {median:.3e} against {EXACT_PF:.3e}; {estimates}"
        )


class TestLimitStateScaling:
    """The trap, and the default that avoids it."""

    def test_the_spread_is_estimated_and_reported(self) -> None:
        transform = MarginalTransform([RESISTANCE, LOAD])
        wrapped = transform.wrap(resistance_minus_load)

        # The spread of R - S under the prior is about 33 physical units.
        assert wrapped.limit_state_scale == pytest.approx(33.0, rel=0.2)

    def test_scaling_does_not_move_the_failure_boundary(self, rng) -> None:
        """Dividing by a positive constant leaves {g <= 0} alone, exactly."""
        transform = MarginalTransform([RESISTANCE, LOAD])
        scaled = transform.wrap(resistance_minus_load)
        unscaled = transform.wrap(resistance_minus_load, scale=False)

        u = rng.standard_normal((5000, 2))
        assert np.array_equal(np.asarray(scaled(u)) <= 0, np.asarray(unscaled(u)) <= 0)

    @pytest.mark.slow
    def test_scaling_is_what_makes_the_estimate_right(self) -> None:
        """Recorded because the failure is quiet: a plausible, wrong answer.

        Without it the smoothed indicator is already hard at sigma0 = 1, so the
        annealing does nothing and the run stops after two iterations with sigma
        still at 0.999.
        """
        transform = MarginalTransform([RESISTANCE, LOAD])

        def median_over_seeds(wrapped):
            return float(
                np.median(
                    [
                        SafeICE(
                            limit_state_function=wrapped,
                            dimension=2,
                            N=1000,
                            max_iterations=15,
                            random_state=seed,
                        ).run(verbose=False)[0]
                        for seed in range(6)
                    ]
                )
            )

        scaled = median_over_seeds(transform.wrap(resistance_minus_load))
        unscaled = median_over_seeds(transform.wrap(resistance_minus_load, scale=False))

        assert EXACT_PF / 1.5 < scaled < EXACT_PF * 1.5, f"{scaled:.3e}"
        assert unscaled < EXACT_PF / 2, (
            f"unscaled gave {unscaled:.3e}, which is close to the answer "
            f"{EXACT_PF:.3e}; if the trap has gone, drop this test"
        )

    def test_setting_sigma0_to_the_spread_is_equivalent(self) -> None:
        """The method only ever sees g/sigma, so the two are the same thing."""
        transform = MarginalTransform([RESISTANCE, LOAD])
        unscaled = transform.wrap(resistance_minus_load, scale=False)
        spread = transform.wrap(resistance_minus_load).limit_state_scale

        by_scaling = SafeICE(
            limit_state_function=transform.wrap(
                resistance_minus_load, scale=float(spread)
            ),
            dimension=2,
            N=1000,
            max_iterations=15,
            random_state=0,
        ).run(verbose=False)[0]
        by_sigma0 = SafeICE(
            limit_state_function=unscaled,
            dimension=2,
            N=1000,
            max_iterations=15,
            sigma0=float(spread),
            random_state=0,
        ).run(verbose=False)[0]

        assert by_scaling == pytest.approx(by_sigma0, rel=1e-12)

    def test_an_explicit_scale_is_used_verbatim(self) -> None:
        transform = MarginalTransform([RESISTANCE, LOAD])
        wrapped = transform.wrap(resistance_minus_load, scale=10.0)

        assert wrapped.limit_state_scale == 10.0
        physical = transform.to_physical(np.zeros(2))
        expected = float(resistance_minus_load(physical)[0]) / 10.0
        assert float(np.asarray(wrapped(np.zeros(2)))) == pytest.approx(expected)

    def test_a_non_positive_scale_is_rejected(self) -> None:
        transform = MarginalTransform([RESISTANCE, LOAD])
        with pytest.raises(ValueError, match="scale must be positive"):
            transform.wrap(resistance_minus_load, scale=-1.0)


class TestArgumentChecking:
    def test_at_least_one_marginal(self) -> None:
        with pytest.raises(ValueError, match="At least one marginal"):
            MarginalTransform([])

    def test_correlation_shape(self) -> None:
        with pytest.raises(ValueError, match="Correlation must be 2x2"):
            MarginalTransform([RESISTANCE, LOAD], correlation=np.eye(3))

    def test_correlation_must_be_symmetric(self) -> None:
        with pytest.raises(ValueError, match="symmetric"):
            MarginalTransform(
                [RESISTANCE, LOAD], correlation=np.array([[1.0, 0.5], [0.2, 1.0]])
            )

    def test_correlation_needs_a_unit_diagonal(self) -> None:
        with pytest.raises(ValueError, match="unit diagonal"):
            MarginalTransform(
                [RESISTANCE, LOAD], correlation=np.array([[2.0, 0.5], [0.5, 2.0]])
            )

    def test_correlation_must_be_positive_definite(self) -> None:
        with pytest.raises(ValueError, match="positive definite"):
            MarginalTransform(
                [RESISTANCE, LOAD], correlation=np.array([[1.0, 1.0], [1.0, 1.0]])
            )

    def test_wrong_number_of_columns(self) -> None:
        transform = MarginalTransform([RESISTANCE, LOAD])
        with pytest.raises(ValueError, match="Expected 2 columns"):
            transform.to_physical(np.zeros((4, 3)))
        with pytest.raises(ValueError, match="Expected 2 columns"):
            transform.to_standard(np.zeros((4, 3)))

    def test_three_dimensional_input_is_rejected(self) -> None:
        transform = MarginalTransform([RESISTANCE, LOAD])
        with pytest.raises(ValueError, match="1-D or 2-D"):
            transform.to_physical(np.zeros((2, 2, 2)))
