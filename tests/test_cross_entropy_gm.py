"""Cross-entropy with a Gaussian mixture: what it does, and where it stops.

Kurtz and Song (2013), reference [25], the older method ICE was introduced to
improve on. The Safe-ICE paper's case for ICE rests on two claims about it, and
having a faithful implementation makes them testable rather than quotable:

* it "discards most samples by relying solely on an elite subset -- diminishing
  statistical efficiency";
* "in high dimensions, its Gaussian proposals collapse onto a thin shell,
  causing numerical instability".

Both hold, and the second is not a marginal effect:

    d      exact       CE-GM            Safe-ICE
    2      2.19e-03    0.72x            1.00x
    10     5.35e-03    0.45x            1.00x
    50     3.61e-03    0.01x            1.01x
    100    2.63e-03    3.5e-10 (0.00x)  1.04x

The tests below check that it works where it should -- low dimension, few modes,
against closed forms -- and record where it does not. The failing cases are the
method's properties, not defects in this implementation, so they are asserted
rather than fixed.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from safe_ice import CrossEntropyGaussianMixture, SafeICE
from safe_ice.problems.benchmarks import BenchmarkProblems


def sphere(beta: float):
    """P(||u|| > beta), a chi-square tail, exact at any dimension."""

    def limit_state(u: np.ndarray) -> np.ndarray:
        return beta - np.linalg.norm(u, axis=-1)

    return limit_state


def median_estimate(cls, seeds, **kwargs) -> float:
    return float(
        np.median([cls(random_state=s, **kwargs).run(verbose=False)[0] for s in seeds])
    )


class TestItWorksWhereItShould:
    """Low dimension, against answers known in closed form."""

    @pytest.mark.parametrize("z", [2.5, 3.0])
    def test_two_mode_system(self, z: float) -> None:
        exact = float(2 * stats.norm.cdf(-z))
        median = median_estimate(
            CrossEntropyGaussianMixture,
            range(5),
            limit_state_function=BenchmarkProblems.two_mode_opposite_directions(z=z),
            dimension=2,
            K=2,
            N=2000,
        )

        assert exact / 3 < median < exact * 3, f"z={z}: {median:.3e} vs {exact:.3e}"

    def test_sphere_at_low_dimension(self) -> None:
        exact = float(stats.chi2.sf(3.5**2, df=2))
        median = median_estimate(
            CrossEntropyGaussianMixture,
            range(5),
            limit_state_function=sphere(3.5),
            dimension=2,
            K=2,
            N=2000,
        )

        assert exact / 3 < median < exact * 3, f"{median:.3e} vs {exact:.3e}"

    def test_returns_a_probability_with_diagnostics(self) -> None:
        pf, results = CrossEntropyGaussianMixture(
            limit_state_function=sphere(3.0), dimension=2, N=2000, random_state=0
        ).run(verbose=False)

        assert 0.0 <= pf <= 1.0
        assert results["failure_probability"] == pf
        assert results["reached_failure_set"] is True
        assert results["n_evaluations"] > 0


class TestTheEliteSubset:
    """Claim one: most samples are thrown away."""

    def test_most_samples_never_inform_the_fit(self) -> None:
        _pf, results = CrossEntropyGaussianMixture(
            limit_state_function=BenchmarkProblems.four_mode_series_system(z=1.0),
            dimension=2,
            K=4,
            N=2000,
            random_state=0,
        ).run(verbose=False)

        discarded = results["samples_discarded"] / results["n_evaluations"]
        assert discarded > 0.5, f"only {discarded:.1%} discarded"

    def test_each_iteration_keeps_the_rho_fraction(self) -> None:
        """The threshold is the rho-quantile, so rho of them pass by construction."""
        _pf, results = CrossEntropyGaussianMixture(
            limit_state_function=sphere(4.0),
            dimension=2,
            N=2000,
            rho=0.1,
            random_state=0,
        ).run(verbose=False)

        # Every iteration but the last, where the threshold is pinned at zero.
        for record in results["iterations"][:-1]:
            assert record["n_elite"] / 2000 == pytest.approx(0.1, abs=0.01)

    def test_a_larger_elite_fraction_discards_less(self) -> None:
        def discarded_fraction(rho: float) -> float:
            _pf, results = CrossEntropyGaussianMixture(
                limit_state_function=sphere(4.0),
                dimension=2,
                N=2000,
                rho=rho,
                random_state=0,
            ).run(verbose=False)
            return float(results["samples_discarded"] / results["n_evaluations"])

        assert discarded_fraction(0.3) < discarded_fraction(0.1)


class TestHighDimensionalCollapse:
    """Claim two: the Gaussian proposal cannot follow the shell.

    The prior's mass concentrates on a shell of radius sqrt(d), and beyond
    modest dimension a Gaussian mixture fitted to a tenth of the samples cannot
    represent it. This is the method's own limit, recorded rather than repaired.
    """

    @pytest.mark.slow
    @pytest.mark.parametrize(("d", "beta"), [(50, 9.0), (100, 12.0)])
    def test_it_collapses_where_safe_ice_does_not(self, d: int, beta: float) -> None:
        exact = float(stats.chi2.sf(beta**2, df=d))

        cross_entropy = median_estimate(
            CrossEntropyGaussianMixture,
            range(3),
            limit_state_function=sphere(beta),
            dimension=d,
            K=2,
            N=2000,
        )
        safe_ice = median_estimate(
            SafeICE,
            range(3),
            limit_state_function=sphere(beta),
            dimension=d,
            N=2000,
            max_iterations=15,
        )

        assert exact / 2 < safe_ice < exact * 2, (
            f"Safe-ICE should be unaffected at d={d}: {safe_ice:.3e} vs {exact:.3e}"
        )
        assert cross_entropy < exact / 10, (
            f"at d={d} cross-entropy gave {cross_entropy:.3e} against {exact:.3e}; "
            "if this has stopped collapsing, the claim it records no longer holds"
        )

    @pytest.mark.slow
    def test_the_collapse_deepens_with_dimension(self) -> None:
        """Not noise: the further out, the worse, monotonically."""
        ratios = []
        for d, beta in ((2, 3.5), (10, 5.0), (50, 9.0)):
            exact = float(stats.chi2.sf(beta**2, df=d))
            estimate = median_estimate(
                CrossEntropyGaussianMixture,
                range(3),
                limit_state_function=sphere(beta),
                dimension=d,
                K=2,
                N=2000,
            )
            ratios.append(estimate / exact)

        assert ratios[0] > ratios[1] > ratios[2], f"ratios by dimension: {ratios}"


class TestMultiModalFailure:
    """A Gaussian mixture fitted on elites does not reliably find every mode."""

    @staticmethod
    def lobes_covered(samples: np.ndarray, g_values: np.ndarray) -> int:
        """How many of the four-mode problem's lobes the failures fall in."""
        failing = samples[np.asarray(g_values) <= 0]
        if failing.size == 0:
            return 0
        quadrant = list(
            zip(
                np.sign(failing[:, 0] + failing[:, 1]).astype(int),
                np.sign(failing[:, 0] - failing[:, 1]).astype(int),
                strict=False,
            )
        )
        counts: dict[tuple[int, int], int] = {}
        for key in quadrant:
            counts[key] = counts.get(key, 0) + 1
        return sum(1 for count in counts.values() if count >= 3)

    @pytest.mark.slow
    def test_safe_ice_finds_every_lobe_and_cross_entropy_does_not(self) -> None:
        """The measured difference: 4 of 4 every run, against a median of 3."""
        problem = BenchmarkProblems.four_mode_series_system(z=1.0)

        def lobes(cls, **kwargs):
            found = []
            for seed in range(6):
                _pf, results = cls(
                    limit_state_function=problem,
                    dimension=2,
                    random_state=seed,
                    **kwargs,
                ).run(verbose=False)
                found.append(
                    self.lobes_covered(
                        results["final_samples"], results["final_g_values"]
                    )
                )
            return found

        cross_entropy = lobes(CrossEntropyGaussianMixture, K=4, N=2000)
        safe_ice = lobes(SafeICE, N=2000, max_iterations=15)

        assert float(np.median(safe_ice)) == 4.0, f"Safe-ICE lobes: {safe_ice}"
        assert float(np.median(cross_entropy)) < 4.0, (
            f"cross-entropy lobes: {cross_entropy}; if it now finds all four, "
            "the multi-modal weakness this records has gone"
        )

    @pytest.mark.slow
    def test_missing_a_lobe_shows_up_as_an_underestimate(self) -> None:
        """Covering three of four lobes returns roughly three quarters of the answer."""
        reference = 6.465e-05
        median = median_estimate(
            CrossEntropyGaussianMixture,
            range(6),
            limit_state_function=BenchmarkProblems.four_mode_series_system(z=1.0),
            dimension=2,
            K=4,
            N=2000,
        )

        assert median < reference * 0.8, f"{median:.3e} against {reference:.3e}"

    @pytest.mark.slow
    def test_no_choice_of_k_rescues_it(self) -> None:
        """Measured at K = 1, 2, 4, 6 and 8: all between 0.46x and 0.60x."""
        reference = 6.465e-05
        for k in (1, 2, 4, 8):
            median = median_estimate(
                CrossEntropyGaussianMixture,
                range(4),
                limit_state_function=BenchmarkProblems.four_mode_series_system(z=1.0),
                dimension=2,
                K=k,
                N=2000,
            )
            assert median < reference, f"K={k} gave {median:.3e}"


class TestArgumentChecking:
    @pytest.mark.parametrize("rho", [0.0, 1.0, 1.5, -0.1])
    def test_rho_must_be_a_probability(self, rho: float) -> None:
        with pytest.raises(ValueError, match="rho must lie"):
            CrossEntropyGaussianMixture(
                limit_state_function=sphere(3.0), dimension=2, rho=rho
            )

    def test_the_elite_set_cannot_be_empty(self) -> None:
        with pytest.raises(ValueError, match="at least one elite"):
            CrossEntropyGaussianMixture(
                limit_state_function=sphere(3.0), dimension=2, N=5, rho=0.1
            )


class TestReportingAndAwkwardInputs:
    def test_verbose_output_names_each_iteration(self, capsys) -> None:
        pf, results = CrossEntropyGaussianMixture(
            limit_state_function=sphere(4.0), dimension=2, N=2000, random_state=0
        ).run(verbose=True)

        out = capsys.readouterr().out
        assert "Cross-entropy with a Gaussian mixture" in out
        assert out.count("iteration ") >= results["n_iterations"]
        assert "discarded" in out
        assert f"{pf:.6e}" in out

    def test_stopping_short_is_warned_about(self, capsys) -> None:
        """An estimate that never reached g <= 0 should not pass for an answer."""
        CrossEntropyGaussianMixture(
            limit_state_function=sphere(8.0),
            dimension=2,
            N=2000,
            max_iterations=1,
            random_state=0,
        ).run(verbose=True)

        assert "stopped before the threshold reached zero" in capsys.readouterr().out

    def test_accepts_a_scalar_only_limit_state(self) -> None:
        """Not every limit state takes a batch; the fallback evaluates row-wise."""

        def scalar_only(u: np.ndarray) -> float:
            point = np.atleast_2d(u)
            if point.shape[0] != 1:
                raise TypeError("one point at a time")
            return float(3.0 - np.linalg.norm(point[0]))

        pf, _results = CrossEntropyGaussianMixture(
            limit_state_function=scalar_only, dimension=2, N=1000, random_state=0
        ).run(verbose=False)

        exact = float(stats.chi2.sf(9.0, df=2))
        assert exact / 4 < pf < exact * 4, f"{pf:.3e} against {exact:.3e}"

    def test_a_generator_can_be_passed_directly(self) -> None:
        """Sharing one generator across estimators has to give a repeatable run."""
        first = CrossEntropyGaussianMixture(
            limit_state_function=sphere(3.5),
            dimension=2,
            N=1000,
            random_state=np.random.default_rng(11),
        ).run(verbose=False)[0]
        second = CrossEntropyGaussianMixture(
            limit_state_function=sphere(3.5),
            dimension=2,
            N=1000,
            random_state=np.random.default_rng(11),
        ).run(verbose=False)[0]

        assert first == second

    def test_nan_is_treated_as_no_failure(self) -> None:
        """A NaN must not be counted as passing a threshold."""
        estimator = CrossEntropyGaussianMixture(
            limit_state_function=lambda u: np.full(np.atleast_2d(u).shape[0], np.nan),
            dimension=2,
            N=1000,
            random_state=0,
        )
        values = estimator._evaluate(np.zeros((4, 2)))

        assert np.all(np.isposinf(values))

    def test_it_gives_up_when_the_elite_set_is_too_small(self, capsys) -> None:
        """Fewer elites than the mixture needs means there is nothing to refit."""
        estimator = CrossEntropyGaussianMixture(
            limit_state_function=sphere(6.0),
            dimension=2,
            K=8,
            N=100,
            rho=0.1,
            random_state=0,
        )
        pf, results = estimator.run(verbose=True)

        assert 0.0 <= pf <= 1.0
        assert "too few elite samples" in capsys.readouterr().out
        assert results["reached_failure_set"] is False
