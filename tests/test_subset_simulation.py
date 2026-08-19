"""Subset simulation, checked against closed forms and against Safe-ICE.

This method exists here to be a second opinion, so the tests are mostly about
whether it agrees with answers obtained some other way. It shares no code with
the importance-sampling machinery -- no proposal, no mixture, no weights, no EM
-- which is what makes agreement between the two informative about the answer
rather than about the implementation.

That paid off immediately on the heat transfer problem, where the paper's
reference of 4.69e-07 came from its own subset simulation:

    our subset simulation   2.808e-07     0.60x the paper
    our Safe-ICE            3.174e-07     0.68x the paper
    the two against each other           0.88x

Two independent methods landing within 12% of each other, both about a third
below the paper, is evidence that the difference is the finite-difference
discretisation of the PDE rather than either estimator.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest
from scipy import stats

from safe_ice import SafeICE, SubsetSimulation
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


class TestAgainstClosedForms:
    """Nowhere for a plausible-looking error to hide.

    The tolerance is a factor of two, which is loose because the method is: at
    N=2000 its coefficient of variation across seeds is around 18%. It tightens
    with the sample count in the way it should -- on the sphere at d=2 the mean
    over twelve runs moves 0.90x, 1.01x, 1.02x of the exact value for N of
    1000, 4000 and 16000, with the CV falling from 18.9% to 6.9%.
    """

    @pytest.mark.parametrize("z", [3.0, 4.5])
    def test_two_mode_system(self, z: float) -> None:
        """Failure probability is 2 Phi(-z) exactly, at any dimension."""
        exact = float(2 * stats.norm.cdf(-z))
        median = median_estimate(
            SubsetSimulation,
            range(5),
            limit_state_function=BenchmarkProblems.two_mode_opposite_directions(z=z),
            dimension=2,
            N=2000,
        )

        assert exact / 2 < median < exact * 2, (
            f"z={z}: {median:.3e} against {exact:.3e}"
        )

    @pytest.mark.parametrize(("d", "beta"), [(2, 3.5), (50, 9.0)])
    def test_sphere_tail(self, d: int, beta: float) -> None:
        exact = float(stats.chi2.sf(beta**2, df=d))
        median = median_estimate(
            SubsetSimulation,
            range(5),
            limit_state_function=sphere(beta),
            dimension=d,
            N=2000,
        )

        assert exact / 2 < median < exact * 2, (
            f"d={d}: {median:.3e} against {exact:.3e}"
        )


class TestAgreesWithSafeICE:
    """The point of having it: an independent check on the same problem."""

    @pytest.mark.slow
    def test_four_mode_problem(self) -> None:
        problem = BenchmarkProblems.four_mode_series_system(z=1.0)

        subset = median_estimate(
            SubsetSimulation,
            range(5),
            limit_state_function=problem,
            dimension=2,
            N=2000,
        )
        importance = median_estimate(
            SafeICE,
            range(5),
            limit_state_function=problem,
            dimension=2,
            N=1000,
            max_iterations=15,
        )

        ratio = subset / importance
        assert 1 / 2 < ratio < 2, (
            f"subset simulation {subset:.3e} against Safe-ICE {importance:.3e}, "
            f"ratio {ratio:.2f}"
        )


class TestLevelStructure:
    """P_F is the product of the conditional probabilities, one per level."""

    @staticmethod
    def run_one(**kwargs):
        return SubsetSimulation(
            limit_state_function=BenchmarkProblems.two_mode_opposite_directions(z=4.0),
            dimension=2,
            N=2000,
            random_state=0,
            **kwargs,
        ).run(verbose=False)

    def test_estimate_is_the_product_of_the_conditionals(self) -> None:
        pf, results = self.run_one()
        assert pf == pytest.approx(
            float(np.prod(results["conditional_probabilities"])), rel=1e-12
        )

    def test_intermediate_conditionals_equal_p0(self) -> None:
        """An intermediate threshold is the p0-quantile, so p0 passes by construction.

        Counting `g <= threshold` instead drifts above p0: rejected MCMC moves
        leave duplicate states, and a duplicate sitting exactly on the threshold
        is counted too. It measured 0.1005 rather than 0.1, and the excess
        compounds over levels in one direction.
        """
        _pf, results = self.run_one()
        conditionals = results["conditional_probabilities"]

        assert len(conditionals) >= 2
        for value in conditionals[:-1]:
            assert value == pytest.approx(0.1, abs=1e-12)

    def test_thresholds_fall_towards_zero(self) -> None:
        _pf, results = self.run_one()
        thresholds = [level["threshold"] for level in results["levels"]]

        assert all(later <= earlier for earlier, later in pairwise(thresholds)), (
            f"thresholds not monotone: {thresholds}"
        )
        assert thresholds[-1] == 0.0
        assert results["reached_failure_set"] is True

    def test_number_of_levels_tracks_the_rarity(self) -> None:
        """About log(P_F)/log(p0) levels, so a rarer event needs more."""
        _pf, common = SubsetSimulation(
            limit_state_function=BenchmarkProblems.two_mode_opposite_directions(z=2.5),
            dimension=2,
            N=2000,
            random_state=0,
        ).run(verbose=False)
        _pf, rare = SubsetSimulation(
            limit_state_function=BenchmarkProblems.two_mode_opposite_directions(z=5.0),
            dimension=2,
            N=2000,
            random_state=0,
        ).run(verbose=False)

        assert rare["n_levels"] > common["n_levels"]

    def test_stopping_early_is_reported(self) -> None:
        """Cut it off before it reaches g <= 0 and it must say so."""
        _pf, results = self.run_one(max_levels=2)

        assert results["n_levels"] == 2
        assert results["reached_failure_set"] is False


class TestTheMarkovChains:
    def test_conditional_samples_stay_inside_the_level(self) -> None:
        """Every state the chains return must satisfy the level's threshold.

        A candidate is kept only if it stays inside, so the seeds' constraint
        propagates. If it did not, the conditional probabilities would not be
        conditional on anything.
        """
        problem = BenchmarkProblems.two_mode_opposite_directions(z=4.0)
        simulation = SubsetSimulation(
            limit_state_function=problem, dimension=2, N=2000, random_state=1
        )

        rng = np.random.default_rng(1)
        samples = rng.standard_normal((2000, 2))
        g_values = np.asarray(problem(samples))
        order = np.argsort(g_values)
        samples, g_values = samples[order], g_values[order]

        n_seeds = 200
        threshold = float(g_values[n_seeds - 1])
        new_samples, new_g = simulation._conditional_samples(
            samples[:n_seeds], g_values[:n_seeds], threshold
        )

        assert new_samples.shape == (2000, 2)
        assert np.all(new_g <= threshold + 1e-12)
        # The recomputed limit state must match what was returned.
        assert np.allclose(np.asarray(problem(new_samples)), new_g)

    def test_chains_actually_move(self) -> None:
        """A proposal width of zero would return the seeds repeated."""
        problem = BenchmarkProblems.two_mode_opposite_directions(z=4.0)
        simulation = SubsetSimulation(
            limit_state_function=problem, dimension=2, N=2000, random_state=1
        )

        rng = np.random.default_rng(1)
        samples = rng.standard_normal((2000, 2))
        g_values = np.asarray(problem(samples))
        order = np.argsort(g_values)
        samples, g_values = samples[order], g_values[order]

        new_samples, _new_g = simulation._conditional_samples(
            samples[:200], g_values[:200], float(g_values[199])
        )

        unique_rows = len({tuple(row) for row in np.round(new_samples, 12)})
        assert unique_rows > 200, f"only {unique_rows} distinct states from 200 seeds"


class TestArgumentChecking:
    @pytest.mark.parametrize(("n", "p0"), [(10, 0.01), (1000, 0.0005)])
    def test_seed_count_must_be_a_whole_number(self, n: int, p0: float) -> None:
        with pytest.raises(ValueError, match="whole number of seeds"):
            SubsetSimulation(limit_state_function=sphere(3.0), dimension=2, N=n, p0=p0)

    @pytest.mark.parametrize(("n", "p0"), [(1000, 0.15), (1000, 0.3)])
    def test_seeds_must_divide_the_sample_count(self, n: int, p0: float) -> None:
        """150 seeds is a whole number, but 1000 places cannot be filled by it.

        Each seed starts a chain of equal length, so an indivisible count leaves
        the level short: 150 seeds with 6 steps each fills 900 of 1000.
        """
        with pytest.raises(ValueError, match="whole number of chains"):
            SubsetSimulation(limit_state_function=sphere(3.0), dimension=2, N=n, p0=p0)

    @pytest.mark.parametrize("p0", [0.0, 1.0, 1.5])
    def test_p0_must_be_a_probability(self, p0: float) -> None:
        with pytest.raises(ValueError):
            SubsetSimulation(
                limit_state_function=sphere(3.0), dimension=2, N=1000, p0=p0
            )


class TestReportingAndAwkwardInputs:
    def test_verbose_output_names_the_levels(self, capsys) -> None:
        pf, results = SubsetSimulation(
            limit_state_function=BenchmarkProblems.two_mode_opposite_directions(z=3.5),
            dimension=2,
            N=2000,
            random_state=0,
        ).run(verbose=True)

        out = capsys.readouterr().out
        assert "Subset simulation" in out
        assert out.count("level ") >= results["n_levels"]
        assert f"{pf:.6e}" in out
        assert "reaches the failure set" in out

    def test_early_stop_is_warned_about(self, capsys) -> None:
        """An estimate that never reached g <= 0 is an upper bound, not an answer."""
        SubsetSimulation(
            limit_state_function=BenchmarkProblems.two_mode_opposite_directions(z=6.0),
            dimension=2,
            N=2000,
            max_levels=2,
            random_state=0,
        ).run(verbose=True)

        assert "upper bound" in capsys.readouterr().out

    def test_accepts_a_scalar_only_limit_state(self) -> None:
        """Not every limit state takes a batch; the fallback evaluates row-wise."""

        def scalar_only(u: np.ndarray) -> float:
            point = np.atleast_2d(u)
            if point.shape[0] != 1:
                raise TypeError("this function only takes one point at a time")
            return float(3.0 - np.linalg.norm(point[0]))

        pf, _results = SubsetSimulation(
            limit_state_function=scalar_only, dimension=2, N=1000, random_state=0
        ).run(verbose=False)

        exact = float(stats.chi2.sf(9.0, df=2))
        assert exact / 3 < pf < exact * 3, f"{pf:.3e} against {exact:.3e}"

    def test_nan_from_the_limit_state_is_treated_as_no_failure(self) -> None:
        """A NaN must not silently count as passing a threshold."""
        simulation = SubsetSimulation(
            limit_state_function=lambda u: np.full(np.atleast_2d(u).shape[0], np.nan),
            dimension=2,
            N=1000,
            random_state=0,
        )
        values = simulation._evaluate(np.zeros((4, 2)))

        assert np.all(np.isposinf(values))
