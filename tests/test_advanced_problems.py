"""The advanced problem generators, checked against closed-form answers.

This module was 238 statements at 0% coverage and was not exported from the
package, so nothing could reach it through the public API and nothing tested
it. Given that every other module examined this way turned out to contain a
defect that changed results, the roadmap recorded it as either to be exported
and tested or removed.

It is correct. Each class below is checked against a value that can be worked
out by hand, so the choice was to export it:

* series and parallel systems of independent components against
  ``1 - prod(1 - p_i)`` and ``prod(p_i)``;
* k-out-of-n against the same two at ``k = 1`` and ``k = n``;
* the Karhunen-Loeve expansion against the covariance it is built from;
* connectivity against a graph whose cut sets are known by inspection.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from safe_ice import (
    NetworkReliabilityProblem,
    StochasticProcessProblem,
    SystemReliabilityProblem,
    TimeVariantProblem,
)

# Reliability indices for three independent components. Each fails when its own
# standard normal coordinate exceeds beta, so p_i = 1 - Phi(beta_i) exactly.
BETAS = (2.0, 2.5, 3.0)
COMPONENT_PF = tuple(float(stats.norm.sf(b)) for b in BETAS)


def component_functions():
    """g_i(u) = beta_i - u_i, so component i fails when u_i >= beta_i."""
    return [
        (lambda u, b=b, i=i: float(b - np.atleast_1d(u)[i]))
        for i, b in enumerate(BETAS)
    ]


@pytest.fixture
def samples(rng):
    return rng.standard_normal((200_000, len(BETAS)))


class TestSystemReliability:
    """Series is the minimum over components, parallel the maximum."""

    def test_series_matches_the_closed_form(self, samples) -> None:
        exact = 1.0 - float(np.prod([1.0 - p for p in COMPONENT_PF]))
        problem = SystemReliabilityProblem(component_functions())

        g = np.asarray(problem.get_series_system()(samples))
        observed = float((g <= 0).mean())

        assert observed == pytest.approx(exact, rel=0.05)

    def test_parallel_is_far_rarer_than_series(self, samples) -> None:
        """prod(p_i) is 1.9e-07 here, too rare to measure at this sample size."""
        problem = SystemReliabilityProblem(component_functions())

        parallel = np.asarray(problem.get_parallel_system()(samples))
        series = np.asarray(problem.get_series_system()(samples))

        assert float((parallel <= 0).mean()) < float((series <= 0).mean())
        # The parallel system fails only where every component does.
        assert np.all(parallel >= series)

    def test_k_out_of_n_spans_series_and_parallel(self, samples) -> None:
        """k=1 must reproduce the series system and k=n the parallel one."""
        problem = SystemReliabilityProblem(component_functions())
        n = len(BETAS)

        one = np.asarray(problem.get_k_out_of_n_system(1)(samples))
        many = np.asarray(problem.get_k_out_of_n_system(n)(samples))
        series = np.asarray(problem.get_series_system()(samples))
        parallel = np.asarray(problem.get_parallel_system()(samples))

        assert float((one <= 0).mean()) == pytest.approx(
            float((series <= 0).mean()), rel=0.05
        )
        assert float((many <= 0).mean()) == pytest.approx(
            float((parallel <= 0).mean()), abs=1e-5
        )

    def test_two_out_of_three_matches_the_closed_form(self, samples) -> None:
        """P(at least two of three fail), summed over the pairs and the triple."""
        p1, p2, p3 = COMPONENT_PF
        exact = (
            p1 * p2 * (1 - p3) + p1 * p3 * (1 - p2) + p2 * p3 * (1 - p1) + p1 * p2 * p3
        )

        problem = SystemReliabilityProblem(component_functions())
        g = np.asarray(problem.get_k_out_of_n_system(2)(samples))

        assert float((g <= 0).mean()) == pytest.approx(exact, rel=0.35)

    def test_correlation_matrix_defaults_to_independence(self) -> None:
        problem = SystemReliabilityProblem(component_functions())
        assert np.array_equal(problem.correlation_matrix, np.eye(len(BETAS)))


class TestTimeVariant:
    """Series is the worst time point, parallel the best."""

    @staticmethod
    def falling_threshold():
        """g(u, t) = (3 - t) - u_0, so failure gets easier as t grows."""
        return lambda u, t: float((3.0 - t) - np.atleast_1d(u)[0])

    def test_series_is_the_worst_time_point(self, rng) -> None:
        times = np.linspace(0.0, 1.0, 5)
        problem = TimeVariantProblem(self.falling_threshold(), times)
        u = rng.standard_normal((200_000, 1))

        g = np.asarray(problem.get_series_system_limit_state()(u))
        # The minimum over t is at t = 1, where failure is u_0 >= 2.
        assert float((g <= 0).mean()) == pytest.approx(
            float(stats.norm.sf(2.0)), rel=0.05
        )

    def test_parallel_is_the_best_time_point(self, rng) -> None:
        times = np.linspace(0.0, 1.0, 5)
        problem = TimeVariantProblem(self.falling_threshold(), times)
        u = rng.standard_normal((200_000, 1))

        g = np.asarray(problem.get_parallel_system_limit_state()(u))
        # The maximum over t is at t = 0, where failure is u_0 >= 3.
        assert float((g <= 0).mean()) == pytest.approx(
            float(stats.norm.sf(3.0)), rel=0.15
        )

    def test_correlation_matrix_is_a_valid_correlation(self) -> None:
        times = np.linspace(0.0, 1.0, 6)
        problem = TimeVariantProblem(self.falling_threshold(), times)
        matrix = problem.correlation_matrix

        assert matrix.shape == (len(times), len(times))
        assert np.diag(matrix) == pytest.approx(np.ones(len(times)))
        assert matrix == pytest.approx(matrix.T)
        assert float(np.min(np.linalg.eigvalsh(matrix))) > -1e-10

    def test_cumulative_damage_accumulates(self, rng) -> None:
        times = np.linspace(0.0, 1.0, 5)
        problem = TimeVariantProblem(self.falling_threshold(), times)
        limit_state = problem.get_cumulative_damage_limit_state(
            lambda u, t: float(np.atleast_1d(u)[0] ** 2 * t), threshold=2.0
        )

        g = np.asarray(limit_state(rng.standard_normal((2000, 1))))
        assert np.all(np.isfinite(g))
        assert float(g.std()) > 0.0
        # Zero input accrues no damage, so g is the whole threshold.
        assert float(np.asarray(limit_state(np.zeros((1, 1))))[0]) == pytest.approx(2.0)


class TestStochasticProcess:
    """The Karhunen-Loeve expansion must reproduce its own covariance."""

    @staticmethod
    def build(n_points: int = 21):
        mesh = np.linspace(0.0, 1.0, n_points)
        covariance = lambda x, y: float(np.exp(-abs(x - y) / 0.3))  # noqa: E731
        return StochasticProcessProblem(lambda x: 0.0, covariance, mesh), mesh

    def test_eigenvalues_sum_to_the_trace(self) -> None:
        """A complete expansion carries all of the variance."""
        problem, mesh = self.build()
        assert float(problem.eigenvalues.sum()) == pytest.approx(
            float(len(mesh)), rel=1e-9
        )

    def test_generated_fields_have_the_target_covariance(self, rng) -> None:
        problem, mesh = self.build(11)
        target = np.array(
            [[float(np.exp(-abs(a - b) / 0.3)) for b in mesh] for a in mesh]
        )

        fields = np.array(
            [
                problem.generate_random_field(rng.standard_normal(len(mesh)))
                for _ in range(20_000)
            ]
        )
        empirical = np.cov(fields.T)

        assert float(np.abs(empirical - target).max()) < 0.05
        assert float(np.diag(empirical).mean()) == pytest.approx(1.0, abs=0.05)

    def test_truncation_reduces_the_variance(self, rng) -> None:
        """Keeping fewer modes must lose energy, never gain it."""
        problem, mesh = self.build(11)
        coefficients = rng.standard_normal((4000, len(mesh)))

        full = np.array([problem.generate_random_field(c) for c in coefficients])
        truncated = np.array(
            [problem.generate_random_field(c, n_kl_terms=2) for c in coefficients]
        )

        assert float(truncated.var()) < float(full.var())

    def test_excursion_limit_state_responds_to_the_field(self, rng) -> None:
        problem, mesh = self.build(11)
        limit_state = problem.get_excursion_limit_state(threshold=2.0)

        g = np.asarray(limit_state(rng.standard_normal((3000, len(mesh)))))
        assert np.all(np.isfinite(g))
        assert float(g.std()) > 0.0
        # g = threshold - max(field), so a zero field leaves the threshold.
        assert float(np.asarray(limit_state(np.zeros((1, len(mesh)))))[0]) == (
            pytest.approx(2.0)
        )


class TestNetworkReliability:
    """Connectivity on a graph whose cut sets can be read off by hand."""

    @staticmethod
    def build():
        # 0-1, 0-2, 1-2, 1-3, 2-3. Isolating node 0 needs edges 0 and 1.
        adjacency = np.array(
            [[0, 1, 1, 0], [1, 0, 1, 1], [1, 1, 0, 1], [0, 1, 1, 0]], dtype=float
        )
        return NetworkReliabilityProblem(adjacency)

    def test_edges_are_extracted_once_each(self) -> None:
        problem = self.build()
        assert problem.n_nodes == 4
        assert problem.edges == [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)]

    def test_intact_network_is_connected(self) -> None:
        problem = self.build()
        limit_state = problem.get_connectivity_limit_state(0, 3)
        assert float(np.asarray(limit_state(np.zeros((1, problem.n_edges))))[0]) == 1.0

    @pytest.mark.parametrize(("cut", "expected"), [((0, 1), -1.0), ((3, 4), -1.0)])
    def test_isolating_an_endpoint_disconnects_it(self, cut, expected) -> None:
        """The default edge fails when its coordinate exceeds 3."""
        problem = self.build()
        limit_state = problem.get_connectivity_limit_state(0, 3)

        u = np.zeros((1, problem.n_edges))
        u[0, list(cut)] = 5.0
        assert float(np.asarray(limit_state(u))[0]) == expected

    def test_cutting_one_edge_leaves_a_path(self) -> None:
        problem = self.build()
        limit_state = problem.get_connectivity_limit_state(0, 3)

        for edge in range(problem.n_edges):
            u = np.zeros((1, problem.n_edges))
            u[0, edge] = 5.0
            assert float(np.asarray(limit_state(u))[0]) == 1.0, (
                f"cutting edge {edge} alone should not disconnect 0 from 3"
            )

    def test_limit_state_is_two_valued(self, rng) -> None:
        """Recorded, not endorsed: this cannot be adapted on.

        Connectivity is either true or false, so g takes only +/-1 and the
        smoothed indicator Phi(-g/sigma) has no gradient for the sigma schedule
        of equation (10) to follow. The class is useful for describing the
        problem, but Safe-ICE cannot estimate it.
        """
        problem = self.build()
        limit_state = problem.get_connectivity_limit_state(0, 3)

        u = rng.standard_normal((500, problem.n_edges))
        u[:100, 0] = 5.0
        u[:100, 1] = 5.0

        assert set(np.unique(np.asarray(limit_state(u)))) <= {-1.0, 1.0}


class TestCorrelatedSystem:
    """Correlated components, via a Cholesky factor of the correlation matrix.

    This method used to hand component ``j`` the one-element slice
    ``u_corr[j:j+1]``, where the series, parallel and k-out-of-n systems all
    pass the whole vector. Any component function that reads past its own first
    entry therefore raised ``IndexError`` -- which is every function that works
    with the other three methods.
    """

    @staticmethod
    def build(correlation):
        return SystemReliabilityProblem(component_functions(), correlation)

    @pytest.mark.parametrize("kind", ["series", "parallel"])
    def test_identity_correlation_reproduces_the_plain_system(
        self, kind, samples
    ) -> None:
        """With R = I there is nothing to correlate, so the answers must match."""
        problem = self.build(np.eye(len(BETAS)))

        correlated = np.asarray(problem.get_correlated_system(kind)(samples))
        plain = np.asarray(
            (
                problem.get_series_system()
                if kind == "series"
                else problem.get_parallel_system()
            )(samples)
        )

        assert np.allclose(correlated, plain)

    def test_positive_correlation_makes_a_series_system_safer(self, samples) -> None:
        """Correlated components fail together, so fewer runs lose any of them."""
        independent = self.build(np.eye(len(BETAS)))
        correlated = self.build(
            np.full((len(BETAS), len(BETAS)), 0.8) + 0.2 * np.eye(3)
        )

        pf_independent = float(
            (
                np.asarray(independent.get_correlated_system("series")(samples)) <= 0
            ).mean()
        )
        pf_correlated = float(
            (
                np.asarray(correlated.get_correlated_system("series")(samples)) <= 0
            ).mean()
        )

        assert pf_correlated < pf_independent

    def test_two_out_of_n_sits_between_series_and_parallel(self, samples) -> None:
        problem = self.build(np.eye(len(BETAS)))

        series = np.asarray(problem.get_correlated_system("series")(samples))
        two = np.asarray(problem.get_correlated_system("2-out-of-n")(samples))
        parallel = np.asarray(problem.get_correlated_system("parallel")(samples))

        assert np.all(series <= two)
        assert np.all(two <= parallel)

    def test_unknown_system_type_is_rejected(self) -> None:
        problem = self.build(np.eye(len(BETAS)))
        with pytest.raises(ValueError, match="Unknown system type"):
            problem.get_correlated_system("nonsense")(np.zeros((1, len(BETAS))))
