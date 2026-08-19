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

import warnings
from itertools import pairwise

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
def test_optimized_batching_preserves_the_density(d: int, rng, seed) -> None:
    """Chunking the density evaluation must not change what it returns.

    OptimizedSafeICE evaluates q_safe in slices of ``batch_size`` rows. A
    batched loop that got its slice bookkeeping wrong would still integrate to
    roughly 1 while returning the wrong value per sample, so this compares
    against the unbatched parent element by element as well.
    """
    params = build_params(rng, 3, d)
    kwargs = {
        "limit_state_function": lambda u: np.sum(u, axis=-1),
        "dimension": d,
        "N": 10,
        "random_state": seed,
    }
    plain = SafeICE(**kwargs)
    # Small enough to force several batches over the 500-row sample below.
    batched = OptimizedSafeICE(**kwargs, batch_size=64)

    u = rng.standard_normal((500, d)) * 2.0
    assert np.allclose(
        batched._evaluate_safe_mixture_density(u, params, 0.5),
        plain._evaluate_safe_mixture_density(u, params, 0.5),
    )

    integral, stderr = integrate_density(
        lambda x: batched._evaluate_safe_mixture_density(x, params, 0.5), d
    )
    assert integral == pytest.approx(1.0, abs=max(TOLERANCE, 3 * stderr))


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


class TestPenalizedWeightUpdate:
    """The cross-entropy penalty must not collapse the mixture on sight."""

    def test_penalty_vanishes_at_uniform_weights(self) -> None:
        """Uniform weights are the maximum-entropy case, so the penalty is 0.

        With the sign inverted the bracket becomes -2 ln K instead, which
        subtracts a flat ~0.3 from every weight at K=20 and zeroes all but the
        largest component on the very first step.
        """
        from safe_ice.optimization.penalized_em import PenalizedEMOptimizer

        K, n = 20, 200
        uniform = np.full(K, 1.0 / K)
        # Responsibilities that are themselves uniform: EM alone would leave
        # the weights untouched, so any change is entirely the penalty.
        weighted_resp = np.full((n, K), 1.0 / K)

        new_pi, _pi_em = PenalizedEMOptimizer()._update_mixture_weights_penalized(
            weighted_resp, uniform, beta=1.0
        )

        assert new_pi == pytest.approx(uniform, abs=1e-12)

    def test_penalty_drains_small_components_into_large_ones(self) -> None:
        """Below-average components shrink, above-average ones grow."""
        from safe_ice.optimization.penalized_em import PenalizedEMOptimizer

        old_pi = np.array([0.7, 0.1, 0.1, 0.1])
        weighted_resp = np.tile(old_pi, (200, 1))

        new_pi, _pi_em = PenalizedEMOptimizer()._update_mixture_weights_penalized(
            weighted_resp, old_pi, beta=1.0
        )

        assert new_pi[0] > old_pi[0]
        assert np.all(new_pi[1:] < old_pi[1:])
        # Equation (21) is a zero-sum redistribution, so it preserves the
        # normalisation without a renormalising step.
        assert new_pi.sum() == pytest.approx(1.0)

    def test_mixture_is_not_collapsed_on_the_first_step(self) -> None:
        """A K=20 mixture must not be reduced to one surviving component."""
        from safe_ice.optimization.penalized_em import PenalizedEMOptimizer

        n_components, n_samples = 20, 500
        uniform = np.full(n_components, 1.0 / n_components)
        rng = np.random.default_rng(7)
        weighted_resp = rng.dirichlet(np.ones(n_components), size=n_samples)

        new_pi, _pi_em = PenalizedEMOptimizer()._update_mixture_weights_penalized(
            weighted_resp, uniform, beta=1.0
        )

        assert int(np.sum(new_pi > 0.0)) > 1  # equation (22) prunes at zero


class TestInitialisationScalesWithDimension:
    """The initial proposal has to sit on top of the target it is fitting.

    The target is the standard normal in R^d, whose radius follows chi_d, and
    chi_d is exactly Nakagami(m = d/2, Omega = d). Fixed initial values happen
    to be about right near d=2 but leave the proposal at radius ~1.1 for any
    dimension, while the target's radius grows like sqrt(d). At d=20 the two
    barely overlapped and estimates came out around 1e-20 for a true 1.5e-2.
    """

    @pytest.mark.parametrize("d", [2, 5, 10, 20])
    def test_initial_radius_tracks_the_target(self, d: int, seed) -> None:
        ice = SafeICE(
            limit_state_function=lambda u: 3.0 - np.linalg.norm(u, axis=-1),
            dimension=d,
            N=4000,
            random_state=seed,
        )
        params = ice._initialize_vmfnm_parameters()
        samples = ice._generate_safe_mixture_samples(params, 0.95)
        mean_radius = float(np.linalg.norm(samples, axis=1).mean())

        target = float(stats.chi.mean(d))
        # Generous: the point is that it scales with d at all, not that it
        # matches exactly. Without scaling the ratio is 0.24 at d=20.
        assert 0.6 < mean_radius / target < 1.7

    @pytest.mark.parametrize("d", [2, 10, 20])
    def test_omega_scales_with_dimension(self, d: int, seed) -> None:
        """E[R^2] = Omega for Nakagami, and the target has E[||u||^2] = d."""
        ice = SafeICE(
            limit_state_function=lambda u: 3.0 - np.linalg.norm(u, axis=-1),
            dimension=d,
            N=10,
            random_state=seed,
        )
        params = ice._initialize_vmfnm_parameters()
        assert 0.5 * d < float(np.mean(params.Omega)) < 2.0 * d

    @pytest.mark.slow
    def test_high_dimensional_sphere_is_usable(self) -> None:
        """d=20 used to return ~1e-20 for a true 1.5e-2."""
        d, beta = 20, 6.0
        true = float(1 - stats.chi2.cdf(beta**2, df=d))

        estimates = []
        for s in range(4):
            ice = SafeICE(
                limit_state_function=lambda u: beta - np.linalg.norm(u, axis=-1),
                dimension=d,
                N=2000,
                max_iterations=15,
                random_state=s,
            )
            pf, _ = ice.run(verbose=False)
            estimates.append(pf)

        within = sum(1 for pf in estimates if true / 4 < pf < true * 4)
        assert within >= 3, f"only {within}/4 seeds usable: {estimates}"


class TestPenaltyCoefficientSchedule:
    """Equation (23) sets beta; it used to be an unrelated heuristic.

    The replaced version measured each weight's deviation from the mean rather
    than its change since the previous iteration, scaled by the number of
    components rather than the number of samples, substituted
    ``(1 - max pi) / min pi`` for the entropy term of equation (24), and
    blended the result with the previous beta. Its own docstring called it a
    "simple adaptive heuristic".
    """

    @staticmethod
    def beta_of(pi_old, pi_new, pi_em, n, d):
        """Equation (23) as implemented."""
        from safe_ice.optimization.penalized_em import PenalizedEMOptimizer

        return PenalizedEMOptimizer._update_beta(
            np.asarray(pi_old), np.asarray(pi_new), np.asarray(pi_em), n, d
        )

    def test_settled_weights_give_the_largest_penalty(self) -> None:
        """The first term is 1 when nothing moved, and falls as drift grows."""
        pi = np.full(4, 0.25)
        settled = self.beta_of(pi, pi, pi, 1000, 2)
        drifting = self.beta_of(
            pi, pi + np.array([0.1, -0.1, 0.05, -0.05]), pi, 1000, 2
        )
        assert settled > drifting

    def test_drift_is_scaled_by_the_sample_count(self) -> None:
        """Equation (23) uses N, not K: more samples means less tolerance."""
        pi = np.full(4, 0.25)
        moved = pi + np.array([0.01, -0.01, 0.005, -0.005])
        assert self.beta_of(pi, moved, pi, 10_000, 2) < self.beta_of(
            pi, moved, pi, 100, 2
        )

    def test_eta_shrinks_with_dimension(self) -> None:
        """eta = min(1, 0.5^(floor(d/2) - 1)) damps the drift term as d grows."""
        pi = np.full(4, 0.25)
        moved = pi + np.array([0.02, -0.02, 0.01, -0.01])
        low_d = self.beta_of(pi, moved, pi, 1000, 2)
        high_d = self.beta_of(pi, moved, pi, 1000, 20)
        assert high_d > low_d  # weaker damping term -> closer to 1

    def test_beta_is_bounded_and_finite(self) -> None:
        rng = np.random.default_rng(3)
        for _ in range(50):
            k = int(rng.integers(2, 12))
            pi_old = rng.dirichlet(np.ones(k))
            pi_em = rng.dirichlet(np.ones(k))
            pi_new = rng.dirichlet(np.ones(k))
            beta = self.beta_of(pi_old, pi_new, pi_em, 1000, int(rng.integers(2, 40)))
            assert np.isfinite(beta)
            assert beta >= 0.0

    def test_single_component_has_nothing_to_prune(self) -> None:
        """E = 1*ln 1 = 0 there, so equation (24)'s denominator vanishes."""
        one = np.array([1.0])
        assert self.beta_of(one, one, one, 1000, 2) == 0.0


class TestSigmaSchedule:
    """Equation (10) picks sigma; the search for it has to be well posed.

    The CV of the intermediate weights rises monotonically as sigma falls, so
    the minimiser is the largest sigma at which it reaches the target. Two
    things make a naive search on the whole interval fail:

    * below some sigma the smoothed indicator underflows for every sample, the
      weights all vanish and the CV is undefined -- on a rare-event problem
      that is most of the interval, and a bounded minimiser handed a
      mostly-infinite objective returns arbitrary points;
    * just above that region only a handful of samples still carry weight, and
      the CV of one surviving sample out of N is about sqrt(N), so it sweeps
      through the target on its way to infinity. Those spurious minima sit at a
      sigma hundreds of times too small.
    """

    @staticmethod
    def estimator(seed, dimension=2, **kwargs):
        from safe_ice.problems.benchmarks import BenchmarkProblems

        return SafeICE(
            limit_state_function=BenchmarkProblems.four_mode_series_system(),
            dimension=dimension,
            N=1000,
            max_iterations=12,
            random_state=seed,
            **kwargs,
        )

    def test_sigma_decreases_monotonically(self) -> None:
        ice = self.estimator(0)
        ice.run(verbose=False)
        sigmas = ice.history["sigma"]

        assert len(sigmas) >= 2
        assert all(later <= earlier for earlier, later in pairwise(sigmas)), (
            f"sigma did not decrease monotonically: {sigmas}"
        )

    def test_first_step_does_not_overshoot(self) -> None:
        """It fell from 1 to 0.373 in one step where 0.999 was the minimiser.

        The proposal is still the initial one at that point, so a large drop
        leaves it unable to follow, and the weight CV never recovers.
        """
        ice = self.estimator(0)
        ice.run(verbose=False)
        sigmas = ice.history["sigma"]

        assert sigmas[1] > 0.5 * sigmas[0], (
            f"sigma fell from {sigmas[0]} to {sigmas[1]} on the first step"
        )

    def test_returned_sigma_is_inside_the_interval(self) -> None:
        """Equation (10) minimises over the open interval (0, sigma_prev)."""
        ice = self.estimator(3)
        params = ice._initialize_vmfnm_parameters()
        samples = ice._generate_safe_mixture_samples(params, 0.0)
        g_values = ice._evaluate_limit_state(samples)

        for sigma_prev in (1.0, 0.5, 0.01):
            chosen = ice._determine_next_sigma(
                samples, g_values, params, 0.0, sigma_prev
            )
            assert 0.0 < chosen <= sigma_prev

    def test_target_is_met_when_it_is_reachable(self) -> None:
        """Where a root exists, the chosen sigma should sit on it."""
        ice = self.estimator(1)
        params = ice._initialize_vmfnm_parameters()
        samples = ice._generate_safe_mixture_samples(params, 0.0)
        g_values = ice._evaluate_limit_state(samples)

        chosen = ice._determine_next_sigma(samples, g_values, params, 0.0, 1.0)
        weights = ice._calculate_intermediate_weights(
            samples, g_values, chosen, params, 0.0
        )
        cv = ice._coefficient_of_variation(weights)

        if chosen < 0.999:  # a root was found rather than the interval held
            assert cv == pytest.approx(ice.delta_target, rel=0.25), (
                f"CV {cv} at sigma {chosen} against target {ice.delta_target}"
            )

    def test_no_seed_collapses_on_a_smooth_problem(self) -> None:
        """The failure this guards against was a run twenty orders of magnitude low.

        On the heat transfer problem two of six seeds returned 1e-27 and 1e-34
        against a reference of 4.69e-07. The four-mode problem is cheap enough
        to check every seed here.
        """
        reference = 6.465e-05
        estimates = [self.estimator(seed).run(verbose=False)[0] for seed in range(8)]

        for seed, pf in enumerate(estimates):
            assert reference / 5 < pf < reference * 5, (
                f"seed {seed} returned {pf:.3e} against {reference:.3e}"
            )


class TestAutomaticSigma0:
    """sigma0 is chosen from the limit state rather than assumed to be 1.

    The smoothed indicator is ``Phi(-g/sigma)``, so only the ratio matters. A
    fixed ``sigma0 = 1`` assumes ``g`` is of order one, which every benchmark in
    the paper satisfies and a limit state in physical units does not.

    The two ways of being wrong are not symmetric, and that is what the rule
    below is built on. Equation (10) searches ``(0, sigma_prev)``, so sigma only
    falls: too large costs an iteration or two and recovers, too small cannot be
    undone. The nonlinear oscillator's spread is 0.0195, so its ``sigma0 = 1``
    is fifty times too large and it still lands within 2% of its reference. A
    resistance minus a load with a spread of 33 has a ``sigma0`` thirty times
    too small and returns a third of the answer.
    """

    @staticmethod
    def estimator(limit_state, dimension=2, **kwargs):
        return SafeICE(
            limit_state_function=limit_state,
            dimension=dimension,
            N=200,
            max_iterations=2,
            random_state=0,
            **kwargs,
        )

    def test_never_falls_below_one(self) -> None:
        """Erring upwards is the safe direction, so 1 is a floor."""
        for scale in (1e-6, 1e-3, 0.02, 0.5):
            tiny = self.estimator(
                lambda u, c=scale: c * (3.0 - np.linalg.norm(u, axis=-1))
            )
            assert tiny.sigma0 == 1.0

    @pytest.mark.parametrize("scale", [10.0, 100.0, 1000.0])
    def test_tracks_a_large_limit_state(self, scale: float) -> None:
        estimator = self.estimator(
            lambda u, c=scale: c * (3.0 - np.linalg.norm(u, axis=-1))
        )
        # The spread of 3 - ||u|| at d=2 is about 0.66, so expect 0.66 * scale.
        assert estimator.sigma0 == pytest.approx(0.66 * scale, rel=0.25)

    def test_leaves_the_paper_benchmarks_alone(self) -> None:
        """Every one of them has a spread below one, so nothing changes."""
        from safe_ice.problems.benchmarks import BenchmarkProblems

        for limit_state, dimension in (
            (BenchmarkProblems.four_mode_series_system(), 2),
            (BenchmarkProblems.three_mode_problem(), 2),
            (BenchmarkProblems.two_mode_opposite_directions(), 2),
            (BenchmarkProblems.nonlinear_oscillator(), 10),
        ):
            assert self.estimator(limit_state, dimension).sigma0 == 1.0

    def test_an_explicit_value_is_respected(self) -> None:
        estimator = self.estimator(
            lambda u: 1000.0 * (3.0 - np.linalg.norm(u, axis=-1)), sigma0=2.5
        )
        assert estimator.sigma0 == 2.5

    def test_a_degenerate_limit_state_falls_back_to_one(self) -> None:
        """A constant g has no spread, and a pilot of one has no variance."""
        constant = self.estimator(lambda u: np.ones(np.atleast_2d(u).shape[0]))
        assert constant.sigma0 == 1.0

        no_pilot = self.estimator(
            lambda u: 1000.0 * (3.0 - np.linalg.norm(u, axis=-1)), sigma0_pilot=1
        )
        assert no_pilot.sigma0 == 1.0

    def test_non_finite_values_do_not_poison_the_estimate(self) -> None:
        """NaNs are dropped rather than turning the spread into a NaN."""

        def sometimes_nan(u):
            values = 1000.0 * (3.0 - np.linalg.norm(np.atleast_2d(u), axis=-1))
            values[::10] = np.nan
            return values

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            estimator = self.estimator(sometimes_nan)

        assert np.isfinite(estimator.sigma0)
        assert estimator.sigma0 > 1.0

    def test_the_pilot_does_not_warn_on_the_user_s_behalf(self) -> None:
        """run() reports on its own samples; warning twice for one problem is noise."""

        def nan_producing(u):
            values = 3.0 - np.linalg.norm(np.atleast_2d(u), axis=-1)
            values[0] = np.nan
            return values

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self.estimator(nan_producing)  # must not raise
