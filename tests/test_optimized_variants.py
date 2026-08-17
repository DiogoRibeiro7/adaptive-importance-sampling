"""OptimizedSafeICE and AdaptiveSafeICE must produce the same answers as SafeICE.

Both are exported from the package root, and both returned ``nan`` for every
seed on every benchmark problem. Nothing caught it: the only test touching
either class checked that its density functions integrated to 1, which they
did. The algorithm around those densities was the part that did not work.

Three defects combined to produce the NaN:

* adaptation weights used a hard indicator ``1{g <= 0} or |g| <= delta``, which
  is zero for every sample until one lands in the failure region. On a
  rare-event problem none do, so every weight was zero, the elite set was
  empty, and the loop stopped after one iteration;
* the smoothing parameter was moved by +/-10% according to whether the number
  of mixture components had changed, rather than by targeting the weight
  coefficient of variation, so it rose from 1.00 to 1.21 over a run instead of
  falling towards the failure region;
* the final estimate was ``sum(1{g<=0} w) / sum(w)`` over samples pooled from
  every iteration, and those weights had already been zeroed outside the
  failure region, so the denominator was the failure mass rather than the
  sample size -- and was exactly zero here.

Both classes now inherit the algorithm from SafeICE and override only sampling,
so the tests below are written as agreement with SafeICE rather than as
reimplementations of the same expectations.
"""

from __future__ import annotations

import numpy as np
import pytest

from safe_ice import AdaptiveSafeICE, OptimizedSafeICE, SafeICE
from safe_ice.problems.benchmarks import BenchmarkProblems

# From crude Monte Carlo over 2e7 samples; see test_benchmark_ground_truth.py.
FOUR_MODE_PF = 5.815e-05

VARIANTS = [SafeICE, OptimizedSafeICE, AdaptiveSafeICE]


def run_variant(cls, problem, d: int, seed: int, **kwargs) -> float:
    """Run one estimator and return its failure probability."""
    estimator = cls(
        limit_state_function=problem,
        dimension=d,
        max_iterations=10,
        random_state=seed,
        **kwargs,
    )
    pf, _ = estimator.run(verbose=False)
    return float(pf)


class TestEstimatesAreFinite:
    """The failure that shipped: a probability that is not a number."""

    @pytest.mark.parametrize("cls", VARIANTS, ids=lambda c: c.__name__)
    def test_estimate_is_a_probability(self, cls) -> None:
        pf = run_variant(cls, BenchmarkProblems.four_mode_series_system(), 2, 0)
        assert np.isfinite(pf), f"{cls.__name__} returned {pf}"
        assert pf > 0.0, f"{cls.__name__} returned {pf}, so it found no failures"


class TestAgreementWithReference:
    """Each variant must land on the Monte Carlo value, not merely be finite."""

    @pytest.mark.slow
    @pytest.mark.parametrize("cls", VARIANTS, ids=lambda c: c.__name__)
    def test_matches_monte_carlo(self, cls) -> None:
        estimates = [
            run_variant(cls, BenchmarkProblems.four_mode_series_system(), 2, s)
            for s in range(6)
        ]
        median = float(np.median(estimates))
        assert FOUR_MODE_PF / 3 < median < FOUR_MODE_PF * 3, (
            f"{cls.__name__}: median {median:.3e} against reference "
            f"{FOUR_MODE_PF:.3e}; estimates {estimates}"
        )


class TestInheritsTheAlgorithm:
    """The overrides must be confined to sampling.

    If a subclass ever reimplements the sigma schedule, the weights or the
    estimator again, these fail and say so.
    """

    ALGORITHM_METHODS = (
        "_determine_next_sigma",
        "_calculate_intermediate_weights",
        "_calculate_stopping_weights",
        "_estimate_failure_probability",
        "_cosine_annealing_schedule",
        "_initialize_vmfnm_parameters",
        "run",
    )

    @pytest.mark.parametrize("cls", [OptimizedSafeICE, AdaptiveSafeICE])
    @pytest.mark.parametrize("method", ALGORITHM_METHODS)
    def test_does_not_override_the_algorithm(self, cls, method: str) -> None:
        assert getattr(cls, method) is getattr(SafeICE, method), (
            f"{cls.__name__} overrides {method}; the answer must come from "
            "SafeICE so that all three variants stay in agreement"
        )

    def test_sigma_falls_over_a_run(self) -> None:
        """Sigma must decrease. The removed rule drove it up, 1.00 -> 1.21."""
        estimator = OptimizedSafeICE(
            limit_state_function=BenchmarkProblems.four_mode_series_system(),
            dimension=2,
            N=1000,
            max_iterations=8,
            random_state=0,
        )
        estimator.run(verbose=False)

        sigmas = estimator.history["sigma"]
        assert len(sigmas) >= 2
        assert sigmas[-1] < sigmas[0], f"sigma did not fall: {sigmas}"


class TestBatchingIsTransparent:
    """batch_size bounds memory; it must not change the answer."""

    def test_small_batches_agree_with_one_batch(self) -> None:
        problem = BenchmarkProblems.four_mode_series_system()
        one_batch = run_variant(OptimizedSafeICE, problem, 2, 3, N=500)
        many_batches = run_variant(
            OptimizedSafeICE, problem, 2, 3, N=500, batch_size=32
        )
        assert one_batch == pytest.approx(many_batches, rel=1e-12)

    def test_limit_state_sees_every_sample(self) -> None:
        """A batching loop that skipped a slice would still return a number."""
        seen: list[int] = []

        def counting_problem(u: np.ndarray) -> np.ndarray:
            seen.append(u.shape[0])
            return np.asarray(
                BenchmarkProblems.four_mode_series_system()(u), dtype=np.float64
            )

        estimator = OptimizedSafeICE(
            limit_state_function=counting_problem,
            dimension=2,
            N=300,
            batch_size=64,
            max_iterations=1,
            random_state=0,
        )
        estimator.run(verbose=False)

        # Two draws of N per run: the iteration itself, then the final estimate.
        assert sum(seen) == 600
        assert max(seen) <= 64


class TestAdaptiveDefaults:
    """AdaptiveSafeICE's remaining job is to pick defaults from the dimension."""

    @pytest.mark.parametrize(("d", "expected_k0"), [(2, 10), (5, 15), (10, 20)])
    def test_k0_grows_with_dimension(self, d: int, expected_k0: int) -> None:
        estimator = AdaptiveSafeICE(
            limit_state_function=lambda u: np.sum(u, axis=-1),
            dimension=d,
            random_state=0,
        )
        assert expected_k0 == estimator.K0

    def test_explicit_arguments_win(self) -> None:
        estimator = AdaptiveSafeICE(
            limit_state_function=lambda u: np.sum(u, axis=-1),
            dimension=10,
            N=123,
            K0=7,
            delta_target=2.5,
            delta_star=0.5,
            random_state=0,
        )
        assert (estimator.N, estimator.K0) == (123, 7)
        assert (estimator.delta_target, estimator.delta_star) == (2.5, 0.5)

    def test_sample_size_is_capped(self) -> None:
        assert AdaptiveSafeICE._compute_adaptive_sample_size(1000) == 50_000

    def test_initialisation_scales_with_dimension(self) -> None:
        """The removed initialiser drew m in (2, 4) at every dimension.

        The norm of a d-dimensional standard normal is Nakagami(d/2, d), so a
        proposal initialised at m ~ 3 sits at radius ~1.5 whether d is 2 or 200.
        """
        estimator = AdaptiveSafeICE(
            limit_state_function=lambda u: np.sum(u, axis=-1),
            dimension=50,
            random_state=0,
        )
        params = estimator._initialize_vmfnm_parameters()

        assert float(np.median(params.m)) > 10.0
        assert float(np.median(params.Omega)) > 20.0
