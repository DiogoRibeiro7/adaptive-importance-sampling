"""Benchmark problems checked against crude Monte Carlo.

None of these have closed-form solutions, so the reference values below come
from crude Monte Carlo over 2e7 standard-normal samples. That is the same
method a user would reach for, run at a sample count the estimator is meant to
make unnecessary, which makes it an independent check rather than a restatement
of the implementation.

    problem                          failures / 2e7      pf        rel. s.e.
    four_mode_series_system                   1163    5.815e-05      2.9%
    three_mode_problem                       69490    3.475e-03      0.4%
    two_mode_opposite_directions             53800    2.690e-03      0.4%
    nakagami_ratio_problem                 1037533    5.188e-02      0.1%

An earlier comment in the test suite put the four-mode reference at 1.22e-5,
which does not match this problem at its default z=3.8.
"""

from __future__ import annotations

import numpy as np
import pytest

from safe_ice import SafeICE
from safe_ice.problems.benchmarks import BenchmarkProblems

# (name, factory, dimension, reference pf, relative standard error of the
# reference). The tolerance below is far wider than the reference error, so the
# latter is recorded for context rather than used arithmetically.
GROUND_TRUTH = [
    (
        "four_mode_series_system",
        BenchmarkProblems.four_mode_series_system,
        2,
        5.815e-05,
    ),
    ("three_mode_problem", BenchmarkProblems.three_mode_problem, 2, 3.475e-03),
    (
        "two_mode_opposite_directions",
        BenchmarkProblems.two_mode_opposite_directions,
        2,
        2.690e-03,
    ),
    ("nakagami_ratio_problem", BenchmarkProblems.nakagami_ratio_problem, 2, 5.188e-02),
]


class TestAgainstMonteCarlo:
    @pytest.mark.slow
    @pytest.mark.parametrize(("name", "factory", "d", "reference"), GROUND_TRUTH)
    def test_estimate_matches_reference(
        self, name: str, factory, d: int, reference: float
    ) -> None:
        """The estimate should sit on the Monte Carlo value, not merely near it."""
        limit_state = factory()

        estimates = []
        for s in range(6):
            ice = SafeICE(
                limit_state_function=limit_state,
                dimension=d,
                N=1000,
                max_iterations=15,
                random_state=s,
            )
            pf, _ = ice.run(verbose=False)
            estimates.append(pf)

        median = float(np.median(estimates))
        assert reference / 3 < median < reference * 3, (
            f"{name}: median {median:.3e} against reference {reference:.3e}; "
            f"estimates {estimates}"
        )

    @pytest.mark.parametrize(("name", "factory", "d", "reference"), GROUND_TRUTH)
    def test_reference_problems_have_a_reachable_failure_region(
        self, name: str, factory, d: int, reference: float, rng
    ) -> None:
        """Crude sampling must find failures at roughly the reference rate.

        Guards the reference values themselves: if a problem definition changes,
        this fails rather than the estimate silently being compared against a
        stale number.
        """
        limit_state = factory()
        u = rng.standard_normal((200_000, d))
        g = np.asarray(limit_state(u)).reshape(-1)
        observed = float((g <= 0).mean())

        if reference > 1e-3:
            # Frequent enough to measure directly at this sample count.
            assert reference / 3 < observed < reference * 3, (
                f"{name}: observed {observed:.3e} against reference {reference:.3e}"
            )
        else:
            # Too rare for 2e5 samples to pin down; only require that the
            # failure region is reachable at all.
            assert observed < 1e-2, f"{name}: observed {observed:.3e}, far too common"


class TestNonlinearOscillator:
    """The oscillator benchmark cannot fail, and so measures nothing.

    Its displacement is computed as force_rms / (k * (1 - alpha)) with
    k = 5e6, giving values around 4e-7 against a threshold of z = 0.05. Reaching
    the threshold needs ||u|| of about 7.3e5, where the norm of a 10-dimensional
    standard normal averages 3.1. Crude Monte Carlo over 2e7 samples finds zero
    failures, and the CLI's `safe-ice benchmark oscillator` reports exactly 0.

    The scaling is inconsistent by a factor of roughly 1e5. Reconstructing the
    intended formulation needs the source paper, so the defect is recorded here
    rather than guessed at.
    """

    def test_failure_region_is_unreachable(self, rng) -> None:
        """Records the defect. Delete this test once the problem is fixed."""
        limit_state = BenchmarkProblems.nonlinear_oscillator_simplified()
        u = rng.standard_normal((50_000, 10))
        g = np.asarray(limit_state(u)).reshape(-1)

        # Every value sits just under the threshold, varying only at ~1e-7.
        assert float(g.min()) > 0.0499
        assert float((g <= 0).mean()) == 0.0

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "The nonlinear oscillator cannot fail: displacement is ~4e-7 "
            "against a threshold of 0.05, so ||u|| of about 7.3e5 would be "
            "needed where chi_10 averages 3.1. Crude Monte Carlo over 2e7 "
            "samples finds no failures. Fixing the scaling requires the source "
            "paper. Remove this marker once the problem is usable."
        ),
    )
    def test_should_be_a_usable_rare_event_benchmark(self, rng) -> None:
        """What the problem ought to do: fail sometimes, rarely."""
        limit_state = BenchmarkProblems.nonlinear_oscillator_simplified()
        u = rng.standard_normal((200_000, 10))
        g = np.asarray(limit_state(u)).reshape(-1)

        observed = float((g <= 0).mean())
        assert 0.0 < observed < 1e-2, f"observed failure rate {observed:.3e}"

    def test_wrapper_delegates_to_the_simplified_form(self, rng) -> None:
        """nonlinear_oscillator is a thin wrapper, so it shares the defect."""
        u = rng.standard_normal((1000, 10))
        wrapper = np.asarray(BenchmarkProblems.nonlinear_oscillator()(u)).reshape(-1)
        direct = np.asarray(
            BenchmarkProblems.nonlinear_oscillator_simplified()(u)
        ).reshape(-1)
        assert np.allclose(wrapper, direct)
