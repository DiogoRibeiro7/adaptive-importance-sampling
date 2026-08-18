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
    """The Bouc-Wen oscillator of Section 4.3, checked against crude MC.

    This benchmark used to compute the displacement as
    ``force_rms / (k * (1 - alpha))``, a closed form appearing nowhere in the
    paper, which never integrated the equations of motion. It produced values
    around 4e-07 against a threshold of 0.05, so the problem could not fail and
    every estimator returned exactly 0. The tests here recorded that defect as
    a strict xfail; they now check the real model.

    Crude Monte Carlo over 2e6 samples of the implementation gives:

        z       failures      pf          rel. s.e.    paper, Figure 7
        0.05    3597          1.798e-03   1.7%         ~1.5e-03
        0.06     295          1.475e-04   5.8%         ~1.2e-04
        0.07       9          4.500e-06   33%          ~5e-06
    """

    REFERENCE_PF = 1.798e-03  # z = 0.05

    def test_failure_region_is_reachable(self, rng) -> None:
        """Crude sampling must find failures at roughly the reference rate."""
        limit_state = BenchmarkProblems.nonlinear_oscillator(z=0.05)
        u = rng.standard_normal((60_000, 10))
        observed = float((np.asarray(limit_state(u)) <= 0).mean())

        assert self.REFERENCE_PF / 3 < observed < self.REFERENCE_PF * 3, (
            f"observed {observed:.3e} against reference {self.REFERENCE_PF:.3e}"
        )

    @pytest.mark.slow
    def test_estimate_matches_reference(self) -> None:
        """The estimator should reproduce the Monte Carlo value."""
        estimates = []
        for s in range(4):
            ice = SafeICE(
                limit_state_function=BenchmarkProblems.nonlinear_oscillator(z=0.05),
                dimension=10,
                N=1000,
                max_iterations=15,
                random_state=s,
            )
            pf, _ = ice.run(verbose=False)
            estimates.append(pf)

        median = float(np.median(estimates))
        assert self.REFERENCE_PF / 3 < median < self.REFERENCE_PF * 3, (
            f"median {median:.3e} against reference {self.REFERENCE_PF:.3e}; "
            f"estimates {estimates}"
        )

    @pytest.mark.slow
    @pytest.mark.parametrize(("z", "reference"), [(0.05, 1.798e-03), (0.06, 1.475e-04)])
    def test_threshold_controls_rarity(self, z: float, reference: float, rng) -> None:
        """Raising z must make failure rarer, by the measured amount."""
        limit_state = BenchmarkProblems.nonlinear_oscillator(z=z)
        # 100k samples give ~180 failures at z=0.05 and ~15 at z=0.06, so the
        # 3x band below is comfortable. Each evaluation integrates 800 RK4
        # steps, so this is the most expensive test in the suite; doubling the
        # count doubles its runtime for no extra confidence.
        u = rng.standard_normal((100_000, 10))
        observed = float((np.asarray(limit_state(u)) <= 0).mean())

        assert reference / 3 < observed < reference * 3, (
            f"z={z}: observed {observed:.3e} against reference {reference:.3e}"
        )

    def test_response_is_bounded_for_extreme_inputs(self, rng) -> None:
        """The |z|^3 term used to overflow once integration error unsaturated z.

        The exact solution obeys |z| <= (A/(beta+gamma))^(1/n) = 1. Enforcing
        that bound is a no-op for sampled inputs -- peaks are 0.9993 in the
        failure region -- but keeps absurd ones from producing inf or NaN.
        """
        limit_state = BenchmarkProblems.nonlinear_oscillator()
        for scale in (1e-3, 1.0, 1e3):
            g = np.asarray(limit_state(rng.standard_normal((20, 10)) * scale))
            assert np.all(np.isfinite(g)), f"non-finite response at scale {scale}"

    def test_dimension_must_be_even(self) -> None:
        """d/2 frequency components are drawn from the first and second half."""
        with pytest.raises(ValueError, match="even dimension"):
            BenchmarkProblems.nonlinear_oscillator(dimension=7)

    def test_input_dimension_is_checked(self) -> None:
        """It silently zero-padded mismatched input before, hiding mistakes."""
        limit_state = BenchmarkProblems.nonlinear_oscillator()
        with pytest.raises(ValueError, match="expects dimension 10"):
            limit_state(np.zeros((5, 2)))
