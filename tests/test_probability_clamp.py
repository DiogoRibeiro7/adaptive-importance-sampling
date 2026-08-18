"""The returned failure probability must be a probability.

The importance-sampling estimator is unbiased but unconstrained: the weights
``p(u) / q_safe(u)`` are a ratio of densities and nothing bounds them by 1, so
a single run can return a value above 1 even though the mean over runs is
exactly right. Callers expect a probability, so the returned value is clamped.

Clamping is not free in principle. Truncating the overshoots without touching
the undershoots biases the mean downwards -- by about 12% on a limit state that
fails everywhere, where the estimator scatters either side of a true value of
1. It is free in practice for the problems this package exists for: rare-event
estimates sit two to four orders of magnitude below 1 and never reach the
clamp at all.

That makes an overshoot worth surfacing rather than hiding. It means the
proposal is not covering the target, so the weights are degenerate and a few
samples carry the whole sum. Returning 1.0 silently would present such a run as
a converged one, so the raw value stays in ``results["pf_unclamped"]`` and a
warning is issued.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from safe_ice import AdaptiveSafeICE, OptimizedSafeICE, SafeICE
from safe_ice.problems.benchmarks import BenchmarkProblems


def always_fails(u: np.ndarray) -> np.ndarray:
    """True failure probability of exactly 1, the estimator's worst case."""
    return np.full(u.shape[0], -1.0)


class TestTheClampItself:
    """Unit-level behaviour of the bound, independent of any run."""

    @pytest.mark.parametrize("value", [0.0, 1e-9, 0.5, 1.0])
    def test_values_inside_the_range_pass_through(self, value: float) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert SafeICE._clamp_to_probability(value) == value

    @pytest.mark.parametrize("value", [1.0000001, 1.07, 4.15])
    def test_values_above_one_are_clamped_and_warned(self, value: float) -> None:
        with pytest.warns(RuntimeWarning, match="clamped"):
            assert SafeICE._clamp_to_probability(value) == 1.0

    def test_the_warning_explains_the_diagnosis(self) -> None:
        """A bare "clamped" message would not tell a user what went wrong."""
        with pytest.warns(RuntimeWarning) as record:
            SafeICE._clamp_to_probability(2.5)

        message = str(record[0].message)
        assert "does not cover the target" in message
        assert "degenerate" in message
        assert "pf_unclamped" in message

    def test_negative_values_are_clamped_too(self) -> None:
        """Unreachable in practice, but must not pass through if it happens."""
        with pytest.warns(RuntimeWarning, match="clamped"):
            assert SafeICE._clamp_to_probability(-0.5) == 0.0


class TestRareEventsAreUnaffected:
    """The regime the package is for never reaches the clamp."""

    @pytest.mark.slow
    @pytest.mark.parametrize("seed", range(4))
    def test_estimates_are_far_below_one(self, seed: int) -> None:
        with warnings.catch_warnings():
            # Any clamp warning here would mean the estimator is degenerate on
            # an ordinary rare-event problem, which is a failure, not a pass.
            warnings.simplefilter("error")
            pf, results = SafeICE(
                limit_state_function=BenchmarkProblems.four_mode_series_system(),
                dimension=2,
                N=1000,
                max_iterations=10,
                random_state=seed,
            ).run(verbose=False)

        assert pf < 1e-3
        assert pf == results["pf_unclamped"], "clamping altered a rare-event estimate"


class TestEveryVariantClamps:
    """OptimizedSafeICE and AdaptiveSafeICE inherit run(), so they must agree."""

    @pytest.mark.parametrize(
        "cls", [SafeICE, OptimizedSafeICE, AdaptiveSafeICE], ids=lambda c: c.__name__
    )
    def test_returned_value_is_a_probability(self, cls) -> None:
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            pf, results = cls(
                limit_state_function=always_fails,
                dimension=2,
                N=100,
                max_iterations=2,
                random_state=2,
            ).run(verbose=False)

        assert 0.0 <= pf <= 1.0, f"{cls.__name__} returned {pf}"
        assert "pf_unclamped" in results, f"{cls.__name__} dropped the raw value"
        assert results["pf_unclamped"] >= pf


class TestTheRawValueSurvives:
    """The point of option three: nothing is hidden."""

    def test_unclamped_value_is_reported_unchanged(self) -> None:
        """On whichever seeds overshoot, the raw value and warning must appear.

        Which seeds overshoot depends on the sampling path, so this scans a
        range rather than pinning one. Hard-coding a seed and its exact value
        made this test fail whenever an unrelated change moved the stream.
        """
        overshooting = 0

        for seed in range(15):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                pf, results = SafeICE(
                    limit_state_function=always_fails,
                    dimension=2,
                    N=100,
                    max_iterations=2,
                    random_state=seed,
                ).run(verbose=False)

            raw = results["pf_unclamped"]
            warned = any("clamped" in str(w.message) for w in caught)

            if raw > 1.0:
                overshooting += 1
                assert pf == 1.0, f"seed {seed}: raw {raw} was not clamped"
                assert warned, f"seed {seed}: clamped {raw} silently"
            else:
                assert pf == raw, f"seed {seed}: altered an in-range estimate"
                assert not warned, f"seed {seed}: warned without clamping"

        assert overshooting > 0, (
            "no seed overshot 1, so the clamp path was never exercised"
        )
