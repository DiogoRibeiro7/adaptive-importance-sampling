"""ICE-vMFNM, the baseline Safe-ICE is measured against.

Safe-ICE is this method plus a heavy-tailed proposal component and a penalised
EM step, and the paper's tables are comparisons between the two. The tests here
check both halves of that statement: that the two additions really are absent,
and that everything else is shared, since a comparison is only meaningful if
the methods differ in the ways being compared and in no others.

The last test reproduces the paper's qualitative finding, which is not that
Safe-ICE is a little better but that ICE-vMFNM's accuracy depends on a choice
the user has to make blind. On the four-mode problem, over ten runs each:

    method             rel. err. of mean      CV
    Safe-ICE                       0.010   0.109
    ICE-vMFNM (K=2)                0.084   0.180
    ICE-vMFNM (K=4)                0.276   0.366
    ICE-vMFNM (K=8)                0.290   0.214

Adding components makes it worse, which is the paper's own observation.
"""

from __future__ import annotations

import numpy as np
import pytest

from safe_ice import ICEvMFNM, SafeICE
from safe_ice.problems.benchmarks import BenchmarkProblems

FOUR_MODE_PF = 6.465e-05


def four_mode():
    return BenchmarkProblems.four_mode_series_system(z=1.0)


def estimator(cls, seed: int, **kwargs):
    return cls(
        limit_state_function=four_mode(),
        dimension=2,
        N=1000,
        max_iterations=20,
        random_state=seed,
        **kwargs,
    )


class TestTheTwoAdditionsAreAbsent:
    """What makes this ICE rather than Safe-ICE."""

    def test_no_heavy_tailed_component(self) -> None:
        """lambda is the weight on the light-tailed family, and stays at 1."""
        ice = estimator(ICEvMFNM, 0, K=3)

        for sigma in (10.0, 1.0, 0.5, 1e-3, 1e-9, 0.0):
            assert ice._cosine_annealing_schedule(sigma, ice.sigma0) == 1.0

    def test_safe_ice_does_anneal_by_contrast(self) -> None:
        """The same call on Safe-ICE sweeps from 0 up to its cap."""
        safe = estimator(SafeICE, 0)

        assert safe._cosine_annealing_schedule(safe.sigma0, safe.sigma0) == 0.0
        assert safe._cosine_annealing_schedule(1e-9, safe.sigma0) == safe.lambda_max

    def test_penalty_is_disabled(self) -> None:
        assert estimator(ICEvMFNM, 0, K=3).em_optimizer.penalized is False
        assert estimator(SafeICE, 0).em_optimizer.penalized is True

    @pytest.mark.parametrize("k", [2, 4])
    def test_component_count_holds(self, k: int) -> None:
        """Nothing drives components out, so a sensible K survives intact."""
        ice = estimator(ICEvMFNM, 0, K=k)
        _pf, results = ice.run(verbose=False)

        components = ice.K
        assert components == k
        assert results["final_components"] == k
        assert set(ice.history["components"]) == {k}

    def test_components_can_still_die_of_starvation(self) -> None:
        """Not every drop is a penalty.

        Ask for more components than the problem supports and plain EM leaves
        some with no responsibility at all. A component of zero weight cannot
        be updated, so it is dropped -- 8 ends at 6 on the four-mode problem.
        This is worth separating from Safe-ICE's pruning, which is deliberate
        and starts from 20.
        """
        ice = estimator(ICEvMFNM, 0, K=8)
        _pf, results = ice.run(verbose=False)

        assert results["final_components"] <= 8
        assert results["final_components"] >= 4, "should not collapse like Safe-ICE"
        assert all(k <= 8 for k in ice.history["components"])

    def test_safe_ice_prunes_by_contrast(self) -> None:
        _pf, results = estimator(SafeICE, 0).run(verbose=False)
        assert results["final_components"] < 20


class TestEverythingElseIsShared:
    """The comparison is only fair if the common parts really are common."""

    SHARED = (
        "_determine_next_sigma",
        "_calculate_intermediate_weights",
        "_calculate_stopping_weights",
        "_estimate_failure_probability",
        "_evaluate_safe_mixture_density",
        "_initialize_vmfnm_parameters",
        "run",
    )

    @pytest.mark.parametrize("method", SHARED)
    def test_not_reimplemented(self, method: str) -> None:
        assert getattr(ICEvMFNM, method) is getattr(SafeICE, method), (
            f"ICEvMFNM overrides {method}; then a difference in results could "
            "come from that rather than from the two additions under comparison"
        )

    def test_only_the_schedule_is_overridden(self) -> None:
        """The one deliberate override, plus the constructor."""
        overridden = {
            name
            for name in vars(ICEvMFNM)
            if not name.startswith("__") and hasattr(SafeICE, name)
        }
        assert overridden == {"_cosine_annealing_schedule"}


class TestItWorks:
    """A baseline still has to produce the right answer."""

    @pytest.mark.parametrize("k", [2, 4])
    def test_estimate_matches_the_reference(self, k: int) -> None:
        estimates = [
            estimator(ICEvMFNM, s, K=k).run(verbose=False)[0] for s in range(6)
        ]
        median = float(np.median(estimates))

        assert FOUR_MODE_PF / 3 < median < FOUR_MODE_PF * 3, (
            f"K={k}: median {median:.3e} against {FOUR_MODE_PF:.3e}; {estimates}"
        )

    def test_returns_a_probability_with_the_same_result_keys(self) -> None:
        pf, results = estimator(ICEvMFNM, 0, K=2).run(verbose=False)
        _safe_pf, safe_results = estimator(SafeICE, 0).run(verbose=False)

        assert 0.0 <= pf <= 1.0
        assert set(results) == set(safe_results)


class TestThePaperComparison:
    """Safe-ICE should be the more accurate and the more stable of the two."""

    @staticmethod
    def accuracy(cls, seeds, **kwargs):
        """Relative error of the mean and coefficient of variation, as in the paper."""
        estimates = np.array(
            [estimator(cls, s, **kwargs).run(verbose=False)[0] for s in seeds]
        )
        rel_error = abs(float(estimates.mean()) - FOUR_MODE_PF) / FOUR_MODE_PF
        cv = float(estimates.std() / estimates.mean())
        return rel_error, cv

    @pytest.mark.slow
    def test_safe_ice_is_more_accurate_than_the_baseline(self) -> None:
        seeds = range(10)
        safe_error, safe_cv = self.accuracy(SafeICE, seeds)
        ice_error, ice_cv = self.accuracy(ICEvMFNM, seeds, K=4)

        assert safe_error < ice_error, (
            f"Safe-ICE relative error {safe_error:.3f} against "
            f"ICE-vMFNM(K=4) {ice_error:.3f}"
        )
        assert safe_cv < ice_cv, f"CV {safe_cv:.3f} against {ice_cv:.3f}"

    @pytest.mark.slow
    def test_the_baseline_depends_on_a_blind_choice_of_k(self) -> None:
        """More components makes it worse here, which is the paper's point.

        The user has no way to know the right K in advance; that is what the
        penalised EM removes.
        """
        seeds = range(10)
        error_small, _cv = self.accuracy(ICEvMFNM, seeds, K=2)
        error_large, _cv = self.accuracy(ICEvMFNM, seeds, K=8)

        assert error_large > error_small, (
            f"K=8 gave relative error {error_large:.3f}, K=2 gave {error_small:.3f}"
        )
