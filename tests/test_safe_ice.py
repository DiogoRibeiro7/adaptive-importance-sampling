"""Tests for the SafeICE algorithm."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from safe_ice import SafeICE
from safe_ice.core.parameters import vMFNMParameters
from safe_ice.problems.benchmarks import BenchmarkProblems


class TestSafeICEInitialization:
    """Test SafeICE initialization and configuration."""

    def test_basic_initialization(self, seed):
        """Test SafeICE can be initialized with minimal parameters."""

        def g(u):
            return 3.5 - np.linalg.norm(u, axis=-1)

        ice = SafeICE(
            limit_state_function=g,
            dimension=2,
            N=100,
            random_state=seed,
        )

        assert ice is not None
        assert ice.d == 2
        assert ice.N == 100
        assert ice.max_iterations == 20  # default value
        assert ice.delta_target == 4.0  # default value

    def test_initialization_with_custom_parameters(self, seed):
        """Test SafeICE initialization with custom parameters."""

        def g(u):
            return 3.5 - np.linalg.norm(u, axis=-1)

        ice = SafeICE(
            limit_state_function=g,
            dimension=5,
            N=500,
            max_iterations=50,
            delta_target=2.0,
            delta_star=1.0,
            sigma0=2.0,
            K0=15,
            em_max_iter=50,
            random_state=seed,
        )

        assert ice.d == 5
        assert ice.N == 500
        assert ice.max_iterations == 50
        assert ice.delta_target == 2.0
        assert ice.delta_star == 1.0
        assert ice.sigma0 == 2.0
        assert ice.K0 == 15


class TestSafeICEExecution:
    """Test SafeICE algorithm execution."""

    def test_single_iteration(self, seed):
        """Test that a single iteration runs without errors."""
        problems = BenchmarkProblems()
        g = problems.four_mode_series_system()

        ice = SafeICE(
            limit_state_function=g,
            dimension=2,
            N=100,
            max_iterations=1,
            random_state=seed,
        )

        # Run one iteration
        pf, results = ice.run(verbose=False)

        assert pf > 0
        assert pf < 1
        assert "final_samples" in results
        assert "final_weights" in results
        assert results["final_samples"].shape[1] == 2
        assert len(results["final_weights"]) == len(results["final_samples"])
        assert np.all(results["final_weights"] >= 0)

    def test_multiple_iterations(self, seed):
        """Test multiple iterations with convergence."""
        problems = BenchmarkProblems()
        g = problems.four_mode_series_system()

        ice = SafeICE(
            limit_state_function=g,
            dimension=2,
            N=200,
            max_iterations=5,
            random_state=seed,
        )

        pf, results = ice.run(verbose=False)

        assert pf > 0
        assert pf < 1e-3  # Known to be a rare event
        assert results["final_samples"].shape[0] >= 200  # At least one iteration
        assert results["final_samples"].shape[0] <= 200 * 5  # At most max_iterations
        assert results["final_samples"].shape[1] == 2
        assert len(results["final_weights"]) == len(results["final_samples"])

    def test_deterministic_limit_state(self, seed):
        """Test with a deterministic limit state function."""

        def g(u):
            # Simple sphere: failure if ||u|| > 3
            return 3.0 - np.linalg.norm(u, axis=-1)

        ice = SafeICE(
            limit_state_function=g,
            dimension=2,
            N=500,
            max_iterations=10,
            random_state=seed,
        )

        pf, _results = ice.run(verbose=False)

        # For a 2D standard normal, P(||u|| > 3) should be small
        assert pf > 0
        assert pf < 0.1  # Should be around 0.01

    def test_high_dimensional_problem(self, seed):
        """Test with higher dimensional problem."""

        def g(u):
            # High-dimensional sphere
            return 4.5 - np.linalg.norm(u, axis=-1)

        ice = SafeICE(
            limit_state_function=g,
            dimension=10,
            N=500,
            max_iterations=3,
            random_state=seed,
        )

        pf, results = ice.run(verbose=False)

        assert pf > 0
        assert results["final_samples"].shape[1] == 10
        assert len(results["final_weights"]) == len(results["final_samples"])

    @pytest.mark.parametrize("dimension", [2, 5, 10])
    def test_dimension_consistency(self, dimension, seed):
        """Test that output dimensions are consistent with input."""

        def g(u):
            return 3.5 - np.linalg.norm(u, axis=-1)

        ice = SafeICE(
            limit_state_function=g,
            dimension=dimension,
            N=100,
            max_iterations=2,
            random_state=seed,
        )

        _pf, results = ice.run(verbose=False)

        assert results["final_samples"].shape[1] == dimension


class TestSafeICEWithInitialParams:
    """Test SafeICE with initial parameter specification."""

    def test_with_initial_params(self, seed):
        """Test SafeICE with user-provided initial parameters."""

        def g(u):
            return 3.5 - np.linalg.norm(u, axis=-1)

        # Create initial parameters
        initial_params = vMFNMParameters(
            K=2,  # Two components
            d=2,  # 2D problem
            pi=np.array([0.6, 0.4]),
            m=np.array([2.0, 2.5]),
            Omega=np.array([1.0, 1.2]),
            mu=np.array([[1.0, 0.0], [0.0, 1.0]]) / np.sqrt(2),
            kappa=np.array([5.0, 3.0]),
        )

        ice = SafeICE(
            limit_state_function=g,
            dimension=2,
            N=200,
            max_iterations=3,
            random_state=seed,
        )

        pf, results = ice.run(initial_params=initial_params, verbose=False)

        assert pf > 0
        assert results["final_samples"].shape[1] == 2


class TestBenchmarkProblems:
    """Test with benchmark problems."""

    def test_four_mode_series_system(self, seed):
        """Test the four-mode series system benchmark."""
        problems = BenchmarkProblems()
        g = problems.four_mode_series_system()

        ice = SafeICE(
            limit_state_function=g,
            dimension=2,
            N=500,
            max_iterations=10,
            random_state=seed,
        )

        pf, _results = ice.run(verbose=False)

        # Crude Monte Carlo over 2e7 samples puts the reference at
        # 6.465e-05 +/- 2.8% for the default z=1.0.
        assert pf > 1e-6
        assert pf < 1e-4

    def test_three_mode_problem(self, seed):
        """Test the three-mode problem benchmark."""
        problems = BenchmarkProblems()
        g = problems.three_mode_problem()

        ice = SafeICE(
            limit_state_function=g,
            dimension=2,
            N=300,
            max_iterations=5,
            random_state=seed,
        )

        pf, _results = ice.run(verbose=False)

        # Reference probability is approximately 2.3e-3
        assert pf > 1e-4
        assert pf < 1e-2

    @pytest.mark.slow
    def test_two_mode_opposite_directions(self, seed):
        """Test the two-mode opposite directions benchmark (slow test)."""
        problems = BenchmarkProblems()
        # Test with dimension 10
        g = problems.two_mode_opposite_directions(dimension=10)

        ice = SafeICE(
            limit_state_function=g,
            dimension=10,
            N=500,
            max_iterations=5,
            random_state=seed,
        )

        pf, results = ice.run(verbose=False)

        assert pf > 0
        assert results["final_samples"].shape[1] == 10


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_limit_state_returns_nan(self, seed):
        """Test handling when limit state function returns NaN."""

        def g(u):
            # Return NaN for some inputs
            result = 3.5 - np.linalg.norm(u, axis=-1)
            result[0] = np.nan
            return result

        ice = SafeICE(
            limit_state_function=g,
            dimension=2,
            N=100,
            max_iterations=2,
            random_state=seed,
        )

        # Should handle NaN gracefully
        with pytest.warns(RuntimeWarning):
            _pf, _results = ice.run(verbose=False)

    def test_limit_state_returns_inf(self, seed):
        """Test handling when limit state function returns infinity."""

        def g(u):
            # Return infinity for some inputs
            result = 3.5 - np.linalg.norm(u, axis=-1)
            result[0] = np.inf
            return result

        ice = SafeICE(
            limit_state_function=g,
            dimension=2,
            N=100,
            max_iterations=2,
            random_state=seed,
        )

        # Should handle infinity gracefully
        pf, _results = ice.run(verbose=False)
        assert pf >= 0
        assert pf <= 1

    def test_all_samples_fail(self, seed):
        """Test when all samples are in failure region."""

        def g(u):
            # Always in failure (negative values)
            return -1.0 * np.ones(len(u))

        ice = SafeICE(
            limit_state_function=g,
            dimension=2,
            N=100,
            max_iterations=2,
            random_state=seed,
        )

        # True P_f is 1.0, which is the estimator's worst case: with no failure
        # boundary anywhere there is nothing for sigma to concentrate on, so the
        # raw estimate scatters either side of 1 and an overshoot warns.
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            pf, results = ice.run(verbose=False)

        assert 0.0 <= pf <= 1.0
        assert results["pf_unclamped"] >= pf
        assert pf > 0.5

    def test_no_samples_fail(self, seed):
        """Test when no samples are in failure region."""

        def g(u):
            # Never in failure (always positive)
            return 10.0 * np.ones(len(u))

        ice = SafeICE(
            limit_state_function=g,
            dimension=2,
            N=100,
            max_iterations=2,
            random_state=seed,
        )

        pf, _results = ice.run(verbose=False)

        # Should give very low probability
        assert pf < 1e-10


class TestConvergence:
    """Test convergence behavior."""

    def test_cv_convergence(self, seed):
        """Test that CV convergence criterion works."""

        def g(u):
            return 3.5 - np.linalg.norm(u, axis=-1)

        ice = SafeICE(
            limit_state_function=g,
            dimension=2,
            N=1000,
            max_iterations=20,
            delta_target=0.1,  # Strict convergence
            cv_tolerance=0.01,
            random_state=seed,
        )

        _pf, results = ice.run(verbose=False)

        # With strict CV, should converge or reach max_iterations
        assert len(results["final_samples"]) <= 1000 * 20

    def test_early_stopping(self, seed):
        """Test that algorithm stops early when converged."""

        def g(u):
            # Simple problem that should converge quickly
            return 2.0 - np.linalg.norm(u, axis=-1)

        ice = SafeICE(
            limit_state_function=g,
            dimension=2,
            N=500,
            max_iterations=20,  # Allow many iterations
            delta_target=0.5,  # Relaxed convergence,
            random_state=seed,
        )

        _pf, results = ice.run(verbose=False)

        # Should stop before max_iterations
        assert len(results["final_samples"]) < 500 * 20


class TestReproducibility:
    """Test deterministic RNG support via random_state."""

    def test_same_seed_same_result(self):
        """Two runs with identical seed produce identical outputs."""

        def g(u):
            return 3.5 - np.linalg.norm(u, axis=-1)

        pf1, _ = SafeICE(
            g,
            dimension=2,
            N=100,
            max_iterations=2,
            random_state=42,
        ).run(verbose=False)

        pf2, _ = SafeICE(
            g,
            dimension=2,
            N=100,
            max_iterations=2,
            random_state=42,
        ).run(verbose=False)

        assert pf1 == pf2

    def test_different_seed_different_result(self):
        """Two runs with different seeds produce different outputs."""

        def g(u):
            return 3.5 - np.linalg.norm(u, axis=-1)

        pf1, _ = SafeICE(
            g,
            dimension=2,
            N=200,
            max_iterations=3,
            random_state=1,
        ).run(verbose=False)

        pf2, _ = SafeICE(
            g,
            dimension=2,
            N=200,
            max_iterations=3,
            random_state=2,
        ).run(verbose=False)

        assert pf1 != pf2

    def test_generator_accepted(self):
        """A np.random.Generator can be passed directly."""

        def g(u):
            return 3.5 - np.linalg.norm(u, axis=-1)

        rng = np.random.default_rng(99)
        pf, _ = SafeICE(
            g,
            dimension=2,
            N=100,
            max_iterations=2,
            random_state=rng,
        ).run(verbose=False)

        assert np.isfinite(pf) and pf >= 0

    def test_none_is_backward_compatible(self):
        """random_state=None uses global np.random (no crash)."""

        def g(u):
            return 3.5 - np.linalg.norm(u, axis=-1)

        pf, _ = SafeICE(
            g,
            dimension=2,
            N=100,
            max_iterations=1,
            random_state=None,
        ).run(verbose=False)

        assert np.isfinite(pf) and pf >= 0
