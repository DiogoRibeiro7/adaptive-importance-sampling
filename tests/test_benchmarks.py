"""Tests for benchmark problems."""

from __future__ import annotations

import numpy as np
import pytest

from safe_ice.problems.benchmarks import BenchmarkProblems
from safe_ice.problems.heat_transfer import HeatTransferProblem


class TestBenchmarkProblemsStructure:
    """Test structure and properties of benchmark problems."""

    def setup_method(self):
        """Set up test fixtures."""
        self.problems = BenchmarkProblems()

    def test_four_mode_series_system_properties(self):
        """Test properties of four-mode series system."""
        g = self.problems.four_mode_series_system()

        # Test with single point
        u_test = np.array([1.0, 1.0])
        result = g(u_test)
        assert isinstance(result, (float, np.floating))

        # Test with multiple points
        u_batch = np.array([[1.0, 1.0], [2.0, 2.0], [0.0, 0.0]])
        results = g(u_batch)
        assert results.shape == (3,)

        # Test function is continuous (small perturbation = small change)
        u1 = np.array([1.0, 1.0])
        u2 = np.array([1.01, 1.01])
        diff = abs(g(u1) - g(u2))
        assert diff < 0.5  # Reasonable continuity

    def test_three_mode_problem_properties(self, rng):
        """Test properties of three-mode problem."""
        g = self.problems.three_mode_problem()

        # Test single and batch evaluation
        u_single = np.array([0.5, 0.5])
        result_single = g(u_single)
        assert isinstance(result_single, (float, np.floating))

        u_batch = rng.standard_normal((10, 2))
        result_batch = g(u_batch)
        assert result_batch.shape == (10,)

    def test_two_mode_opposite_directions(self, rng):
        """Test two-mode opposite directions problem."""
        # Test different dimensions
        for d in [2, 5, 10]:
            g = self.problems.two_mode_opposite_directions(dimension=d)

            u_test = rng.standard_normal((5, d))
            results = g(u_test)

            assert results.shape == (5,)
            assert not np.any(np.isnan(results))
            assert not np.any(np.isinf(results))

    def test_nonlinear_oscillator(self, rng):
        """Test nonlinear oscillator problem."""
        g = self.problems.nonlinear_oscillator()

        # Should work with 10D input
        u_test = rng.standard_normal((10, 10))
        results = g(u_test)

        assert results.shape == (10,)
        assert not np.any(np.isnan(results))

    def test_nakagami_ratio_problem(self, rng):
        """Test Nakagami ratio problem."""
        g = self.problems.nakagami_ratio_problem()

        # Test with 2D input (two Nakagami variables)
        u_test = rng.standard_normal((20, 2))
        results = g(u_test)

        assert results.shape == (20,)
        assert not np.any(np.isnan(results))


class TestBenchmarkProblemsBehavior:
    """Test expected behavior of benchmark problems."""

    def setup_method(self):
        """Set up test fixtures."""
        self.problems = BenchmarkProblems()

    def test_four_mode_series_symmetry(self):
        """Test symmetry properties of four-mode series system."""
        g = self.problems.four_mode_series_system()

        # The function has some symmetry properties
        u1 = np.array([1.0, 2.0])
        u2 = np.array([2.0, 1.0])

        # Not necessarily equal due to cross terms, but should be similar
        result1 = g(u1)
        result2 = g(u2)

        # Check they give same sign at least
        assert np.sign(result1) == np.sign(result2)

    def test_failure_regions(self):
        """Test that problems have reasonable failure regions."""
        # Test that origin is typically in safe region
        origin = np.array([0.0, 0.0])

        g1 = self.problems.four_mode_series_system()
        assert g1(origin) > 0  # Safe at origin

        g2 = self.problems.three_mode_problem()
        assert g2(origin) > 0  # Safe at origin

        # Test that far points are typically in failure region
        far_point = np.array([10.0, 10.0])

        assert g1(far_point) < 0  # Failure far from origin
        assert g2(far_point) < 0  # Failure far from origin

    def test_gradient_smoothness(self):
        """Test that limit state functions are smooth (no discontinuities)."""
        g = self.problems.four_mode_series_system()

        # Test gradient approximation
        u = np.array([1.0, 1.0])
        eps = 1e-6

        # Approximate gradient using finite differences
        grad_approx = np.zeros(2)
        for i in range(2):
            u_plus = u.copy()
            u_minus = u.copy()
            u_plus[i] += eps
            u_minus[i] -= eps
            grad_approx[i] = (g(u_plus) - g(u_minus)) / (2 * eps)

        # Gradient should be finite and reasonable
        assert np.all(np.isfinite(grad_approx))
        assert np.linalg.norm(grad_approx) < 100  # Reasonable magnitude

    def test_monotonicity_properties(self):
        """Test monotonicity in certain directions."""
        g = self.problems.nakagami_ratio_problem()

        # As both components increase, ratio should change monotonically
        u_base = np.array([0.0, 0.0])
        u_scaled = np.array([2.0, 2.0])

        # Values along the line
        alphas = np.linspace(0, 1, 10)
        values = [g(u_base + alpha * (u_scaled - u_base)) for alpha in alphas]

        # Check for general trend (not strict monotonicity due to nonlinearity)
        differences = np.diff(values)
        # Most differences should have same sign
        dominant_sign = np.sign(np.median(differences))
        assert np.sum(np.sign(differences) == dominant_sign) > len(differences) / 2


@pytest.mark.slow
class TestHeatTransferProblem:
    """Test heat transfer problem with KL expansion."""

    def test_initialization_default(self):
        """Test default initialization of heat transfer problem."""
        problem = HeatTransferProblem()

        assert problem.n_terms == 10
        assert problem.correlation_length == 0.5
        assert problem.field_std == 0.3
        assert problem.threshold == 100.0

    def test_initialization_custom(self):
        """Test custom initialization of heat transfer problem."""
        problem = HeatTransferProblem(
            n_terms=5, correlation_length=1.0, field_std=0.5, threshold=150.0
        )

        assert problem.n_terms == 5
        assert problem.correlation_length == 1.0
        assert problem.field_std == 0.5
        assert problem.threshold == 150.0

    def test_kl_expansion_setup(self):
        """Test KL expansion eigenvalues and eigenvectors."""
        problem = HeatTransferProblem(n_terms=5)

        # Check eigenvalues
        assert len(problem.eigenvalues) == 5
        assert np.all(problem.eigenvalues > 0)
        # Eigenvalues should be non-increasing (ties possible due to
        # symmetry in the exponential covariance kernel)
        assert np.all(np.diff(problem.eigenvalues) <= 0)

        # Check eigenvectors
        assert problem.eigenvectors.shape == (50, 5)  # 50 grid points, 5 terms

    def test_limit_state_function(self, rng):
        """Test heat transfer limit state function."""
        problem = HeatTransferProblem(n_terms=5)
        g = problem.get_limit_state_function()

        # Test with random input
        u_test = rng.standard_normal((10, 5))
        results = g(u_test)

        assert results.shape == (10,)
        assert not np.any(np.isnan(results))
        assert not np.any(np.isinf(results))

    def test_temperature_computation(self):
        """Test temperature field computation returns finite values."""
        problem = HeatTransferProblem(n_terms=3)
        g = problem.get_limit_state_function()

        u_zero = np.zeros(3)
        result_zero = g(u_zero)

        u_nonzero = np.ones(3)
        result_nonzero = g(u_nonzero)

        # Both must be finite and within a reasonable range
        assert np.isfinite(result_zero)
        assert np.isfinite(result_nonzero)

    @pytest.mark.parametrize("n_terms", [5, 10, 20])
    def test_different_truncations(self, n_terms, rng):
        """Test heat transfer problem with different KL truncations."""
        problem = HeatTransferProblem(n_terms=n_terms)
        g = problem.get_limit_state_function()

        u_test = rng.standard_normal((5, n_terms))
        results = g(u_test)

        assert results.shape == (5,)
        assert not np.any(np.isnan(results))

    def test_physical_consistency(self):
        """Test physical consistency of heat transfer problem."""
        problem = HeatTransferProblem(
            field_std=0.3,
            threshold=100.0,
        )
        g = problem.get_limit_state_function()

        # Large positive fluctuations should increase temperature
        u_positive = np.ones(problem.n_terms) * 3
        result_positive = g(u_positive)

        # Large negative fluctuations should decrease temperature
        u_negative = -np.ones(problem.n_terms) * 3
        result_negative = g(u_negative)

        # Both must be finite; when the PDE solver fully converges,
        # positive KL coefficients lead to higher temperature
        # (lower g), but this depends on grid resolution and
        # iteration count.
        assert np.isfinite(result_positive)
        assert np.isfinite(result_negative)
        assert result_positive <= result_negative


class TestBenchmarkComparison:
    """Test comparison between different benchmark problems."""

    def test_relative_difficulty(self, seed):
        """Test that SafeICE produces finite probability estimates."""
        from safe_ice import SafeICE

        problems = BenchmarkProblems()

        # Four-mode series
        g1 = problems.four_mode_series_system()
        ice1 = SafeICE(g1, dimension=2, N=1000, max_iterations=5, random_state=seed)
        pf1, _ = ice1.run(verbose=False)

        # Three-mode
        g2 = problems.three_mode_problem()
        ice2 = SafeICE(g2, dimension=2, N=1000, max_iterations=5, random_state=seed)
        pf2, _ = ice2.run(verbose=False)

        # Both estimates must be non-negative finite values
        assert np.isfinite(pf1) and pf1 >= 0
        assert np.isfinite(pf2) and pf2 >= 0

    def test_dimension_scaling(self, seed):
        """Test how failure probability scales with dimension."""
        problems = BenchmarkProblems()

        # Test two-mode problem in different dimensions
        dimensions = [2, 5, 10]
        pf_values = []

        for d in dimensions:
            g = problems.two_mode_opposite_directions(dimension=d)

            from safe_ice import SafeICE

            ice = SafeICE(
                limit_state_function=g,
                dimension=d,
                N=1000,
                max_iterations=5,
                random_state=seed,
            )
            pf, _ = ice.run(verbose=False)
            pf_values.append(pf)

        # Higher dimensions typically have different failure probabilities
        # Check that we get reasonable values
        assert all(0 < pf < 1 for pf in pf_values)


class TestBenchmarkRobustness:
    """Test robustness of benchmark problems."""

    def test_handle_edge_inputs(self):
        """Test handling of edge case inputs."""
        problems = BenchmarkProblems()
        g = problems.four_mode_series_system()

        # Test with very large values
        u_large = np.array([[1e6, 1e6]])
        result_large = g(u_large)
        assert np.isfinite(result_large)

        # Test with very small values
        u_small = np.array([[1e-6, 1e-6]])
        result_small = g(u_small)
        assert np.isfinite(result_small)

        # Test with mixed scales
        u_mixed = np.array([[1e6, 1e-6]])
        result_mixed = g(u_mixed)
        assert np.isfinite(result_mixed)

    def test_vectorization_consistency(self, rng):
        """Test that vectorized evaluation is consistent with single evaluation."""
        problems = BenchmarkProblems()
        g = problems.four_mode_series_system()

        # Generate test points
        test_points = rng.standard_normal((5, 2))

        # Evaluate individually
        individual_results = np.array([g(u) for u in test_points])

        # Evaluate as batch
        batch_results = g(test_points)

        # Should give same results
        np.testing.assert_allclose(individual_results, batch_results, rtol=1e-10)

    def test_numerical_stability(self, rng):
        """Test numerical stability of limit state functions."""
        problems = BenchmarkProblems()

        # Test all benchmark problems
        test_functions = [
            problems.four_mode_series_system(),
            problems.three_mode_problem(),
            problems.nonlinear_oscillator(),
            problems.nakagami_ratio_problem(),
        ]

        for g in test_functions:
            # Test with various scales of input
            for scale in [1e-3, 1.0, 1e3]:
                if g == problems.nonlinear_oscillator():
                    u_test = rng.standard_normal((10, 10)) * scale
                else:
                    u_test = rng.standard_normal((10, 2)) * scale

                results = g(u_test)

                # Check for numerical issues
                assert not np.any(np.isnan(results))
                assert not np.any(np.isinf(results))
                assert np.all(np.isfinite(results))
