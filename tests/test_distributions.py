# tests/test_distributions.py
"""Tests for distribution implementations."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest
from scipy.stats import chi2, kstest  # noqa: F401  (may be used in future tests)

from safe_ice.core.parameters import vMFNMParameters
from safe_ice.distributions.mixture import vMFNMDistribution
from safe_ice.distributions.nakagami import (
    InverseNakagamiDistribution,
    NakagamiDistribution,
)
from safe_ice.distributions.vmf import VonMisesFisherSampler


class TestVonMisesFisherSampler:
    """Tests for von Mises-Fisher distribution sampler."""

    def test_sample_shape(self, rng) -> None:
        """Test that samples have correct shape."""
        mu: npt.NDArray[np.float64] = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        kappa: float = 2.0
        n_samples: int = 100

        samples = VonMisesFisherSampler.sample(mu, kappa, n_samples, rng=rng)
        assert samples.shape == (n_samples, 3)

    def test_samples_are_unit_vectors(self, rng) -> None:
        """Test that all samples are unit vectors."""
        mu: npt.NDArray[np.float64] = np.array([1.0, 0.0], dtype=np.float64)
        kappa: float = 5.0
        n_samples: int = 50

        samples = VonMisesFisherSampler.sample(mu, kappa, n_samples, rng=rng)
        norms = np.linalg.norm(samples, axis=1)

        np.testing.assert_allclose(norms, 1.0, rtol=1e-10)

    def test_zero_kappa_uniform(self, rng) -> None:
        """Test that kappa=0 gives uniform distribution."""
        mu: npt.NDArray[np.float64] = np.array([1.0, 0.0], dtype=np.float64)
        kappa: float = 0.0
        n_samples: int = 1000

        samples = VonMisesFisherSampler.sample(mu, kappa, n_samples, rng=rng)

        # Check mean direction is close to zero
        mean_direction = np.mean(samples, axis=0)
        assert np.linalg.norm(mean_direction) < 0.2

    def test_high_kappa_concentration(self, rng) -> None:
        """Test that high kappa concentrates around mean."""
        mu: npt.NDArray[np.float64] = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        kappa: float = 50.0
        n_samples: int = 100

        samples = VonMisesFisherSampler.sample(mu, kappa, n_samples, rng=rng)

        # All samples should be close to mu
        dot_products = np.dot(samples, mu)
        assert np.all(dot_products > 0.9)

    def test_circular_vmf(self, rng) -> None:
        """Test 2D circular case."""
        mu: npt.NDArray[np.float64] = np.array([1.0, 0.0], dtype=np.float64)
        kappa: float = 3.0
        n_samples: int = 200

        samples = VonMisesFisherSampler.sample(mu, kappa, n_samples, rng=rng)

        assert samples.shape == (n_samples, 2)
        norms = np.linalg.norm(samples, axis=1)
        np.testing.assert_allclose(norms, 1.0, rtol=1e-10)


class TestNakagamiDistribution:
    """Tests for Nakagami distribution."""

    def test_pdf_positive(self) -> None:
        """Test that PDF is positive for r > 0."""
        m, Omega = 2.0, 1.0
        r: npt.NDArray[np.float64] = np.linspace(0.1, 5.0, 50, dtype=np.float64)

        pdf_values = NakagamiDistribution.pdf(r, m, Omega)

        assert np.all(pdf_values >= 0)

    def test_pdf_zero_at_zero(self) -> None:
        """Test that PDF is zero at r = 0."""
        m, Omega = 2.0, 1.0

        pdf_value = NakagamiDistribution.pdf(0.0, m, Omega)

        assert pdf_value == 0.0

    def test_sample_shape(self, rng) -> None:
        """Test sample shape."""
        m, Omega = 1.5, 2.0
        n_samples = 100

        samples = NakagamiDistribution.sample(m, Omega, n_samples, rng=rng)

        assert samples.shape == (n_samples,)
        assert np.all(samples > 0)

    def test_sample_mean_approximation(self, rng) -> None:
        """Test that sample mean approximates theoretical mean."""
        m, Omega = 2.0, 1.0
        n_samples = 10000

        samples = NakagamiDistribution.sample(m, Omega, n_samples, rng=rng)
        sample_mean = float(np.mean(samples))

        # Theoretical mean: E[R] = (Gamma(m + 0.5) / Gamma(m)) * sqrt(Omega / m)
        from scipy.special import gamma

        theoretical_mean = float((gamma(m + 0.5) / gamma(m)) * np.sqrt(Omega / m))

        # Allow 5% error due to sampling
        np.testing.assert_allclose(sample_mean, theoretical_mean, rtol=0.05)

    def test_cdf_bounds(self) -> None:
        """Test CDF is between 0 and 1."""
        m, Omega = 2.0, 1.0
        r: npt.NDArray[np.float64] = np.linspace(0, 5, 50, dtype=np.float64)

        cdf_values = NakagamiDistribution.cdf(r, m, Omega)

        assert np.all(cdf_values >= 0)
        assert np.all(cdf_values <= 1)
        assert np.allclose(cdf_values[-1], 1.0, atol=0.01)


class TestInverseNakagamiDistribution:
    """Tests for Inverse Nakagami distribution."""

    def test_pdf_positive(self) -> None:
        """Test that PDF is positive for r > 0."""
        m, Omega = 2.0, 1.0
        r: npt.NDArray[np.float64] = np.linspace(0.1, 5.0, 50, dtype=np.float64)

        pdf_values = InverseNakagamiDistribution.pdf(r, m, Omega)

        assert np.all(pdf_values >= 0)

    def test_sample_positive(self, rng) -> None:
        """Test that all samples are positive."""
        m, Omega = 1.5, 2.0
        n_samples = 100

        samples = InverseNakagamiDistribution.sample(m, Omega, n_samples, rng=rng)

        assert samples.shape == (n_samples,)
        assert np.all(samples > 0)

    def test_heavy_tail_property(self, rng) -> None:
        """Test that distribution has heavy tails."""
        m, Omega = 2.0, 1.0
        n_samples = 10000

        samples = InverseNakagamiDistribution.sample(m, Omega, n_samples, rng=rng)

        # Heavy-tailed: should have some very large values
        assert float(np.max(samples)) > 5.0
        # Should also have small values
        assert float(np.min(samples)) < 0.5


class TestvMFNMDistribution:
    """Tests for vMFNM mixture distribution."""

    @pytest.fixture
    def simple_params_2d(self) -> vMFNMParameters:
        """Simple 2D vMFNM parameters."""
        return vMFNMParameters(
            pi=np.array([0.6, 0.4], dtype=np.float64),
            m=np.array([2.0, 1.5], dtype=np.float64),
            Omega=np.array([1.0, 1.5], dtype=np.float64),
            mu=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64),
            kappa=np.array([3.0, 2.0], dtype=np.float64),
        )

    def test_initialization(self, simple_params_2d: vMFNMParameters) -> None:
        """Test distribution initialization."""
        dist = vMFNMDistribution(simple_params_2d)

        assert dist.params.K == 2
        assert dist.params.d == 2

    def test_pdf_positive(self, simple_params_2d: vMFNMParameters, rng) -> None:
        """Test that PDF is non-negative."""
        dist = vMFNMDistribution(simple_params_2d)

        x = rng.standard_normal((10, 2))
        pdf_values = dist.pdf(x)

        assert np.all(pdf_values >= 0)

    def test_sample_shape(self, simple_params_2d: vMFNMParameters, rng) -> None:
        """Test sample shape."""
        dist = vMFNMDistribution(simple_params_2d)
        n_samples = 100

        samples = dist.sample(n_samples, rng=rng)

        assert samples.shape == (n_samples, 2)

    def test_sample_pdf_consistency(
        self, simple_params_2d: vMFNMParameters, rng
    ) -> None:
        """Test that samples have reasonable PDF values."""
        dist = vMFNMDistribution(simple_params_2d)
        n_samples = 50

        samples = dist.sample(n_samples, rng=rng)
        pdf_values = dist.pdf(samples)

        # All PDF values should be positive
        assert np.all(pdf_values > 0)

    def test_log_likelihood(self, simple_params_2d: vMFNMParameters, rng) -> None:
        """Test log-likelihood computation."""
        dist = vMFNMDistribution(simple_params_2d)

        samples = dist.sample(100, rng=rng)
        log_likelihood = float(dist.log_likelihood(samples))

        # Should be finite and negative
        assert np.isfinite(log_likelihood)
        assert log_likelihood < 0

    @pytest.mark.parametrize("dimension", [2, 5, 10])
    def test_different_dimensions(self, dimension: int, rng) -> None:
        """Test distribution works in different dimensions."""
        K = 3
        params = vMFNMParameters(
            pi=(np.ones(K, dtype=np.float64) / K),
            m=rng.uniform(1, 3, K).astype(np.float64),
            Omega=rng.uniform(0.5, 2, K).astype(np.float64),
            mu=rng.standard_normal((K, dimension)).astype(np.float64),
            kappa=rng.uniform(0.5, 3, K).astype(np.float64),
        )

        dist = vMFNMDistribution(params)
        samples = dist.sample(50, rng=rng)

        assert samples.shape == (50, dimension)


class TestIntegration:
    """Integration tests for distributions."""

    def test_vmfnm_mixture_normalization(self, rng) -> None:
        """Test that vMFNM PDF approximately integrates to 1."""
        # This is a Monte Carlo integration test
        params = vMFNMParameters(
            pi=np.array([1.0], dtype=np.float64),
            m=np.array([2.0], dtype=np.float64),
            Omega=np.array([1.0], dtype=np.float64),
            mu=np.array([[1.0, 0.0]], dtype=np.float64),
            kappa=np.array([2.0], dtype=np.float64),
        )

        dist = vMFNMDistribution(params)

        # Generate many samples from a wide proposal N(0, 9I)
        n_samples = 100000
        proposal_std = 3.0
        samples = (rng.standard_normal((n_samples, 2)) * proposal_std).astype(
            np.float64
        )

        # Approximate ∫ f(x) dx via importance sampling with the actual
        # proposal density q(x) = N(0, σ²I).
        pdf_values = dist.pdf(samples)
        d = 2
        proposal_var = proposal_std**2
        proposal_pdf = (1.0 / (2.0 * np.pi * proposal_var) ** (d / 2.0)) * np.exp(
            -0.5 * np.sum(samples**2, axis=1) / proposal_var
        )

        integral_estimate = float(np.mean(pdf_values / proposal_pdf))

        # Allow large tolerance due to Monte Carlo approximation
        # This is more of a sanity check
        assert 0.5 < integral_estimate < 2.0

    @pytest.mark.slow
    def test_vmfnm_sampling_consistency(self, rng) -> None:
        """Test sampling and PDF evaluation are consistent."""
        params = vMFNMParameters(
            pi=np.array([0.5, 0.5], dtype=np.float64),
            m=np.array([2.0, 1.5], dtype=np.float64),
            Omega=np.array([1.0, 1.5], dtype=np.float64),
            mu=np.array([[1.0, 0.0], [-1.0, 0.0]], dtype=np.float64),
            kappa=np.array([3.0, 2.0], dtype=np.float64),
        )

        dist = vMFNMDistribution(params)

        # Generate samples
        samples = dist.sample(5000, rng=rng)

        # Samples should roughly follow the distribution
        # Test: mean of samples should be near zero (symmetric mixture)
        sample_mean = np.mean(samples, axis=0)
        assert np.linalg.norm(sample_mean) < 0.2


if __name__ == "__main__":
    # Running tests directly: python tests/test_distributions.py
    pytest.main([__file__, "-v"])
