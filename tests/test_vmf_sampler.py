"""The von Mises-Fisher sampler is checked against its closed-form moment.

The mean resultant length of vMF_d(mu, kappa) has an exact expression,

    E[x . mu] = A_d(kappa) = I_{d/2}(kappa) / I_{d/2-1}(kappa),

which pins the concentration without needing a full distributional test. It is
computed here with the exponentially scaled Bessel function ``ive``, whose ratio
equals the ratio of the unscaled ones and does not overflow.

A "high-concentration" shortcut used to intercept kappa >= 30 and replace Wood's
rejection sampler with a Gaussian around mu of scale ``0.2 / sqrt(kappa)``. The
0.2 is arbitrary and about five times too small, and it ignores the dimension,
so samples clustered far too tightly: at d=20, kappa=50 the mean resultant
length came out at 0.9925 against a true 0.8263. Wood's sampler is exact at any
concentration, so the shortcut only ever introduced error.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import ive

from safe_ice.distributions.vmf import VonMisesFisherSampler

# Deliberately straddles the old kappa >= 30 cutoff.
CONCENTRATIONS = [0.5, 2.0, 10.0, 29.0, 30.0, 31.0, 50.0, 200.0]


def mean_resultant_length(kappa: float, d: int) -> float:
    """A_d(kappa) = I_{d/2}(kappa) / I_{d/2-1}(kappa)."""
    return float(ive(d / 2.0, kappa) / ive(d / 2.0 - 1.0, kappa))


def assert_concentration(d: int, kappa: float, rng, n: int) -> None:
    """Check E[x . mu] against A_d(kappa) within the sampling error."""
    mu = np.zeros(d)
    mu[0] = 1.0

    samples = VonMisesFisherSampler.sample(mu, kappa, n, rng=rng)
    projections = samples @ mu

    expected = mean_resultant_length(kappa, d)
    # Tolerance comes from the sampling error of the mean, with a floor so the
    # low-kappa cases, where the expectation is near zero, are not held to an
    # unreachable relative accuracy.
    standard_error = float(np.std(projections) / np.sqrt(len(projections)))
    assert float(np.mean(projections)) == pytest.approx(
        expected, abs=max(5.0 * standard_error, 0.005)
    )


class TestMeanResultantLength:
    @pytest.mark.parametrize("kappa", [2.0, 29.0, 31.0, 200.0])
    def test_concentration_matches_closed_form(self, kappa: float, rng) -> None:
        """Fast check at one dimension, straddling the old cutoff."""
        assert_concentration(20, kappa, rng, n=15_000)

    @pytest.mark.slow
    @pytest.mark.parametrize("d", [2, 3, 20])
    @pytest.mark.parametrize("kappa", CONCENTRATIONS)
    def test_concentration_across_dimensions(self, d: int, kappa: float, rng) -> None:
        """Across dimensions; slow because each case needs many samples.

        Wood's sampler draws one point per Python-loop iteration, so sample
        counts here dominate the suite's runtime. The dimension list is kept
        short deliberately; the fast test above covers d=20 in more detail.
        """
        assert_concentration(d, kappa, rng, n=20_000)

    @pytest.mark.parametrize("kappa", [29.0, 30.0, 31.0])
    def test_no_jump_across_the_old_cutoff(self, kappa: float, rng) -> None:
        """Concentration must vary smoothly through kappa = 30."""
        d = 20
        mu = np.zeros(d)
        mu[0] = 1.0
        samples = VonMisesFisherSampler.sample(mu, kappa, 15_000, rng=rng)
        got = float(np.mean(samples @ mu))
        assert got == pytest.approx(mean_resultant_length(kappa, d), abs=0.01)


class TestSamplerInvariants:
    @pytest.mark.parametrize("d", [2, 3, 10])
    @pytest.mark.parametrize("kappa", [0.0, 1.0, 50.0])
    def test_samples_are_unit_vectors(self, d: int, kappa: float, rng) -> None:
        mu = np.zeros(d)
        mu[0] = 1.0
        samples = VonMisesFisherSampler.sample(mu, kappa, 500, rng=rng)
        assert samples.shape == (500, d)
        assert np.allclose(np.linalg.norm(samples, axis=1), 1.0)

    @pytest.mark.parametrize("d", [3, 10])
    def test_zero_concentration_is_uniform(self, d: int, rng) -> None:
        """With kappa=0 there is no preferred direction."""
        mu = np.zeros(d)
        mu[0] = 1.0
        samples = VonMisesFisherSampler.sample(mu, 0.0, 15_000, rng=rng)
        assert float(np.mean(samples @ mu)) == pytest.approx(0.0, abs=0.02)

    @pytest.mark.parametrize("kappa", [2.0, 50.0])
    def test_direction_follows_mu(self, kappa: float, rng) -> None:
        """Concentration must track mu, not a fixed axis."""
        d = 10
        rotated = np.zeros(d)
        rotated[3] = 1.0

        samples = VonMisesFisherSampler.sample(rotated, kappa, 15_000, rng=rng)
        along = float(np.mean(samples @ rotated))
        assert along == pytest.approx(mean_resultant_length(kappa, d), abs=0.01)

        # Orthogonal directions carry no systematic component.
        other = np.zeros(d)
        other[0] = 1.0
        assert float(np.mean(samples @ other)) == pytest.approx(0.0, abs=0.02)

    def test_non_unit_mu_is_normalised(self, rng) -> None:
        d, kappa = 5, 20.0
        mu = np.zeros(d)
        mu[0] = 7.5  # deliberately not a unit vector
        samples = VonMisesFisherSampler.sample(mu, kappa, 15_000, rng=rng)
        unit = mu / np.linalg.norm(mu)
        assert float(np.mean(samples @ unit)) == pytest.approx(
            mean_resultant_length(kappa, d), abs=0.01
        )
