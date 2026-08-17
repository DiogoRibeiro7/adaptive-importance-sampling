"""The Nakagami densities are checked against scipy across the shape range.

The radial part of the proposal is a Nakagami, and the initialiser sets its
shape from the problem dimension (chi_d is Nakagami(m=d/2, Omega=d)), so m
grows with d and large shapes are reachable rather than hypothetical.

A normal approximation used to replace the exact form for m > 170, on the
theory that the exact one would overflow. It does not overflow, and the
approximation was wrong by 43% at m=200 and 145% at m=500. These tests compare
against scipy.stats.nakagami, which is the same distribution under a different
parameterisation:

    Nakagami(m, Omega)  ==  scipy.stats.nakagami(nu=m, scale=sqrt(Omega))
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from safe_ice.distributions.nakagami import (
    InverseNakagamiDistribution,
    NakagamiDistribution,
)

# Spans the old m > 170 cutoff deliberately.
SHAPES = [0.5, 1.0, 2.0, 20.0, 169.0, 170.0, 171.0, 200.0, 500.0, 1000.0]


def radii_near_mode(m: float, omega: float, width: float = 0.4, n: int = 150):
    """Sample radii around the mode, where the density carries its mass."""
    mode = np.sqrt(omega * (m - 0.5) / m) if m > 0.5 else 0.1
    return np.linspace(max(mode - width, 1e-3), mode + width, n)


class TestNakagamiAgainstScipy:
    @pytest.mark.parametrize("m", SHAPES)
    @pytest.mark.parametrize("omega", [1.0, 5.0])
    def test_pdf_matches_reference(self, m: float, omega: float) -> None:
        r = radii_near_mode(m, omega)
        reference = stats.nakagami.pdf(r, m, scale=np.sqrt(omega))
        got = np.asarray(NakagamiDistribution.pdf(r, m, omega), dtype=np.float64)

        denominator = np.maximum(np.abs(reference), 1e-300)
        assert np.max(np.abs(got - reference) / denominator) < 1e-9

    @pytest.mark.parametrize("m", SHAPES)
    @pytest.mark.parametrize("omega", [1.0, 5.0])
    def test_cdf_matches_reference(self, m: float, omega: float) -> None:
        r = radii_near_mode(m, omega, width=0.6)
        reference = stats.nakagami.cdf(r, m, scale=np.sqrt(omega))
        got = np.asarray(NakagamiDistribution.cdf(r, m, omega), dtype=np.float64)

        assert np.max(np.abs(got - reference)) < 1e-9

    @pytest.mark.parametrize("m", [200.0, 500.0])
    def test_no_discontinuity_across_the_old_cutoff(self, m: float) -> None:
        """The density must not jump when m crosses 170."""
        omega = 5.0
        r = np.array([np.sqrt(omega * (m - 0.5) / m)])
        below = float(np.asarray(NakagamiDistribution.pdf(r, 169.0, omega))[0])
        above = float(np.asarray(NakagamiDistribution.pdf(r, 171.0, omega))[0])
        # Both are evaluated at the mode for shape m, so neither is the peak;
        # the point is only that they are the same order of magnitude.
        assert 1e-3 < below / above < 1e3

    @pytest.mark.parametrize("m", SHAPES)
    def test_pdf_stays_finite_and_non_negative(self, m: float) -> None:
        r = np.linspace(1e-3, 20.0, 500)
        got = np.asarray(NakagamiDistribution.pdf(r, m, 5.0), dtype=np.float64)
        assert np.all(np.isfinite(got))
        assert np.all(got >= 0.0)


class TestInverseNakagami:
    @pytest.mark.parametrize("m", [2.0, 20.0, 200.0])
    def test_pdf_integrates_to_one(self, m: float) -> None:
        """Y = 1/R, so the density must still be normalised."""
        y = np.linspace(1e-4, 80.0, 400_000)
        values = np.asarray(
            InverseNakagamiDistribution.pdf(y, m, 5.0), dtype=np.float64
        )
        assert np.trapezoid(values, y) == pytest.approx(1.0, abs=0.02)

    @pytest.mark.parametrize("m", [2.0, 200.0])
    def test_samples_match_the_density(self, m: float, rng) -> None:
        """Sampling draws R ~ Nakagami then returns 1/R; check E[1/Y^2]."""
        omega = 5.0
        samples = InverseNakagamiDistribution.sample(m, omega, 20_000, rng=rng)
        # 1/Y = R and E[R^2] = Omega
        assert float(np.mean((1.0 / samples) ** 2)) == pytest.approx(omega, rel=0.05)


class TestNakagamiSampling:
    @pytest.mark.parametrize("m", [1.0, 20.0, 200.0])
    def test_second_moment_is_omega(self, m: float, rng) -> None:
        omega = 3.0
        samples = NakagamiDistribution.sample(m, omega, 20_000, rng=rng)
        assert float(np.mean(samples**2)) == pytest.approx(omega, rel=0.03)

    @pytest.mark.parametrize("m", [2.0, 99.0, 101.0, 200.0, 1000.0])
    def test_samples_follow_the_distribution(self, m: float, rng) -> None:
        """A moment check is not enough: the shape has to be right too.

        A normal approximation used to take over for m > 100 whose variance was
        four times too large, so samples had twice the correct spread. E[R^2]
        was still within 0.3% of Omega, so only a distributional test catches
        it. This one rejected at p = 0 for every m above the old cutoff.
        """
        omega = 5.0
        samples = NakagamiDistribution.sample(m, omega, 20_000, rng=rng)

        result = stats.kstest(
            samples, lambda x: stats.nakagami.cdf(x, m, scale=np.sqrt(omega))
        )
        assert result.pvalue > 0.001, f"KS statistic {result.statistic:.4f}"

    @pytest.mark.parametrize("m", [2.0, 200.0])
    def test_sample_spread_matches_the_distribution(self, m: float, rng) -> None:
        omega = 5.0
        samples = NakagamiDistribution.sample(m, omega, 40_000, rng=rng)
        expected = float(stats.nakagami.std(m, scale=np.sqrt(omega)))
        # The old approximation gave exactly twice this.
        assert float(np.std(samples)) == pytest.approx(expected, rel=0.05)

    @pytest.mark.parametrize("m", [2.0, 200.0])
    def test_inverse_sampler_inherits_the_fix(self, m: float, rng) -> None:
        """InverseNakagami draws R then returns 1/R, so 1/Y must be Nakagami."""
        omega = 5.0
        y = InverseNakagamiDistribution.sample(m, omega, 20_000, rng=rng)

        result = stats.kstest(
            1.0 / y, lambda x: stats.nakagami.cdf(x, m, scale=np.sqrt(omega))
        )
        assert result.pvalue > 0.001, f"KS statistic {result.statistic:.4f}"

    def test_chi_identity(self, rng) -> None:
        """chi_d is exactly Nakagami(m=d/2, Omega=d), which the initialiser uses."""
        for d in (2, 5, 20):
            samples = NakagamiDistribution.sample(d / 2.0, float(d), 20_000, rng=rng)
            assert float(np.mean(samples)) == pytest.approx(
                float(stats.chi.mean(d)), rel=0.03
            )
