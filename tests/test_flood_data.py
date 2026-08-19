"""The committed USGS record, and the reliability problem built on it.

`notebooks/05_flood_risk_real_data.ipynb` has its outputs saved, so a change in
either the data or the estimators would leave it quietly stale. These tests pin
the things that notebook asserts, so the staleness shows up here first.

The record is real; the channel hydraulics are stylised. That distinction is
worth keeping visible, and is why the tests below check the data against the
published record and the hydraulics only against themselves.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from safe_ice import SafeICE, SubsetSimulation
from safe_ice.transforms import MarginalTransform

DATA = (
    Path(__file__).resolve().parent.parent / "data" / "usgs_01646500_annual_peaks.csv"
)
CFS_TO_CUMECS = 0.028316846592


@pytest.fixture(scope="module")
def record():
    rows = list(csv.DictReader(DATA.open(encoding="utf-8")))
    return {
        "years": np.array([int(r["water_year"]) for r in rows]),
        "cfs": np.array([float(r["peak_discharge_cfs"]) for r in rows]),
        "cumecs": np.array([float(r["peak_discharge_m3s"]) for r in rows]),
    }


class TestTheRecord:
    def test_the_file_is_present_and_complete(self, record) -> None:
        assert len(record["years"]) == 95
        assert record["years"].min() == 1931
        assert record["years"].max() == 2025

    def test_years_are_unique_and_ordered(self, record) -> None:
        years = record["years"]
        assert len(set(years.tolist())) == len(years)
        assert np.all(np.diff(years) > 0)

    def test_the_unit_conversion_is_right(self, record) -> None:
        assert record["cumecs"] == pytest.approx(
            record["cfs"] * CFS_TO_CUMECS, abs=1e-3
        )

    def test_discharges_are_positive_and_plausible(self, record) -> None:
        """A gauge record with a zero or a negative in it has been mis-parsed."""
        assert np.all(record["cumecs"] > 0)
        assert record["cumecs"].min() == pytest.approx(866.0, abs=1.0)
        assert record["cumecs"].max() == pytest.approx(13705.0, abs=1.0)


class TestTheFloodDistribution:
    def test_gumbel_is_not_rejected(self, record) -> None:
        """The notebook's choice has to survive its own goodness-of-fit test."""
        params = stats.gumbel_r.fit(record["cumecs"])
        result = stats.kstest(record["cumecs"], stats.gumbel_r.cdf, args=params)

        assert result.pvalue > 0.05, f"KS p-value {result.pvalue:.4f}"

    def test_return_levels_are_ordered_and_exceed_the_record(self, record) -> None:
        flood = stats.gumbel_r(*stats.gumbel_r.fit(record["cumecs"]))
        levels = [float(flood.ppf(1 - 1 / t)) for t in (2, 10, 100, 500)]

        assert levels == sorted(levels)
        # A 500-year estimate from 95 years of record must extrapolate beyond it.
        assert levels[-1] > float(record["cumecs"].max()) * 0.8


class TestTheLeveeProblem:
    """The notebook's setup, checked three ways."""

    CHANNEL_WIDTH = 300.0
    CREST = 10.0

    @staticmethod
    def depth(q, n, s, width=300.0):
        return (q * n / (width * np.sqrt(s))) ** 0.6

    def freeboard(self, x):
        return self.CREST - self.depth(
            np.maximum(x[:, 0], 1.0), x[:, 1], x[:, 2], self.CHANNEL_WIDTH
        )

    @pytest.fixture
    def transform(self, record):
        flood = stats.gumbel_r(*stats.gumbel_r.fit(record["cumecs"]))
        return MarginalTransform(
            [
                flood,
                stats.lognorm(s=0.15, scale=0.040),
                stats.lognorm(s=0.15, scale=0.003),
            ]
        )

    def test_the_crest_clears_the_hundred_year_flood(self, record) -> None:
        """Otherwise overtopping is not a rare event and the example is pointless."""
        flood = stats.gumbel_r(*stats.gumbel_r.fit(record["cumecs"]))
        hundred_year_depth = self.depth(float(flood.ppf(0.99)), 0.040, 0.003)

        assert hundred_year_depth < self.CREST
        assert hundred_year_depth == pytest.approx(6.5, abs=0.3)

    @pytest.mark.slow
    def test_crude_monte_carlo_gives_the_expected_probability(self, transform) -> None:
        samples = transform.sample(2_000_000, random_state=20240117)
        observed = float((self.freeboard(samples) <= 0).mean())

        assert observed == pytest.approx(5.7e-05, rel=0.4)

    @pytest.mark.slow
    @pytest.mark.parametrize("estimator", [SafeICE, SubsetSimulation])
    def test_the_estimators_agree_with_counting(self, estimator, transform) -> None:
        """Two methods sharing no machinery, against a reference that just counts."""
        wrapped = transform.wrap(self.freeboard)
        estimates = [
            estimator(limit_state_function=wrapped, dimension=3, random_state=seed).run(
                verbose=False
            )[0]
            for seed in range(5)
        ]
        median = float(np.median(estimates))

        assert 5.7e-05 / 2 < median < 5.7e-05 * 2, (
            f"{estimator.__name__}: {median:.3e} against 5.7e-05; {estimates}"
        )

    @pytest.mark.slow
    def test_the_answer_depends_on_the_flood_distribution(self, record) -> None:
        """The notebook's closing point, and the one worth not losing.

        Ninety-five years cannot separate a Gumbel tail from a lognormal one --
        neither is rejected -- but they disagree about the tail by nearly an
        order of magnitude, which dwarfs the scatter between estimators.
        """
        peaks = record["cumecs"]
        roughness = stats.lognorm(s=0.15, scale=0.040)
        slope = stats.lognorm(s=0.15, scale=0.003)

        def estimate(fitted):
            wrapped = MarginalTransform([fitted, roughness, slope]).wrap(self.freeboard)
            return float(
                np.median(
                    [
                        SafeICE(
                            limit_state_function=wrapped,
                            dimension=3,
                            N=1000,
                            max_iterations=15,
                            random_state=seed,
                        ).run(verbose=False)[0]
                        for seed in range(5)
                    ]
                )
            )

        gumbel = estimate(stats.gumbel_r(*stats.gumbel_r.fit(peaks)))
        lognormal = estimate(stats.lognorm(*stats.lognorm.fit(peaks, floc=0)))

        assert lognormal > gumbel * 3, (
            f"lognormal {lognormal:.3e} against Gumbel {gumbel:.3e}; if these have "
            "converged, the notebook's closing point needs rewriting"
        )
