"""Regression tests for the vectorised numerical helpers.

The mixture density and the EM E-step evaluate whole batches in a handful of
NumPy passes. These tests pin the batched results to the scalar reference
formulas so a future optimisation cannot silently change the numbers.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.special import ive

from safe_ice.core.parameters import vMFNMParameters
from safe_ice.distributions._numeric import (
    LOG_MAX,
    RADIUS_FLOOR,
    exp_clamped,
    radii_and_directions,
    sphere_surface_area,
    vmf_pdf_batch,
)
from safe_ice.distributions.mixture import vMFNMDistribution
from safe_ice.distributions.nakagami import NakagamiDistribution


def reference_vmf_pdf(x, mu, kappa):
    """Scalar von Mises-Fisher density, written directly from the definition."""
    d = int(np.size(x))
    if kappa <= 0.0:
        return 1.0 / sphere_surface_area(d)
    nu = d / 2.0 - 1.0
    ive_val = float(ive(nu, kappa))
    if ive_val <= 0.0 or not np.isfinite(ive_val):
        return 0.0
    log_norm = (
        nu * math.log(kappa)
        - (d / 2.0) * math.log(2.0 * math.pi)
        - math.log(ive_val)
        - kappa
    )
    log_pdf = log_norm + kappa * float(np.dot(x, mu))
    if not np.isfinite(log_pdf):
        return 0.0
    if log_pdf < -745.0:
        return 0.0
    if log_pdf > 700.0:
        return math.exp(700.0)
    return math.exp(log_pdf)


class TestRadiiAndDirections:
    """Polar decomposition of a batch of points."""

    def test_matches_per_row_computation(self, rng):
        X = rng.standard_normal((50, 4)) * 3.0
        radii, directions = radii_and_directions(X)

        for i, row in enumerate(X):
            assert radii[i] == pytest.approx(float(np.linalg.norm(row)), rel=1e-12)
            assert directions[i] == pytest.approx(row / np.linalg.norm(row), rel=1e-12)

    def test_directions_are_unit_vectors(self, rng):
        X = rng.standard_normal((40, 6)) * 0.01
        _, directions = radii_and_directions(X)
        assert np.allclose(np.linalg.norm(directions, axis=1), 1.0)

    def test_degenerate_rows_use_first_basis_vector(self):
        X = np.zeros((3, 3))
        X[1] = 1e-15  # below the floor
        X[2, 0] = 1.0  # ordinary row
        radii, directions = radii_and_directions(X)

        assert radii[0] == RADIUS_FLOOR
        assert radii[1] == RADIUS_FLOOR
        assert np.array_equal(directions[0], [1.0, 0.0, 0.0])
        assert np.array_equal(directions[1], [1.0, 0.0, 0.0])
        assert radii[2] == pytest.approx(1.0)


class TestExpClamped:
    """Guard clauses around exponentiating log-densities."""

    @pytest.mark.parametrize("bad", [np.inf, -np.inf, np.nan])
    def test_non_finite_maps_to_zero(self, bad):
        # +inf must yield 0 rather than saturating: the finiteness check comes first.
        assert exp_clamped(np.array([bad])) == 0.0

    def test_underflow_maps_to_zero(self):
        assert exp_clamped(np.array([-800.0])) == 0.0

    def test_saturates_instead_of_overflowing(self):
        got = exp_clamped(np.array([5000.0]))
        assert got == pytest.approx(math.exp(LOG_MAX))
        assert np.isfinite(got).all()

    def test_ordinary_values_pass_through(self, rng):
        vals = rng.uniform(-700.0, 700.0, size=64)
        assert exp_clamped(vals) == pytest.approx(np.exp(vals), rel=1e-12)


class TestVmfPdfBatch:
    """Batched vMF density against the scalar reference."""

    @pytest.mark.parametrize("d", [2, 3, 5, 10])
    @pytest.mark.parametrize("kappa", [0.0, 0.5, 7.5, 250.0])
    def test_matches_reference(self, d, kappa, rng):
        mu = rng.standard_normal(d)
        mu /= np.linalg.norm(mu)
        X = rng.standard_normal((25, d))
        X /= np.linalg.norm(X, axis=1, keepdims=True)

        got = vmf_pdf_batch(X, mu, kappa)
        want = np.array([reference_vmf_pdf(x, mu, kappa) for x in X])
        assert got == pytest.approx(want, rel=1e-12)

    def test_zero_concentration_is_uniform(self, rng):
        d = 4
        X = rng.standard_normal((10, d))
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        got = vmf_pdf_batch(X, np.eye(d)[0], 0.0)
        assert got == pytest.approx(1.0 / sphere_surface_area(d))

    @pytest.mark.parametrize("d", [2, 3, 7])
    def test_integrates_to_one_over_the_sphere(self, d, rng):
        # Uniform directions on S^{d-1} have density 1/area, so the mean of
        # pdf/uniform_density estimates the integral over the sphere.
        mu = np.eye(d)[0]
        kappa = 3.0
        X = rng.standard_normal((200_000, d))
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        integral = np.mean(vmf_pdf_batch(X, mu, kappa)) * sphere_surface_area(d)
        assert integral == pytest.approx(1.0, rel=0.02)


class TestSphereSurfaceArea:
    """Known closed-form values."""

    def test_circle_and_sphere(self):
        assert sphere_surface_area(2) == pytest.approx(2.0 * math.pi)
        assert sphere_surface_area(3) == pytest.approx(4.0 * math.pi)


class TestMixturePdfVectorisation:
    """The batched mixture density must equal the per-sample formula."""

    @pytest.mark.parametrize("d", [2, 3, 6])
    @pytest.mark.parametrize("K", [1, 4])
    def test_matches_per_sample_loop(self, d, K, rng):
        pi = rng.dirichlet(np.ones(K))
        params = vMFNMParameters(
            pi=pi,
            m=rng.uniform(0.5, 5.0, size=K),
            Omega=rng.uniform(0.5, 10.0, size=K),
            mu=rng.standard_normal((K, d)),
            kappa=rng.uniform(0.0, 40.0, size=K),
        )
        dist = vMFNMDistribution(params)

        X = rng.standard_normal((40, d)) * 2.0
        got = dist.pdf(X)

        # Reference: the original per-sample, per-component formulation.
        want = np.zeros(X.shape[0])
        for i, row in enumerate(X):
            r = float(np.linalg.norm(row))
            a = row / r if r > RADIUS_FLOOR else np.eye(d)[0]
            r = max(r, RADIUS_FLOOR)
            jacobian = r ** (d - 1)
            total = 0.0
            for k in range(K):
                nak = float(
                    NakagamiDistribution.pdf(
                        r, float(params.m[k]), float(params.Omega[k])
                    )
                )
                vmf = reference_vmf_pdf(a, params.mu[k], float(params.kappa[k]))
                total += float(params.pi[k]) * nak * vmf / jacobian
            want[i] = total

        assert got == pytest.approx(want, rel=1e-10)

    def test_single_row_input_is_accepted(self, rng):
        d, K = 3, 2
        params = vMFNMParameters(
            pi=np.full(K, 1.0 / K),
            m=np.full(K, 2.0),
            Omega=np.full(K, 3.0),
            mu=rng.standard_normal((K, d)),
            kappa=np.full(K, 1.5),
        )
        dist = vMFNMDistribution(params)
        row = rng.standard_normal(d)

        assert dist.pdf(row) == pytest.approx(dist.pdf(row.reshape(1, -1)))


class TestEStepVectorisation:
    """Responsibilities must be a valid posterior for every sample."""

    def test_rows_sum_to_one(self, rng):
        from safe_ice.optimization.penalized_em import PenalizedEMOptimizer

        d, K, n = 4, 3, 60
        params = vMFNMParameters(
            pi=rng.dirichlet(np.ones(K)),
            m=rng.uniform(0.5, 4.0, size=K),
            Omega=rng.uniform(0.5, 8.0, size=K),
            mu=rng.standard_normal((K, d)),
            kappa=rng.uniform(0.0, 20.0, size=K),
        )
        X = rng.standard_normal((n, d)) * 2.0
        radii, directions = radii_and_directions(X)
        weights = rng.uniform(0.0, 2.0, size=n)

        resp = PenalizedEMOptimizer()._e_step(X, radii, directions, params, weights)

        assert resp.shape == (n, K)
        assert np.all(resp >= 0.0)
        assert np.allclose(resp.sum(axis=1), 1.0)

    def test_unreachable_samples_fall_back_to_uniform(self, rng):
        from safe_ice.optimization.penalized_em import PenalizedEMOptimizer

        d, K = 3, 4
        params = vMFNMParameters(
            pi=np.full(K, 1.0 / K),
            m=np.full(K, 2.0),
            Omega=np.full(K, 1.0),
            mu=rng.standard_normal((K, d)),
            kappa=np.full(K, 1.0),
        )
        # A radius this large has negligible density under every component.
        X = np.full((2, d), 1e6)
        radii, directions = radii_and_directions(X)
        weights = np.ones(2)

        resp = PenalizedEMOptimizer()._e_step(X, radii, directions, params, weights)
        assert resp == pytest.approx(np.full((2, K), 1.0 / K))
