# safe_ice/distributions/_numeric.py
"""Shared vectorised numerical helpers for the mixture and EM code paths.

These exist so the mixture density and the EM E-step can evaluate whole sample
batches in a few NumPy passes instead of one Python call per (sample,
component) pair. The guard clauses mirror the scalar reference implementations
exactly, including the order in which the non-finite and saturation cases are
tested, so results are unchanged.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.special import gamma, ive

NDArrayF = npt.NDArray[np.float64]

#: Radii at or below this are treated as degenerate and clamped.
RADIUS_FLOOR = 1e-12

#: ``exp`` of a float64 overflows above this exponent; the reference code
#: saturates rather than returning ``inf``.
LOG_MAX = 700.0

#: Below this exponent ``exp`` underflows to zero in float64.
LOG_MIN = -745.0


def radii_and_directions(X: NDArrayF) -> tuple[NDArrayF, NDArrayF]:
    """Split rows of ``X`` into radii and unit directions.

    Rows whose norm falls at or below :data:`RADIUS_FLOOR` are degenerate: the
    direction is undefined, so the first basis vector is used and the radius is
    clamped to the floor.

    Returns
    -------
    radii : ndarray of shape (n,)
    directions : ndarray of shape (n, d)
    """
    norms = np.linalg.norm(X, axis=1)
    degenerate = norms <= RADIUS_FLOOR

    directions = np.zeros_like(X)
    finite_rows = ~degenerate
    directions[finite_rows] = X[finite_rows] / norms[finite_rows, None]
    directions[degenerate, 0] = 1.0

    radii = np.where(degenerate, RADIUS_FLOOR, norms)
    return radii, directions


def exp_clamped(log_pdf: NDArrayF) -> NDArrayF:
    """Exponentiate log-densities, mapping non-finite values to zero.

    Mirrors the scalar guards: non-finite exponents yield ``0`` (checked first,
    so ``+inf`` yields ``0`` rather than saturating), exponents below
    :data:`LOG_MIN` underflow to ``0``, and exponents above :data:`LOG_MAX`
    saturate at ``exp(LOG_MAX)``.
    """
    out = np.zeros_like(log_pdf, dtype=np.float64)
    finite = np.isfinite(log_pdf)
    saturated = finite & (log_pdf > LOG_MAX)
    normal = finite & (log_pdf >= LOG_MIN) & (log_pdf <= LOG_MAX)

    out[normal] = np.exp(log_pdf[normal])
    out[saturated] = np.exp(LOG_MAX)
    return out


def sphere_surface_area(d: int) -> float:
    """Surface area of the unit sphere S^{d-1} embedded in R^d."""
    return float(2.0 * (np.pi ** (d / 2.0)) / float(gamma(d / 2.0)))


def vmf_pdf_batch(X: NDArrayF, mu: NDArrayF, kappa: float) -> NDArrayF:
    """von Mises-Fisher density at unit rows of ``X``, w.r.t. surface measure.

    Normalised so that the integral over S^{d-1} with respect to the surface
    area measure is one. A non-positive concentration gives the uniform density
    on the sphere; ``kappa`` is a concentration parameter and is never negative
    for a well-formed mixture.

    Uses the exponentially-scaled Bessel function ``ive`` so that large
    ``kappa`` does not overflow: ``ive(v, k) = iv(v, k) * exp(-k)``.
    """
    n, d = int(X.shape[0]), int(X.shape[1])

    if kappa <= 0.0:
        return np.full(n, 1.0 / sphere_surface_area(d), dtype=np.float64)

    nu = d / 2.0 - 1.0
    ive_val = float(ive(nu, kappa))
    if ive_val <= 0.0 or not np.isfinite(ive_val):
        return np.zeros(n, dtype=np.float64)

    log_norm = (
        nu * float(np.log(kappa))
        - (d / 2.0) * float(np.log(2.0 * np.pi))
        - float(np.log(ive_val))
        - float(kappa)
    )
    log_pdf: NDArrayF = log_norm + float(kappa) * (X @ mu)
    return exp_clamped(log_pdf)
