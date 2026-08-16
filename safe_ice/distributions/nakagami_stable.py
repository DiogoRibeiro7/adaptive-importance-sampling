# safe_ice/distributions/nakagami_stable.py
"""Numerically stable Nakagami and Inverse Nakagami distributions.

This module provides numerically stable implementations using log-space computations
to avoid overflow/underflow issues with extreme parameter values.

Typing policy
-------------
- Public functions accept either scalars or arrays (ArrayLike).
- If the input is a scalar, return a Python `float`.
- If the input is an array, return `NDArray[np.float64]`.

We expose precise typing with @overload so mypy understands both cases.
"""

from __future__ import annotations

from typing import overload

import numpy as np
import numpy.typing as npt
from scipy.special import gammainc, loggamma  # normalized lower incomplete gamma

NDArrayF = npt.NDArray[np.float64]


def _as_float(x: np.ndarray) -> float:
    """Extract a Python float from a 0-d/1-elem array safely."""
    return float(np.asarray(x, dtype=np.float64).reshape(1)[0])


# =============================================================================
# Nakagami(m, Omega) on r > 0
# pdf(r) = 2 * (m^m / (Gamma(m) * Omega^m)) * r^{2m-1} * exp(-m r^2 / Omega), r >= 0
#
# Numerically stable version using log-space:
# log(pdf) = log(2) + m*log(m) - loggamma(m) - m*log(Omega) + (2m-1)*log(r) - m*r^2/Omega
# =============================================================================
class NakagamiDistribution:
    """Nakagami(m, Omega) distribution with numerical stability."""

    # -------------------- PDF --------------------
    @overload
    @staticmethod
    def pdf(r: float, m: float, Omega: float) -> float: ...
    @overload
    @staticmethod
    def pdf(r: NDArrayF, m: float, Omega: float) -> NDArrayF: ...

    @staticmethod
    def pdf(r: npt.ArrayLike, m: float, Omega: float) -> float | NDArrayF:
        r_arr: NDArrayF = np.asarray(r, dtype=np.float64)
        m = float(m)
        Omega = float(Omega)

        # Initialize output
        out: NDArrayF = np.zeros_like(r_arr, dtype=np.float64)
        mask = r_arr > 0.0  # PDF is 0 for r <= 0

        if not np.any(mask):
            if np.isscalar(r) or np.asarray(r).ndim == 0:
                return 0.0
            return out

        xm = r_arr[mask]

        # Compute log PDF for numerical stability
        # log(pdf) = log(2) + m*log(m) - loggamma(m) - m*log(Omega) + (2m-1)*log(xm) - m*xm^2/Omega
        try:
            log_pdf = (
                np.log(2.0)
                + m * np.log(m)
                - loggamma(m)
                - m * np.log(Omega)
                + (2.0 * m - 1.0) * np.log(xm)
                - m * (xm**2) / Omega
            )

            # Convert back from log-space, handling potential underflow
            pdf_values = np.exp(log_pdf)
            # Handle numerical underflow
            pdf_values = np.where(np.isfinite(pdf_values), pdf_values, 0.0)
            out[mask] = pdf_values

        except (OverflowError, FloatingPointError):
            # For extreme values where even log-space fails, use approximations or return 0
            # This typically happens when m is extremely large (>170)
            # In such cases, the distribution becomes extremely concentrated
            # For practical purposes, we can return 0 for most values

            # Check if we're near the mode
            mode = np.sqrt(Omega * (m - 0.5) / m) if m > 0.5 else 0.0
            near_mode = np.abs(xm - mode) < 0.1 * mode if mode > 0 else False

            # Return small non-zero values only near the mode
            out[mask] = np.where(near_mode, 1e-10, 0.0)

        # Return scalar if input was scalar
        if np.isscalar(r) or np.asarray(r).ndim == 0:
            return _as_float(out)
        return out

    # -------------------- Log PDF --------------------
    @staticmethod
    def log_pdf(r: npt.ArrayLike, m: float, Omega: float) -> float | NDArrayF:
        """Compute log PDF directly for numerical stability."""
        r_arr: NDArrayF = np.asarray(r, dtype=np.float64)
        m = float(m)
        Omega = float(Omega)

        # Initialize with -inf (log(0))
        out: NDArrayF = np.full_like(r_arr, -np.inf, dtype=np.float64)
        mask = r_arr > 0.0

        if not np.any(mask):
            if np.isscalar(r) or np.asarray(r).ndim == 0:
                return -np.inf
            return out

        xm = r_arr[mask]

        # Compute log PDF
        log_pdf = (
            np.log(2.0)
            + m * np.log(m)
            - loggamma(m)
            - m * np.log(Omega)
            + (2.0 * m - 1.0) * np.log(xm)
            - m * (xm**2) / Omega
        )

        out[mask] = log_pdf

        if np.isscalar(r) or np.asarray(r).ndim == 0:
            return _as_float(out)
        return out

    # -------------------- CDF --------------------
    @overload
    @staticmethod
    def cdf(r: float, m: float, Omega: float) -> float: ...
    @overload
    @staticmethod
    def cdf(r: NDArrayF, m: float, Omega: float) -> NDArrayF: ...

    @staticmethod
    def cdf(r: npt.ArrayLike, m: float, Omega: float) -> float | NDArrayF:
        r_arr: NDArrayF = np.asarray(r, dtype=np.float64)
        m = float(m)
        Omega = float(Omega)

        # For r >= 0: F(r) = P(m, m r^2 / Omega); for r < 0 => 0
        out: NDArrayF = np.zeros_like(r_arr, dtype=np.float64)
        mask = r_arr >= 0.0

        if np.any(mask):
            z = (m * (r_arr[mask] ** 2)) / Omega

            # Handle potential overflow in z
            z = np.minimum(z, 1e10)  # Cap at a large value to avoid overflow

            try:
                out[mask] = gammainc(m, z)  # regularized lower incomplete gamma
            except (OverflowError, FloatingPointError):
                # For extreme values, use approximation
                # When z >> m, gammainc(m, z) ≈ 1
                # When z << m, gammainc(m, z) ≈ 0
                out[mask] = np.where(z > m + 10 * np.sqrt(m), 1.0, 0.0)

        if np.isscalar(r) or np.asarray(r).ndim == 0:
            return _as_float(out)
        return out

    # -------------------- Sample --------------------
    @staticmethod
    def sample(m: float, Omega: float, n: int = 1) -> NDArrayF:
        """Draw `n` Nakagami samples as sqrt of a Gamma(m, Omega/m)."""
        m = float(m)
        Omega = float(Omega)
        n = int(n)

        # Handle extreme m values
        if m > 100:
            # For very large m, Nakagami approaches a narrow distribution
            # Use normal approximation: mean ≈ sqrt(Omega), variance ≈ Omega/(4m)
            mean = np.sqrt(Omega)
            std = np.sqrt(Omega / (4 * m))
            samples = np.random.normal(mean, std, n)
            # Ensure positive values
            return np.abs(samples).astype(np.float64, copy=False)

        # Standard approach for reasonable m values
        # X ~ Gamma(shape=m, scale=Omega/m)  (NumPy parameterizes by shape & scale)
        x = np.random.gamma(shape=m, scale=(Omega / m), size=n).astype(
            np.float64, copy=False
        )
        return np.sqrt(x).astype(np.float64, copy=False)


# =============================================================================
# Inverse Nakagami via transformation: if R ~ Nakagami(m, Omega), define Y = 1 / R.
# Then:
#   pdf_Y(y) = pdf_R(1/y) * (1 / y^2),  y > 0
#   CDF_Y(y) = P(Y <= y) = P(1 / R <= y) = P(R >= 1 / y) = 1 - F_R(1 / y)
# Sampling: draw R ~ Nakagami(m, Omega), return 1 / R.
# This produces a heavy-tailed distribution as desired in Safe-ICE.
# =============================================================================
class InverseNakagamiDistribution:
    """Inverse Nakagami distribution with numerical stability."""

    # -------------------- PDF --------------------
    @overload
    @staticmethod
    def pdf(y: float, m: float, Omega: float) -> float: ...
    @overload
    @staticmethod
    def pdf(y: NDArrayF, m: float, Omega: float) -> NDArrayF: ...

    @staticmethod
    def pdf(y: npt.ArrayLike, m: float, Omega: float) -> float | NDArrayF:
        y_arr: NDArrayF = np.asarray(y, dtype=np.float64)
        m = float(m)
        Omega = float(Omega)

        out: NDArrayF = np.zeros_like(y_arr, dtype=np.float64)
        mask = y_arr > 0.0

        if not np.any(mask):
            if np.isscalar(y) or np.asarray(y).ndim == 0:
                return 0.0
            return out

        ym = y_arr[mask]

        # pdf_Y(y) = f_R(1/y) * (1 / y^2)
        # Use log-space for numerical stability
        r_inv = 1.0 / ym

        # Get log PDF of Nakagami at 1/y
        log_fr = NakagamiDistribution.log_pdf(r_inv, m, Omega)

        # log(pdf_Y) = log(f_R(1/y)) - 2*log(y)
        log_pdf_y = log_fr - 2.0 * np.log(ym)

        # Convert back from log-space
        pdf_values = np.exp(log_pdf_y)
        pdf_values = np.where(np.isfinite(pdf_values), pdf_values, 0.0)
        out[mask] = pdf_values

        if np.isscalar(y) or np.asarray(y).ndim == 0:
            return _as_float(out)
        return out

    # -------------------- CDF --------------------
    @overload
    @staticmethod
    def cdf(y: float, m: float, Omega: float) -> float: ...
    @overload
    @staticmethod
    def cdf(y: NDArrayF, m: float, Omega: float) -> NDArrayF: ...

    @staticmethod
    def cdf(y: npt.ArrayLike, m: float, Omega: float) -> float | NDArrayF:
        y_arr: NDArrayF = np.asarray(y, dtype=np.float64)
        m = float(m)
        Omega = float(Omega)

        out: NDArrayF = np.zeros_like(y_arr, dtype=np.float64)
        # For y <= 0, CDF = 0. For y > 0: P(Y <= y) = P(R >= 1/y) = 1 - F_R(1/y).
        mask = y_arr > 0.0

        if np.any(mask):
            ym = y_arr[mask]
            r_thresh = 1.0 / ym
            Fr = NakagamiDistribution.cdf(r_thresh, m, Omega)  # ndarray
            out[mask] = 1.0 - Fr

        if np.isscalar(y) or np.asarray(y).ndim == 0:
            return _as_float(out)
        return out

    # -------------------- Sample --------------------
    @staticmethod
    def sample(m: float, Omega: float, n: int = 1) -> NDArrayF:
        """Draw `n` samples via inverse transform Y = 1 / R, R ~ Nakagami(m, Omega)."""
        r = NakagamiDistribution.sample(m, Omega, n)
        # Avoid division by zero (practically not needed, r>0 a.s.; still guard)
        eps = np.finfo(np.float64).tiny
        return 1.0 / np.maximum(r, eps)
