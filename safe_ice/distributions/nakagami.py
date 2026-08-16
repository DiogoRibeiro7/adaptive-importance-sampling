# safe_ice/distributions/nakagami.py
"""Nakagami and Inverse Nakagami distributions with scalar/array-friendly APIs.

Typing policy
-------------
- Public functions accept either scalars or arrays (ArrayLike).
- If the input is a scalar, return a Python `float`.
- If the input is an array, return `NDArray[np.float64]`.

We expose precise typing with @overload so mypy understands both cases.
"""

from __future__ import annotations

from typing import Any, overload

import numpy as np
import numpy.typing as npt
from scipy.special import gammainc, loggamma  # normalized lower incomplete gamma

from ..typing import RNGLike

NDArrayF = npt.NDArray[np.float64]


def _as_float(x: np.ndarray) -> float:
    """Extract a Python float from a 0-d/1-elem array safely."""
    return float(np.asarray(x, dtype=np.float64).reshape(1)[0])


# =============================================================================
# Nakagami(m, Omega) on r > 0
# pdf(r) = 2 * (m^m / (Gamma(m) * Omega^m)) * r^{2m-1} * exp(-m r^2 / Omega), r >= 0
# CDF F(r) = P(m, m r^2 / Omega) where P is the regularized lower incomplete gamma
# Sampling: if X ~ Gamma(shape=m, scale=Omega/m), then R = sqrt(X) ~ Nakagami(m, Omega).
# =============================================================================
class NakagamiDistribution:
    """Nakagami(m, Omega) distribution utilities."""

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

        # Use log-space computation for numerical stability
        # log(pdf) = log(2) + m*log(m) - loggamma(m) - m*log(Omega) + (2m-1)*log(xm) - m*xm^2/Omega
        try:
            # For extreme m values (> 170), use approximation to avoid overflow
            if m > 170:
                # For very large m, the distribution becomes extremely concentrated
                # Use a normal approximation around the mode
                mode = np.sqrt(Omega * (m - 0.5) / m) if m > 0.5 else 0.0
                # Variance approximation for large m
                variance = Omega / (4 * m)
                std = np.sqrt(variance)

                # Use normal PDF approximation
                pdf_values = (1.0 / (std * np.sqrt(2 * np.pi))) * np.exp(
                    -0.5 * ((xm - mode) / std) ** 2
                )
                out[mask] = pdf_values
            else:
                # Standard log-space computation for reasonable m values
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
            # Fallback: return 0 for numerical issues
            out[mask] = 0.0

        # Return scalar if input was scalar
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

            try:
                # For extreme m values, use approximation
                if m > 170:
                    # Use normal CDF approximation
                    mode = np.sqrt(Omega * (m - 0.5) / m) if m > 0.5 else 0.0
                    variance = Omega / (4 * m)
                    std = np.sqrt(variance)

                    # Normal CDF
                    from scipy.stats import norm

                    out[mask] = norm.cdf(r_arr[mask], loc=mode, scale=std)
                else:
                    # Handle potential overflow in z
                    z = np.minimum(z, 700)  # Cap to avoid overflow in gammainc

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
    def sample(
        m: float, Omega: float, n: int = 1, rng: RNGLike | None = None
    ) -> NDArrayF:
        """Draw `n` Nakagami samples as sqrt of a Gamma(m, Omega/m).

        Parameters
        ----------
        rng : numpy random generator, optional
            If *None*, the global ``np.random`` state is used.
        """
        _rng: Any = rng if rng is not None else np.random
        m = float(m)
        Omega = float(Omega)
        n = int(n)

        if m > 100:
            mean = np.sqrt(Omega * (m - 0.5) / m) if m > 0.5 else 0.0
            variance = Omega * (1 - 1 / (4 * m)) / m if m > 0.25 else Omega
            std = np.sqrt(variance)
            samples = _rng.normal(mean, std, n)
            return np.asarray(np.abs(samples), dtype=np.float64)

        x = np.asarray(_rng.gamma(shape=m, scale=(Omega / m), size=n), dtype=np.float64)
        return np.asarray(np.sqrt(x), dtype=np.float64)


# =============================================================================
# Inverse Nakagami via transformation: if R ~ Nakagami(m, Omega), define Y = 1 / R.
# Then:
#   pdf_Y(y) = pdf_R(1/y) * (1 / y^2),  y > 0
#   CDF_Y(y) = P(Y <= y) = P(1 / R <= y) = P(R >= 1 / y) = 1 - F_R(1 / y)
# Sampling: draw R ~ Nakagami(m, Omega), return 1 / R.
# This produces a heavy-tailed distribution as desired in Safe-ICE.
# =============================================================================
class InverseNakagamiDistribution:
    """Inverse Nakagami distribution defined by Y = 1 / R, R ~ Nakagami(m, Omega)."""

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
        r_inv = 1.0 / ym

        # Get PDF of Nakagami at 1/y (now numerically stable)
        fr = NakagamiDistribution.pdf(r_inv, m, Omega)  # returns ndarray here

        # Compute final PDF
        pdf_values = fr * (1.0 / (ym**2))
        # Handle any numerical issues
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
        ym = y_arr[mask]
        r_thresh = 1.0 / ym
        Fr = NakagamiDistribution.cdf(r_thresh, m, Omega)  # ndarray
        out[mask] = 1.0 - Fr

        if np.isscalar(y) or np.asarray(y).ndim == 0:
            return _as_float(out)
        return out

    # -------------------- Sample --------------------
    @staticmethod
    def sample(
        m: float, Omega: float, n: int = 1, rng: RNGLike | None = None
    ) -> NDArrayF:
        """Draw `n` samples via Y = 1/R, R ~ Nakagami(m, Omega).

        Parameters
        ----------
        rng : numpy random generator, optional
            Forwarded to :meth:`NakagamiDistribution.sample`.
        """
        r = NakagamiDistribution.sample(m, Omega, n, rng=rng)
        eps = np.finfo(np.float64).tiny
        return 1.0 / np.maximum(r, eps)
