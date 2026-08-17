# safe_ice/distributions/vmf.py
"""von Mises–Fisher distribution sampler (Wood, 1994)."""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

NDArrayF = npt.NDArray[np.float64]

# Type alias accepted for the ``rng`` parameter.
RNGLike = np.random.Generator | np.random.RandomState


def _default_rng(
    rng: RNGLike | None = None,
) -> RNGLike:
    """Return *rng* if given, else the legacy ``np.random`` module."""
    return rng if rng is not None else np.random  # type: ignore[return-value]


class VonMisesFisherSampler:
    """Exact von Mises–Fisher sampler using Wood's algorithm."""

    @staticmethod
    def sample(
        mu: npt.ArrayLike,
        kappa: float,
        n_samples: int = 1,
        rng: RNGLike | None = None,
    ) -> NDArrayF:
        """Sample from a vMF_d(mu, kappa).

        Parameters
        ----------
        mu : array_like
            Mean direction, a length-d array. It need not be a unit vector;
            it is normalised internally.
        kappa : float
            Concentration, must be non-negative.
        n_samples : int
            Number of samples to draw.
        rng : numpy random generator, optional
            If *None*, the global ``np.random`` state is used.

        Returns
        -------
        ndarray of shape (n_samples, d)
            Unit vectors on S^{d-1}.
        """
        _rng = _default_rng(rng)
        mu_arr: NDArrayF = np.asarray(mu, dtype=np.float64).reshape(-1)
        d: int = int(mu_arr.shape[0])

        # Normalize mu (if zero, raise)
        mu_norm: float = float(np.linalg.norm(mu_arr))
        if mu_norm == 0.0:
            raise ValueError("mu must be non-zero.")
        mu_unit: NDArrayF = mu_arr / mu_norm

        # kappa == 0 => uniform on sphere
        if kappa == 0.0:
            samples: NDArrayF = np.asarray(
                _rng.normal(loc=0.0, scale=1.0, size=(n_samples, d)),
                dtype=np.float64,
            )
            # normalize each row
            norms: NDArrayF = np.linalg.norm(samples, axis=1, keepdims=True).astype(
                np.float64, copy=False
            )
            eps: float = float(np.finfo(np.float64).tiny)
            norms = np.maximum(norms, eps)
            return (samples / norms).astype(np.float64, copy=False)

        # Special case: d == 2 (circular Von Mises) — must come before the
        # high-κ tangent-space shortcut so that 2-D calls always use the
        # exact circular sampler.
        if d == 2:
            return VonMisesFisherSampler._sample_circular(
                mu_unit, float(kappa), n_samples, _rng
            )

        # General case d >= 3. Wood's rejection sampler is exact and holds up at
        # any concentration: measured against I_{d/2}(k)/I_{d/2-1}(k) it is
        # accurate to ~1e-6 out to kappa = 1e4.
        #
        # A "high-concentration" shortcut used to intercept kappa >= 30 with a
        # Gaussian around mu using scale 0.2/sqrt(kappa). The 0.2 is arbitrary
        # and roughly five times too small -- the tangent-space standard
        # deviation is 1/sqrt(kappa) -- and it ignored the dimension entirely,
        # so samples clustered far too tightly around mu. At d=20, kappa=50 the
        # mean resultant length came out at 0.9925 against a true 0.8263.
        w: NDArrayF = VonMisesFisherSampler._sample_w_wood_batch(
            float(kappa), d, n_samples, _rng
        )

        # Directions v ~ Unif(S^{d-2}) in R^{d-1}, one row per sample.
        v_raw: NDArrayF = np.asarray(
            _rng.normal(loc=0.0, scale=1.0, size=(n_samples, d - 1)),
            dtype=np.float64,
        )
        v_norms: NDArrayF = np.linalg.norm(v_raw, axis=1, keepdims=True)
        # A row of exact zeros has probability zero, but guard it anyway.
        degenerate = (v_norms <= 0.0).reshape(-1)
        v_norms = np.maximum(v_norms, float(np.finfo(np.float64).tiny))
        v: NDArrayF = v_raw / v_norms
        if np.any(degenerate):
            v[degenerate] = 0.0
            v[degenerate, 0] = 1.0

        # Points on S^{d-1} aligned with e_d: the first d-1 coordinates scale by
        # sqrt(1 - w^2), the last one is w.
        radial: NDArrayF = np.sqrt(np.maximum(0.0, 1.0 - w * w)).reshape(-1, 1)
        xy: NDArrayF = np.concatenate([radial * v, w.reshape(-1, 1)], axis=1)

        # Rotate e_d -> mu with a single Householder reflection, applied to
        # every row at once.
        return VonMisesFisherSampler._householder_rotation_batch(xy, mu_unit)

    @staticmethod
    def _sample_w_wood_batch(kappa: float, d: int, n: int, rng: RNGLike) -> NDArrayF:
        """Sample n values of w with Wood (1994), rejecting in batches.

        Mathematically identical to drawing them one at a time; the acceptance
        rate is high, so a modest over-draw usually finishes in one pass.
        """
        b: float = (d - 1.0) / (
            2.0 * kappa + math.sqrt(4.0 * kappa * kappa + (d - 1.0) ** 2)
        )
        x0: float = (1.0 - b) / (1.0 + b)
        c: float = kappa * x0 + (d - 1.0) * math.log(1.0 - x0 * x0)
        a_beta: float = (d - 1.0) / 2.0

        if n <= 0:
            return np.zeros(0, dtype=np.float64)

        kept: list[NDArrayF] = []
        remaining: int = n
        while remaining > 0:
            # Over-draw a little so a single pass usually suffices.
            draw = max(int(remaining * 1.3) + 8, 16)
            z: NDArrayF = np.asarray(
                rng.beta(a=a_beta, b=a_beta, size=draw), dtype=np.float64
            )
            w_try: NDArrayF = (1.0 - (1.0 + b) * z) / (1.0 - (1.0 - b) * z)
            u: NDArrayF = np.asarray(rng.uniform(0.0, 1.0, size=draw), dtype=np.float64)

            # 1 - x0 * w_try is strictly positive: |w_try| <= 1 and 0 < x0 < 1.
            accept = (kappa * w_try + (d - 1.0) * np.log1p(-x0 * w_try) - c) >= np.log(
                u
            )

            taken = w_try[accept][:remaining]
            kept.append(taken)
            remaining -= int(taken.shape[0])

        return np.concatenate(kept).astype(np.float64, copy=False)

    @staticmethod
    def _householder_rotation_batch(x: NDArrayF, mu: NDArrayF) -> NDArrayF:
        """Apply the reflection mapping e_d to mu, to every row of x."""
        d: int = int(mu.shape[0])
        if x.shape[1] != d:
            raise ValueError("x rows and mu must have the same dimension.")

        e_d: NDArrayF = np.zeros(d, dtype=np.float64)
        e_d[-1] = 1.0

        u: NDArrayF = e_d - mu
        u_norm: float = float(np.linalg.norm(u))
        if u_norm < 1e-15:
            # mu already is e_d, so the reflection is the identity.
            return x.astype(np.float64, copy=False)

        u = u / u_norm
        return (x - 2.0 * np.outer(x @ u, u)).astype(np.float64, copy=False)

    @staticmethod
    def _sample_circular(
        mu: NDArrayF,
        kappa: float,
        n_samples: int,
        rng: RNGLike,
    ) -> NDArrayF:
        """Sample on S^1 (d == 2)."""
        if mu.shape[0] != 2:
            raise ValueError("Circular case requires mu with dimension 2.")
        angles: NDArrayF = np.asarray(
            rng.vonmises(mu=0.0, kappa=kappa, size=n_samples),
            dtype=np.float64,
        )
        mu_angle: float = float(np.arctan2(mu[1], mu[0]))
        angles = (angles + mu_angle).astype(np.float64, copy=False)
        return np.column_stack([np.cos(angles), np.sin(angles)]).astype(
            np.float64, copy=False
        )
