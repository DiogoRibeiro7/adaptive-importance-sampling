# safe_ice/distributions/mixture.py
"""von Mises–Fisher–Nakagami mixture distribution."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from ..core.parameters import vMFNMParameters
from ..typing import RNGLike
from ._numeric import radii_and_directions, sphere_surface_area, vmf_pdf_batch
from .nakagami import NakagamiDistribution
from .vmf import VonMisesFisherSampler

NDArrayF = npt.NDArray[np.float64]

# ``sphere_surface_area`` moved to ``_numeric`` but is re-exported here, where
# it has always been importable from.
__all__ = ["sphere_surface_area", "vMFNMDistribution"]


class vMFNMDistribution:
    """Complete von Mises–Fisher–Nakagami mixture distribution."""

    def __init__(self, params: vMFNMParameters) -> None:
        self.params = params
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        K, d = self.params.K, self.params.d
        assert self.params.pi.shape == (K,)
        assert self.params.m.shape == (K,)
        assert self.params.Omega.shape == (K,)
        assert self.params.mu.shape == (K, d)
        assert self.params.kappa.shape == (K,)

        assert np.allclose(float(np.sum(self.params.pi)), 1.0)
        assert np.all(self.params.pi >= 0)
        assert np.all(self.params.m > 0)
        assert np.all(self.params.Omega > 0)
        assert np.all(self.params.kappa >= 0)

        # Normalize mu_k to unit vectors
        for k in range(K):
            norm = float(np.linalg.norm(self.params.mu[k]))
            if norm > 0.0:
                self.params.mu[k] = self.params.mu[k] / norm

    def pdf(self, x: npt.ArrayLike) -> NDArrayF:
        """Compute mixture PDF at rows of x. Returns (n,) float64 array."""
        X = np.asarray(x, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        n, d = X.shape
        assert d == self.params.d

        radii, directions = radii_and_directions(X)
        jacobian = radii ** (d - 1)

        # One vectorised pass per component rather than one call per (sample,
        # component) pair. Accumulating sequentially over k keeps the summation
        # order of the original per-sample loop; batched norms and BLAS matvecs
        # still shift results by ~1e-13 relative, far below sampling noise.
        out = np.zeros(n, dtype=np.float64)
        for k in range(self.params.K):
            nak = np.asarray(
                NakagamiDistribution.pdf(
                    radii, float(self.params.m[k]), float(self.params.Omega[k])
                ),
                dtype=np.float64,
            )
            vmf = self._vmf_pdf_many(
                directions, self.params.mu[k], float(self.params.kappa[k])
            )
            out += float(self.params.pi[k]) * nak * vmf / jacobian
        return out

    def _vmf_pdf(self, x: NDArrayF, mu: NDArrayF, kappa: float) -> float:
        """von Mises–Fisher pdf at unit x w.r.t. the surface area measure."""
        x_arr = np.asarray(x, dtype=np.float64).reshape(1, -1)
        return float(self._vmf_pdf_many(x_arr, mu, kappa)[0])

    def _vmf_pdf_many(self, X: NDArrayF, mu: NDArrayF, kappa: float) -> NDArrayF:
        """von Mises–Fisher pdf at unit rows of X w.r.t. the surface area measure."""
        return vmf_pdf_batch(X, mu, kappa)

    def sample(self, n_samples: int, rng: RNGLike | None = None) -> NDArrayF:
        """Sample from the mixture. Returns (n_samples, d).

        Parameters
        ----------
        rng : numpy random generator, optional
            If *None*, the global ``np.random`` state is used.
        """
        _rng: Any = rng if rng is not None else np.random
        n, d = int(n_samples), self.params.d
        samples = np.zeros((n, d), dtype=np.float64)

        comp = _rng.choice(self.params.K, size=n, p=self.params.pi)
        for i in range(n):
            k = int(comp[i])
            r_i = float(
                NakagamiDistribution.sample(
                    float(self.params.m[k]),
                    float(self.params.Omega[k]),
                    1,
                    rng=rng,
                )[0]
            )
            a_i = VonMisesFisherSampler.sample(
                self.params.mu[k],
                float(self.params.kappa[k]),
                1,
                rng=rng,
            )[0]
            samples[i] = r_i * a_i
        return samples

    def log_likelihood(self, x: npt.ArrayLike) -> float:
        pdf_vals = self.pdf(x)
        # Floor at the smallest positive float rather than 1e-15: densities on
        # R^d fall below 1e-15 routinely once d is around 20, and flooring there
        # caps every term at log(1e-15) = -34.5, flattening real differences.
        floor = float(np.finfo(np.float64).tiny)
        return float(np.sum(np.log(np.maximum(pdf_vals, floor))))
