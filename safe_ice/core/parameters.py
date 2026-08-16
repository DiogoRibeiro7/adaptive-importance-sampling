# safe_ice/core/parameters.py
"""Parameter dataclasses for the Safe-ICE algorithm.

We use explicit NumPy typing so mypy knows array shapes/dtypes and to avoid
`Any` leakage when accessing attributes like `.shape`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

# Explicit type aliases for clarity and mypy friendliness
NDArrayF = npt.NDArray[np.float64]


@dataclass(init=False)
class vMFNMParameters:
    """Parameters for the von Mises–Fisher + Nakagami mixture (vMFNM).

    Attributes
    ----------
    pi : NDArrayF
        Mixture weights (shape: (K,)), non-negative and sum to 1.
    m : NDArrayF
        Nakagami shape parameters (shape: (K,)), typically m >= 0.5.
    Omega : NDArrayF
        Nakagami scale parameters (shape: (K,)), Omega > 0.
    mu : NDArrayF
        vMF mean directions (shape: (K, d)); each row is expected to be unit-norm.
    kappa : NDArrayF
        vMF concentration parameters (shape: (K,)), kappa >= 0.
    """

    pi: NDArrayF  # (K,)
    m: NDArrayF  # (K,)
    Omega: NDArrayF  # (K,)
    mu: NDArrayF  # (K, d)
    kappa: NDArrayF  # (K,)

    def __init__(
        self,
        pi: NDArrayF,
        m: NDArrayF,
        Omega: NDArrayF,
        mu: NDArrayF,
        kappa: NDArrayF,
        K: int | None = None,
        d: int | None = None,
    ) -> None:
        # K/d are accepted for backward compatibility with older call sites/tests.
        self.pi = np.asarray(pi, dtype=np.float64)
        self.m = np.asarray(m, dtype=np.float64)
        self.Omega = np.asarray(Omega, dtype=np.float64)
        self.mu = np.asarray(mu, dtype=np.float64)
        self.kappa = np.asarray(kappa, dtype=np.float64)

        if K is not None and int(K) != int(self.pi.shape[0]):
            raise ValueError("Provided K is inconsistent with parameter array shapes.")
        if d is not None and int(d) != int(self.mu.shape[1]):
            raise ValueError("Provided d is inconsistent with mu shape.")

    @property
    def K(self) -> int:
        """Number of mixture components K."""
        # Cast to Python int so mypy doesn't infer numpy scalar / Any.
        return int(self.pi.shape[0])

    @property
    def d(self) -> int:
        """Ambient dimension d."""
        # Indexing into `.shape` can be typed as `Any` without casts; make it explicit.
        return int(self.mu.shape[1])
