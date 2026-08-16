# safe_ice/problems/heat_transfer.py
"""Heat transfer problem with Karhunen–Loève expansion."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import numpy.typing as npt

# Typed aliases
NDArrayF = npt.NDArray[np.float64]
NDArrayB = npt.NDArray[np.bool_]


class HeatTransferProblem:
    """Complete heat transfer problem implementation from Section 4.5."""

    def __init__(
        self,
        grid_size: int = 21,
        correlation_length: float = 0.5,
        n_terms: int = 10,
        field_std: float = 0.3,
        threshold: float = 100.0,
        heat_source: float = 2000.0,
    ) -> None:
        """Initialize heat transfer problem.

        Parameters
        ----------
        grid_size : int
            Discretization grid size.
        correlation_length : float
            Correlation length for the random field.
        n_terms : int
            Number of Karhunen-Loeve expansion terms.
        field_std : float
            Standard deviation for the lognormal conductivity field.
        threshold : float
            Failure threshold used in the limit state function.
        heat_source : float
            Heat source magnitude.
        """
        self.grid_size = int(grid_size)
        self.l = float(correlation_length)
        self.correlation_length = float(correlation_length)  # compatibility alias
        self.n_terms = int(n_terms)
        self.field_std = float(field_std)
        self.threshold = float(threshold)
        self.Q = float(heat_source)

        # Domain parameters: (x_min, x_max, y_min, y_max)
        self.domain = (-0.5, 0.5, -0.5, 0.5)

        self._setup_discretization()
        self._setup_kl_expansion()

    def _setup_discretization(self) -> None:
        """Setup finite-difference discretization (typed to avoid Any propagation)."""
        x: NDArrayF = np.linspace(
            self.domain[0], self.domain[1], self.grid_size, dtype=np.float64
        )
        y: NDArrayF = np.linspace(
            self.domain[2], self.domain[3], self.grid_size, dtype=np.float64
        )
        X, Y = np.meshgrid(x, y)
        self.X: NDArrayF = np.asarray(X, dtype=np.float64)
        self.Y: NDArrayF = np.asarray(Y, dtype=np.float64)

        # Grid points (N x 2)
        self.grid_points: NDArrayF = np.asarray(
            np.column_stack([self.X.ravel(), self.Y.ravel()]), dtype=np.float64
        )
        self.n_points = int(self.grid_points.shape[0])

    def _setup_kl_expansion(self) -> None:
        """Setup Karhunen–Loève expansion for a lognormal random field."""
        # Exponential covariance: k(x,x') = exp(-||x - x'|| / l)
        distances: NDArrayF = np.sqrt(
            np.sum(
                (self.grid_points[:, None, :] - self.grid_points[None, :, :]) ** 2,
                axis=2,
            )
        )
        C: NDArrayF = np.exp(-distances / np.float64(self.l))

        # Eigendecomposition (float64 arrays)
        eigenvals, eigenvecs = np.linalg.eigh(C)

        # Sort by descending eigenvalue and keep first n_terms
        idx = np.argsort(eigenvals)[::-1]
        self.eigenvals = eigenvals[idx][: self.n_terms].astype(np.float64, copy=False)
        self.eigenvecs = eigenvecs[:, idx][:, : self.n_terms].astype(
            np.float64, copy=False
        )
        # Compatibility alias for eigenvalues (not affected by normalization).
        self.eigenvalues = self.eigenvals

        # --- Vectorized, type-stable column normalization ---
        # Norms as a (1, n_terms) array (keepdims=True prevents scalar return)
        norms: NDArrayF = np.linalg.norm(self.eigenvecs, axis=0, keepdims=True).astype(
            np.float64, copy=False
        )

        # Avoid division by zero in a vectorized, typed-safe way
        eps = np.finfo(np.float64).tiny  # strictly positive float64
        norms = np.maximum(norms, eps)

        # Broadcasted normalization: (n_points, n_terms) / (1, n_terms)
        self.eigenvecs = self.eigenvecs / norms

        # Assign eigenvectors alias AFTER normalization so it exposes
        # the normalized modes, not the stale pre-normalization ones.
        self.eigenvectors = self.eigenvecs[:50, :]

    def generate_permeability_field(self, xi: npt.ArrayLike) -> NDArrayF:
        """Generate lognormal permeability field from the KL expansion."""
        # Mean and std for lognormal field
        mu_kappa = 1.0
        sigma_kappa = self.field_std

        # Lognormal parameters
        a_kappa = np.log((mu_kappa**2) / np.sqrt(mu_kappa**2 + sigma_kappa**2))
        b_kappa = np.sqrt(np.log(1.0 + (sigma_kappa**2) / (mu_kappa**2)))

        # KL expansion coefficients
        np.asarray(xi, dtype=np.float64)[: self.n_terms]
        # KL expansion coefficients: turn the triple product into a matvec
        # shapes: eigenvecs (n_points, n_terms) @ coeffs (n_terms,) -> (n_points,)
        coeffs: NDArrayF = np.multiply(
            np.sqrt(self.eigenvals, dtype=np.float64),
            np.asarray(xi, dtype=np.float64)[: self.n_terms],
            dtype=np.float64,
        )
        f_field: NDArrayF = np.asarray(self.eigenvecs @ coeffs, dtype=np.float64)

        # Lognormal field
        kappa_field: NDArrayF = np.exp(a_kappa + b_kappa * f_field, dtype=np.float64)

        return np.asarray(
            kappa_field.reshape(self.grid_size, self.grid_size), dtype=np.float64
        )

    def solve_heat_equation(self, kappa_field: npt.ArrayLike) -> NDArrayF:
        """Solve heat equation using a simple finite-difference iterative scheme."""
        h = 1.0 / float(self.grid_size - 1)  # grid spacing

        # Temperature and conductivity fields
        T: NDArrayF = np.zeros((self.grid_size, self.grid_size), dtype=np.float64)
        kappa: NDArrayF = np.asarray(kappa_field, dtype=np.float64)

        # Heat source region A = (0.2, 0.3) × (0.2, 0.3)
        x_indices = np.where((self.X >= 0.2) & (self.X <= 0.3))
        np.where((self.Y >= 0.2) & (self.Y <= 0.3))
        source_mask: NDArrayB = np.zeros_like(T, dtype=bool)
        source_mask[x_indices[0], x_indices[1]] = True  # rectangular mask

        # Iterative Jacobi-like update
        for _ in range(1000):
            T_old: NDArrayF = T.copy()

            # Interior updates
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                for i in range(1, self.grid_size - 1):
                    for j in range(1, self.grid_size - 1):
                        laplacian = (
                            kappa[i + 1, j] * (T_old[i + 1, j] - T_old[i, j])
                            - kappa[i - 1, j] * (T_old[i, j] - T_old[i - 1, j])
                        ) / h**2 + (
                            kappa[i, j + 1] * (T_old[i, j + 1] - T_old[i, j])
                            - kappa[i, j - 1] * (T_old[i, j] - T_old[i, j - 1])
                        ) / h**2

                        source_term = self.Q if source_mask[i, j] else 0.0
                        updated = T_old[i, j] + 0.01 * (laplacian + source_term)
                        if not np.isfinite(updated):
                            updated = 0.0
                        # Clamp to keep iteration numerically stable.
                        T[i, j] = float(np.clip(updated, -1e6, 1e6))

            # Boundary conditions
            T[0, :] = 0.0  # Bottom: Dirichlet
            T[-1, :] = T[-2, :]  # Top: Neumann (zero gradient)
            T[:, 0] = 0.0  # Left: Dirichlet
            T[:, -1] = 0.0  # Right: Dirichlet

            # Convergence check
            if float(np.max(np.abs(T - T_old))) < 1e-6:
                break

        return T

    def create_limit_state_function(
        self, threshold: float | None = None
    ) -> Callable[[npt.ArrayLike], float | NDArrayF]:
        """Return g(u) = threshold − average temperature on region B."""
        limit_threshold = self.threshold if threshold is None else float(threshold)

        def limit_state_function(u: npt.ArrayLike) -> float | NDArrayF:
            arr = np.asarray(u, dtype=np.float64)
            if arr.ndim == 1:
                samples = arr.reshape(1, -1)
                is_single = True
            elif arr.ndim == 2:
                samples = arr
                is_single = False
            else:
                raise ValueError("Input must be a 1-D or 2-D array.")

            if samples.shape[1] != self.n_terms:
                raise ValueError(f"Expected input dimension {self.n_terms}.")

            g_values = np.zeros(samples.shape[0], dtype=np.float64)
            # Generate permeability field from KL coefficients
            x_mask = (self.X >= -0.3) & (self.X <= -0.2)
            y_mask = (self.Y >= -0.3) & (self.Y <= 0.2)
            eval_mask = x_mask & y_mask

            for i, sample in enumerate(samples):
                kappa_field = self.generate_permeability_field(sample)
                T_field = self.solve_heat_equation(kappa_field)
                T_avg = float(np.mean(T_field[eval_mask], dtype=np.float64))
                g_values[i] = limit_threshold - T_avg

            if is_single:
                return float(g_values[0])
            return g_values

        return limit_state_function

    def get_limit_state_function(
        self, threshold: float | None = None
    ) -> Callable[[npt.ArrayLike], float | NDArrayF]:
        """Compatibility alias for create_limit_state_function."""
        return self.create_limit_state_function(threshold=threshold)
