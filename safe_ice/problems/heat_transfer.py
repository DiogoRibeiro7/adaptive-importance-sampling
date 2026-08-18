# safe_ice/problems/heat_transfer.py
"""Heat transfer problem with Karhunen–Loève expansion."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import numpy.typing as npt
from scipy import sparse
from scipy.sparse.linalg import spsolve

# Typed aliases
NDArrayF = npt.NDArray[np.float64]
NDArrayB = npt.NDArray[np.bool_]


class HeatTransferProblem:
    """Complete heat transfer problem implementation from Section 4.5."""

    def __init__(
        self,
        grid_size: int = 21,
        correlation_length: float = 0.2,
        n_terms: int = 10,
        field_std: float = 0.3,
        threshold: float = 10.0,
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
        # Equation (46): k(x, x') = exp(-||x - x'||^2 / l^2). This was
        # exp(-||x - x'|| / l), the exponential kernel, which gives a far
        # rougher field than the squared-exponential the paper specifies.
        sq_distances: NDArrayF = np.sum(
            (self.grid_points[:, None, :] - self.grid_points[None, :, :]) ** 2,
            axis=2,
        )
        C: NDArrayF = np.exp(-sq_distances / np.float64(self.l) ** 2)

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

        # Fix the sign of each mode. An eigenvector is only defined up to sign,
        # and LAPACK's choice varies between builds, so the same coefficients
        # produced mirror-image fields on different machines: a test asserting
        # that positive coefficients raise the conductivity passed locally and
        # failed in CI. Anchoring on the largest-magnitude entry makes the
        # expansion reproducible.
        dominant = np.argmax(np.abs(self.eigenvecs), axis=0)
        signs = np.sign(self.eigenvecs[dominant, np.arange(self.eigenvecs.shape[1])])
        signs[signs == 0.0] = 1.0
        self.eigenvecs = self.eigenvecs * signs

        # Assign eigenvectors alias AFTER normalization so it exposes
        # the normalized modes, not the stale pre-normalization ones.
        self.eigenvectors = self.eigenvecs[:50, :]

    def _region_mask(
        self, x_lo: float, x_hi: float, y_lo: float, y_hi: float
    ) -> NDArrayB:
        """Grid nodes inside a rectangle, robust to floating-point edges.

        The bounds are multiples of 0.05 and 0.1, which a linspace reproduces
        exactly at some grid sizes and misses by an ulp at others. A bare
        ``>=`` comparison therefore captured a different number of nodes
        depending on ``grid_size``: at 31 the nominal average temperature on
        region B came out at 2.02 against 5.04, 4.79 and 4.75 for 21, 41 and
        51, purely because the region had lost a row of nodes.
        """
        tol = 1e-9
        x_min, x_max = x_lo - tol, x_hi + tol
        y_min, y_max = y_lo - tol, y_hi + tol
        inside = (
            np.greater_equal(self.X, x_min)
            & np.less_equal(self.X, x_max)
            & np.greater_equal(self.Y, y_min)
            & np.less_equal(self.Y, y_max)
        )
        return np.asarray(inside, dtype=bool)

    def generate_permeability_field(self, xi: npt.ArrayLike) -> NDArrayF:
        """Generate lognormal permeability field from the KL expansion."""
        # Mean and std for lognormal field
        mu_kappa = 1.0
        sigma_kappa = self.field_std

        # Lognormal parameters
        a_kappa = np.log((mu_kappa**2) / np.sqrt(mu_kappa**2 + sigma_kappa**2))
        b_kappa = np.sqrt(np.log(1.0 + (sigma_kappa**2) / (mu_kappa**2)))

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
        """Solve -div(kappa grad T) = I_A Q by a direct sparse solve.

        Five-point conservative finite differences with conductivity averaged
        onto the cell faces.

        Boundary conditions follow Figure 11: zero Dirichlet on the top edge,
        zero Neumann on the other three. The text of section 4.5 says the
        opposite -- "zero Neumann boundary on the top part and zero Dirichlet
        boundary in the rest" -- and the two cannot both hold. The figure is
        the consistent one: with heat able to escape through three edges the
        nominal average temperature on region B is 0.49 against a failure
        threshold of 10, which is 51 standard deviations away and so could
        never produce the reported probability of 4.69e-07. With the top edge
        as the only sink it is 5.32, and failure is a rare event rather than an
        impossible one.

        This replaces an explicit relaxation, ``T += 0.01 * (laplacian + Q)``
        run for 1000 sweeps with the result clamped to +/-1e6. The elliptic
        problem has no time derivative to march, and the fixed pseudo-step was
        about sixteen times the explicit stability limit for this grid
        (``dt * kappa / h^2 = 4`` against a 2-D limit of ``0.25``), so it
        diverged: 380 of 441 nodes ended pinned at the clamp, and the resulting
        limit state returned exactly ``threshold`` for every input, with no
        dependence on the random field at all.
        """
        kappa: NDArrayF = np.asarray(kappa_field, dtype=np.float64)
        n = self.grid_size
        h = (self.domain[1] - self.domain[0]) / float(n - 1)

        # Heat source on A = (0.2, 0.3) x (0.2, 0.3). The y extent used to be
        # computed and then discarded, leaving a full-height strip.
        source = self._region_mask(0.2, 0.3, 0.2, 0.3)
        # Spread Q over the source nodes so the discrete heat input equals the
        # physical one, Q times the area of A, at any grid size. Assigning Q to
        # each node instead makes the input (0.1 + h)^2 / 0.1^2 times too
        # large, because a closed region picks up an extra row of nodes on each
        # side: 2.25x at grid_size 21 and 1.44x at 51. That was the whole of
        # the apparent grid dependence -- dividing it out leaves the nominal
        # average temperature on B at 4.55 for every grid tested.
        source_area = 0.1 * 0.1
        n_source = int(np.count_nonzero(source))
        source_density = self.Q * source_area / (n_source * h * h) if n_source else 0.0

        def index(row: int, col: int) -> int:
            return row * n + col

        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []
        rhs: NDArrayF = np.zeros(n * n, dtype=np.float64)

        inv_h2 = 1.0 / (h * h)

        for i in range(n):
            for j in range(n):
                node = index(i, j)

                # Dirichlet on the top edge.
                if i == n - 1:
                    rows.append(node)
                    cols.append(node)
                    vals.append(1.0)
                    continue

                diagonal = 0.0
                for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ni, nj = i + di, j + dj
                    # Zero Neumann on the bottom, left and right edges: the
                    # ghost node mirrors its interior counterpart.
                    if ni < 0:
                        ni = 1
                    if nj < 0:
                        nj = 1
                    elif nj > n - 1:
                        nj = n - 2

                    face_kappa = 0.5 * (kappa[i, j] + kappa[ni, nj])
                    coefficient = face_kappa * inv_h2
                    diagonal += coefficient
                    rows.append(node)
                    cols.append(index(ni, nj))
                    vals.append(-coefficient)

                rows.append(node)
                cols.append(node)
                vals.append(diagonal)
                rhs[node] = source_density if source[i, j] else 0.0

        matrix = sparse.csr_matrix(
            (vals, (rows, cols)), shape=(n * n, n * n), dtype=np.float64
        )
        solution: NDArrayF = np.asarray(spsolve(matrix, rhs), dtype=np.float64)
        return solution.reshape(n, n)

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
            # Section 4.5 calls B "a squared domain" but writes its extent as
            # (-0.3, -0.2) x (-0.3, 0.2), which is not square. Figure 11 draws
            # a small square in the lower left, so the second bound is read as
            # -0.2.
            eval_mask = self._region_mask(-0.3, -0.2, -0.3, -0.2)

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
