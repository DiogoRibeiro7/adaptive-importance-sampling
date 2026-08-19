"""Map a problem stated in physical units into the space the estimator works in.

The estimator draws from a standard normal prior, so a limit state written in
terms of real quantities -- a discharge in cubic metres per second, a yield
strength in megapascals -- cannot be handed to it directly. The Safe-ICE paper
says as much in its introduction: "The prior is typically Gaussian; otherwise a
Nataf or Rosenblatt transformation can be applied to map the original
distributions to Gaussian ones."

:class:`MarginalTransform` is that map. Given the marginal distribution of each
input, and optionally the correlation between them, it converts back and forth
and wraps a physical limit state into one the estimator accepts.

Independent inputs
------------------
With independent marginals the map is one-dimensional and exact:

.. math::

    u_i = \\Phi^{-1}(F_i(x_i)), \\qquad x_i = F_i^{-1}(\\Phi(u_i))

Correlated inputs
-----------------
Correlation needs the Nataf transformation. Prescribing a correlation
``rho_x`` between two physical variables does not mean the underlying normals
are correlated by the same amount: the marginal transforms are non-linear, so
they distort it. The Gaussian correlation ``rho_z`` that produces a given
``rho_x`` solves

.. math::

    \\rho_x = \\int\\!\\!\\int
        \\frac{F_i^{-1}(\\Phi(z_i)) - \\mu_i}{\\sigma_i}
        \\frac{F_j^{-1}(\\Phi(z_j)) - \\mu_j}{\\sigma_j}
        \\varphi_2(z_i, z_j; \\rho_z)\\, dz_i\\, dz_j

which is solved here per pair by quadrature and a bracketed root find. The
distortion is real: two lognormals with a 30% coefficient of variation asked
for a physical correlation of 0.8 need a Gaussian correlation of 0.806, and
asked for -0.5 need -0.535. It grows with the skew of the marginals.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np
from scipy import stats
from scipy.optimize import brentq

from .typing import LimitStateFunction, NDArrayF

__all__ = ["Marginal", "MarginalTransform"]

#: How far from the boundary the uniform intermediate is kept. Phi(u) reaches
#: exactly 0 or 1 in floating point for |u| beyond about 8.3, and a ppf given
#: exactly 0 or 1 returns an infinity, which then propagates into the limit
#: state. The estimator's heavy-tailed proposal does reach that far out.
_PROBABILITY_MARGIN = 1e-15


class Marginal(Protocol):
    """What this module needs of a marginal distribution.

    Any frozen ``scipy.stats`` distribution satisfies it, as does anything else
    providing the same four methods.
    """

    def cdf(self, x: Any) -> Any:
        """Cumulative distribution function."""
        ...

    def ppf(self, q: Any) -> Any:
        """Inverse cumulative distribution function."""
        ...

    def mean(self) -> Any:
        """Distribution mean."""
        ...

    def std(self) -> Any:
        """Distribution standard deviation."""
        ...


class MarginalTransform:
    """Convert between physical space and independent standard normal space.

    Parameters
    ----------
    marginals:
        One distribution per input, in order. Frozen ``scipy.stats``
        distributions are the usual choice, e.g.
        ``[lognorm(s=0.15, scale=200.0), gumbel_r(loc=30.0, scale=4.0)]``.
    correlation:
        Target correlation matrix between the *physical* variables. Omit it for
        independent inputs. Must be symmetric with unit diagonal.
    quadrature_points:
        Nodes per axis in the Gauss-Hermite rule used to solve for the Gaussian
        correlations. The default is ample; raising it costs setup time only.

    Attributes
    ----------
    gaussian_correlation:
        The correlation the underlying normals need in order to produce
        ``correlation`` in physical space. Equal to ``correlation`` only when
        the marginals are themselves normal.

    Examples
    --------
    Wrap a limit state written in physical units and hand it to the estimator::

        from scipy.stats import lognorm
        from safe_ice import SafeICE
        from safe_ice.transforms import MarginalTransform

        transform = MarginalTransform(
            [lognorm(s=0.15, scale=200.0), lognorm(s=0.25, scale=80.0)]
        )


        def capacity_exceeded(x):
            return x[:, 0] - x[:, 1]  # resistance minus load


        estimator = SafeICE(
            limit_state_function=transform.wrap(capacity_exceeded),
            dimension=2,
        )
    """

    def __init__(
        self,
        marginals: Sequence[Marginal],
        correlation: NDArrayF | None = None,
        quadrature_points: int = 40,
    ) -> None:
        self.marginals = list(marginals)
        self.d = len(self.marginals)
        if self.d == 0:
            raise ValueError("At least one marginal is required.")

        if correlation is None:
            self.correlation: NDArrayF = np.eye(self.d, dtype=np.float64)
            self.gaussian_correlation: NDArrayF = np.eye(self.d, dtype=np.float64)
        else:
            self.correlation = self._validated_correlation(correlation)
            self.gaussian_correlation = self._solve_gaussian_correlation(
                int(quadrature_points)
            )

        # z = u @ L.T turns independent normals into ones with the required
        # Gaussian correlation.
        self._cholesky: NDArrayF = np.linalg.cholesky(self.gaussian_correlation)

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    def to_physical(self, u: NDArrayF) -> NDArrayF:
        """Map independent standard normals to physical values."""
        points = self._as_2d(u)
        if points.shape[1] != self.d:
            raise ValueError(f"Expected {self.d} columns, got {points.shape[1]}.")

        z = points @ self._cholesky.T
        probabilities = np.clip(
            stats.norm.cdf(z), _PROBABILITY_MARGIN, 1.0 - _PROBABILITY_MARGIN
        )

        physical: NDArrayF = np.empty_like(probabilities, dtype=np.float64)
        for i, marginal in enumerate(self.marginals):
            physical[:, i] = np.asarray(marginal.ppf(probabilities[:, i]))
        return physical

    def to_standard(self, x: NDArrayF) -> NDArrayF:
        """Map physical values back to independent standard normals."""
        points = self._as_2d(x)
        if points.shape[1] != self.d:
            raise ValueError(f"Expected {self.d} columns, got {points.shape[1]}.")

        z = np.empty_like(points)
        for i, marginal in enumerate(self.marginals):
            probabilities = np.clip(
                np.asarray(marginal.cdf(points[:, i])),
                _PROBABILITY_MARGIN,
                1.0 - _PROBABILITY_MARGIN,
            )
            z[:, i] = stats.norm.ppf(probabilities)

        # Undo z = u @ L.T without forming the inverse.
        return np.linalg.solve(self._cholesky, z.T).T

    def wrap(
        self,
        limit_state: LimitStateFunction,
        scale: bool | float = True,
        pilot: int = 512,
        random_state: int | np.random.Generator | None = 0,
    ) -> LimitStateFunction:
        """Turn a limit state written in physical units into one in u-space.

        The returned function is what the estimators take. Failure is still
        ``g <= 0``; only the coordinates change.

        Parameters
        ----------
        limit_state:
            The limit state in physical units. Receives an ``(n, d)`` array of
            physical values.
        scale:
            Divide the result by a constant, so that its spread is of order one.
            ``True`` estimates that constant from a pilot sample, a float uses
            the value given, and ``False`` leaves the limit state alone.

            This matters more than it looks. Safe-ICE smooths the failure
            indicator as ``Phi(-g/sigma)`` and starts from ``sigma0 = 1``, which
            assumes ``g`` is of order one. A limit state in physical units is
            not: a resistance minus a load in newtons has a spread in the
            hundreds, and ``Phi(-g/1)`` is then already a hard indicator, so the
            smoothing the method depends on does nothing. On the lognormal
            resistance problem below that turned a true 1.82e-05 into 5.1e-06,
            with individual runs between 0.03x and 0.61x of the answer -- wrong,
            but not obviously so.

            Dividing by a positive constant cannot change the answer, since
            ``{g <= 0}`` and ``{g/c <= 0}`` are the same set, so this is a pure
            reparameterisation. With it, the same problem gives 1.05x with runs
            spanning 0.94x to 1.06x. Setting ``sigma0`` to the spread instead is
            exactly equivalent -- the method only ever sees ``g/sigma`` -- but
            doing it here means the scale is handled where the physical units
            enter.
        pilot:
            Samples used to estimate the spread when ``scale`` is ``True``. Each
            costs one limit-state evaluation, which is worth knowing if that is
            a finite-element solve.
        random_state:
            Seed for the pilot sample, so the scaling is reproducible.
        """

        def in_u_space(u: NDArrayF) -> NDArrayF:
            physical = self.to_physical(np.asarray(u, dtype=np.float64))
            return np.asarray(limit_state(physical), dtype=np.float64).reshape(-1)

        divisor = 1.0
        if scale is True:
            sample = self.sample(int(pilot), random_state=random_state)
            spread = float(
                np.std(np.asarray(limit_state(sample), dtype=np.float64).reshape(-1))
            )
            if np.isfinite(spread) and spread > 0.0:
                divisor = spread
        elif scale is not False:
            divisor = float(scale)
            if divisor <= 0.0:
                raise ValueError(f"scale must be positive, got {divisor}.")

        def transformed(u: NDArrayF) -> float | NDArrayF:
            values = in_u_space(u) / divisor
            if np.ndim(u) == 1:
                return float(values[0])
            return values

        transformed.limit_state_scale = divisor  # type: ignore[attr-defined]
        return transformed

    def sample(
        self, n: int, random_state: int | np.random.Generator | None = None
    ) -> NDArrayF:
        """Draw physical samples with the requested marginals and correlation.

        Useful for checking a transform against the data it was fitted to, and
        for crude Monte Carlo in physical space.
        """
        if isinstance(random_state, np.random.Generator):
            rng = random_state
        else:
            rng = np.random.default_rng(random_state)
        return self.to_physical(
            np.asarray(rng.standard_normal((int(n), self.d)), dtype=np.float64)
        )

    # -------------------------------------------------------------------------
    # Internals
    # -------------------------------------------------------------------------
    @staticmethod
    def _as_2d(values: NDArrayF) -> NDArrayF:
        array = np.asarray(values, dtype=np.float64)
        if array.ndim == 1:
            return array.reshape(1, -1)
        if array.ndim != 2:
            raise ValueError("Input must be a 1-D or 2-D array.")
        return array

    def _validated_correlation(self, correlation: NDArrayF) -> NDArrayF:
        matrix = np.asarray(correlation, dtype=np.float64)
        if matrix.shape != (self.d, self.d):
            raise ValueError(
                f"Correlation must be {self.d}x{self.d}, got {matrix.shape}."
            )
        if not np.allclose(matrix, matrix.T):
            raise ValueError("Correlation must be symmetric.")
        if not np.allclose(np.diag(matrix), 1.0):
            raise ValueError("Correlation must have a unit diagonal.")
        if float(np.min(np.linalg.eigvalsh(matrix))) <= 0.0:
            raise ValueError("Correlation must be positive definite.")
        return matrix

    def _solve_gaussian_correlation(self, quadrature_points: int) -> NDArrayF:
        """Find the Gaussian correlations that yield the physical ones.

        One bracketed root find per off-diagonal pair. The integrand is
        evaluated on a tensor Gauss-Hermite grid, which is cheap because the
        standardised marginal values at the nodes do not depend on ``rho_z`` and
        so are computed once per variable.
        """
        nodes, weights = np.polynomial.hermite_e.hermegauss(quadrature_points)
        weights = weights / np.sqrt(2.0 * np.pi)

        # Standardised physical value at each node, per variable.
        standardised = np.empty((self.d, quadrature_points), dtype=np.float64)
        for i, marginal in enumerate(self.marginals):
            probabilities = np.clip(
                stats.norm.cdf(nodes), _PROBABILITY_MARGIN, 1.0 - _PROBABILITY_MARGIN
            )
            values = np.asarray(marginal.ppf(probabilities), dtype=np.float64)
            mean = float(np.asarray(marginal.mean()))
            std = float(np.asarray(marginal.std()))
            if not np.isfinite(mean) or not np.isfinite(std) or std <= 0.0:
                raise ValueError(
                    f"Marginal {i} needs a finite mean and a positive finite "
                    "standard deviation for the Nataf correlation solve."
                )
            standardised[i] = (values - mean) / std

        gaussian = np.eye(self.d, dtype=np.float64)

        for i in range(self.d):
            for j in range(i + 1, self.d):
                target = float(self.correlation[i, j])
                if target == 0.0:
                    continue

                def mismatch(
                    rho: float, i: int = i, j: int = j, t: float = target
                ) -> float:
                    return self._pair_correlation(rho, standardised, weights, i, j) - t

                # The map from rho_z to rho_x is increasing and passes through
                # the target, so a bracket at the extremes suffices.
                low, high = -0.999, 0.999
                if mismatch(low) > 0.0 or mismatch(high) < 0.0:
                    raise ValueError(
                        f"A physical correlation of {target} is not attainable "
                        f"for marginals {i} and {j}. Non-normal marginals bound "
                        "the correlation they can express."
                    )
                rho_z = float(brentq(mismatch, low, high, xtol=1e-10))
                gaussian[i, j] = gaussian[j, i] = rho_z

        return gaussian

    @staticmethod
    def _pair_correlation(
        rho: float,
        standardised: NDArrayF,
        weights: NDArrayF,
        i: int,
        j: int,
    ) -> float:
        """Physical correlation produced by a Gaussian correlation of ``rho``.

        The two standard normals are written as ``z_i`` and
        ``rho z_i + sqrt(1 - rho^2) z_j`` with ``z_i, z_j`` independent, so the
        double integral becomes a weighted sum over the same node grid. The
        second variable's standardised values have to be interpolated, since the
        combination lands between nodes.
        """
        nodes, _ = np.polynomial.hermite_e.hermegauss(standardised.shape[1])
        combined = rho * nodes[:, None] + np.sqrt(1.0 - rho**2) * nodes[None, :]
        second = np.interp(combined, nodes, standardised[j])
        outer_weights = weights[:, None] * weights[None, :]
        return float(np.sum(outer_weights * standardised[i][:, None] * second))
