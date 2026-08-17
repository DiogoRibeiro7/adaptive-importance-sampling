"""Safe-ICE with dimension-dependent defaults."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..typing import LimitStateFunction
from .safe_ice_optimized import OptimizedSafeICE


class AdaptiveSafeICE(OptimizedSafeICE):
    """Safe-ICE that picks its sample size and mixture size from the dimension.

    The defaults of :class:`SafeICE` are tuned for small problems. This
    subclass chooses ``N``, ``K0``, ``delta_target`` and ``delta_star`` from the
    dimension instead, which is the only thing it does: the algorithm itself is
    inherited unchanged from :class:`OptimizedSafeICE`, and any of the four can
    still be passed explicitly to override the choice.

    Parameters
    ----------
    limit_state_function:
        Function g(u) such that failure occurs when g(u) <= 0.
    dimension:
        Problem dimension.
    N:
        Samples per iteration. Chosen from the dimension when omitted.
    auto_tune, adaptive_schedule:
        Accepted for backwards compatibility and currently have no effect.
    **kwargs:
        Forwarded to :class:`OptimizedSafeICE`.

    Notes
    -----
    This class used to carry its own ``run``, annealing schedule, convergence
    test and parameter initialisation. That loop did not converge -- see the
    module docstring of :mod:`safe_ice.core.safe_ice_optimized` -- and its
    initialisation drew the radial parameters from fixed ranges (m in (2, 4),
    Omega in (1, 3)) regardless of dimension, which places the proposal at a
    radius of roughly 1.5 whether the problem is 2- or 200-dimensional. The
    inherited initialisation scales them as m = d/2 and Omega = d, matching the
    identity that the norm of a d-dimensional standard normal is
    Nakagami(d/2, d).
    """

    def __init__(
        self,
        limit_state_function: LimitStateFunction,
        dimension: int,
        N: int | None = None,
        max_iterations: int = 30,
        auto_tune: bool = True,
        adaptive_schedule: bool = True,
        random_state: int | np.random.Generator | None = None,
        **kwargs: Any,
    ) -> None:
        if N is None:
            N = self._compute_adaptive_sample_size(dimension)

        kwargs.setdefault("K0", self._compute_adaptive_k0(dimension))
        kwargs.setdefault(
            "delta_target", self._compute_adaptive_delta_target(dimension)
        )
        kwargs.setdefault("delta_star", self._compute_adaptive_delta_star(dimension))

        super().__init__(
            limit_state_function=limit_state_function,
            dimension=dimension,
            N=N,
            max_iterations=max_iterations,
            random_state=random_state,
            **kwargs,
        )

        self.auto_tune = bool(auto_tune)
        self.adaptive_schedule = bool(adaptive_schedule)

    # -------------------------------------------------------------------------
    # Dimension-dependent defaults
    # -------------------------------------------------------------------------
    @staticmethod
    def _compute_adaptive_sample_size(d: int) -> int:
        """Samples per iteration for dimension ``d``, capped at 50000."""
        if d <= 2:
            base_N = 500
        elif d <= 5:
            base_N = 1000
        elif d <= 10:
            base_N = 2000
        elif d <= 20:
            base_N = 3000
        elif d <= 50:
            base_N = 5000
        else:
            base_N = 10000

        return min(int(base_N * np.sqrt(d / 2)), 50000)

    @staticmethod
    def _compute_adaptive_k0(d: int) -> int:
        """Initial number of mixture components for dimension ``d``."""
        if d <= 2:
            return 10
        if d <= 5:
            return 15
        if d <= 10:
            return 20
        if d <= 20:
            return 30
        return min(50, 2 * d)

    @staticmethod
    def _compute_adaptive_delta_target(d: int) -> float:
        """Target coefficient of variation for dimension ``d``."""
        if d <= 2:
            return 3.0
        if d <= 5:
            return 3.5
        if d <= 10:
            return 4.0
        if d <= 20:
            return 4.5
        return float(min(6.0, 3.0 + np.log(d)))

    @staticmethod
    def _compute_adaptive_delta_star(d: int) -> float:
        """Convergence threshold for dimension ``d``."""
        if d <= 2:
            return 1.5
        if d <= 10:
            return 1.0
        return 0.75
