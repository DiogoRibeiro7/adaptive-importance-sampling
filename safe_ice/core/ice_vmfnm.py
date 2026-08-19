"""The ICE-vMFNM baseline that Safe-ICE is measured against.

Improved cross-entropy importance sampling with a von Mises-Fisher-Nakagami
mixture, from

    Papaioannou, Geyer and Straub, "Improved cross entropy-based importance
    sampling with a flexible mixture model", Reliability Engineering & System
    Safety 191:106564, 2019

which is reference [26] of the Safe-ICE paper. Safe-ICE is this method plus two
additions, and the paper's tables are comparisons between the two:

* a heavy-tailed component mixed into the proposal with weight ``1 - lambda``,
  annealed towards the light-tailed family as sigma falls;
* a cross-entropy penalty in the M-step that prunes redundant components, so
  the number of components need not be chosen in advance.

Removing both recovers ICE-vMFNM exactly, which is how it is implemented here:
``lambda`` is held at 1 so the proposal is the vMFNM mixture alone, and the
penalty coefficient is held at 0 so the M-step is the plain weighted EM update
of equation (19). Everything else -- the smoothed indicator, the schedule for
sigma, the weights and the final estimator -- is shared with
:class:`~safe_ice.core.safe_ice.SafeICE`.

Sharing it is deliberate. A comparison is only meaningful if the two methods
differ in the ways being compared and in no others, and reimplementing the
common parts would leave room for them to differ by accident.
"""

from __future__ import annotations

import numpy as np

from ..optimization.penalized_em import PenalizedEMOptimizer
from ..typing import LimitStateFunction
from .safe_ice import SafeICE


class ICEvMFNM(SafeICE):
    """Improved cross-entropy importance sampling with a vMFNM mixture.

    Parameters
    ----------
    limit_state_function:
        Function g(u) such that failure occurs when g(u) <= 0.
    dimension:
        Problem dimension.
    K:
        Number of mixture components. This method has no mechanism for adapting
        it, which is the shortcoming Safe-ICE's penalised EM addresses; the
        paper reports results for several values to show that the choice
        matters. Its own experiments use 2 and 4 for the four-mode problem, 3
        and 6 for the three-mode one, and 1 and 3 for the oscillator.

        Nothing here drives components out, so ``K`` normally holds for the
        whole run. It can still fall if plain EM leaves a component with no
        responsibility at all, since a component with zero weight cannot be
        updated and is dropped: asking for 8 components on the four-mode
        problem ends with 6. That is a different mechanism from Safe-ICE's,
        which prunes deliberately and from 20 down.
    delta_target, delta_star, max_iterations, N, sigma0, em_max_iter:
        As in :class:`SafeICE`.
    random_state:
        Seed or generator, for reproducibility.

    Notes
    -----
    ``lambda_max`` is not accepted: there is no heavy-tailed component for it to
    bound. Reading ``self.lambda_max`` still returns 1.0, since that is the
    weight this method places on the light-tailed family at every iteration.
    """

    def __init__(
        self,
        limit_state_function: LimitStateFunction,
        dimension: int,
        K: int = 2,
        delta_target: float = 4.0,
        delta_star: float = 1.5,
        max_iterations: int = 20,
        N: int = 1000,
        sigma0: float = 1.0,
        em_max_iter: int = 20,
        random_state: int | np.random.Generator | None = None,
    ) -> None:
        super().__init__(
            limit_state_function=limit_state_function,
            dimension=dimension,
            K0=K,
            delta_target=delta_target,
            delta_star=delta_star,
            max_iterations=max_iterations,
            N=N,
            sigma0=sigma0,
            em_max_iter=em_max_iter,
            lambda_max=1.0,
            random_state=random_state,
        )

        # The plain weighted EM update of equation (19). With no penalty
        # nothing drives components out deliberately, though EM can still lose
        # one that ends up with no responsibility at all.
        self.em_optimizer = PenalizedEMOptimizer(
            max_em_iterations=int(em_max_iter), penalized=False
        )

    @property
    def K(self) -> int:
        """The number of mixture components this method was asked for."""
        return int(self.K0)

    def _cosine_annealing_schedule(self, sigma: float, M: float) -> float:  # noqa: ARG002
        """Always 1: the proposal is the vMFNM mixture, with no safety component.

        Safe-ICE anneals this from 0 to just under 1 so that early iterations
        explore with heavy tails. ICE-vMFNM has no such component, so the
        weight on the light-tailed family is 1 throughout. ``sigma`` and ``M``
        are accepted to match the signature being overridden.
        """
        return 1.0
