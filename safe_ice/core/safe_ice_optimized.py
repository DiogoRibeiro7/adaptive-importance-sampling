"""Safe-ICE with vectorised sample generation.

:class:`~safe_ice.core.safe_ice.SafeICE` draws its proposal one sample at a
time. This subclass overrides only that step: component assignments are drawn
in bulk and each component's samples are generated in a single call. Everything
that determines the answer -- the sigma schedule, the penalised EM update, the
weights and the final estimator -- is inherited, so the two classes agree up to
Monte Carlo error.

This module previously reimplemented the whole algorithm rather than
subclassing it, and the copy did not work. Its sigma rule moved sigma by +/-10%
according to whether the number of mixture components had changed, instead of
targeting the weight coefficient of variation, so sigma rose from 1.00 to 1.21
over a run rather than falling. The proposal never concentrated on the failure
region. Its adaptation weights used a hard indicator, which is zero for every
sample until one lands in the failure region -- on a rare-event problem, none
do -- so the elite set was empty and the loop stopped after one iteration. Its
final estimate then divided by the sum of those weights, which was zero.
``run()`` returned NaN for every seed on every benchmark.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from ..distributions.nakagami import InverseNakagamiDistribution, NakagamiDistribution
from ..distributions.vmf import VonMisesFisherSampler
from ..typing import LimitStateFunction
from .parameters import vMFNMParameters
from .safe_ice import SafeICE

# Type alias for NumPy float arrays
NDArrayF = npt.NDArray[np.float64]


class OptimizedSafeICE(SafeICE):
    """Safe-ICE with batched proposal sampling.

    Parameters
    ----------
    limit_state_function:
        Function g(u) such that failure occurs when g(u) <= 0.
    dimension:
        Problem dimension.
    enable_caching:
        Cache the matched inverse-Nakagami scale per component within an
        iteration. It depends only on the component's parameters, so it is
        recomputed once per component rather than once per sample.
    enable_parallel:
        Accepted for backwards compatibility and currently has no effect.
        Sampling is vectorised unconditionally.
    batch_size:
        Upper bound on the number of samples passed to the limit-state
        function at once. Defaults to ``min(N, 10000)``.

    All remaining parameters are those of :class:`SafeICE` and are forwarded
    unchanged.

    Notes
    -----
    With the default ``batch_size`` this class produces the same estimates as
    :class:`SafeICE` up to the random stream; it is not a different algorithm.
    """

    def __init__(
        self,
        limit_state_function: LimitStateFunction,
        dimension: int,
        K0: int = 20,
        delta_target: float = 4.0,
        delta_star: float = 1.5,
        max_iterations: int = 20,
        N: int = 1000,
        sigma0: float = 1.0,
        em_max_iter: int = 20,
        enable_caching: bool = True,
        enable_parallel: bool = True,
        batch_size: int | None = None,
        lambda_max: float = 0.95,
        random_state: int | np.random.Generator | None = None,
    ) -> None:
        super().__init__(
            limit_state_function=limit_state_function,
            dimension=dimension,
            K0=K0,
            delta_target=delta_target,
            delta_star=delta_star,
            max_iterations=max_iterations,
            N=N,
            sigma0=sigma0,
            em_max_iter=em_max_iter,
            lambda_max=lambda_max,
            random_state=random_state,
        )

        self.enable_caching = bool(enable_caching)
        self.enable_parallel = bool(enable_parallel)
        self.batch_size = int(batch_size) if batch_size else min(int(N), 10000)

        # Heavy-tailed shape, as in SafeICE._sample_heavy_tailed_component.
        self.m_IN = max(1, math.ceil(math.sqrt(self.d)))

        self._omega_in_cache: dict[int, float] = {}

    # -------------------------------------------------------------------------
    # Sampling (the actual optimisation)
    # -------------------------------------------------------------------------
    def _generate_safe_mixture_samples(
        self, params: vMFNMParameters, lambda_val: float
    ) -> NDArrayF:
        """Draw N samples from q_safe in two batches instead of N draws."""
        self._omega_in_cache = {}

        samples: NDArrayF = np.zeros((self.N, self.d), dtype=np.float64)

        uniform_draws = np.asarray(self._rng.uniform(size=self.N), dtype=np.float64)
        light_mask = uniform_draws < float(lambda_val)
        n_light = int(np.sum(light_mask))
        n_heavy = int(self.N - n_light)

        if n_light > 0:
            samples[light_mask] = self._sample_vmfnm_batch(params, n_light)
        if n_heavy > 0:
            samples[~light_mask] = self._sample_heavy_tailed_batch(params, n_heavy)

        return samples

    def _sample_vmfnm_batch(self, params: vMFNMParameters, n_samples: int) -> NDArrayF:
        """Sample the light-tailed vMF-Nakagami mixture, grouped by component."""
        samples: NDArrayF = np.zeros((n_samples, self.d), dtype=np.float64)
        component_indices = self._rng.choice(params.K, size=n_samples, p=params.pi)

        for k in range(params.K):
            mask = component_indices == k
            n_k = int(np.sum(mask))
            if n_k == 0:
                continue

            radii = NakagamiDistribution.sample(
                float(params.m[k]), float(params.Omega[k]), n_k, rng=self._rng
            )
            directions = VonMisesFisherSampler.sample(
                params.mu[k], float(params.kappa[k]), n_k, rng=self._rng
            )
            samples[mask] = np.asarray(radii, dtype=np.float64)[:, np.newaxis] * (
                np.asarray(directions, dtype=np.float64)
            )

        return samples

    def _sample_heavy_tailed_batch(
        self, params: vMFNMParameters, n_samples: int
    ) -> NDArrayF:
        """Sample the heavy-tailed safety component, grouped by component."""
        samples: NDArrayF = np.zeros((n_samples, self.d), dtype=np.float64)
        component_indices = self._rng.choice(params.K, size=n_samples, p=params.pi)

        for k in range(params.K):
            mask = component_indices == k
            n_k = int(np.sum(mask))
            if n_k == 0:
                continue

            omega_in = self._get_cached_omega_in(
                k, float(params.m[k]), float(params.Omega[k])
            )
            radii = InverseNakagamiDistribution.sample(
                float(self.m_IN), omega_in, n_k, rng=self._rng
            )
            directions = VonMisesFisherSampler.sample(
                params.mu[k], float(params.kappa[k]), n_k, rng=self._rng
            )
            samples[mask] = np.asarray(radii, dtype=np.float64)[:, np.newaxis] * (
                np.asarray(directions, dtype=np.float64)
            )

        return samples

    def _get_cached_omega_in(self, k: int, m_N: float, Omega_N: float) -> float:
        """Matched inverse-Nakagami scale for component ``k``."""
        if not self.enable_caching:
            return self._calculate_matched_omega_inverse_nakagami(
                m_N, Omega_N, float(self.m_IN)
            )

        cached = self._omega_in_cache.get(k)
        if cached is None:
            cached = self._calculate_matched_omega_inverse_nakagami(
                m_N, Omega_N, float(self.m_IN)
            )
            self._omega_in_cache[k] = cached
        return cached

    def _clear_caches(self) -> None:
        """Drop cached per-component values."""
        self._omega_in_cache = {}

    # -------------------------------------------------------------------------
    # Batched limit-state evaluation
    # -------------------------------------------------------------------------
    def _evaluate_limit_state(self, samples: NDArrayF) -> NDArrayF:
        """Evaluate g in chunks of at most ``batch_size`` rows."""
        n_samples = samples.shape[0]
        if n_samples <= self.batch_size:
            return super()._evaluate_limit_state(samples)

        evaluate = super()._evaluate_limit_state
        g_values: NDArrayF = np.zeros(n_samples, dtype=np.float64)
        for start in range(0, n_samples, self.batch_size):
            end = min(start + self.batch_size, n_samples)
            g_values[start:end] = evaluate(samples[start:end])
        return g_values

    # -------------------------------------------------------------------------
    # Density evaluation, chunked to bound peak memory
    # -------------------------------------------------------------------------
    def _evaluate_safe_mixture_density(
        self, samples: NDArrayF, params: vMFNMParameters, lambda_val: float
    ) -> NDArrayF:
        """Evaluate q_safe in chunks of at most ``batch_size`` rows."""
        n_samples = samples.shape[0]
        if n_samples <= self.batch_size:
            return super()._evaluate_safe_mixture_density(samples, params, lambda_val)

        evaluate = super()._evaluate_safe_mixture_density
        densities: NDArrayF = np.zeros(n_samples, dtype=np.float64)
        for start in range(0, n_samples, self.batch_size):
            end = min(start + self.batch_size, n_samples)
            densities[start:end] = evaluate(samples[start:end], params, lambda_val)
        return densities
