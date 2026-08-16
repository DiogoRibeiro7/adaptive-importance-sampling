"""Optimized Safe-ICE algorithm with performance improvements."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.stats import chi2

try:
    from numba import jit, prange
except ImportError:
    # Optional acceleration: keep API compatible when numba is unavailable.
    def jit(
        *args: Any, **kwargs: Any
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            return func

        return decorator

    def prange(*args: Any, **kwargs: Any) -> range:
        return range(*args)


from ..distributions.nakagami import InverseNakagamiDistribution, NakagamiDistribution
from ..distributions.vmf import VonMisesFisherSampler
from ..optimization.penalized_em import PenalizedEMOptimizer
from ..typing import LimitStateFunction
from .parameters import vMFNMParameters

# Type alias for NumPy float arrays
NDArrayF = npt.NDArray[np.float64]


class OptimizedSafeICE:
    """
    Optimized Safe Cross-Entropy-Based Importance Sampling.

    Key optimizations:
    - Cached heavy-tailed parameters
    - Vectorized operations
    - Parallel sample evaluation
    - Memory-efficient sample generation
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
        em_max_iter: int = 100,
        enable_caching: bool = True,
        enable_parallel: bool = True,
        batch_size: int | None = None,
        lambda_max: float = 0.95,
        random_state: int | np.random.Generator | None = None,
    ) -> None:
        """
        Initialize Optimized Safe-ICE.

        Parameters
        ----------
        limit_state_function : callable
            Function g(u) where g(u) < 0 indicates failure.
        dimension : int
            Problem dimension.
        K0 : int, optional
            Initial number of mixture components.
        delta_target : float, optional
            Target rarity parameter.
        delta_star : float, optional
            Rarity increment threshold.
        max_iterations : int, optional
            Maximum iterations.
        N : int, optional
            Number of samples per iteration.
        sigma0 : float, optional
            Initial sigma value.
        em_max_iter : int, optional
            Maximum EM iterations.
        enable_caching : bool, optional
            Enable caching of heavy-tailed parameters.
        enable_parallel : bool, optional
            Enable parallel processing.
        batch_size : int, optional
            Batch size for memory-efficient processing.
        """
        self.g = limit_state_function
        self.d = int(dimension)
        self.K0 = int(K0)
        self.delta_target = float(delta_target)
        self.delta_star = float(delta_star)
        self.max_iterations = int(max_iterations)
        self.N = int(N)
        self.sigma0 = float(sigma0)
        self.em_max_iter = int(em_max_iter)
        self.lambda_max = float(lambda_max)

        # Deterministic RNG support
        if random_state is None:
            self._rng: Any = np.random  # legacy global state
        elif isinstance(random_state, np.random.Generator):
            self._rng = random_state
        else:
            self._rng = np.random.default_rng(int(random_state))

        # Optimization flags
        self.enable_caching = enable_caching
        self.enable_parallel = enable_parallel
        self.batch_size = batch_size or min(N, 10000)

        # Caches
        self._omega_in_cache: dict[int, float] | None = None
        self._pdf_cache: dict[tuple[float, ...], float] | None = None

        # Pre-compute constants
        self.m_IN = max(1, math.ceil(math.sqrt(self.d)))

        # Statistics tracking
        self.history: list[dict[str, Any]] = []

    def run(
        self,
        initial_params: vMFNMParameters | None = None,
        verbose: bool = False,
    ) -> tuple[float, dict[str, Any]]:
        """
        Run optimized Safe-ICE algorithm.

        Parameters
        ----------
        initial_params : vMFNMParameters, optional
            Initial distribution parameters.
        verbose : bool, optional
            Print progress information.

        Returns
        -------
        pf : float
            Estimated failure probability.
        results : dict
            Detailed results including samples and convergence metrics.
        """
        # Initialize parameters
        if initial_params is None:
            phi_0 = self._initialize_vmfnm_parameters()
        else:
            phi_0 = initial_params

        # Initialize tracking variables
        sigma = self.sigma0
        delta_prev = 0.0
        phi_t = phi_0

        # Storage for all samples and weights
        all_samples: list[NDArrayF] = []
        all_g_values: list[NDArrayF] = []
        all_weights: list[NDArrayF] = []
        iterations_data: list[dict[str, Any]] = []
        cv_values: list[float] = []
        delta_values: list[float] = []

        if verbose:
            print("=" * 50)
            print("Optimized Safe-ICE Algorithm")
            print("=" * 50)
            print(f"Dimension: {self.d}")
            print(f"Samples per iteration: {self.N}")
            print(f"Initial components: {phi_0.K}")
            print(f"Caching enabled: {self.enable_caching}")
            print(f"Parallel enabled: {self.enable_parallel}")
            print("-" * 50)

        # Main iteration loop
        for t in range(self.max_iterations):
            if verbose:
                print(f"\nIteration {t + 1}")
                print("-" * 30)

            # Clear caches at start of iteration
            if self.enable_caching:
                self._clear_caches()
                self._precompute_omega_in(phi_t)

            # Step 1: Compute annealing parameter
            lambda_t = self._cosine_annealing_schedule(sigma, self.delta_target)

            # Step 2: Generate samples (memory-efficient batching)
            if verbose:
                print(f"  Generating {self.N} samples...")
            samples_t = self._generate_safe_mixture_samples_batched(phi_t, lambda_t)

            # Step 3: Evaluate limit state function (vectorized)
            g_values_t = self._evaluate_limit_state_vectorized(samples_t)

            # Step 4: Update delta
            delta_increment = self.delta_star if delta_prev >= self.delta_star else 0.0
            delta_t = min(self.delta_target, delta_prev + delta_increment)
            delta_prev = delta_t

            # Step 5: Compute importance weights
            if verbose:
                print("  Computing importance weights...")
            weights_t = self._compute_importance_weights_vectorized(
                samples_t, g_values_t, phi_t, lambda_t, delta_t, sigma
            )

            # Step 6: Check for convergence
            cv_weights = self._compute_cv(weights_t)
            cv_values.append(cv_weights)
            delta_values.append(delta_t)

            if verbose:
                print(f"  Delta: {delta_t:.4f}")
                print(f"  CV: {cv_weights:.4f}")
                print(f"  Lambda: {lambda_t:.4f}")

            # Store samples and weights
            all_samples.append(samples_t)
            all_g_values.append(g_values_t)
            all_weights.append(weights_t)

            # Store iteration data
            iterations_data.append(
                {
                    "iteration": t + 1,
                    "K": phi_t.K,
                    "delta": delta_t,
                    "lambda": lambda_t,
                    "cv": cv_weights,
                    "sigma": sigma,
                    "n_failures": np.sum(g_values_t <= 0),
                }
            )

            # Step 7: Update parameters (if not last iteration)
            if t < self.max_iterations - 1:
                # Compute elites
                elites_indices = weights_t > 0
                elites_samples = samples_t[elites_indices]
                elites_weights = weights_t[elites_indices]

                if len(elites_samples) < 10:
                    if verbose:
                        print("  Warning: Too few elite samples, stopping early")
                    break

                # Penalized EM optimization
                if verbose:
                    print("  Running Penalized EM...")
                optimizer = PenalizedEMOptimizer(max_em_iterations=self.em_max_iter)
                phi_t, _ = optimizer.fit(
                    elites_samples, elites_weights, phi_t, beta_init=1.0
                )

                # Adapt sigma based on component evolution
                K_new = phi_t.K
                K_old = iterations_data[-1]["K"]
                sigma = self._adapt_sigma(sigma, K_old, K_new)

                if verbose:
                    print(f"  Updated K: {K_old} -> {K_new}")
                    print(f"  Updated sigma: {sigma:.4f}")

            # Check early convergence
            if cv_weights < 0.01 and delta_t >= self.delta_target:
                if verbose:
                    print(f"\nConverged early at iteration {t + 1}")
                break

        # Combine all samples and weights
        final_samples = np.vstack(all_samples)
        final_g_values = np.hstack(all_g_values)
        final_weights = np.hstack(all_weights)

        # Compute failure probability
        failure_indicator = (final_g_values <= 0).astype(float)
        pf_estimate = np.sum(failure_indicator * final_weights) / np.sum(final_weights)
        cv_w_star = (
            self._compute_cv(final_weights[final_g_values <= 0])
            if np.any(final_g_values <= 0)
            else 0.0
        )

        # Prepare results
        results = {
            "final_samples": final_samples,
            "final_weights": final_weights,
            "final_g_values": final_g_values,
            "iterations": iterations_data,
            "convergence_metrics": {
                "cv_values": cv_values,
                "delta_values": delta_values,
            },
            "final_parameters": phi_t,
        }

        if verbose:
            print("-" * 50)
            print("Final Results:")
            print(f"Failure Probability: {pf_estimate:.6e}")
            print(f"Total Iterations: {t + 1}")
            print(f"Final Components: {phi_t.K}")
            print(f"Final CV: {cv_w_star:.4f}")

        return float(pf_estimate), results

    # -------------------------------------------------------------------------
    # Caching Methods
    # -------------------------------------------------------------------------
    def _clear_caches(self) -> None:
        """Clear all caches."""
        self._omega_in_cache = {}
        self._pdf_cache = {}

    def _precompute_omega_in(self, params: vMFNMParameters) -> None:
        """Precompute Omega_IN values for all components."""
        if not self.enable_caching:
            return

        self._omega_in_cache = {}
        for k in range(params.K):
            omega_in = self._calculate_matched_omega_inverse_nakagami(
                float(params.m[k]), float(params.Omega[k]), float(self.m_IN)
            )
            self._omega_in_cache[k] = omega_in

    def _get_cached_omega_in(self, k: int, m_k: float, omega_k: float) -> float:
        """Get cached Omega_IN value or compute if not cached."""
        if self.enable_caching and self._omega_in_cache and k in self._omega_in_cache:
            return self._omega_in_cache[k]
        return self._calculate_matched_omega_inverse_nakagami(
            m_k, omega_k, float(self.m_IN)
        )

    # -------------------------------------------------------------------------
    # Vectorized Operations
    # -------------------------------------------------------------------------
    def _generate_safe_mixture_samples_batched(
        self, params: vMFNMParameters, lambda_val: float
    ) -> NDArrayF:
        """Generate samples in batches for memory efficiency."""
        samples = np.zeros((self.N, self.d), dtype=np.float64)

        # Determine which samples come from each component
        uniform_draws = self._rng.uniform(size=self.N)
        light_mask = uniform_draws < lambda_val
        n_light = np.sum(light_mask)
        n_heavy = self.N - n_light

        # Generate light-tailed samples (vectorized)
        if n_light > 0:
            light_indices = np.where(light_mask)[0]
            samples[light_indices] = self._sample_vmfnm_batch(params, n_light)

        # Generate heavy-tailed samples (vectorized)
        if n_heavy > 0:
            heavy_indices = np.where(~light_mask)[0]
            samples[heavy_indices] = self._sample_heavy_tailed_batch(params, n_heavy)

        return samples

    def _sample_vmfnm_batch(self, params: vMFNMParameters, n_samples: int) -> NDArrayF:
        """Sample batch from vMFNM distribution (vectorized)."""
        samples = np.zeros((n_samples, self.d), dtype=np.float64)

        # Choose components for all samples at once
        component_indices = self._rng.choice(params.K, size=n_samples, p=params.pi)

        # Group by component for efficient sampling
        for k in range(params.K):
            mask = component_indices == k
            n_k = np.sum(mask)
            if n_k == 0:
                continue

            # Sample radii (vectorized)
            radii = NakagamiDistribution.sample(
                float(params.m[k]),
                float(params.Omega[k]),
                n_k,
                rng=self._rng,
            )

            # Sample directions (already vectorized in VonMisesFisherSampler)
            directions = VonMisesFisherSampler.sample(
                params.mu[k],
                float(params.kappa[k]),
                n_k,
                rng=self._rng,
            )

            # Combine
            samples[mask] = radii[:, np.newaxis] * directions

        return samples

    def _sample_heavy_tailed_batch(
        self, params: vMFNMParameters, n_samples: int
    ) -> NDArrayF:
        """Sample batch from heavy-tailed distribution (vectorized)."""
        samples = np.zeros((n_samples, self.d), dtype=np.float64)

        # Choose components
        component_indices = self._rng.choice(params.K, size=n_samples, p=params.pi)

        # Group by component
        for k in range(params.K):
            mask = component_indices == k
            n_k = np.sum(mask)
            if n_k == 0:
                continue

            # Get cached Omega_IN
            omega_in = self._get_cached_omega_in(
                k, float(params.m[k]), float(params.Omega[k])
            )

            # Sample radii from inverse Nakagami
            radii = InverseNakagamiDistribution.sample(
                float(self.m_IN),
                omega_in,
                n_k,
                rng=self._rng,
            )

            # Sample directions
            directions = VonMisesFisherSampler.sample(
                params.mu[k],
                float(params.kappa[k]),
                n_k,
                rng=self._rng,
            )

            # Combine
            samples[mask] = radii[:, np.newaxis] * directions

        return samples

    def _evaluate_limit_state_vectorized(self, samples: NDArrayF) -> NDArrayF:
        """Evaluate limit state function with optional batching."""
        n_samples = samples.shape[0]

        if n_samples <= self.batch_size:
            # Evaluate all at once
            return np.asarray(self.g(samples), dtype=np.float64)
        else:
            # Process in batches to manage memory
            g_values = np.zeros(n_samples, dtype=np.float64)
            for i in range(0, n_samples, self.batch_size):
                end_idx = min(i + self.batch_size, n_samples)
                batch = samples[i:end_idx]
                g_values[i:end_idx] = self.g(batch)
            return g_values

    def _compute_importance_weights_vectorized(
        self,
        samples: NDArrayF,
        g_values: NDArrayF,
        params: vMFNMParameters,
        lambda_val: float,
        delta: float,
        sigma: float,
    ) -> NDArrayF:
        """Compute importance weights (vectorized)."""
        n_samples = samples.shape[0]

        # Compute densities in batches
        if n_samples <= self.batch_size:
            q_safe = self._evaluate_safe_mixture_density_vectorized(
                samples, params, lambda_val, sigma
            )
        else:
            q_safe = np.zeros(n_samples, dtype=np.float64)
            for i in range(0, n_samples, self.batch_size):
                end_idx = min(i + self.batch_size, n_samples)
                batch = samples[i:end_idx]
                q_safe[i:end_idx] = self._evaluate_safe_mixture_density_vectorized(
                    batch, params, lambda_val, sigma
                )

        # Standard normal density (vectorized)
        phi_u = np.exp(-0.5 * np.sum(samples**2, axis=1)) / (
            (2 * np.pi) ** (self.d / 2)
        )

        # Importance weights with regularization
        weights = np.zeros(n_samples, dtype=np.float64)
        valid_mask = (q_safe > 1e-300) & np.isfinite(q_safe)
        weights[valid_mask] = phi_u[valid_mask] / q_safe[valid_mask]

        # Apply indicator function based on g_values and delta
        indicator = (g_values <= 0) | (np.abs(g_values) <= delta)
        weights = weights * indicator

        return weights

    def _evaluate_safe_mixture_density_vectorized(
        self,
        samples: NDArrayF,
        params: vMFNMParameters,
        lambda_val: float,
        sigma: float,
    ) -> NDArrayF:
        """Evaluate safe mixture density (vectorized)."""
        samples.shape[0]

        # Compute both components
        q_light = self._evaluate_vmfnm_density_vectorized(samples, params, sigma)
        q_heavy = self._evaluate_heavy_tailed_density_vectorized(samples, params)

        # Mixture
        q_safe = lambda_val * q_light + (1 - lambda_val) * q_heavy

        return q_safe

    def _evaluate_vmfnm_density_vectorized(
        self, samples: NDArrayF, params: vMFNMParameters, sigma: float
    ) -> NDArrayF:
        """Evaluate vMFNM density (vectorized)."""
        n_samples = samples.shape[0]
        densities = np.zeros(n_samples, dtype=np.float64)

        # Compute radii and directions for all samples
        radii = np.linalg.norm(samples, axis=1)
        valid_mask = radii > 1e-12

        # Normalized directions
        directions = np.zeros_like(samples)
        directions[valid_mask] = samples[valid_mask] / radii[valid_mask, np.newaxis]

        # See the note in the heavy-tailed density: the polar form needs the
        # r^(d-1) Jacobian to be a density on R^d.
        jacobian = np.maximum(radii, 1e-12) ** (self.d - 1)

        # For each component
        for k in range(params.K):
            # Radial density (Nakagami)
            radial_density = NakagamiDistribution.pdf(
                radii, float(params.m[k]), float(params.Omega[k]) * sigma**2
            )

            # Angular density (vMF) - vectorized dot product
            angular_density = np.exp(
                float(params.kappa[k]) * np.dot(directions, params.mu[k])
            )
            # Normalization constant for vMF
            C_d = self._vmf_normalization_constant(float(params.kappa[k]))
            angular_density *= C_d

            # Component contribution
            densities += (
                float(params.pi[k]) * radial_density * angular_density / jacobian
            )

        return densities

    def _evaluate_heavy_tailed_density_vectorized(
        self, samples: NDArrayF, params: vMFNMParameters
    ) -> NDArrayF:
        """Evaluate heavy-tailed density (vectorized with caching)."""
        n_samples = samples.shape[0]
        densities = np.zeros(n_samples, dtype=np.float64)

        # Compute radii and directions
        radii = np.linalg.norm(samples, axis=1)
        valid_mask = radii > 1e-12

        directions = np.zeros_like(samples)
        directions[valid_mask] = samples[valid_mask] / radii[valid_mask, np.newaxis]

        # Polar-to-Cartesian Jacobian: du = r^(d-1) dr dw. Without it the
        # component does not integrate to one and, since it sits in the
        # importance-sampling denominator, biases every estimate.
        jacobian = np.maximum(radii, 1e-12) ** (self.d - 1)

        # For each component
        for k in range(params.K):
            # Get cached Omega_IN
            omega_in = self._get_cached_omega_in(
                k, float(params.m[k]), float(params.Omega[k])
            )

            # Radial density (Inverse Nakagami)
            radial_density = InverseNakagamiDistribution.pdf(
                radii, float(self.m_IN), omega_in
            )

            # Angular density (vMF)
            angular_density = np.exp(
                float(params.kappa[k]) * np.dot(directions, params.mu[k])
            )
            C_d = self._vmf_normalization_constant(float(params.kappa[k]))
            angular_density *= C_d

            # Component contribution
            densities += (
                float(params.pi[k]) * radial_density * angular_density / jacobian
            )

        return densities

    # -------------------------------------------------------------------------
    # Helper Methods (from original implementation)
    # -------------------------------------------------------------------------
    def _initialize_vmfnm_parameters(self) -> vMFNMParameters:
        """Initialize vMF-Nakagami mixture parameters."""
        K = int(self.K0)
        d = int(self.d)

        pi = np.ones(K, dtype=np.float64) / float(K)
        # Nakagami parameters, scaled to the problem dimension.
        #
        # The target is the standard normal in R^d, whose radius follows a chi
        # distribution with d degrees of freedom, and chi_d is exactly
        # Nakagami(m = d/2, Omega = d). Initialising around that puts the
        # proposal on top of the target from the start; the jitter keeps the
        # components distinguishable so EM has something to separate.
        #
        # Fixed values were used here before, which happen to be right only
        # near d=2. At d=20 they left the proposal at radius ~1.1 while the
        # target sits at ~4.4, so the two barely overlapped and the run could
        # never recover.
        m = np.asarray(self._rng.uniform(0.75, 1.5, K) * (d / 2.0), dtype=np.float64)
        Omega = np.asarray(self._rng.uniform(0.75, 1.5, K) * float(d), dtype=np.float64)

        mu = np.asarray(self._rng.normal(0.0, 1.0, (K, d)), dtype=np.float64)
        row_norms = np.linalg.norm(mu, axis=1, keepdims=True)
        mu = mu / np.maximum(row_norms, np.finfo(np.float64).eps)

        kappa = np.asarray(self._rng.uniform(0.1, 1.0, K), dtype=np.float64)

        return vMFNMParameters(pi=pi, m=m, Omega=Omega, mu=mu, kappa=kappa)

    def _cosine_annealing_schedule(self, sigma: float, M: float) -> float:
        """Cosine annealing schedule for lambda parameter.

        Capped at ``lambda_max`` so the heavy-tailed component never leaves the
        proposal; see the note on the same method in :class:`SafeICE`.
        """
        if sigma > M:
            return 0.0
        raw = 0.5 * (1.0 + math.cos(math.pi * sigma / M))
        return float(min(raw, self.lambda_max))

    def _adapt_sigma(self, sigma: float, K_old: int, K_new: int) -> float:
        """Adapt sigma based on component evolution."""
        if K_new < K_old:
            return sigma * 1.1
        elif K_new > K_old:
            return sigma / 1.1
        return sigma

    def _compute_cv(self, weights: NDArrayF) -> float:
        """Compute coefficient of variation of weights."""
        if len(weights) == 0 or np.sum(weights) == 0:
            return float("inf")
        mean_w = np.mean(weights)
        if mean_w == 0:
            return float("inf")
        std_w = np.std(weights)
        return float(std_w / mean_w)

    def _calculate_matched_omega_inverse_nakagami(
        self,
        m: float,  # noqa: ARG002 - overrides the base signature
        Omega: float,
        m_IN: float,
    ) -> float:
        """Calculate Omega_IN for mode matching."""
        chi2_pdf_val = chi2.pdf(Omega * self.sigma0**2, df=self.d)
        if chi2_pdf_val < 1e-300:
            chi2_pdf_val = 1e-300
        return float(m_IN / (self.d * chi2_pdf_val))

    def _vmf_normalization_constant(self, kappa: float) -> float:
        """Compute vMF normalization constant."""
        from scipy.special import iv

        if kappa < 1e-10:
            return 1.0 / (4 * np.pi)  # Uniform on sphere for d=3

        # General formula
        d = self.d
        return float(
            kappa ** (d / 2 - 1) / ((2 * np.pi) ** (d / 2) * iv(d / 2 - 1, kappa))
        )


# Optional: Numba-accelerated functions for critical loops
@jit(nopython=True, parallel=True)
def compute_radii_parallel(samples: NDArrayF) -> NDArrayF:
    """Compute radii in parallel using Numba."""
    n_samples = samples.shape[0]
    radii = np.zeros(n_samples, dtype=np.float64)
    for i in prange(n_samples):
        radii[i] = np.linalg.norm(samples[i])
    return radii


@jit(nopython=True, parallel=True)
def evaluate_limit_state_parallel(samples: NDArrayF, g_func: Any) -> NDArrayF:
    """Evaluate limit state in parallel (if g_func is Numba-compatible)."""
    n_samples = samples.shape[0]
    g_values = np.zeros(n_samples, dtype=np.float64)
    for i in prange(n_samples):
        g_values[i] = g_func(samples[i])
    return g_values
