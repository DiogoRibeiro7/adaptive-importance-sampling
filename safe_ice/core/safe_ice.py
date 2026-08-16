# safe_ice/core/safe_ice.py
"""Main Safe-ICE algorithm implementation.

This module implements the Safe Cross-Entropy Importance Sampling (Safe-ICE)
procedure end-to-end, with careful typing to satisfy mypy under strict NumPy
stubs. It keeps NumPy arrays as explicit float64 ndarrays (NDArrayF) and
converts NumPy scalars to Python floats at API boundaries to avoid Any leaks.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable
from typing import Any, TypedDict

import numpy as np
import numpy.typing as npt
import scipy.stats as stats
from scipy.optimize import minimize_scalar
from scipy.special import loggamma

from ..distributions._numeric import radii_and_directions, vmf_pdf_batch
from ..distributions.mixture import vMFNMDistribution
from ..distributions.nakagami import (
    InverseNakagamiDistribution,
    NakagamiDistribution,
)
from ..distributions.vmf import VonMisesFisherSampler
from ..optimization.penalized_em import PenalizedEMOptimizer
from .parameters import vMFNMParameters

# ----------------------------
# Typing aliases and structures
# ----------------------------
NDArrayF = npt.NDArray[np.float64]


class _SafeICEHistory(TypedDict):
    """History container for monitoring algorithm progress."""

    sigma: list[float]
    cv: list[float]
    components: list[int]
    lambda_val: list[float]
    pf_estimates: list[float]


class SafeICE:
    """Complete Safe Cross-Entropy Importance Sampling implementation.

    Parameters
    ----------
    limit_state_function:
        Function g(u) such that failure occurs when g(u) <= 0.
    dimension:
        Problem dimension d.
    K0:
        Initial number of mixture components in the vMF-Nakagami mixture.
    delta_target:
        Target coefficient of variation used when adapting sigma.
    delta_star:
        CV stopping threshold.
    max_iterations:
        Maximum number of outer ICE iterations.
    N:
        Number of samples per iteration.
    sigma0:
        Initial smoothing parameter for the smoothed indicator.
    em_max_iter:
        Maximum EM iterations per ICE step.
    """

    def __init__(
        self,
        limit_state_function: Callable[[NDArrayF], float | NDArrayF],
        dimension: int,
        K0: int = 20,
        delta_target: float = 4.0,
        delta_star: float = 1.5,
        max_iterations: int = 20,
        N: int = 1000,
        sigma0: float = 1.0,
        em_max_iter: int = 100,
        cv_tolerance: float = 0.01,
        lambda_max: float = 0.95,
        random_state: int | np.random.Generator | None = None,
    ) -> None:
        self.g = limit_state_function
        self.d = int(dimension)
        self.K0 = int(K0)
        self.delta_target = float(delta_target)
        self.delta_star = float(delta_star)
        self.max_iterations = int(max_iterations)
        self.N = int(N)
        self.sigma0 = float(sigma0)
        self.cv_tolerance = float(cv_tolerance)
        self.lambda_max = float(lambda_max)

        # Deterministic RNG support
        if random_state is None:
            self._rng: Any = np.random  # legacy global state
        elif isinstance(random_state, np.random.Generator):
            self._rng = random_state
        else:
            self._rng = np.random.default_rng(int(random_state))

        # Initialize EM optimizer
        self.em_optimizer = PenalizedEMOptimizer(max_em_iterations=int(em_max_iter))

        # History tracking (typed)
        self.history: _SafeICEHistory = {
            "sigma": [],
            "cv": [],
            "components": [],
            "lambda_val": [],
            "pf_estimates": [],
        }

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    def run(
        self,
        initial_params: vMFNMParameters | None = None,
        verbose: bool = True,
    ) -> tuple[float, dict[str, Any]]:
        """Execute the complete Safe-ICE algorithm.

        Returns
        -------
        Tuple[float, Dict[str, Any]]
            Estimated failure probability and a results dictionary containing
            diagnostics, final parameters, and traces.
        """
        if verbose:
            print("Safe-ICE Algorithm")
            print(f"Problem dimension: {self.d}")
            print(f"Initial components: {self.K0}")
            print(f"Samples per iteration: {self.N}")
            print("-" * 50)

        phi_t: vMFNMParameters = (
            initial_params
            if initial_params is not None
            else self._initialize_vmfnm_parameters()
        )
        sigma_t: float = float(self.sigma0)
        M: float = float(self.sigma0)  # cosine annealing scale
        lambda_t: float = self._cosine_annealing_schedule(sigma_t, M)
        iteration_records: list[dict[str, Any]] = []
        all_samples: list[NDArrayF] = []
        all_g_values: list[NDArrayF] = []

        for t in range(self.max_iterations):
            if verbose:
                print(
                    f"Iteration {t + 1:2d}: "
                    f"sigma={sigma_t:.6f}, lambda={lambda_t:.3f}, "
                    f"K={phi_t.K}"
                )

            # Generate samples from safe importance mixture
            samples: NDArrayF = self._generate_safe_mixture_samples(phi_t, lambda_t)
            g_values: NDArrayF = self._evaluate_limit_state(samples)

            # Stopping weights and CV
            stopping_weights: NDArrayF = self._calculate_stopping_weights(
                samples, g_values, sigma_t, phi_t, lambda_t
            )
            cv_w_star: float = self._coefficient_of_variation(stopping_weights)

            # Record history
            self.history["sigma"].append(float(sigma_t))
            self.history["cv"].append(float(cv_w_star))
            self.history["components"].append(int(phi_t.K))
            self.history["lambda_val"].append(float(lambda_t))
            iteration_records.append(
                {
                    "iteration": t + 1,
                    "K": int(phi_t.K),
                    "sigma": float(sigma_t),
                    "lambda": float(lambda_t),
                }
            )
            all_samples.append(samples)
            all_g_values.append(g_values)

            if verbose:
                print(f"           CV={cv_w_star:.4f}")

            # Convergence check using delta_star
            if cv_w_star <= self.delta_star and (t + 1) >= 2:
                if verbose:
                    print(
                        f"  Converged: CV {cv_w_star:.4f} "
                        f"<= delta_star {self.delta_star}"
                    )
                break

            # Adapt sigma
            sigma_t = self._determine_next_sigma(
                samples, g_values, phi_t, lambda_t, sigma_t
            )

            # Update mixture parameters via penalized EM
            phi_t = self._update_parameters_penalized_em(
                samples, g_values, phi_t, sigma_t, lambda_t
            )

            # Update lambda via cosine annealing
            lambda_t = self._cosine_annealing_schedule(sigma_t, M)

        # --- Final estimate via importance sampling (not crude MC) ---
        final_samples: NDArrayF = self._generate_safe_mixture_samples(phi_t, lambda_t)
        final_g_values: NDArrayF = self._evaluate_limit_state(final_samples)
        pf_estimate: float = self._estimate_failure_probability(
            final_samples, final_g_values, phi_t, lambda_t
        )

        self.history["pf_estimates"].append(float(pf_estimate))

        # Collect all iteration samples for diagnostics
        all_samples_arr: NDArrayF = np.vstack(all_samples)
        all_g_arr: NDArrayF = np.hstack(all_g_values)

        results: dict[str, Any] = {
            "failure_probability": float(pf_estimate),
            "iterations": iteration_records,
            "final_components": int(phi_t.K),
            "final_sigma": float(sigma_t),
            "final_cv": float(self.history["cv"][-1]),
            "final_lambda": float(lambda_t),
            "final_samples": all_samples_arr,
            "final_weights": np.ones(all_samples_arr.shape[0], dtype=np.float64),
            "final_g_values": all_g_arr,
            "history": self.history,
            "convergence_metrics": {
                "cv_values": list(self.history["cv"]),
                "delta_values": list(self.history["sigma"]),
                "sigma_values": list(self.history["sigma"]),
                "lambda_values": list(self.history["lambda_val"]),
                "pf_estimates": list(self.history["pf_estimates"]),
            },
            "final_parameters": phi_t,
        }

        if verbose:
            print("-" * 50)
            print("Final Results:")
            print(f"Failure Probability: {pf_estimate:.6e}")
            print(f"Total Iterations: {len(iteration_records)}")
            print(f"Final Components: {phi_t.K}")
            print(f"Final CV: {self.history['cv'][-1]:.4f}")

        return float(pf_estimate), results

    def _evaluate_limit_state(self, samples: NDArrayF) -> NDArrayF:
        """Evaluate limit state on a batch; fallback to row-wise when needed."""
        try:
            raw = self.g(samples)
            arr = np.asarray(raw, dtype=np.float64)
            if arr.ndim == 0:
                arr = np.full(samples.shape[0], float(arr), dtype=np.float64)
            elif arr.ndim > 1:
                arr = np.asarray(arr).reshape(-1)
            if arr.shape[0] != samples.shape[0]:
                raise ValueError("Limit-state output shape mismatch")
        except Exception:
            arr = np.asarray(
                [
                    float(np.asarray(self.g(s.reshape(1, -1))).reshape(-1)[0])
                    for s in samples
                ],
                dtype=np.float64,
            )

        if np.any(np.isnan(arr)):
            warnings.warn(
                "Limit state returned NaN values; converting to +inf.",
                RuntimeWarning,
                stacklevel=2,
            )
            arr = np.where(np.isnan(arr), np.inf, arr)
        arr = np.where(np.isposinf(arr), np.inf, arr)
        arr = np.where(np.isneginf(arr), -np.inf, arr)
        return arr.astype(np.float64, copy=False)

    # -------------------------------------------------------------------------
    # Initialization & Schedules
    # -------------------------------------------------------------------------
    def _initialize_vmfnm_parameters(self) -> vMFNMParameters:
        """Initialize vMF-Nakagami mixture parameters."""
        K = int(self.K0)
        d = int(self.d)

        # Equal mixture weights
        pi: NDArrayF = np.ones(K, dtype=np.float64) / float(K)

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
        m: NDArrayF = np.asarray(
            self._rng.uniform(0.75, 1.5, K) * (d / 2.0), dtype=np.float64
        )
        Omega: NDArrayF = np.asarray(
            self._rng.uniform(0.75, 1.5, K) * float(d), dtype=np.float64
        )

        # von Mises-Fisher parameters
        mu: NDArrayF = np.asarray(self._rng.normal(0.0, 1.0, (K, d)), dtype=np.float64)
        # Normalize each row to unit norm
        row_norms: NDArrayF = np.linalg.norm(mu, axis=1, keepdims=True).astype(
            np.float64, copy=False
        )
        eps: float = float(np.finfo(np.float64).tiny)
        mu = mu / np.maximum(row_norms, eps)

        # Small initial concentrations
        kappa: NDArrayF = np.asarray(self._rng.uniform(0.1, 1.0, K), dtype=np.float64)

        return vMFNMParameters(pi=pi, m=m, Omega=Omega, mu=mu, kappa=kappa)

    def _cosine_annealing_schedule(self, sigma: float, M: float) -> float:
        """Cosine annealing schedule for lambda parameter.

        The raw schedule reaches exactly 1 as sigma goes to 0, which drops the
        heavy-tailed component from the proposal altogether. That component is
        the "safe" part of Safe-ICE: it keeps the proposal's tails heavier than
        the target so the importance weights stay bounded. Without it a single
        sample from the tail can carry essentially the whole estimate. The
        result is capped at ``lambda_max`` so some heavy-tailed mass always
        remains.
        """
        if sigma > M:
            return 0.0
        # Return Python float to avoid numpy floating[Any]
        raw = 0.5 * (1.0 + math.cos(math.pi * sigma / M))
        return float(min(raw, self.lambda_max))

    # -------------------------------------------------------------------------
    # Sampling
    # -------------------------------------------------------------------------
    def _generate_safe_mixture_samples(
        self, params: vMFNMParameters, lambda_val: float
    ) -> NDArrayF:
        """Generate samples from the safe mixture q_safe(u; φ)."""
        samples: NDArrayF = np.zeros((self.N, self.d), dtype=np.float64)

        for i in range(self.N):
            if float(self._rng.uniform()) < float(lambda_val):
                # Sample from light-tailed vMFNM component
                samples[i] = self._sample_vmfnm_component(params)
            else:
                # Sample from heavy-tailed component
                samples[i] = self._sample_heavy_tailed_component(params)

        return samples

    def _sample_vmfnm_component(self, params: vMFNMParameters) -> NDArrayF:
        """Sample a single vector from the vMF-Nakagami mixture."""
        k = int(self._rng.choice(params.K, p=params.pi))

        r: float = float(
            np.asarray(
                NakagamiDistribution.sample(
                    float(params.m[k]),
                    float(params.Omega[k]),
                    1,
                    rng=self._rng,
                ),
                dtype=np.float64,
            )[0]
        )

        a: NDArrayF = np.asarray(
            VonMisesFisherSampler.sample(
                params.mu[k],
                float(params.kappa[k]),
                1,
                rng=self._rng,
            )[0],
            dtype=np.float64,
        )

        return (r * a).astype(np.float64, copy=False)

    def _sample_heavy_tailed_component(self, params: vMFNMParameters) -> NDArrayF:
        """Sample a single vector from the heavy-tailed inverse-Nakagami component."""
        k = int(self._rng.choice(params.K, p=params.pi))

        # Heavy-tailed parameters: m_IN = ceil(sqrt(d))
        m_IN = max(1, math.ceil(math.sqrt(self.d)))

        # Match modes between Nakagami and Inverse Nakagami (Equation 34)
        Omega_IN = float(
            self._calculate_matched_omega_inverse_nakagami(
                float(params.m[k]), float(params.Omega[k]), float(m_IN)
            )
        )

        # Radius from Inverse Nakagami
        r: float = float(
            np.asarray(
                InverseNakagamiDistribution.sample(
                    float(m_IN),
                    Omega_IN,
                    1,
                    rng=self._rng,
                ),
                dtype=np.float64,
            )[0]
        )

        # Direction from vMF
        a: NDArrayF = np.asarray(
            VonMisesFisherSampler.sample(
                params.mu[k],
                float(params.kappa[k]),
                1,
                rng=self._rng,
            )[0],
            dtype=np.float64,
        )

        return (r * a).astype(np.float64, copy=False)

    # -------------------------------------------------------------------------
    # Weights, CV, Sigma adaptation
    # -------------------------------------------------------------------------
    def _calculate_matched_omega_inverse_nakagami(
        self, m_N: float, Omega_N: float, m_IN: float
    ) -> float:
        """Calculate Omega_IN to match modes (Equation 34)."""
        # gamma_ratio_squared = [Γ(m_N)/Γ(m_N+1/2)]^2 (stable log-domain form)
        log_ratio = 2.0 * (float(loggamma(m_N)) - float(loggamma(m_N + 0.5)))
        gamma_ratio_squared: float = float(np.exp(np.clip(log_ratio, -700.0, 700.0)))
        Omega_IN: float = (
            float((2.0 * m_IN) / (2.0 * m_IN + 1.0))
            * gamma_ratio_squared
            * float(m_N / Omega_N)
        )
        # Ensure strictly positive
        return float(max(Omega_IN, 1e-6))

    def _calculate_stopping_weights(
        self,
        samples: NDArrayF,  # noqa: ARG002 - kept for signature parity
        g_values: NDArrayF,
        sigma: float,
        params: vMFNMParameters,  # noqa: ARG002 - kept for signature parity
        lambda_val: float,  # noqa: ARG002 - kept for signature parity
    ) -> NDArrayF:
        """Calculate stopping weights W*_t(u) = I_ΩF(u) / h_t(u).

        Only ``g_values`` and ``sigma`` enter the formula; the remaining
        arguments mirror the other per-iteration helpers so callers can pass
        the same tuple to each of them.
        """
        # Indicator 1{g(u) <= 0}
        indicators: NDArrayF = (g_values <= 0.0).astype(np.float64, copy=False)

        # Smoothed indicator h_t(u) = Φ(-g(u)/σ_t)
        h_values: NDArrayF = np.asarray(
            stats.norm.cdf(-g_values / float(sigma)), dtype=np.float64
        )

        # Avoid divide-by-zero via max with tiny
        weights: NDArrayF = (indicators / np.maximum(h_values, 1e-15)).astype(
            np.float64, copy=False
        )
        return weights

    def _coefficient_of_variation(self, weights: NDArrayF) -> float:
        """Coefficient of variation of a weight vector."""
        if weights.size == 0:
            return float("inf")

        mean_w: float = float(np.mean(weights))
        if mean_w <= 0.0:
            return float("inf")

        std_w: float = float(np.std(weights))
        return float(std_w / mean_w)

    def _determine_next_sigma(
        self,
        samples: NDArrayF,
        g_values: NDArrayF,
        params: vMFNMParameters,
        lambda_val: float,
        sigma_prev: float,
    ) -> float:
        """Determine next smoothing parameter σ by solving the 1-D problem (10)."""

        def cv_objective(sigma: float) -> float:
            """Objective: (δ_W_t(σ) - δ_target)^2 for 0 < σ < σ_prev."""
            if sigma >= sigma_prev or sigma <= 0.0:
                return 1e10  # reject invalid region
            weights = self._calculate_intermediate_weights(
                samples, g_values, sigma, params, lambda_val
            )
            cv = self._coefficient_of_variation(weights)
            return float((cv - self.delta_target) ** 2)

        # Minimize over (tiny, sigma_prev)
        try:
            result = minimize_scalar(
                cv_objective,
                bounds=(1e-8, float(sigma_prev) * 0.999),
                method="bounded",
            )
            new_sigma = float(result.x)
        except Exception:
            # Fallback: conservative reduction
            new_sigma = float(sigma_prev) * 0.8

        return float(max(new_sigma, 1e-8))

    def _calculate_intermediate_weights(
        self,
        samples: NDArrayF,
        g_values: NDArrayF,
        sigma: float,
        params: vMFNMParameters,
        lambda_val: float,
    ) -> NDArrayF:
        """Calculate intermediate importance weights W_t(u_i, σ)."""
        # h(u;σ) = Φ(-g/σ)
        h_values: NDArrayF = np.asarray(
            stats.norm.cdf(-g_values / float(sigma)), dtype=np.float64
        )

        # Prior density p(u)
        prior_densities: NDArrayF = self._evaluate_prior_density(samples)

        # Safe mixture density q_safe(u; φ)
        safe_densities: NDArrayF = self._evaluate_safe_mixture_density(
            samples, params, lambda_val
        )

        # W_t = h * p / q_safe
        weights: NDArrayF = (
            h_values * prior_densities / np.maximum(safe_densities, 1e-15)
        ).astype(np.float64, copy=False)
        return weights

    def _update_parameters_penalized_em(
        self,
        samples: NDArrayF,
        g_values: NDArrayF,
        params: vMFNMParameters,
        sigma: float,
        lambda_val: float,
    ) -> vMFNMParameters:
        """Update vMFNM parameters using penalized EM with importance weights."""
        weights: NDArrayF = self._calculate_intermediate_weights(
            samples, g_values, sigma, params, lambda_val
        )
        # Fit returns (updated_params, final_K). We only need parameters here.
        updated_params, _final_K = self.em_optimizer.fit(samples, weights, params)
        return updated_params

    # -------------------------------------------------------------------------
    # Densities and final estimate
    # -------------------------------------------------------------------------
    def _estimate_failure_probability(
        self,
        samples: NDArrayF,
        g_values: NDArrayF,
        params: vMFNMParameters,
        lambda_val: float,
    ) -> float:
        """Final failure probability estimate via equation (36)."""
        # Indicator 1{g <= 0}
        indicators: NDArrayF = (g_values <= 0.0).astype(np.float64, copy=False)

        # Densities
        prior_densities: NDArrayF = self._evaluate_prior_density(samples)
        safe_densities: NDArrayF = self._evaluate_safe_mixture_density(
            samples, params, lambda_val
        )

        # Importance weights
        importance_weights: NDArrayF = (
            prior_densities / np.maximum(safe_densities, 1e-15)
        ).astype(np.float64, copy=False)

        # Monte Carlo estimate
        pf_estimate: float = float(np.mean(indicators * importance_weights))
        return pf_estimate

    def _evaluate_prior_density(self, samples: NDArrayF) -> NDArrayF:
        """Evaluate standard Gaussian prior density p(u) = N(0, I_d)."""
        vals = stats.multivariate_normal.pdf(
            samples, mean=np.zeros(self.d), cov=np.eye(self.d)
        )
        return np.asarray(vals, dtype=np.float64)

    def _evaluate_safe_mixture_density(
        self, samples: NDArrayF, params: vMFNMParameters, lambda_val: float
    ) -> NDArrayF:
        """Evaluate safe mixture density q_safe(u; φ)."""
        # Light-tailed component (vMF-Nakagami mixture)
        vmfnm_dist = vMFNMDistribution(params)
        light_densities: NDArrayF = np.asarray(
            vmfnm_dist.pdf(samples), dtype=np.float64
        )

        # Heavy-tailed component (inverse Nakagami radial + vMF angular)
        heavy_densities: NDArrayF = self._evaluate_heavy_tailed_density(samples, params)

        # Combine
        safe_densities: NDArrayF = (
            float(lambda_val) * light_densities
            + (1.0 - float(lambda_val)) * heavy_densities
        ).astype(np.float64, copy=False)
        return safe_densities

    def _evaluate_heavy_tailed_density(
        self, samples: NDArrayF, params: vMFNMParameters
    ) -> NDArrayF:
        """Evaluate density of the heavy-tailed component for each sample.

        The component is written in polar form as a radial density times an
        angular one. Converting that back to a density on R^d needs the
        Jacobian of the polar map, ``du = r^(d-1) dr dw``, exactly as
        :meth:`vMFNMDistribution.pdf` does. Without it the component does not
        integrate to one, and since it is part of the importance-sampling
        denominator every estimate is scaled by the error.
        """
        n_samples = int(samples.shape[0])
        radii, directions = radii_and_directions(samples)
        jacobian = radii ** (self.d - 1)

        # The inverse-Nakagami shape depends only on the problem dimension, so
        # it is hoisted out of the per-component loop.
        m_IN = max(1, math.ceil(math.sqrt(self.d)))

        densities: NDArrayF = np.zeros(n_samples, dtype=np.float64)
        for k in range(params.K):
            # Radial: Inverse Nakagami with mode-matching parameters
            Omega_IN = float(
                self._calculate_matched_omega_inverse_nakagami(
                    float(params.m[k]), float(params.Omega[k]), float(m_IN)
                )
            )
            radial_density: NDArrayF = np.asarray(
                InverseNakagamiDistribution.pdf(radii, int(m_IN), Omega_IN),
                dtype=np.float64,
            )

            # Angular: vMF density at each direction
            angular_density: NDArrayF = vmf_pdf_batch(
                directions, params.mu[k], float(params.kappa[k])
            )

            densities += (
                float(params.pi[k]) * radial_density * angular_density / jacobian
            )

        return densities

    # -------------------------------------------------------------------------
    # vMF single-point PDF
    # -------------------------------------------------------------------------
    def _vmf_pdf_single(self, x: NDArrayF, mu: NDArrayF, kappa: float) -> float:
        """von Mises–Fisher PDF for a single point x on S^{d-1}.

        Returns the density with respect to the surface area measure on the
        unit sphere, so that ∫_{S^{d-1}} f(a) dω(a) = 1.
        """
        x_arr: NDArrayF = np.asarray(x, dtype=np.float64).reshape(1, -1)
        return float(vmf_pdf_batch(x_arr, mu, kappa)[0])
