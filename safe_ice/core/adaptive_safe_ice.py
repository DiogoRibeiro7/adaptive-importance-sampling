"""Adaptive Safe-ICE with automatic parameter tuning."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize_scalar

from ..optimization.penalized_em import PenalizedEMOptimizer
from ..typing import LimitStateFunction
from .parameters import vMFNMParameters
from .safe_ice_optimized import OptimizedSafeICE

NDArrayF = npt.NDArray[np.float64]


class AdaptiveSafeICE(OptimizedSafeICE):
    """
    Adaptive Safe-ICE with automatic parameter tuning.

    Features:
    - Auto-tuned penalty coefficient β
    - Adaptive annealing schedule
    - Dimension-dependent initialization
    - Dynamic convergence criteria
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
        """
        Initialize Adaptive Safe-ICE.

        Parameters
        ----------
        limit_state_function : callable
            Function g(u) where g(u) < 0 indicates failure.
        dimension : int
            Problem dimension.
        N : int, optional
            Number of samples per iteration. Auto-computed if None.
        max_iterations : int, optional
            Maximum iterations.
        auto_tune : bool, optional
            Enable automatic parameter tuning.
        adaptive_schedule : bool, optional
            Enable adaptive annealing schedule.
        random_state : int or np.random.Generator, optional
            Random state for reproducibility.
        **kwargs
            Additional arguments passed to OptimizedSafeICE.
        """
        # Auto-compute N based on dimension if not provided
        if N is None:
            N = self._compute_adaptive_sample_size(dimension)

        # Auto-compute K0 based on dimension
        if "K0" not in kwargs:
            kwargs["K0"] = self._compute_adaptive_k0(dimension)

        # Auto-compute delta parameters based on dimension
        if "delta_target" not in kwargs:
            kwargs["delta_target"] = self._compute_adaptive_delta_target(dimension)

        if "delta_star" not in kwargs:
            kwargs["delta_star"] = self._compute_adaptive_delta_star(dimension)

        # Initialize parent class
        super().__init__(
            limit_state_function=limit_state_function,
            dimension=dimension,
            N=N,
            max_iterations=max_iterations,
            random_state=random_state,
            **kwargs,
        )

        self.auto_tune = auto_tune
        self.adaptive_schedule = adaptive_schedule

        # Adaptive parameters
        self.beta_history: list[float] = []
        self.learning_rate_history: list[float] = []
        self.convergence_history: list[dict[str, float]] = []

    def _compute_adaptive_sample_size(self, d: int) -> int:
        """
        Compute adaptive sample size based on dimension.

        Parameters
        ----------
        d : int
            Dimension.

        Returns
        -------
        int
            Adaptive sample size.
        """
        # Empirical formula based on dimension
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

        # Scale with sqrt(d) for high dimensions
        N = int(base_N * np.sqrt(d / 2))

        return min(N, 50000)  # Cap at 50k samples

    def _compute_adaptive_k0(self, d: int) -> int:
        """
        Compute adaptive initial components based on dimension.

        Parameters
        ----------
        d : int
            Dimension.

        Returns
        -------
        int
            Initial number of components.
        """
        # More components for higher dimensions
        if d <= 2:
            K0 = 10
        elif d <= 5:
            K0 = 15
        elif d <= 10:
            K0 = 20
        elif d <= 20:
            K0 = 30
        else:
            K0 = min(50, 2 * d)

        return K0

    def _compute_adaptive_delta_target(self, d: int) -> float:
        """
        Compute adaptive target delta based on dimension.

        Parameters
        ----------
        d : int
            Dimension.

        Returns
        -------
        float
            Target delta.
        """
        # Higher delta for higher dimensions
        if d <= 2:
            delta_target = 3.0
        elif d <= 5:
            delta_target = 3.5
        elif d <= 10:
            delta_target = 4.0
        elif d <= 20:
            delta_target = 4.5
        else:
            delta_target = min(6.0, 3.0 + np.log(d))

        return delta_target

    def _compute_adaptive_delta_star(self, d: int) -> float:
        """
        Compute adaptive delta star based on dimension.

        Parameters
        ----------
        d : int
            Dimension.

        Returns
        -------
        float
            Delta star (increment threshold).
        """
        # Adaptive increment based on dimension
        if d <= 2:
            delta_star = 1.5
        elif d <= 10:
            delta_star = 1.0
        else:
            delta_star = 0.75

        return delta_star

    def _auto_tune_beta(
        self,
        samples: NDArrayF,
        weights: NDArrayF,
        params: vMFNMParameters,
        iteration: int,
    ) -> float:
        """
        Auto-tune penalty coefficient β.

        Parameters
        ----------
        samples : np.ndarray
            Elite samples.
        weights : np.ndarray
            Importance weights.
        params : vMFNMParameters
            Current parameters.
        iteration : int
            Current iteration number.

        Returns
        -------
        float
            Optimal beta value.
        """

        # Objective function for beta tuning
        def objective(beta: float) -> float:
            """Objective: Balance between likelihood and sparsity."""
            optimizer = PenalizedEMOptimizer(max_em_iterations=20)

            try:
                new_params, _ = optimizer.fit(samples, weights, params, beta_init=beta)

                # Compute effective sample size
                ess = np.sum(weights) ** 2 / np.sum(weights**2)

                # Penalty for too many or too few components
                K = new_params.K
                K_target = max(2, min(self.K0 // 2, int(np.sqrt(len(samples)))))
                K_penalty = abs(K - K_target) / K_target

                # Combined objective
                return -ess / len(samples) + 0.1 * K_penalty

            except Exception:
                return float("inf")

        # Search for optimal beta
        if iteration == 0:
            # Initial broad search
            beta_range = (0.1, 5.0)
        else:
            # Refined search around previous value
            prev_beta = self.beta_history[-1] if self.beta_history else 1.0
            beta_range = (max(0.1, prev_beta * 0.5), min(10.0, prev_beta * 2.0))

        result = minimize_scalar(
            objective, bounds=beta_range, method="bounded", options={"maxiter": 10}
        )

        optimal_beta = result.x if result.success else 1.0
        self.beta_history.append(optimal_beta)

        return optimal_beta

    def _adaptive_annealing_schedule(
        self, sigma: float, iteration: int, convergence_rate: float
    ) -> float:
        """
        Adaptive annealing schedule based on convergence.

        Parameters
        ----------
        sigma : float
            Current sigma value.
        iteration : int
            Current iteration.
        convergence_rate : float
            Rate of convergence (CV change).

        Returns
        -------
        float
            Lambda value for mixture.
        """
        if not self.adaptive_schedule:
            # Use standard cosine annealing
            return self._cosine_annealing_schedule(sigma, self.delta_target)

        # Adaptive schedule based on convergence
        if iteration == 0:
            base_lambda = 0.9
        else:
            # Adjust based on convergence rate
            if convergence_rate < 0.01:
                # Fast convergence - more exploration
                base_lambda = 0.7
            elif convergence_rate < 0.05:
                # Moderate convergence
                base_lambda = 0.8
            else:
                # Slow convergence - more exploitation
                base_lambda = 0.9

        # Apply decay based on iteration
        decay = np.exp(-iteration / 10)
        lambda_val = base_lambda * (1 - 0.3 * (1 - decay))

        return float(max(0.5, min(0.95, lambda_val)))

    def _compute_convergence_metrics(
        self, weights: NDArrayF, g_values: NDArrayF, iteration: int
    ) -> dict[str, float]:
        """
        Compute comprehensive convergence metrics.

        Parameters
        ----------
        weights : np.ndarray
            Importance weights.
        g_values : np.ndarray
            Limit state values.
        iteration : int
            Current iteration.

        Returns
        -------
        dict
            Convergence metrics.
        """
        # Basic CV
        cv = self._compute_cv(weights)

        # Effective sample size
        ess = np.sum(weights) ** 2 / np.sum(weights**2) if np.sum(weights) > 0 else 0

        # Failure sample ratio
        failure_ratio = np.sum(g_values <= 0) / len(g_values)

        # Convergence rate (if we have history)
        if self.convergence_history:
            prev_cv = self.convergence_history[-1]["cv"]
            cv_change = abs(cv - prev_cv) / max(prev_cv, 1e-10)
        else:
            cv_change = 1.0

        metrics = {
            "cv": cv,
            "ess": ess,
            "failure_ratio": failure_ratio,
            "cv_change": cv_change,
            "iteration": iteration,
        }

        self.convergence_history.append(metrics)

        return metrics

    def _adaptive_stopping_criterion(
        self, metrics: dict[str, float], iteration: int
    ) -> bool:
        """
        Adaptive stopping criterion based on multiple metrics.

        Parameters
        ----------
        metrics : dict
            Convergence metrics.
        iteration : int
            Current iteration.

        Returns
        -------
        bool
            True if should stop.
        """
        # Minimum iterations
        if iteration < 3:
            return False

        # Maximum iterations
        if iteration >= self.max_iterations - 1:
            return True

        # Check CV convergence
        cv_converged = metrics["cv"] < 0.05

        # Check CV stability (small changes)
        if len(self.convergence_history) >= 3:
            recent_cvs = [m["cv"] for m in self.convergence_history[-3:]]
            cv_stable = bool(np.std(recent_cvs) < 0.01)
        else:
            cv_stable = False

        # Check ESS
        ess_sufficient = metrics["ess"] > 0.5 * self.N

        # Adaptive criterion
        if cv_converged and cv_stable:
            return True

        return bool(iteration > 10 and cv_stable and ess_sufficient)

    def run(
        self,
        initial_params: vMFNMParameters | None = None,
        verbose: bool = False,
    ) -> tuple[float, dict[str, Any]]:
        """
        Run Adaptive Safe-ICE algorithm.

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
            Detailed results including adaptive metrics.
        """
        # Initialize parameters adaptively
        if initial_params is None:
            if verbose:
                print("Initializing parameters adaptively...")
            initial_params = self._initialize_adaptive_parameters()

        # Initialize tracking
        phi_t = initial_params
        sigma = self.sigma0
        delta_prev = 0.0

        # Storage
        all_samples: list[NDArrayF] = []
        all_g_values: list[NDArrayF] = []
        all_weights: list[NDArrayF] = []
        iterations_data: list[dict[str, Any]] = []

        if verbose:
            print("=" * 50)
            print("Adaptive Safe-ICE Algorithm")
            print("=" * 50)
            print(f"Dimension: {self.d}")
            print(f"Adaptive N: {self.N}")
            print(f"Adaptive K0: {self.K0}")
            print(f"Auto-tune: {self.auto_tune}")
            print(f"Adaptive schedule: {self.adaptive_schedule}")
            print("-" * 50)

        # Main adaptive loop
        for t in range(self.max_iterations):
            if verbose:
                print(f"\nIteration {t + 1}")
                print("-" * 30)

            # Clear caches
            if self.enable_caching:
                self._clear_caches()
                self._precompute_omega_in(phi_t)

            # Compute convergence rate for adaptive schedule
            conv_rate = self.convergence_history[-1]["cv_change"] if t > 0 else 1.0

            # Adaptive annealing
            lambda_t = self._adaptive_annealing_schedule(sigma, t, conv_rate)

            # Generate samples
            if verbose:
                print(f"  Generating {self.N} samples (lambda={lambda_t:.3f})...")
            samples_t = self._generate_safe_mixture_samples_batched(phi_t, lambda_t)

            # Evaluate limit state
            g_values_t = self._evaluate_limit_state_vectorized(samples_t)

            # Update delta adaptively
            if t > 0 and self.convergence_history[-1]["failure_ratio"] < 0.01:
                # Few failures - increase delta more aggressively
                delta_increment = self.delta_star * 1.5
            else:
                delta_increment = (
                    self.delta_star if delta_prev >= self.delta_star else 0.0
                )

            delta_t = min(self.delta_target, delta_prev + delta_increment)
            delta_prev = delta_t

            # Compute weights
            weights_t = self._compute_importance_weights_vectorized(
                samples_t, g_values_t, phi_t, lambda_t, delta_t, sigma
            )

            # Compute convergence metrics
            metrics = self._compute_convergence_metrics(weights_t, g_values_t, t)

            if verbose:
                print(f"  Delta: {delta_t:.4f}")
                print(f"  CV: {metrics['cv']:.4f}")
                print(f"  ESS: {metrics['ess']:.1f}")
                print(f"  Failures: {metrics['failure_ratio']:.2%}")

            # Store results
            all_samples.append(samples_t)
            all_g_values.append(g_values_t)
            all_weights.append(weights_t)

            iterations_data.append(
                {
                    "iteration": t + 1,
                    "K": phi_t.K,
                    "delta": delta_t,
                    "lambda": lambda_t,
                    "metrics": metrics,
                    "sigma": sigma,
                }
            )

            # Check adaptive stopping
            if self._adaptive_stopping_criterion(metrics, t):
                if verbose:
                    print(f"\nAdaptive convergence achieved at iteration {t + 1}")
                break

            # Update parameters (if not last iteration)
            if t < self.max_iterations - 1:
                # Compute elites
                elites_indices = weights_t > 0
                elites_samples = samples_t[elites_indices]
                elites_weights = weights_t[elites_indices]

                if len(elites_samples) < 10:
                    if verbose:
                        print("  Warning: Too few elite samples, stopping early")
                    break

                # Auto-tune beta if enabled
                if self.auto_tune:
                    beta = self._auto_tune_beta(
                        elites_samples, elites_weights, phi_t, t
                    )
                    if verbose:
                        print(f"  Auto-tuned beta: {beta:.3f}")
                else:
                    beta = 1.0

                # Penalized EM optimization
                optimizer = PenalizedEMOptimizer(max_em_iterations=self.em_max_iter)
                phi_t, _ = optimizer.fit(
                    elites_samples, elites_weights, phi_t, beta_init=beta
                )

                # Adaptive sigma update
                K_new = phi_t.K
                K_old = iterations_data[-1]["K"]

                # More aggressive sigma adaptation
                if K_new < K_old - 2:
                    sigma *= 1.2
                elif K_new > K_old + 2:
                    sigma /= 1.2
                else:
                    sigma = self._adapt_sigma(sigma, K_old, K_new)

                if verbose:
                    print(f"  Updated K: {K_old} -> {K_new}")
                    print(f"  Updated sigma: {sigma:.4f}")

        # Combine results
        final_samples = np.vstack(all_samples)
        final_g_values = np.hstack(all_g_values)
        final_weights = np.hstack(all_weights)

        # Compute failure probability
        failure_indicator = (final_g_values <= 0).astype(float)
        pf_estimate = np.sum(failure_indicator * final_weights) / np.sum(final_weights)

        # Prepare results with adaptive metrics
        results = {
            "final_samples": final_samples,
            "final_weights": final_weights,
            "final_g_values": final_g_values,
            "iterations": iterations_data,
            "convergence_metrics": {
                "history": self.convergence_history,
                "cv_values": [m["cv"] for m in self.convergence_history],
                "ess_values": [m["ess"] for m in self.convergence_history],
            },
            "adaptive_metrics": {
                "beta_history": self.beta_history,
                "auto_tuned_N": self.N,
                "auto_tuned_K0": self.K0,
            },
            "final_parameters": phi_t,
        }

        if verbose:
            print("-" * 50)
            print("Final Results:")
            print(f"Failure Probability: {pf_estimate:.6e}")
            print(f"Total Iterations: {t + 1}")
            print(f"Final Components: {phi_t.K}")
            print(
                f"Average beta: {np.mean(self.beta_history):.3f}"
                if self.beta_history
                else ""
            )

        return float(pf_estimate), results

    def _initialize_adaptive_parameters(self) -> vMFNMParameters:
        """
        Initialize parameters adaptively based on problem characteristics.

        Returns
        -------
        vMFNMParameters
            Adaptively initialized parameters.
        """
        K = int(self.K0)
        d = int(self.d)

        # Adaptive mixture weights (favor fewer initial components)
        alpha = np.arange(K, 0, -1)
        pi = alpha / np.sum(alpha)

        # Adaptive Nakagami parameters based on dimension
        if d <= 2:
            m_range = (1.0, 2.0)
            omega_range = (0.5, 1.5)
        elif d <= 10:
            m_range = (1.5, 3.0)
            omega_range = (0.8, 2.0)
        else:
            m_range = (2.0, 4.0)
            omega_range = (1.0, 3.0)

        m = self._rng.uniform(m_range[0], m_range[1], K).astype(np.float64)
        Omega = self._rng.uniform(omega_range[0], omega_range[1], K).astype(np.float64)

        # Adaptive vMF initialization (spread directions)
        mu = np.zeros((K, d), dtype=np.float64)

        if d == 2:
            # Uniform angles in 2D
            angles = np.linspace(0, 2 * np.pi, K, endpoint=False)
            mu[:, 0] = np.cos(angles)
            mu[:, 1] = np.sin(angles)
        else:
            # Random orthogonal directions for high-d
            for k in range(min(K, d)):
                vec = self._rng.standard_normal(d)
                # Orthogonalize against previous directions
                for j in range(k):
                    vec -= np.dot(vec, mu[j]) * mu[j]
                mu[k] = vec / np.linalg.norm(vec)

            # Random directions for remaining components
            for k in range(d, K):
                mu[k] = self._rng.standard_normal(d)
                mu[k] /= np.linalg.norm(mu[k])

        # Adaptive concentration (lower for exploration)
        kappa_base = 0.5 if d <= 5 else 0.2
        kappa = self._rng.uniform(kappa_base, kappa_base * 2, K).astype(np.float64)

        return vMFNMParameters(pi=pi, m=m, Omega=Omega, mu=mu, kappa=kappa)
