"""Performance evaluation utilities for Safe-ICE."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from ..core.safe_ice import SafeICE
from ..typing import LimitStateFunction

NDArrayF = npt.NDArray[np.float64]
NDArrayI = npt.NDArray[np.int64]


def _evaluate_batch(
    limit_state_func: LimitStateFunction, samples: NDArrayF
) -> NDArrayF:
    """Evaluate a limit-state function on a batch, falling back to row-wise.

    Mirrors :meth:`safe_ice.core.safe_ice.SafeICE._evaluate_limit_state` so
    that vectorised and scalar-only limit-state functions are both accepted.
    """
    try:
        arr = np.asarray(limit_state_func(samples), dtype=np.float64)
        if arr.ndim == 0:
            arr = np.full(samples.shape[0], float(arr), dtype=np.float64)
        elif arr.ndim > 1:
            arr = arr.reshape(-1)
        if arr.shape[0] != samples.shape[0]:
            raise ValueError("Limit-state output shape mismatch")
    except Exception:
        arr = np.asarray(
            [
                float(np.asarray(limit_state_func(s.reshape(1, -1))).reshape(-1)[0])
                for s in samples
            ],
            dtype=np.float64,
        )
    return arr


class PerformanceEvaluator:
    """Comprehensive performance evaluation and comparison"""

    @staticmethod
    def run_monte_carlo_reference(
        limit_state_func: LimitStateFunction,
        dimension: int,
        n_samples: int = 1000000,
        random_state: int | np.random.Generator | None = None,
    ) -> tuple[float, float]:
        """Run a crude Monte Carlo simulation to obtain a reference estimate.

        Parameters
        ----------
        limit_state_func : callable
            Function g(u); failure occurs where g(u) <= 0.
        dimension : int
            Problem dimension.
        n_samples : int
            Number of standard-normal samples to draw.
        random_state : int, numpy Generator, or None
            Seed or generator for reproducible runs. ``None`` uses NumPy's
            global random state.

        Returns
        -------
        tuple of (float, float)
            The estimated failure probability and its standard error.
        """
        if random_state is None:
            rng: Any = np.random
        elif isinstance(random_state, np.random.Generator):
            rng = random_state
        else:
            rng = np.random.default_rng(int(random_state))

        # The target density is the standard normal, so sample it directly
        # rather than going through the general multivariate routine.
        samples = rng.standard_normal((n_samples, dimension))
        g_values = _evaluate_batch(limit_state_func, samples)

        indicators = (g_values <= 0).astype(float)
        pf_mc = float(np.mean(indicators))
        pf_std = float(np.sqrt(pf_mc * (1 - pf_mc) / n_samples))

        return pf_mc, pf_std

    @staticmethod
    def compare_methods(
        limit_state_func: LimitStateFunction,
        dimension: int,
        reference_pf: float | None = None,
        n_runs: int = 10,
        safe_ice_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Comprehensive method comparison"""
        if safe_ice_params is None:
            safe_ice_params = {}

        print(f"Performance Comparison - {dimension}D Problem")
        print("=" * 50)

        # Safe-ICE results
        safe_ice_results = []
        safe_ice_iterations = []
        safe_ice_components = []

        for run in range(n_runs):
            print(f"Safe-ICE Run {run + 1}/{n_runs}")

            safe_ice = SafeICE(limit_state_func, dimension, **safe_ice_params)
            pf_estimate, results = safe_ice.run(verbose=False)

            safe_ice_results.append(pf_estimate)
            # ``results["iterations"]`` is the list of per-iteration records,
            # so the iteration count is its length.
            safe_ice_iterations.append(len(results["iterations"]))
            safe_ice_components.append(results["final_components"])

        # Statistics
        safe_ice_mean = np.mean(safe_ice_results)
        safe_ice_std = np.std(safe_ice_results)
        safe_ice_cv = safe_ice_std / safe_ice_mean if safe_ice_mean > 0 else np.inf

        results_dict = {
            "safe_ice": {
                "estimates": safe_ice_results,
                "mean": safe_ice_mean,
                "std": safe_ice_std,
                "cv": safe_ice_cv,
                "mean_iterations": np.mean(safe_ice_iterations),
                "mean_components": np.mean(safe_ice_components),
            }
        }

        # Print results
        print(f"\nSafe-ICE Results ({n_runs} runs):")
        print(f"  Mean Pf: {safe_ice_mean:.6e}")
        print(f"  Std Pf:  {safe_ice_std:.6e}")
        print(f"  CV:      {safe_ice_cv:.4f}")
        print(f"  Avg Iterations: {np.mean(safe_ice_iterations):.1f}")
        print(f"  Avg Components: {np.mean(safe_ice_components):.1f}")

        if reference_pf is not None:
            relative_error = abs(safe_ice_mean - reference_pf) / reference_pf
            print(f"  Relative Error: {relative_error:.4f}")
            results_dict["safe_ice"]["relative_error"] = relative_error

        return results_dict
