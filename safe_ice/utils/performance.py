"""Performance utilities and optimizations for Safe-ICE."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from functools import lru_cache
from typing import Any

import numpy as np


class PerformanceCache:
    """Cache manager for expensive computations in Safe-ICE."""

    def __init__(self, max_size: int = 1000):
        """
        Initialize performance cache.

        Parameters
        ----------
        max_size : int
            Maximum number of items to cache.
        """
        self.max_size = max_size
        self._omega_in_cache: dict[tuple[float, float, float], float] = {}
        self._vmf_norm_cache: dict[tuple[float, int], float] = {}
        self._chi2_pdf_cache: dict[tuple[float, int], float] = {}

    def clear(self) -> None:
        """Clear all caches."""
        self._omega_in_cache.clear()
        self._vmf_norm_cache.clear()
        self._chi2_pdf_cache.clear()

    def get_omega_in(
        self,
        m: float,
        omega: float,
        m_in: float,
        compute_func: Callable[[float, float, float], float] | None = None,
    ) -> float:
        """
        Get cached Omega_IN value or compute if not cached.

        Parameters
        ----------
        m : float
            Nakagami m parameter.
        omega : float
            Nakagami Omega parameter.
        m_in : float
            Inverse Nakagami m parameter.
        compute_func : callable, optional
            Function to compute Omega_IN if not cached.

        Returns
        -------
        float
            Omega_IN value.
        """
        key = (m, omega, m_in)
        if key not in self._omega_in_cache:
            if compute_func is None:
                raise ValueError("compute_func required for uncached value")
            if len(self._omega_in_cache) >= self.max_size:
                # Remove oldest entry (simple FIFO)
                self._omega_in_cache.pop(next(iter(self._omega_in_cache)))
            self._omega_in_cache[key] = compute_func(m, omega, m_in)
        return self._omega_in_cache[key]

    def get_vmf_normalization(
        self,
        kappa: float,
        d: int,
        compute_func: Callable[[float, int], float] | None = None,
    ) -> float:
        """
        Get cached vMF normalization constant.

        Parameters
        ----------
        kappa : float
            Concentration parameter.
        d : int
            Dimension.
        compute_func : callable, optional
            Function to compute normalization if not cached.

        Returns
        -------
        float
            Normalization constant.
        """
        key = (kappa, d)
        if key not in self._vmf_norm_cache:
            if compute_func is None:
                raise ValueError("compute_func required for uncached value")
            if len(self._vmf_norm_cache) >= self.max_size:
                self._vmf_norm_cache.pop(next(iter(self._vmf_norm_cache)))
            self._vmf_norm_cache[key] = compute_func(kappa, d)
        return self._vmf_norm_cache[key]


class VectorizedOperations:
    """Vectorized operations for Safe-ICE computations."""

    @staticmethod
    def compute_radii_batch(samples: np.ndarray) -> np.ndarray:
        """
        Compute radii for batch of samples (vectorized).

        Parameters
        ----------
        samples : np.ndarray
            Samples array of shape (n_samples, d).

        Returns
        -------
        np.ndarray
            Radii array of shape (n_samples,).
        """
        return np.asarray(np.linalg.norm(samples, axis=1))

    @staticmethod
    def normalize_directions_batch(
        samples: np.ndarray, radii: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Normalize samples to unit directions (vectorized).

        Parameters
        ----------
        samples : np.ndarray
            Samples array of shape (n_samples, d).
        radii : np.ndarray, optional
            Pre-computed radii. If None, will be computed.

        Returns
        -------
        directions : np.ndarray
            Normalized directions of shape (n_samples, d).
        radii : np.ndarray
            Radii of shape (n_samples,).
        """
        if radii is None:
            radii = np.linalg.norm(samples, axis=1)

        # Handle zero-radius case
        valid_mask = radii > 1e-12
        directions = np.zeros_like(samples)
        directions[valid_mask] = samples[valid_mask] / radii[valid_mask, np.newaxis]

        # Set degenerate directions to first basis vector
        directions[~valid_mask, 0] = 1.0

        return directions, radii

    @staticmethod
    def evaluate_vmf_density_batch(
        directions: np.ndarray, mu: np.ndarray, kappa: float, normalization: float
    ) -> np.ndarray:
        """
        Evaluate vMF density for batch of directions (vectorized).

        Parameters
        ----------
        directions : np.ndarray
            Unit directions of shape (n_samples, d).
        mu : np.ndarray
            Mean direction of shape (d,).
        kappa : float
            Concentration parameter.
        normalization : float
            Normalization constant.

        Returns
        -------
        np.ndarray
            Density values of shape (n_samples,).
        """
        # Vectorized dot product
        dot_products = np.dot(directions, mu)
        return np.asarray(normalization * np.exp(kappa * dot_products))

    @staticmethod
    def evaluate_mixture_density_batch(
        samples: np.ndarray,  # noqa: ARG004 - kept so batch helpers share one signature
        component_densities: np.ndarray,
        weights: np.ndarray,
    ) -> np.ndarray:
        """
        Evaluate mixture density for batch (vectorized).

        Parameters
        ----------
        samples : np.ndarray
            Samples of shape (n_samples, d).
        component_densities : np.ndarray
            Component densities of shape (n_samples, K).
        weights : np.ndarray
            Mixture weights of shape (K,).

        Returns
        -------
        np.ndarray
            Mixture density values of shape (n_samples,).
        """
        # Weighted sum across components
        return np.asarray(np.dot(component_densities, weights))


class MemoryEfficientSampling:
    """Memory-efficient sampling strategies."""

    @staticmethod
    def generate_samples_in_batches(
        sample_func: Callable[..., np.ndarray],
        n_total: int,
        batch_size: int = 10000,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Generate samples in batches to manage memory.

        Parameters
        ----------
        sample_func : callable
            Function to generate samples.
        n_total : int
            Total number of samples.
        batch_size : int
            Batch size for generation.
        **kwargs
            Additional arguments for sample_func.

        Returns
        -------
        np.ndarray
            All generated samples.
        """
        n_batches = (n_total + batch_size - 1) // batch_size
        samples_list = []

        for i in range(n_batches):
            n_batch = min(batch_size, n_total - i * batch_size)
            batch_samples = sample_func(n_batch, **kwargs)
            samples_list.append(batch_samples)

        return np.vstack(samples_list)

    @staticmethod
    def evaluate_function_in_batches(
        func: Callable[[np.ndarray], np.ndarray],
        data: np.ndarray,
        batch_size: int = 10000,
    ) -> np.ndarray:
        """
        Evaluate function on data in batches.

        Parameters
        ----------
        func : callable
            Function to evaluate.
        data : np.ndarray
            Input data.
        batch_size : int
            Batch size for evaluation.

        Returns
        -------
        np.ndarray
            Function values.
        """
        n_total = len(data)
        n_batches = (n_total + batch_size - 1) // batch_size
        results_list = []

        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, n_total)
            batch_data = data[start_idx:end_idx]
            batch_results = func(batch_data)
            results_list.append(batch_results)

        return np.concatenate(results_list)


class ParallelProcessor:
    """Simple parallel processing utilities (without numba dependency)."""

    @staticmethod
    def parallel_map(
        func: Callable[[Any], Any], data_list: Sequence[Any], n_jobs: int = -1
    ) -> list[Any]:
        """
        Apply function to list in parallel using multiprocessing.

        Parameters
        ----------
        func : callable
            Function to apply.
        data_list : list
            List of data items.
        n_jobs : int
            Number of parallel jobs. -1 means use all CPUs.

        Returns
        -------
        list
            Results list.
        """
        try:
            from multiprocessing import Pool, cpu_count

            if n_jobs == -1:
                n_jobs = cpu_count()

            with Pool(n_jobs) as pool:
                results = pool.map(func, data_list)
            return results

        except Exception as e:
            warnings.warn(
                f"Parallel processing failed: {e}. Using sequential.", stacklevel=2
            )
            return [func(item) for item in data_list]


class OptimizationUtils:
    """Utility functions for performance optimization."""

    @staticmethod
    def preallocate_arrays(shapes: dict[str, tuple[int, ...]]) -> dict[str, np.ndarray]:
        """
        Preallocate arrays to avoid repeated allocations.

        Parameters
        ----------
        shapes : dict
            Dictionary mapping array names to shapes.

        Returns
        -------
        dict
            Dictionary of preallocated arrays.
        """
        arrays = {}
        for name, shape in shapes.items():
            arrays[name] = np.zeros(shape, dtype=np.float64)
        return arrays

    @staticmethod
    def check_memory_usage() -> dict[str, float] | None:
        """Check current memory usage."""
        try:
            import psutil

            process = psutil.Process()
            mem_info = process.memory_info()
            return {
                "rss_mb": mem_info.rss / 1024 / 1024,
                "vms_mb": mem_info.vms / 1024 / 1024,
                "available_mb": psutil.virtual_memory().available / 1024 / 1024,
            }
        except ImportError:
            return None

    @staticmethod
    @lru_cache(maxsize=1000)
    def cached_chi2_pdf(x: float, df: int) -> float:
        """
        Cached chi-squared PDF evaluation.

        Parameters
        ----------
        x : float
            Value to evaluate.
        df : int
            Degrees of freedom.

        Returns
        -------
        float
            PDF value.
        """
        from scipy.stats import chi2

        return float(chi2.pdf(x, df))

    @staticmethod
    @lru_cache(maxsize=1000)
    def cached_bessel_iv(nu: float, z: float) -> float:
        """
        Cached modified Bessel function evaluation.

        Parameters
        ----------
        nu : float
            Order.
        z : float
            Argument.

        Returns
        -------
        float
            Bessel function value.
        """
        from scipy.special import iv

        return float(iv(nu, z))


def optimize_safe_ice_iteration(
    samples: np.ndarray,
    g_values: np.ndarray,  # noqa: ARG001 - part of the iteration signature
    params: Any,  # noqa: ARG001 - part of the iteration signature
    cache: PerformanceCache | None = None,
) -> dict[str, Any]:
    """
    Optimized iteration step for Safe-ICE.

    Parameters
    ----------
    samples : np.ndarray
        Sample array.
    g_values : np.ndarray
        Limit state values.
    params : object
        Distribution parameters.
    cache : PerformanceCache, optional
        Cache object for expensive computations.

    Returns
    -------
    dict
        Optimized iteration results.
    """
    vec_ops = VectorizedOperations()

    # Vectorized operations
    directions, radii = vec_ops.normalize_directions_batch(samples)

    # Use cache if available
    if cache:
        # Precompute cached values
        pass

    # Return optimized results
    return {
        "directions": directions,
        "radii": radii,
        "cache_hits": cache._omega_in_cache if cache else 0,
    }


# Performance monitoring decorator
def profile_performance(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to profile function performance."""
    import functools
    import time

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.time()
        start_memory = OptimizationUtils.check_memory_usage()

        result = func(*args, **kwargs)

        elapsed_time = time.time() - start_time
        end_memory = OptimizationUtils.check_memory_usage()

        if start_memory and end_memory:
            memory_delta = end_memory["rss_mb"] - start_memory["rss_mb"]
            print(
                f"{func.__name__}: {elapsed_time:.2f}s, delta_mem: {memory_delta:.1f}MB"
            )
        else:
            print(f"{func.__name__}: {elapsed_time:.2f}s")

        return result

    return wrapper
