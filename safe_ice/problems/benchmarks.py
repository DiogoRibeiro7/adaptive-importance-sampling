# safe_ice/problems/benchmarks.py
"""Benchmark problems for Safe-ICE testing."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import numpy.typing as npt


class BenchmarkProblems:
    """Complete implementation of benchmark problems from the paper."""

    @staticmethod
    def _as_2d_input(u: npt.ArrayLike) -> tuple[npt.NDArray[np.float64], bool]:
        """Normalize inputs to (n, d) and track if original input was a single sample."""
        arr = np.asarray(u, dtype=np.float64)
        if arr.ndim == 1:
            return arr.reshape(1, -1), True
        if arr.ndim != 2:
            raise ValueError("Input must be a 1-D or 2-D array.")
        return arr, False

    @staticmethod
    def four_mode_series_system(
        z: float = 3.8,
    ) -> Callable[[npt.ArrayLike], float | npt.NDArray[np.float64]]:
        """Four-mode series system from Section 4.1 (Equation 37)."""

        def limit_state_function(
            u: npt.ArrayLike,
        ) -> float | npt.NDArray[np.float64]:
            arr, is_single = BenchmarkProblems._as_2d_input(u)
            if arr.shape[1] != 2:
                raise ValueError("four_mode_series_system expects dimension 2.")

            u1 = arr[:, 0]
            u2 = arr[:, 1]

            g1 = 0.1 * (u1 - u2) ** 2 - (u1 + u2) / np.sqrt(2.0) + 3.0
            g2 = 0.1 * (u1 - u2) ** 2 + (u1 + u2) / np.sqrt(2.0) + 3.0
            g3 = u1 - u2 + np.sqrt(3.5)
            g4 = u2 - u1 + np.sqrt(3.5)

            g_min = np.minimum(np.minimum(g1, g2), np.minimum(g3, g4)) + z
            if is_single:
                return float(g_min[0])
            return np.asarray(g_min, dtype=np.float64)

        return limit_state_function

    @staticmethod
    def three_mode_problem(
        z: float = 3.0,
    ) -> Callable[[npt.ArrayLike], float | npt.NDArray[np.float64]]:
        """Three-mode problem from Section 4.2 (Equation 38)."""

        def limit_state_function(
            u: npt.ArrayLike,
        ) -> float | npt.NDArray[np.float64]:
            arr, is_single = BenchmarkProblems._as_2d_input(u)
            if arr.shape[1] != 2:
                raise ValueError("three_mode_problem expects dimension 2.")

            u1 = arr[:, 0]
            u2 = arr[:, 1]

            g1 = z - 1.0 - u2 + np.exp(-(u1**2) / 10.0) + (u1 / 5.0) ** 4
            g2 = (z**2) / 2.0 - u1 * u2
            g_min = np.minimum(g1, g2)
            if is_single:
                return float(g_min[0])
            return np.asarray(g_min, dtype=np.float64)

        return limit_state_function

    @staticmethod
    def nonlinear_oscillator_simplified(
        d: int = 10, z: float = 0.05
    ) -> Callable[[npt.ArrayLike], float | npt.NDArray[np.float64]]:
        """Simplified nonlinear oscillator problem.

        .. warning::

           This problem cannot fail as parameterised, so it measures nothing.
           Displacement is computed as ``force_rms / (k * (1 - alpha))`` with
           ``k = 5e6``, giving values around ``4e-7`` against a threshold of
           ``z = 0.05``. Reaching the threshold needs ``||u||`` of about
           ``7.3e5``, where the norm of a 10-dimensional standard normal
           averages ``3.1``. Crude Monte Carlo over 2e7 samples finds zero
           failures, and any estimator will return ``0``.

           The scaling is inconsistent by a factor of roughly ``1e5``.
           Reconstructing the intended formulation needs the source paper, so
           the defect is documented rather than guessed at. See
           ``tests/test_benchmark_ground_truth.py::TestNonlinearOscillator``.
        """

        def limit_state_function(
            u: npt.ArrayLike,
        ) -> float | npt.NDArray[np.float64]:
            # Parameters from the paper (kept as floats)
            k = 5e6  # stiffness
            alpha = 0.1  # force partition
            S = 0.005  # white-noise intensity

            # Frequency discretization
            d_eff = max(int(d), 2)
            omega_cut = 15.0 * float(np.pi)
            domega = omega_cut / (d_eff / 2.0)

            # Force amplitude
            sigma = float(np.sqrt(2.0 * S * domega))

            # Construct force “RMS” proxy from u
            arr, is_single = BenchmarkProblems._as_2d_input(u)
            if arr.shape[1] < d_eff:
                padded = np.zeros((arr.shape[0], d_eff), dtype=np.float64)
                padded[:, : arr.shape[1]] = arr
                arr_eff = padded
            else:
                arr_eff = arr[:, :d_eff]

            force_rms = sigma * np.sqrt(np.sum(arr_eff**2, axis=1))

            # Simplified response calculation
            response_scale = force_rms / (k * (1.0 - alpha))

            # Approximate maximum displacement
            max_displacement = response_scale * (1.0 + 0.5 * force_rms / (k * 0.04))
            g_values = z - max_displacement

            if is_single:
                return float(g_values[0])
            return np.asarray(g_values, dtype=np.float64)

        return limit_state_function

    @staticmethod
    def nonlinear_oscillator(
        dimension: int = 10, z: float = 0.05
    ) -> Callable[[npt.ArrayLike], float | npt.NDArray[np.float64]]:
        """Compatibility wrapper for nonlinear oscillator benchmark.

        .. warning::

           Delegates to :meth:`nonlinear_oscillator_simplified` and shares its
           defect: the failure region is unreachable, so this returns a
           probability of zero for any estimator.
        """
        return BenchmarkProblems.nonlinear_oscillator_simplified(d=dimension, z=z)

    @staticmethod
    def two_mode_opposite_directions(
        dimension: int = 2,
        z: float = 3.0,
    ) -> Callable[[npt.ArrayLike], float | npt.NDArray[np.float64]]:
        """Two-mode problem with opposite directions (Equation 43)."""

        d = int(dimension)

        def limit_state_function(
            u: npt.ArrayLike,
        ) -> float | npt.NDArray[np.float64]:
            arr, is_single = BenchmarkProblems._as_2d_input(u)
            if arr.shape[1] != d:
                raise ValueError(f"two_mode_opposite_directions expects dimension {d}.")

            sum_u = np.sum(arr, axis=1)
            scale = np.sqrt(float(d))
            g1 = z - sum_u / scale
            g2 = z + sum_u / scale
            g_min = np.minimum(g1, g2)
            if is_single:
                return float(g_min[0])
            return np.asarray(g_min, dtype=np.float64)

        return limit_state_function

    @staticmethod
    def nakagami_ratio_problem(
        threshold: float = 0.1,
    ) -> Callable[[npt.ArrayLike], float | npt.NDArray[np.float64]]:
        """Simple ratio-style benchmark using two transformed standard normals."""

        def limit_state_function(
            u: npt.ArrayLike,
        ) -> float | npt.NDArray[np.float64]:
            arr, is_single = BenchmarkProblems._as_2d_input(u)
            if arr.shape[1] != 2:
                raise ValueError("nakagami_ratio_problem expects dimension 2.")

            log_ratio = np.clip(arr[:, 1] - arr[:, 0], -700.0, 700.0)
            g_values = np.exp(log_ratio) - threshold

            if is_single:
                return float(g_values[0])
            return np.asarray(g_values, dtype=np.float64)

        return limit_state_function
