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
        z: float = 1.0,
    ) -> Callable[[npt.ArrayLike], float | npt.NDArray[np.float64]]:
        """Four-mode series system from Section 4.1 (Equation 37).

        Failure is ``g(u) + z <= 0``, so larger ``z`` makes it rarer. Crude
        Monte Carlo over 2e7 samples, against the paper's Figure 4:

            z      failures        pf        rel. s.e.
            0.0       44376    2.2188e-03      0.5%
            0.5        8249    4.1245e-04      1.1%
            1.0        1293    6.4650e-05      2.8%
            1.5         187    9.3500e-06      7.3%
            2.0          21    1.0500e-06     21.8%

        Notes
        -----
        The last two branches of equation (37) are ``u1 - u2 + 7/sqrt(2)``.
        They were written as ``sqrt(3.5)``, reading the fraction ``7/sqrt(2)``
        as ``sqrt(7/2)``: 1.8708 instead of 4.9497. That put the failure region
        far closer to the origin, giving 1.88e-01 at ``z=0`` where the paper
        reports about 1e-03 -- above the top of Figure 4's axis. The default
        ``z`` was 3.8, chosen against the broken function; 1.0 is one of the
        thresholds tabulated in the paper's Table 1.
        """

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
            g3 = u1 - u2 + 7.0 / np.sqrt(2.0)
            g4 = u2 - u1 + 7.0 / np.sqrt(2.0)

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
    def nonlinear_oscillator(
        dimension: int = 10, z: float = 0.05, t_end: float = 8.0, dt: float = 0.01
    ) -> Callable[[npt.ArrayLike], float | npt.NDArray[np.float64]]:
        r"""Hysteretic single-degree-of-freedom oscillator, Section 4.3.

        A Bouc-Wen oscillator (equation 39) driven by white-noise ground
        acceleration discretised in the frequency domain (equation 41):

        .. math::

            m\ddot{x} + c\dot{x} + k[\alpha x + (1-\alpha) x_y z] = f(t)

        with the hysteretic variable following the Bouc-Wen law (equation 40).
        The equations of motion are integrated with the classical fourth-order
        Runge-Kutta method, and the limit state (equation 42) is

        .. math::

            g(u) = z - x(t_{end})

        so failure is a displacement at ``t_end`` exceeding the threshold.

        Parameters
        ----------
        dimension:
            Number of frequency components, which is also the dimension of
            ``u``. Must be even; the paper uses 10.
        z:
            Displacement threshold in metres. The paper varies it from 0.05 to
            0.08, giving failure probabilities from about 1.8e-03 to 1.5e-07.
        t_end:
            Time at which the displacement is compared against ``z``.
        dt:
            Runge-Kutta step. The paper uses 0.01 s.

        Notes
        -----
        This previously computed the displacement as
        ``force_rms / (k * (1 - alpha))``, a closed form that appears nowhere
        in the paper and never integrated the equations of motion. It returned
        values around 4e-07 against a threshold of 0.05, so the problem could
        not fail and every estimator returned exactly 0. Crude Monte Carlo over
        the implementation below gives 1.798e-03 at ``z=0.05``, 1.475e-04 at
        ``0.06`` and 4.5e-06 at ``0.07``, matching the paper's Figure 7.
        """
        d = int(dimension)
        if d < 2 or d % 2 != 0:
            raise ValueError(
                f"nonlinear_oscillator needs an even dimension of at least 2, got {d}."
            )

        # Structural parameters (Section 4.3). SI units throughout.
        mass = 6e4  # kg
        stiffness = 5e6  # N/m
        damping_ratio = 0.05
        yield_displacement = 0.04  # m
        alpha = 0.1  # elastic / hysteretic split of the restoring force
        damping = 2.0 * mass * damping_ratio * float(np.sqrt(stiffness / mass))

        # Bouc-Wen law (equation 40).
        bw_a, bw_beta, bw_gamma, bw_n = 1.0, 0.5, 0.5, 3
        # The exact solution satisfies |z| <= (A / (beta + gamma))^(1/n); the
        # hysteretic variable saturates there. Measured peaks are 0.9264 for
        # ordinary samples and 0.9993 in the failure region, so enforcing the
        # bound changes nothing that is sampled. It matters only for extreme
        # inputs, where discretisation error lets z drift past saturation and
        # the |z|^3 term then amplifies it until the integration overflows.
        hyst_bound = (bw_a / (bw_beta + bw_gamma)) ** (1.0 / bw_n)

        # Frequency discretisation of the load (equation 41). The cut-off is
        # 15*pi, which is exactly the highest retained frequency d/2 * domega.
        half = d // 2
        domega = 30.0 * float(np.pi) / d
        omega = np.arange(1, half + 1, dtype=np.float64) * domega
        intensity = 0.005  # white-noise intensity S, m^2/s^3
        sigma = float(np.sqrt(2.0 * intensity * domega))

        # RK4 evaluates the load at t, t + dt/2 and t + dt, so the load is
        # tabulated once on the half-step grid and reused for every call.
        n_steps = round(float(t_end) / float(dt))
        sample_times = np.arange(2 * n_steps + 1, dtype=np.float64) * (float(dt) / 2.0)
        cos_table = np.cos(np.outer(sample_times, omega))
        sin_table = np.sin(np.outer(sample_times, omega))
        force_amplitude = -mass * sigma

        def limit_state_function(
            u: npt.ArrayLike,
        ) -> float | npt.NDArray[np.float64]:
            arr, is_single = BenchmarkProblems._as_2d_input(u)
            if arr.shape[1] != d:
                raise ValueError(f"nonlinear_oscillator expects dimension {d}.")

            u_cos = arr[:, :half]
            u_sin = arr[:, half:]

            def load(index: int) -> npt.NDArray[np.float64]:
                combined = u_cos @ cos_table[index] + u_sin @ sin_table[index]
                return np.asarray(force_amplitude * combined, dtype=np.float64)

            def derivatives(
                x: npt.NDArray[np.float64],
                v: npt.NDArray[np.float64],
                hyst: npt.NDArray[np.float64],
                f: npt.NDArray[np.float64],
            ) -> tuple[
                npt.NDArray[np.float64],
                npt.NDArray[np.float64],
                npt.NDArray[np.float64],
            ]:
                # Clamp on use, not only on the accepted state: RK4's
                # intermediate stages are unclamped, and with a large enough
                # load h * dz overshoots saturation there, after which the
                # |z|^3 feedback amplifies the overshoot until it overflows.
                hyst = np.clip(hyst, -hyst_bound, hyst_bound)
                abs_hyst = np.abs(hyst)
                d_hyst = (
                    bw_a * v
                    - bw_beta * np.abs(v) * abs_hyst ** (bw_n - 1) * hyst
                    - bw_gamma * v * abs_hyst**bw_n
                ) / yield_displacement
                d_v = (
                    f
                    - damping * v
                    - stiffness
                    * (alpha * x + (1.0 - alpha) * yield_displacement * hyst)
                ) / mass
                return v, d_v, d_hyst

            n = arr.shape[0]
            x = np.zeros(n, dtype=np.float64)
            v = np.zeros(n, dtype=np.float64)
            hyst = np.zeros(n, dtype=np.float64)
            h = float(dt)

            for step in range(n_steps):
                f_start = load(2 * step)
                f_mid = load(2 * step + 1)
                f_end = load(2 * step + 2)

                x1, v1, h1 = derivatives(x, v, hyst, f_start)
                x2, v2, h2 = derivatives(
                    x + h / 2 * x1, v + h / 2 * v1, hyst + h / 2 * h1, f_mid
                )
                x3, v3, h3 = derivatives(
                    x + h / 2 * x2, v + h / 2 * v2, hyst + h / 2 * h2, f_mid
                )
                x4, v4, h4 = derivatives(x + h * x3, v + h * v3, hyst + h * h3, f_end)

                x = x + h / 6 * (x1 + 2 * x2 + 2 * x3 + x4)
                v = v + h / 6 * (v1 + 2 * v2 + 2 * v3 + v4)
                hyst = np.clip(
                    hyst + h / 6 * (h1 + 2 * h2 + 2 * h3 + h4),
                    -hyst_bound,
                    hyst_bound,
                )

            g_values = float(z) - x
            if is_single:
                return float(g_values[0])
            return np.asarray(g_values, dtype=np.float64)

        return limit_state_function

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
