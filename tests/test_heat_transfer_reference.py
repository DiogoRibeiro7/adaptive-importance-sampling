"""The heat conduction problem of section 4.5, checked against the paper.

This module was the least covered in the package and had never been checked
against an independent answer. It did not work: the temperature field was
produced by an explicit relaxation, ``T += 0.01 * (laplacian + Q)``, run for
1000 sweeps and clamped to +/-1e6. An elliptic problem has no time derivative
to march, and that fixed pseudo-step was about sixteen times the explicit
stability limit for the default grid, so it diverged -- 380 of 441 nodes ended
pinned at the clamp, and the limit state returned exactly ``threshold`` for
every input, with no dependence on the random field at all.

Four parameters also disagreed with the paper: the correlation length was 0.5
against ``l = 0.2``, the failure threshold 100 against the 10 of equation (48),
the covariance was ``exp(-r/l)`` against equation (46)'s ``exp(-r^2/l^2)``, and
the heat source's y extent was computed and then discarded, leaving a
full-height strip instead of the square A.

With those corrected and the field solved directly, Safe-ICE estimates the
failure probability at 2.83e-07 against the paper's 4.69e-07 -- a factor of
0.60, using finite differences on a 21x21 grid where the paper uses finite
elements with 25040 triangles.
"""

from __future__ import annotations

import numpy as np
import pytest

from safe_ice import SafeICE
from safe_ice.problems.heat_transfer import HeatTransferProblem

# Section 4.5, estimated by subset simulation over 50 runs.
PAPER_REFERENCE_PF = 4.69e-07


class TestParametersMatchThePaper:
    """Each of these was wrong, and none was covered by a test."""

    def test_correlation_length(self) -> None:
        assert float(HeatTransferProblem().l) == pytest.approx(0.2)

    def test_failure_threshold(self) -> None:
        """Equation (48): g(u) = 10 - mean temperature on B."""
        assert float(HeatTransferProblem().threshold) == pytest.approx(10.0)

    def test_heat_source_and_field_statistics(self) -> None:
        problem = HeatTransferProblem()
        heat_source = float(problem.Q)
        assert heat_source == pytest.approx(2000.0)
        assert float(problem.field_std) == pytest.approx(0.3)
        assert problem.n_terms == 10

    def test_covariance_is_squared_exponential(self) -> None:
        """Equation (46) is exp(-r^2/l^2); the code had exp(-r/l)."""
        problem = HeatTransferProblem(grid_size=5)
        points = problem.grid_points
        r = float(np.linalg.norm(points[0] - points[-1]))

        squared_exponential = float(np.exp(-(r**2) / problem.l**2))
        exponential = float(np.exp(-r / problem.l))

        sq = np.sum((points[:, None, :] - points[None, :, :]) ** 2, axis=2)
        kernel = np.exp(-sq / problem.l**2)

        assert float(kernel[0, -1]) == pytest.approx(squared_exponential)
        assert float(kernel[0, -1]) != pytest.approx(exponential)


class TestTheSolver:
    """The field is now a direct solve, so it can be checked exactly."""

    @pytest.mark.parametrize("grid_size", [21, 41])
    def test_reproduces_the_analytic_solution(self, grid_size: int) -> None:
        """Uniform conductivity and a uniform source reduce this to 1-D.

        With -T'' = q, zero gradient at y = -0.5 and T = 0 at y = +0.5, the
        solution is T(y) = q (0.375 - y^2/2 - y/2). The source is spread so
        that its integral is Q times the region area, and here the region is
        the whole unit domain, so q = Q. Nothing about the previous relaxation
        could have passed this.
        """

        class UniformSource(HeatTransferProblem):
            def _region_mask(self, *bounds: float):
                return np.ones_like(self.X, dtype=bool)

        problem = UniformSource(grid_size=grid_size)
        problem.Q = 1.0
        temperature = problem.solve_heat_equation(np.ones((grid_size, grid_size)))

        # The solver spreads Q over the nominal area of A, so recover the
        # density it actually applied rather than assuming one.
        h = 1.0 / (grid_size - 1)
        n_source = grid_size * grid_size
        density = problem.Q * (0.1 * 0.1) / (n_source * h * h)

        y = np.linspace(-0.5, 0.5, grid_size)
        exact = density * (0.375 - y**2 / 2.0 - y / 2.0)
        column = temperature[:, grid_size // 2]

        assert column == pytest.approx(exact, rel=1e-9)

    def test_dirichlet_edge_is_zero(self) -> None:
        """Figure 11: T = 0 on the top edge, zero gradient on the other three."""
        problem = HeatTransferProblem()
        temperature = problem.solve_heat_equation(
            problem.generate_permeability_field(np.zeros(10))
        )

        assert temperature[-1, :] == pytest.approx(0.0, abs=1e-12)
        # The insulated edges are the far side of the domain from the only
        # sink, so they must be hot, not pinned to zero.
        assert float(np.min(temperature[0, 1:-1])) > 0.0
        assert np.all(np.isfinite(temperature))

    def test_extreme_samples_stay_finite(self) -> None:
        """The estimator's heavy tails reach far enough to underflow kappa.

        ``kappa = exp(a + b f)`` goes to zero for a large enough negative
        field, which leaves the conduction matrix singular and the solve
        returning NaN. A run would then fail outright on the sample that hit
        it, so the exponent is clipped.
        """
        limit_state = HeatTransferProblem().create_limit_state_function()
        rng = np.random.default_rng(0)
        for scale in (1.0, 10.0, 100.0, 1000.0):
            values = np.asarray(limit_state(rng.standard_normal((10, 10)) * scale))
            assert np.all(np.isfinite(values)), f"non-finite at scale {scale}"

    def test_field_is_finite_and_bounded(self) -> None:
        """The relaxation left 380 of 441 nodes pinned at a +/-1e6 clamp."""
        problem = HeatTransferProblem()
        rng = np.random.default_rng(0)
        for _ in range(5):
            temperature = problem.solve_heat_equation(
                problem.generate_permeability_field(rng.standard_normal(10))
            )
            assert np.all(np.isfinite(temperature))
            assert float(np.max(np.abs(temperature))) < 1e4


class TestRegionsAndGridIndependence:
    @pytest.mark.parametrize("grid_size", [21, 31, 41, 51])
    def test_regions_are_squares(self, grid_size: int) -> None:
        """The bounds are exact multiples that a linspace can miss by an ulp."""
        problem = HeatTransferProblem(grid_size=grid_size)
        source = problem._region_mask(0.2, 0.3, 0.2, 0.3)
        region_b = problem._region_mask(-0.3, -0.2, -0.3, -0.2)

        for mask in (source, region_b):
            count = int(np.count_nonzero(mask))
            side = round(float(np.sqrt(count)))
            assert side * side == count, f"{count} nodes is not a square"
            assert side >= 3

    @pytest.mark.slow
    @pytest.mark.parametrize("grid_size", [21, 31, 41, 51])
    def test_temperature_on_b_is_grid_independent(self, grid_size: int) -> None:
        """Assigning Q per node made the heat input depend on the grid.

        A closed region picks up an extra row of nodes on each side, so the
        discrete input was ((0.1 + h) / 0.1)^2 times too large: 2.25x at
        grid_size 21 against 1.44x at 51. Spreading Q over the region's area
        removes it, and the average temperature on B settles at 4.553.
        """
        problem = HeatTransferProblem(grid_size=grid_size)
        temperature = problem.solve_heat_equation(
            problem.generate_permeability_field(np.zeros(10))
        )
        region_b = problem._region_mask(-0.3, -0.2, -0.3, -0.2)

        assert float(temperature[region_b].mean()) == pytest.approx(4.553, abs=0.01)


class TestLimitState:
    def test_depends_on_the_random_field(self) -> None:
        """It used to return exactly ``threshold`` for every input."""
        problem = HeatTransferProblem()
        limit_state = problem.create_limit_state_function()
        rng = np.random.default_rng(1)
        values = np.asarray(limit_state(rng.standard_normal((40, 10))))

        assert float(values.std()) > 0.1
        assert len(np.unique(values)) == values.size

    def test_input_dimension_is_checked(self) -> None:
        limit_state = HeatTransferProblem().create_limit_state_function()
        with pytest.raises(ValueError, match="dimension"):
            limit_state(np.zeros((3, 4)))

    @pytest.mark.slow
    def test_estimate_is_near_the_paper_reference(self) -> None:
        """Within an order of magnitude, given a different discretisation.

        The paper solves the field with finite elements over 25040 triangles;
        this is second-order finite differences on 21x21. The median over
        seeds is about 3e-07 against the paper's 4.69e-07.

        The median rather than each run: on this problem the estimator
        collapses on a minority of seeds. Sigma falls too far in the first two
        iterations -- 1 to 0.37 to 0.13 -- and then stalls, the weight CV never
        comes down from 6-8 against a target of 4, and the run ends many orders
        of magnitude low. Two of six seeds did that in one measurement. It is
        a property of the sigma schedule of equation (10) on a 10-dimensional
        problem with a smooth limit state, not of this module; see ROADMAP.md.
        """
        limit_state = HeatTransferProblem().create_limit_state_function()

        estimates = []
        for seed in range(3):
            ice = SafeICE(
                limit_state_function=limit_state,
                dimension=10,
                N=1000,
                max_iterations=15,
                random_state=seed,
            )
            pf, _ = ice.run(verbose=False)
            estimates.append(pf)

        median = float(np.median(estimates))
        assert PAPER_REFERENCE_PF / 10 < median < PAPER_REFERENCE_PF * 10, (
            f"median {median:.3e} against the paper's {PAPER_REFERENCE_PF:.3e}; "
            f"estimates {estimates}"
        )
