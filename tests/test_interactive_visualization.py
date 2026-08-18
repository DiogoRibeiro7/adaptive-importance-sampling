"""Smoke tests for the Plotly visualiser.

This module was 165 statements at 0% coverage. It draws figures rather than
computing results, so a defect here misleads rather than corrupts, and the
tests are correspondingly shallow: each entry point is called on the output of
a real run and the figure it returns is inspected for the traces it claims to
draw. That is enough to catch the failure these had gone uncaught for -- an
exception on a key that ``run()`` does not actually produce.

``show=False`` throughout: a test must never open a browser window.
"""

from __future__ import annotations

import numpy as np
import pytest

from safe_ice import SafeICE

plotly = pytest.importorskip(
    "plotly", reason="interactive visualisation needs the viz extra"
)

from safe_ice.analysis.interactive_visualization import (  # noqa: E402
    InteractiveVisualizer,
    create_interactive_dashboard,
)


@pytest.fixture(scope="module")
def results():
    """A real run, so the figures are fed the keys run() actually returns."""
    ice = SafeICE(
        limit_state_function=lambda u: 3.0 - np.linalg.norm(u, axis=-1),
        dimension=3,
        N=200,
        max_iterations=4,
        random_state=0,
    )
    _pf, output = ice.run(verbose=False)
    return output


@pytest.fixture
def visualizer():
    return InteractiveVisualizer()


class TestConvergencePlot:
    def test_returns_a_figure_with_traces(self, visualizer, results) -> None:
        figure = visualizer.plot_convergence_interactive(results, show=False)
        assert len(figure.data) > 0

    def test_does_not_mutate_the_results(self, visualizer, results) -> None:
        before = set(results)
        visualizer.plot_convergence_interactive(results, show=False)
        assert set(results) == before


class TestSampleEvolution:
    def test_three_dimensional_run(self, visualizer, results) -> None:
        figure = visualizer.plot_sample_evolution_3d(results, show=False)
        assert len(figure.data) > 0

    def test_two_dimensional_run_is_handled(self, visualizer) -> None:
        """The 3-D plot has to cope with a problem that has only two axes."""
        ice = SafeICE(
            limit_state_function=lambda u: 3.0 - np.linalg.norm(u, axis=-1),
            dimension=2,
            N=200,
            max_iterations=3,
            random_state=1,
        )
        _pf, output = ice.run(verbose=False)

        figure = visualizer.plot_sample_evolution_3d(output, show=False)
        assert figure is not None


class TestMixtureEvolution:
    def test_default_dimensions(self, visualizer, results) -> None:
        figure = visualizer.plot_mixture_evolution(results, show=False)
        assert figure is not None

    def test_explicit_dimension_pair(self, visualizer, results) -> None:
        figure = visualizer.plot_mixture_evolution(
            results, dimension_indices=(1, 2), show=False
        )
        assert figure is not None


class TestParameterSensitivity:
    def test_plots_one_series_per_parameter(self, visualizer, results) -> None:
        figure = visualizer.create_parameter_sensitivity_plot(
            {"N": [100.0, 200.0, 400.0]},
            [results, results, results],
            show=False,
        )
        assert figure is not None


class TestDashboard:
    def test_builds_without_a_limit_state(self, results) -> None:
        assert create_interactive_dashboard(results) is None


class TestRealtimeMonitor:
    def test_returns_a_widget(self, visualizer) -> None:
        """FigureWidget needs anywidget in some Plotly versions; skip if absent."""
        try:
            widget = visualizer.plot_realtime_monitor(show=False)
        except (ImportError, ValueError) as exc:  # pragma: no cover - env dependent
            pytest.skip(f"FigureWidget unavailable: {exc}")
        assert widget is not None
