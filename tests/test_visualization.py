"""Regression tests for the Matplotlib analysis helpers."""

from __future__ import annotations

import numpy as np
import pytest

from safe_ice import SafeICE
from safe_ice.analysis.visualization import AdvancedAnalysis


def test_component_evolution_uses_recorded_lambda_and_threshold() -> None:
    ice = SafeICE(
        limit_state_function=lambda u: 3.0 - np.linalg.norm(u, axis=-1),
        dimension=2,
        N=100,
        max_iterations=3,
        delta_star=0.75,
        random_state=4,
    )
    _pf, results = ice.run(verbose=False)

    figure = AdvancedAnalysis.analyze_component_evolution(results, show=False)

    lambda_axis = figure.axes[2]
    assert list(lambda_axis.lines[0].get_ydata()) == pytest.approx(
        results["history"]["lambda_val"]
    )

    cv_axis = figure.axes[3]
    target_line = cv_axis.lines[1]
    assert list(target_line.get_ydata()) == pytest.approx([0.75, 0.75])
