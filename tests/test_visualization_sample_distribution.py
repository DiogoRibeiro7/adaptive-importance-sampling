"""Regression tests for Matplotlib sample-distribution analysis."""

from __future__ import annotations

import numpy as np

from safe_ice import SafeICE
from safe_ice.analysis.visualization import AdvancedAnalysis


def test_sample_distribution_supports_batch_only_limit_state() -> None:
    def batch_only(u: np.ndarray) -> np.ndarray:
        if u.ndim != 2:
            raise ValueError("batch input required")
        return 3.0 - np.linalg.norm(u, axis=1)

    ice = SafeICE(
        limit_state_function=batch_only,
        dimension=2,
        N=100,
        max_iterations=2,
        random_state=5,
    )
    _pf, results = ice.run(verbose=False)

    figure = AdvancedAnalysis.analyze_sample_distribution(
        results, batch_only, show=False
    )

    assert figure is not None
    assert len(figure.axes) == 2
    assert len(figure.axes[0].collections) > 0
