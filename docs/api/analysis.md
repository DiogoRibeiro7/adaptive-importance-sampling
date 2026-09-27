# Analysis

Tools for checking an estimate and looking inside a run.
[Running and diagnosing](../guide/running.md#diagnostics) shows them in use.

::: safe_ice.PerformanceEvaluator

::: safe_ice.AdvancedAnalysis

## Interactive plots

The Plotly helpers live in `safe_ice.analysis.interactive_visualization` and are
not re-exported from the top-level package, because Plotly is an optional
dependency. Install the `viz` extra to use them:

```bash
pip install "safe-ice[viz] @ git+https://github.com/DiogoRibeiro7/adaptive-importance-sampling.git"
```

::: safe_ice.analysis.interactive_visualization.InteractiveVisualizer
    options:
      heading_level: 3

::: safe_ice.analysis.interactive_visualization.create_interactive_dashboard
    options:
      heading_level: 3
