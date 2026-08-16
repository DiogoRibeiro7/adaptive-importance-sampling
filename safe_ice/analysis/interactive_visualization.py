"""Interactive visualization tools for Safe-ICE analysis."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    warnings.warn(
        "Plotly not available. Interactive visualizations disabled.", stacklevel=2
    )


class InteractiveVisualizer:
    """Interactive visualization tools using Plotly."""

    def __init__(self) -> None:
        """Initialize interactive visualizer."""
        if not PLOTLY_AVAILABLE:
            raise ImportError(
                "Plotly is required for interactive visualization. "
                "Install with: pip install plotly"
            )

    def plot_convergence_interactive(
        self, results: dict[str, Any], show: bool = True
    ) -> go.Figure:
        """
        Create interactive convergence plot.

        Parameters
        ----------
        results : dict
            Results from Safe-ICE run.
        show : bool
            Whether to display the plot.

        Returns
        -------
        plotly.graph_objects.Figure
            Interactive figure.
        """
        iterations = results.get("iterations", [])
        metrics = results.get("convergence_metrics", {})

        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Coefficient of Variation",
                "Number of Components",
                "Delta Evolution",
                "Failure Samples",
            ),
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
            ],
        )

        # Extract data
        iter_nums = list(range(1, len(iterations) + 1))
        cv_values = metrics.get("cv_values", [])
        delta_values = metrics.get("delta_values", [])
        K_values = [it["K"] for it in iterations]
        n_failures = [it.get("n_failures", 0) for it in iterations]

        # Plot 1: CV convergence
        fig.add_trace(
            go.Scatter(
                x=iter_nums,
                y=cv_values,
                mode="lines+markers",
                name="CV",
                line={"color": "blue", "width": 2},
                marker={"size": 8},
                hovertemplate="Iteration: %{x}<br>CV: %{y:.4f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

        # Add convergence threshold line
        fig.add_hline(
            y=0.05,
            line_dash="dash",
            line_color="red",
            annotation_text="Target CV",
            row=1,
            col=1,
        )

        # Plot 2: Number of components
        fig.add_trace(
            go.Bar(
                x=iter_nums,
                y=K_values,
                name="Components",
                marker_color="green",
                hovertemplate="Iteration: %{x}<br>K: %{y}<extra></extra>",
            ),
            row=1,
            col=2,
        )

        # Plot 3: Delta evolution
        fig.add_trace(
            go.Scatter(
                x=iter_nums,
                y=delta_values,
                mode="lines+markers",
                name="Delta",
                line={"color": "orange", "width": 2},
                marker={"size": 8},
                fill="tozeroy",
                hovertemplate="Iteration: %{x}<br>Delta: %{y:.3f}<extra></extra>",
            ),
            row=2,
            col=1,
        )

        # Plot 4: Failure samples
        fig.add_trace(
            go.Scatter(
                x=iter_nums,
                y=n_failures,
                mode="lines+markers",
                name="Failures",
                line={"color": "red", "width": 2},
                marker={"size": 8},
                hovertemplate="Iteration: %{x}<br>Failures: %{y}<extra></extra>",
            ),
            row=2,
            col=2,
        )

        # Update layout
        fig.update_layout(
            title="Safe-ICE Convergence Analysis",
            showlegend=False,
            height=600,
            hovermode="x unified",
        )

        # Update axes
        fig.update_xaxes(title_text="Iteration", row=2, col=1)
        fig.update_xaxes(title_text="Iteration", row=2, col=2)
        fig.update_yaxes(title_text="CV", row=1, col=1)
        fig.update_yaxes(title_text="K", row=1, col=2)
        fig.update_yaxes(title_text="Delta", row=2, col=1)
        fig.update_yaxes(title_text="Count", row=2, col=2)

        if show:
            fig.show()

        return fig

    def plot_sample_evolution_3d(
        self, results: dict[str, Any], show: bool = True
    ) -> go.Figure:
        """
        Create 3D visualization of sample evolution.

        Parameters
        ----------
        results : dict
            Results from Safe-ICE run.
        show : bool
            Whether to display the plot.

        Returns
        -------
        plotly.graph_objects.Figure
            3D interactive figure.
        """
        samples = results["final_samples"]
        g_values = results["final_g_values"]
        results.get("iterations", [])

        if samples.shape[1] < 2:
            warnings.warn("3D visualization requires at least 2D samples", stacklevel=2)
            return None

        # Use first 3 dimensions (or pad with zeros)
        if samples.shape[1] >= 3:
            x, y, z = samples[:, 0], samples[:, 1], samples[:, 2]
        else:
            x, y = samples[:, 0], samples[:, 1]
            z = np.zeros_like(x)

        # Create figure
        fig = go.Figure()

        # Color by g-values
        colorscale = [[0, "blue"], [0.5, "white"], [1, "red"]]

        # Add scatter plot
        fig.add_trace(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="markers",
                marker={
                    "size": 3,
                    "color": g_values,
                    "colorscale": colorscale,
                    "showscale": True,
                    "colorbar": {"title": "g(u)"},
                    "opacity": 0.7,
                },
                text=[f"g={g:.3f}" for g in g_values],
                hovertemplate="x: %{x:.2f}<br>y: %{y:.2f}<br>z: %{z:.2f}<br>%{text}<extra></extra>",
            )
        )

        # Add failure boundary (g=0)
        if samples.shape[1] >= 3:
            # Create mesh for failure surface (simplified)
            theta = np.linspace(0, 2 * np.pi, 20)
            phi = np.linspace(0, np.pi, 20)
            r = 3.0  # Approximate radius

            x_mesh = r * np.outer(np.sin(phi), np.cos(theta))
            y_mesh = r * np.outer(np.sin(phi), np.sin(theta))
            z_mesh = r * np.outer(np.cos(phi), np.ones(theta.size))

            fig.add_trace(
                go.Surface(
                    x=x_mesh,
                    y=y_mesh,
                    z=z_mesh,
                    opacity=0.2,
                    colorscale="Greys",
                    showscale=False,
                    name="Failure boundary",
                )
            )

        # Update layout
        fig.update_layout(
            title="3D Sample Distribution",
            scene={
                "xaxis_title": "u₁",
                "yaxis_title": "u₂",
                "zaxis_title": "u₃" if samples.shape[1] >= 3 else "z",
                "camera": {"eye": {"x": 1.5, "y": 1.5, "z": 1.5}},
            },
            height=700,
        )

        if show:
            fig.show()

        return fig

    def plot_mixture_evolution(
        self,
        results: dict[str, Any],
        dimension_indices: tuple[int, int] = (0, 1),
        show: bool = True,
    ) -> go.Figure:
        """
        Animate mixture evolution over iterations.

        Parameters
        ----------
        results : dict
            Results from Safe-ICE run.
        dimension_indices : tuple
            Dimensions to plot (for 2D projection).
        show : bool
            Whether to display the plot.

        Returns
        -------
        plotly.graph_objects.Figure
            Animated figure.
        """
        iterations = results.get("iterations", [])
        if not iterations:
            warnings.warn("No iteration data available", stacklevel=2)
            return None

        # Create frames for animation
        frames = []
        d1, d2 = dimension_indices

        # Generate grid for density evaluation
        x_range = np.linspace(-5, 5, 50)
        y_range = np.linspace(-5, 5, 50)
        X, Y = np.meshgrid(x_range, y_range)

        for i, iter_data in enumerate(iterations):
            if "parameters" not in iter_data:
                continue

            params = iter_data["parameters"]

            # Compute mixture density on grid (simplified)
            Z = np.zeros_like(X)

            # This is a simplified visualization
            # In practice, you'd evaluate the actual mixture density
            for k in range(params.K):
                if d1 < params.mu.shape[1] and d2 < params.mu.shape[1]:
                    mu_k = params.mu[k, [d1, d2]]
                    # Approximate with Gaussian
                    for ii in range(X.shape[0]):
                        for jj in range(X.shape[1]):
                            point = np.array([X[ii, jj], Y[ii, jj]])
                            dist = np.linalg.norm(point - mu_k * 3)
                            Z[ii, jj] += params.pi[k] * np.exp(-0.5 * dist**2)

            frame = go.Frame(
                data=[
                    go.Contour(
                        x=x_range, y=y_range, z=Z, colorscale="Viridis", showscale=False
                    )
                ],
                name=f"Iteration {i + 1}",
            )
            frames.append(frame)

        # Create initial figure
        fig = go.Figure(data=[frames[0].data[0]] if frames else [], frames=frames)

        # Add animation controls
        fig.update_layout(
            title="Mixture Evolution Animation",
            xaxis_title=f"u_{d1 + 1}",
            yaxis_title=f"u_{d2 + 1}",
            updatemenus=[
                {
                    "type": "buttons",
                    "showactive": False,
                    "buttons": [
                        {
                            "label": "Play",
                            "method": "animate",
                            "args": [
                                None,
                                {
                                    "frame": {"duration": 500, "redraw": True},
                                    "fromcurrent": True,
                                },
                            ],
                        },
                        {
                            "label": "Pause",
                            "method": "animate",
                            "args": [
                                [None],
                                {
                                    "frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate",
                                },
                            ],
                        },
                    ],
                }
            ],
            sliders=[
                {
                    "steps": [
                        {
                            "args": [
                                [f.name],
                                {
                                    "frame": {"duration": 0, "redraw": True},
                                    "mode": "immediate",
                                },
                            ],
                            "label": f.name,
                            "method": "animate",
                        }
                        for f in frames
                    ],
                    "active": 0,
                    "len": 0.9,
                    "x": 0.05,
                    "xanchor": "left",
                    "y": -0.1,
                    "yanchor": "top",
                }
            ],
        )

        if show:
            fig.show()

        return fig

    def plot_realtime_monitor(self, show: bool = True) -> go.FigureWidget:
        """
        Create real-time monitoring dashboard.

        Returns
        -------
        plotly.graph_objects.FigureWidget
            Interactive widget for real-time updates.
        """
        # Create widget figure
        fig = go.FigureWidget(
            make_subplots(
                rows=2,
                cols=2,
                subplot_titles=(
                    "Failure Probability",
                    "Convergence (CV)",
                    "Components (K)",
                    "Sample Efficiency",
                ),
            )
        )

        # Initialize empty traces
        fig.add_trace(
            go.Scatter(x=[], y=[], mode="lines+markers", name="Pf"), row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=[], y=[], mode="lines+markers", name="CV"), row=1, col=2
        )
        fig.add_trace(go.Bar(x=[], y=[], name="K"), row=2, col=1)
        fig.add_trace(
            go.Scatter(x=[], y=[], mode="lines+markers", name="ESS"), row=2, col=2
        )

        # Update layout
        fig.update_layout(
            title="Real-Time Safe-ICE Monitor", height=600, showlegend=False
        )

        if show:
            fig.show()

        return fig

    def create_parameter_sensitivity_plot(
        self,
        parameter_ranges: dict[str, list[float]],
        results_list: list[dict[str, Any]],
        show: bool = True,
    ) -> go.Figure:
        """
        Create parameter sensitivity analysis plot.

        Parameters
        ----------
        parameter_ranges : dict
            Dictionary of parameter names and values tested.
        results_list : list
            List of results for each parameter combination.
        show : bool
            Whether to display the plot.

        Returns
        -------
        plotly.graph_objects.Figure
            Interactive sensitivity plot.
        """
        # Extract failure probabilities
        pf_values = [r.get("pf", 0) for r in results_list]

        # Create parallel coordinates plot
        data_dict = parameter_ranges.copy()
        data_dict["Failure Probability"] = pf_values

        fig = go.Figure(
            data=go.Parcoords(
                line={
                    "color": pf_values,
                    "colorscale": "Viridis",
                    "showscale": True,
                    "colorbar": {"title": "Pf"},
                },
                dimensions=[
                    {
                        "label": name,
                        "values": values,
                        "range": [min(values), max(values)],
                    }
                    for name, values in data_dict.items()
                ],
            )
        )

        fig.update_layout(title="Parameter Sensitivity Analysis", height=500)

        if show:
            fig.show()

        return fig


def create_interactive_dashboard(
    results: dict[str, Any],
    limit_state_func: Any | None = None,  # noqa: ARG001 - reserved for plotting the failure boundary
) -> None:
    """
    Create comprehensive interactive dashboard.

    Parameters
    ----------
    results : dict
        Results from Safe-ICE run.
    limit_state_func : callable, optional
        Limit state function for additional analysis.
    """
    if not PLOTLY_AVAILABLE:
        warnings.warn("Plotly not available. Cannot create dashboard.", stacklevel=2)
        return

    InteractiveVisualizer()

    # Create dashboard with tabs
    from plotly import graph_objects as go
    from plotly.subplots import make_subplots

    # Create figure with subplots
    fig = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=(
            "Convergence History",
            "Sample Distribution",
            "Component Evolution",
            "Weight Distribution",
            "Failure Probability",
            "Performance Metrics",
        ),
        specs=[
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "bar"}, {"type": "histogram"}],
            [{"type": "scatter"}, {"type": "indicator"}],
        ],
        row_heights=[0.35, 0.35, 0.3],
        vertical_spacing=0.12,
        horizontal_spacing=0.15,
    )

    # Extract data
    iterations = results.get("iterations", [])
    metrics = results.get("convergence_metrics", {})
    samples = results.get("final_samples", np.array([]))
    weights = results.get("final_weights", np.array([]))
    g_values = results.get("final_g_values", np.array([]))

    # 1. Convergence plot
    iter_nums = list(range(1, len(iterations) + 1))
    cv_values = metrics.get("cv_values", [])

    fig.add_trace(
        go.Scatter(
            x=iter_nums,
            y=cv_values,
            mode="lines+markers",
            name="CV",
            line={"color": "blue"},
        ),
        row=1,
        col=1,
    )

    # 2. Sample distribution (2D projection if needed)
    if samples.shape[1] >= 2:
        failure_mask = g_values <= 0
        fig.add_trace(
            go.Scatter(
                x=samples[~failure_mask, 0],
                y=samples[~failure_mask, 1],
                mode="markers",
                marker={"size": 3, "color": "blue", "opacity": 0.5},
                name="Safe",
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=samples[failure_mask, 0],
                y=samples[failure_mask, 1],
                mode="markers",
                marker={"size": 5, "color": "red", "opacity": 0.8},
                name="Failure",
            ),
            row=1,
            col=2,
        )

    # 3. Component evolution
    K_values = [it["K"] for it in iterations]
    fig.add_trace(go.Bar(x=iter_nums, y=K_values, marker_color="green"), row=2, col=1)

    # 4. Weight distribution
    fig.add_trace(
        go.Histogram(x=weights[weights > 0], nbinsx=50, marker_color="orange"),
        row=2,
        col=2,
    )

    # 5. Failure probability evolution
    pf_evolution = []
    for i in range(len(iterations)):
        # Approximate pf at each iteration
        iter_weights = weights[: min((i + 1) * results.get("N", 1000), len(weights))]
        iter_g = g_values[: len(iter_weights)]
        if np.sum(iter_weights) > 0:
            pf_i = np.sum((iter_g <= 0) * iter_weights) / np.sum(iter_weights)
            pf_evolution.append(pf_i)

    fig.add_trace(
        go.Scatter(
            x=iter_nums[: len(pf_evolution)],
            y=pf_evolution,
            mode="lines+markers",
            line={"color": "red"},
        ),
        row=3,
        col=1,
    )

    # 6. Performance indicator
    final_pf = (
        np.sum((g_values <= 0) * weights) / np.sum(weights)
        if np.sum(weights) > 0
        else 0
    )

    fig.add_trace(
        go.Indicator(
            value=final_pf,
            mode="gauge+number",
            title={"text": "Final Pf"},
            number={"valueformat": ".2e"},
            gauge={"axis": {"range": [0, 0.1]}},
        ),
        row=3,
        col=2,
    )

    # Update layout
    fig.update_layout(title="Safe-ICE Analysis Dashboard", height=900, showlegend=False)

    # Update axes labels
    fig.update_xaxes(title_text="Iteration", row=1, col=1)
    fig.update_xaxes(title_text="u₁", row=1, col=2)
    fig.update_xaxes(title_text="Iteration", row=2, col=1)
    fig.update_xaxes(title_text="Weight", row=2, col=2)
    fig.update_xaxes(title_text="Iteration", row=3, col=1)

    fig.update_yaxes(title_text="CV", row=1, col=1)
    fig.update_yaxes(title_text="u₂", row=1, col=2)
    fig.update_yaxes(title_text="K", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=2)
    fig.update_yaxes(title_text="Pf", row=3, col=1)

    fig.show()

    print("\nDashboard created successfully!")
    print("Interactive features:")
    print("- Hover over points for details")
    print("- Zoom/pan with mouse")
    print("- Double-click to reset view")
    print("- Click legend items to toggle visibility")
