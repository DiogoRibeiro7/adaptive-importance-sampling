"""Visualization and advanced analysis tools for Safe-ICE."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from ..typing import LimitStateFunction


class AdvancedAnalysis:
    """Advanced analysis tools for Safe-ICE results"""

    @staticmethod
    def analyze_component_evolution(results: dict[str, Any]) -> None:
        """Analyze how mixture components evolve during optimization"""
        history = results["history"]

        _fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Component count evolution
        axes[0, 0].plot(history["components"], "b-o")
        axes[0, 0].set_title("Component Count Evolution")
        axes[0, 0].set_xlabel("Iteration")
        axes[0, 0].set_ylabel("Number of Components")
        axes[0, 0].grid(True, alpha=0.3)

        # Sigma evolution
        axes[0, 1].semilogy(history["sigma"], "g-s")
        axes[0, 1].set_title("Smoothing Parameter Evolution")
        axes[0, 1].set_xlabel("Iteration")
        axes[0, 1].set_ylabel("σ")
        axes[0, 1].grid(True, alpha=0.3)

        # Lambda evolution
        axes[1, 0].plot(history["lambda"], "r-^")
        axes[1, 0].set_title("Cosine Annealing Schedule")
        axes[1, 0].set_xlabel("Iteration")
        axes[1, 0].set_ylabel("λ (Light-tail Weight)")
        axes[1, 0].grid(True, alpha=0.3)

        # CV evolution with target
        axes[1, 1].semilogy(history["cv"], "m-d")
        axes[1, 1].axhline(y=1.5, color="k", linestyle="--", alpha=0.7, label="Target")
        axes[1, 1].set_title("Coefficient of Variation")
        axes[1, 1].set_xlabel("Iteration")
        axes[1, 1].set_ylabel("CV")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def analyze_sample_distribution(
        results: dict[str, Any], problem_func: LimitStateFunction
    ) -> None:
        """Analyze final sample distribution (for 2D problems)"""
        if results["final_samples"].shape[1] != 2:
            print("Sample distribution analysis only available for 2D problems")
            return

        samples = results["final_samples"]
        g_values = results["final_g_values"]

        # Separate failure and safe samples
        failure_samples = samples[g_values <= 0]
        safe_samples = samples[g_values > 0]

        _fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Sample scatter plot
        axes[0].scatter(
            safe_samples[:, 0],
            safe_samples[:, 1],
            c="blue",
            alpha=0.6,
            s=20,
            label="Safe samples",
        )
        if len(failure_samples) > 0:
            axes[0].scatter(
                failure_samples[:, 0],
                failure_samples[:, 1],
                c="red",
                alpha=0.8,
                s=30,
                label="Failure samples",
            )

        # Add failure boundary (approximate)
        x_range = np.linspace(-6, 6, 100)
        y_range = np.linspace(-6, 6, 100)
        X_grid, Y_grid = np.meshgrid(x_range, y_range)
        Z_grid = np.zeros_like(X_grid)

        for i in range(len(x_range)):
            for j in range(len(y_range)):
                Z_grid[j, i] = problem_func(np.array([X_grid[j, i], Y_grid[j, i]]))

        axes[0].contour(
            X_grid, Y_grid, Z_grid, levels=[0], colors="black", linewidths=2
        )
        axes[0].set_xlabel("u₁")
        axes[0].set_ylabel("u₂")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_title("Final Sample Distribution")
        axes[0].axis("equal")

        # G-function histogram
        axes[1].hist(g_values, bins=50, alpha=0.7, color="skyblue", edgecolor="black")
        axes[1].axvline(
            x=0, color="red", linestyle="--", linewidth=2, label="Failure boundary"
        )
        axes[1].set_xlabel("g(u)")
        axes[1].set_ylabel("Frequency")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_title("Limit State Function Distribution")

        plt.tight_layout()
        plt.show()

        # Statistics
        failure_rate = len(failure_samples) / len(samples)
        print("Final sample statistics:")
        print(f"  Total samples: {len(samples)}")
        print(f"  Failure samples: {len(failure_samples)} ({failure_rate:.1%})")
        print(f"  G-function range: [{np.min(g_values):.3f}, {np.max(g_values):.3f}]")
