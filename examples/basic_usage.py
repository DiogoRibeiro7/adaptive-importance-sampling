"""Basic usage example of Safe-ICE algorithm."""

import sys

import matplotlib.pyplot as plt
import numpy as np

# This script prints mathematical notation. Windows consoles still default to a
# legacy code page, where that would raise UnicodeEncodeError, so make sure
# stdout can carry it.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from safe_ice import SafeICE


def main():
    """Demonstrate basic Safe-ICE usage with a simple sphere problem."""

    # Define a simple limit state function
    def sphere_limit_state(u):
        """
        Spherical limit state function.
        Failure occurs when ||u|| > beta.
        """
        beta = 3.0
        return beta - np.linalg.norm(u, axis=-1)

    print("=" * 60)
    print("Safe-ICE Basic Usage Example")
    print("=" * 60)
    print("\nProblem: Spherical limit state with beta = 3.0")
    print("Dimension: 2")
    print("Analytical solution: P(||U|| > 3) ≈ 0.0111")

    # Initialize Safe-ICE
    ice = SafeICE(
        limit_state_function=sphere_limit_state,
        dimension=2,
        N=1000,  # Samples per iteration
        max_iterations=10,  # Maximum iterations
        delta_target=3.0,  # Target rarity parameter
        delta_star=1.5,  # Rarity increment threshold
    )

    print("\nRunning Safe-ICE algorithm...")
    print("-" * 40)

    # Run the algorithm
    pf, results = ice.run(verbose=True)

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Estimated failure probability: {pf:.6e}")
    print("Analytical failure probability: 1.11e-2")
    print(f"Relative error: {abs(pf - 0.0111) / 0.0111:.2%}")
    print(f"\nTotal samples used: {len(results['final_samples'])}")
    print(f"Number of iterations: {len(results['iterations'])}")

    # Analyze convergence
    metrics = results["convergence_metrics"]
    print("\nConvergence metrics:")
    print(f"  Final CV: {metrics['cv_values'][-1]:.4f}")
    print(f"  Final delta: {metrics['delta_values'][-1]:.4f}")

    # Component evolution
    print("\nComponent evolution:")
    for i, iter_data in enumerate(results["iterations"]):
        print(f"  Iteration {i + 1}: K = {iter_data['K']} components")

    # Visualize results for 2D problem
    if results["final_samples"].shape[1] == 2:
        visualize_2d_results(results, sphere_limit_state)


def visualize_2d_results(results, limit_state_func):  # noqa: ARG001 - kept for call-site symmetry
    """Visualize results for 2D problems."""
    samples = results["final_samples"]
    g_values = results["final_g_values"]

    # Separate failure and safe samples
    failure_mask = g_values <= 0
    safe_samples = samples[~failure_mask]
    failure_samples = samples[failure_mask]

    # Create figure
    _fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: Sample distribution
    ax = axes[0]
    if len(safe_samples) > 0:
        ax.scatter(
            safe_samples[:, 0],
            safe_samples[:, 1],
            c="blue",
            alpha=0.5,
            s=10,
            label="Safe",
        )
    if len(failure_samples) > 0:
        ax.scatter(
            failure_samples[:, 0],
            failure_samples[:, 1],
            c="red",
            alpha=0.8,
            s=20,
            label="Failure",
        )

    # Add failure boundary
    theta = np.linspace(0, 2 * np.pi, 100)
    r = 3.0  # beta value
    x_boundary = r * np.cos(theta)
    y_boundary = r * np.sin(theta)
    ax.plot(x_boundary, y_boundary, "k-", linewidth=2, label="Failure boundary")

    ax.set_xlabel("u₁")
    ax.set_ylabel("u₂")
    ax.set_title("Sample Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis("equal")
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)

    # Plot 2: Convergence
    ax = axes[1]
    metrics = results["convergence_metrics"]
    iterations = range(1, len(metrics["cv_values"]) + 1)

    ax.plot(iterations, metrics["cv_values"], "b-o", label="CV")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Coefficient of Variation")
    ax.set_title("Convergence History")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Plot 3: G-values histogram
    ax = axes[2]
    ax.hist(g_values, bins=50, alpha=0.7, color="skyblue", edgecolor="black")
    ax.axvline(x=0, color="red", linestyle="--", linewidth=2, label="Failure threshold")
    ax.set_xlabel("g(u)")
    ax.set_ylabel("Frequency")
    ax.set_title("Limit State Values Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("\nVisualization complete!")
    print(
        f"Failure samples: {len(failure_samples)} / {len(samples)} = {len(failure_samples) / len(samples):.1%}"
    )


if __name__ == "__main__":
    main()
