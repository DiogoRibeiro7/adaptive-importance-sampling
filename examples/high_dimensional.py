"""Example of Safe-ICE for high-dimensional problems."""

import time

import matplotlib.pyplot as plt
import numpy as np

from safe_ice import SafeICE
from safe_ice.problems.benchmarks import BenchmarkProblems


def linear_limit_state_hd(dimension):
    """Create a linear limit state function in high dimensions."""

    def g(u):
        # Linear combination with decreasing weights
        weights = np.ones(dimension) / np.sqrt(dimension)
        beta = 3.0
        return beta - np.dot(u, weights)

    return g


def nonlinear_limit_state_hd(dimension):
    """Create a nonlinear limit state function in high dimensions."""

    def g(u):
        # Nonlinear combination
        beta = 4.5
        # Sum of squares normalized by dimension
        term1 = np.sum(u**2, axis=-1) / dimension
        # Cross terms
        if u.ndim == 2:
            term2 = 0.1 * np.sum(u[:, :-1] * u[:, 1:], axis=1)
        else:
            term2 = 0.1 * np.sum(u[:-1] * u[1:])
        return beta - np.sqrt(term1) - term2

    return g


def sphere_limit_state_hd(dimension):
    """Create a spherical limit state function in high dimensions."""

    def g(u):
        # Scaled sphere
        beta = np.sqrt(dimension) * 2.5
        return beta - np.linalg.norm(u, axis=-1)

    return g


def run_dimension_study():
    """Study Safe-ICE performance across different dimensions."""

    print("=" * 70)
    print("HIGH-DIMENSIONAL RELIABILITY ANALYSIS")
    print("=" * 70)

    dimensions = [2, 5, 10, 20, 50, 100]
    results = []

    for d in dimensions:
        print(f"\n{'=' * 70}")
        print(f"DIMENSION: {d}")
        print("-" * 70)

        # Use nonlinear limit state
        g = nonlinear_limit_state_hd(d)

        # Adjust parameters based on dimension
        n_samples = min(1000 * d, 10000)  # Scale with dimension
        k0 = min(d * 5, 50)  # Initial components

        print("Configuration:")
        print(f"  Samples per iteration: {n_samples}")
        print(f"  Initial components: {k0}")
        print("  Max iterations: 20")

        # Initialize Safe-ICE
        ice = SafeICE(
            limit_state_function=g,
            dimension=d,
            N=n_samples,
            K0=k0,
            max_iterations=20,
            delta_target=3.0,
            delta_star=1.5,
        )

        # Run algorithm
        print("\nRunning Safe-ICE...")
        start_time = time.time()
        pf, ice_results = ice.run(verbose=False)
        elapsed_time = time.time() - start_time

        # Store results
        result = {
            "dimension": d,
            "pf": pf,
            "time": elapsed_time,
            "total_samples": len(ice_results["final_samples"]),
            "iterations": len(ice_results["iterations"]),
            "final_K": ice_results["iterations"][-1]["K"],
            "cv": ice_results["convergence_metrics"]["cv_values"][-1],
        }
        results.append(result)

        print("\nResults:")
        print(f"  Failure probability: {pf:.6e}")
        print(f"  Total samples: {result['total_samples']}")
        print(f"  Iterations: {result['iterations']}")
        print(f"  Final components: {result['final_K']}")
        print(f"  Final CV: {result['cv']:.4f}")
        print(f"  Time: {elapsed_time:.2f} seconds")

    return results


def test_specific_problems():
    """Test specific high-dimensional benchmark problems."""

    print("\n" + "=" * 70)
    print("SPECIFIC HIGH-DIMENSIONAL BENCHMARKS")
    print("=" * 70)

    problems = BenchmarkProblems()

    # Test 1: Nonlinear oscillator (10D)
    print(f"\n{'=' * 70}")
    print("Nonlinear Oscillator (10D)")
    print("-" * 70)

    g = problems.nonlinear_oscillator()
    ice = SafeICE(
        limit_state_function=g, dimension=10, N=3000, K0=30, max_iterations=15
    )

    pf, results = ice.run(verbose=True)
    print(f"\nFailure probability: {pf:.6e}")

    # Test 2: Two-mode opposite directions (various dimensions)
    for d in [10, 20, 50]:
        print(f"\n{'=' * 70}")
        print(f"Two-Mode Opposite Directions ({d}D)")
        print("-" * 70)

        g = problems.two_mode_opposite_directions(dimension=d)
        ice = SafeICE(
            limit_state_function=g, dimension=d, N=2000, K0=20, max_iterations=10
        )

        pf, results = ice.run(verbose=False)
        print(f"Failure probability: {pf:.6e}")
        print(f"Final components: {results['iterations'][-1]['K']}")


def visualize_dimension_scaling(results):
    """Visualize how performance scales with dimension."""

    dimensions = [r["dimension"] for r in results]

    _fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Plot 1: Failure probability vs dimension
    ax = axes[0, 0]
    pf_values = [r["pf"] for r in results]
    ax.semilogy(dimensions, pf_values, "b-o")
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Failure Probability")
    ax.set_title("Failure Probability Scaling")
    ax.grid(True, alpha=0.3)

    # Plot 2: Total samples vs dimension
    ax = axes[0, 1]
    samples = [r["total_samples"] for r in results]
    ax.plot(dimensions, samples, "r-s")
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Total Samples")
    ax.set_title("Sample Requirement")
    ax.grid(True, alpha=0.3)

    # Plot 3: Computational time vs dimension
    ax = axes[0, 2]
    times = [r["time"] for r in results]
    ax.plot(dimensions, times, "g-^")
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Computational Time")
    ax.grid(True, alpha=0.3)

    # Plot 4: Number of iterations vs dimension
    ax = axes[1, 0]
    iterations = [r["iterations"] for r in results]
    ax.plot(dimensions, iterations, "m-d")
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Iterations")
    ax.set_title("Convergence Speed")
    ax.grid(True, alpha=0.3)

    # Plot 5: Final components vs dimension
    ax = axes[1, 1]
    final_K = [r["final_K"] for r in results]
    ax.plot(dimensions, final_K, "c-*")
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Final K")
    ax.set_title("Final Number of Components")
    ax.grid(True, alpha=0.3)

    # Plot 6: CV vs dimension
    ax = axes[1, 2]
    cv_values = [r["cv"] for r in results]
    ax.plot(dimensions, cv_values, "y-p")
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Coefficient of Variation")
    ax.set_title("Estimation Accuracy")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Print scaling summary
    print("\n" + "=" * 70)
    print("DIMENSION SCALING SUMMARY")
    print("=" * 70)
    print(
        f"{'Dim':<6} {'Pf':<12} {'Samples':<10} {'Time(s)':<10} {'K_final':<8} {'CV':<8}"
    )
    print("-" * 70)
    for r in results:
        print(
            f"{r['dimension']:<6} {r['pf']:<12.2e} {r['total_samples']:<10} "
            f"{r['time']:<10.2f} {r['final_K']:<8} {r['cv']:<8.4f}"
        )


def main():
    """Run high-dimensional examples."""

    # Run dimension scaling study
    results = run_dimension_study()

    # Visualize results
    visualize_dimension_scaling(results)

    # Test specific problems
    test_specific_problems()

    print("\n" + "=" * 70)
    print("HIGH-DIMENSIONAL ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
