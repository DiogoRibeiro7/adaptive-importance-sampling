"""Compare Safe-ICE performance with standard Monte Carlo on benchmark problems."""

import time

import matplotlib.pyplot as plt
import numpy as np

from safe_ice import SafeICE
from safe_ice.analysis.performance import PerformanceEvaluator
from safe_ice.problems.benchmarks import BenchmarkProblems


def run_safe_ice(limit_state_func, dimension, n_samples, max_iterations):
    """Run Safe-ICE algorithm."""
    ice = SafeICE(
        limit_state_function=limit_state_func,
        dimension=dimension,
        N=n_samples,
        max_iterations=max_iterations,
    )

    start_time = time.time()
    pf, results = ice.run(verbose=False)
    elapsed_time = time.time() - start_time

    return pf, results, elapsed_time


def run_monte_carlo(limit_state_func, dimension, n_samples):
    """Run standard Monte Carlo simulation."""
    start_time = time.time()

    # Generate standard normal samples
    u = np.random.default_rng(0).standard_normal(n_samples, dimension)

    # Evaluate limit state function
    g_values = limit_state_func(u)

    # Estimate failure probability
    n_failures = np.sum(g_values <= 0)
    pf = n_failures / n_samples

    # Compute standard error
    std_error = np.sqrt((1 - pf) / n_failures) if n_failures > 0 else 0

    elapsed_time = time.time() - start_time

    return pf, std_error, elapsed_time


def main():
    """Compare Safe-ICE with Monte Carlo on benchmark problems."""

    print("=" * 70)
    print("SAFE-ICE vs MONTE CARLO COMPARISON")
    print("=" * 70)

    # Initialize benchmark problems
    problems = BenchmarkProblems()
    PerformanceEvaluator()

    # Define test cases
    test_cases = [
        {
            "name": "Four-Mode Series System",
            "func": problems.four_mode_series_system(),
            "dimension": 2,
            "ref_pf": 1.22e-5,
            "n_samples_ice": 1000,
            "n_samples_mc": 1000000,
            "max_iterations": 15,
        },
        {
            "name": "Three-Mode Problem",
            "func": problems.three_mode_problem(),
            "dimension": 2,
            "ref_pf": 2.3e-3,
            "n_samples_ice": 500,
            "n_samples_mc": 100000,
            "max_iterations": 10,
        },
    ]

    results_comparison = []

    for test_case in test_cases:
        print(f"\n{'=' * 70}")
        print(f"Problem: {test_case['name']}")
        print(f"Dimension: {test_case['dimension']}")
        print(f"Reference Pf: {test_case['ref_pf']:.2e}")
        print("-" * 70)

        # Run Safe-ICE
        print(f"\nRunning Safe-ICE (N={test_case['n_samples_ice']})...")
        pf_ice, results_ice, time_ice = run_safe_ice(
            test_case["func"],
            test_case["dimension"],
            test_case["n_samples_ice"],
            test_case["max_iterations"],
        )

        total_samples_ice = len(results_ice["final_samples"])
        cv_ice = results_ice["convergence_metrics"]["cv_values"][-1]

        print(f"  Pf estimate: {pf_ice:.6e}")
        print(f"  Total samples: {total_samples_ice}")
        print(f"  Final CV: {cv_ice:.4f}")
        print(f"  Time: {time_ice:.2f} seconds")
        print(
            f"  Relative error: {abs(pf_ice - test_case['ref_pf']) / test_case['ref_pf']:.2%}"
        )

        # Run Monte Carlo
        print(f"\nRunning Monte Carlo (N={test_case['n_samples_mc']})...")
        pf_mc, std_mc, time_mc = run_monte_carlo(
            test_case["func"], test_case["dimension"], test_case["n_samples_mc"]
        )

        print(f"  Pf estimate: {pf_mc:.6e}")
        print(f"  Standard error: {std_mc:.4f}")
        print(f"  Time: {time_mc:.2f} seconds")
        if pf_mc > 0:
            print(
                f"  Relative error: {abs(pf_mc - test_case['ref_pf']) / test_case['ref_pf']:.2%}"
            )

        # Compute efficiency metrics
        efficiency_ratio = test_case["n_samples_mc"] / total_samples_ice
        speedup = time_mc / time_ice if time_ice > 0 else float("inf")

        print(f"\n{'EFFICIENCY METRICS':^70}")
        print("-" * 70)
        print(f"  Sample efficiency ratio: {efficiency_ratio:.1f}x")
        print(f"  Computational speedup: {speedup:.1f}x")

        # Store results
        results_comparison.append(
            {
                "problem": test_case["name"],
                "pf_ice": pf_ice,
                "pf_mc": pf_mc,
                "pf_ref": test_case["ref_pf"],
                "samples_ice": total_samples_ice,
                "samples_mc": test_case["n_samples_mc"],
                "time_ice": time_ice,
                "time_mc": time_mc,
                "efficiency": efficiency_ratio,
                "speedup": speedup,
            }
        )

    # Visualize comparison
    visualize_comparison(results_comparison)


def visualize_comparison(results):
    """Create comparison plots."""
    n_problems = len(results)

    _fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Failure probability estimates
    ax = axes[0, 0]
    x_pos = np.arange(n_problems)
    width = 0.25

    pf_ice = [r["pf_ice"] for r in results]
    pf_mc = [r["pf_mc"] for r in results]
    pf_ref = [r["pf_ref"] for r in results]

    ax.bar(x_pos - width, pf_ice, width, label="Safe-ICE", color="blue", alpha=0.7)
    ax.bar(x_pos, pf_mc, width, label="Monte Carlo", color="red", alpha=0.7)
    ax.bar(x_pos + width, pf_ref, width, label="Reference", color="green", alpha=0.7)

    ax.set_xlabel("Problem")
    ax.set_ylabel("Failure Probability")
    ax.set_title("Failure Probability Estimates")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([r["problem"] for r in results], rotation=45, ha="right")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Sample count comparison
    ax = axes[0, 1]
    samples_ice = [r["samples_ice"] for r in results]
    samples_mc = [r["samples_mc"] for r in results]

    ax.bar(
        x_pos - width / 2, samples_ice, width, label="Safe-ICE", color="blue", alpha=0.7
    )
    ax.bar(
        x_pos + width / 2,
        samples_mc,
        width,
        label="Monte Carlo",
        color="red",
        alpha=0.7,
    )

    ax.set_xlabel("Problem")
    ax.set_ylabel("Number of Samples")
    ax.set_title("Sample Count Comparison")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([r["problem"] for r in results], rotation=45, ha="right")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Computational time
    ax = axes[1, 0]
    time_ice = [r["time_ice"] for r in results]
    time_mc = [r["time_mc"] for r in results]

    ax.bar(
        x_pos - width / 2, time_ice, width, label="Safe-ICE", color="blue", alpha=0.7
    )
    ax.bar(
        x_pos + width / 2, time_mc, width, label="Monte Carlo", color="red", alpha=0.7
    )

    ax.set_xlabel("Problem")
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Computational Time")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([r["problem"] for r in results], rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Efficiency metrics
    ax = axes[1, 1]
    efficiency = [r["efficiency"] for r in results]
    speedup = [r["speedup"] for r in results]

    ax.bar(
        x_pos - width / 2,
        efficiency,
        width,
        label="Sample Efficiency",
        color="green",
        alpha=0.7,
    )
    ax.bar(
        x_pos + width / 2, speedup, width, label="Speedup", color="orange", alpha=0.7
    )

    ax.set_xlabel("Problem")
    ax.set_ylabel("Ratio")
    ax.set_title("Efficiency Metrics (Higher is Better)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([r["problem"] for r in results], rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add horizontal line at y=1
    ax.axhline(y=1, color="black", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()

    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Problem':<30} {'Efficiency':<15} {'Speedup':<15}")
    print("-" * 70)
    for r in results:
        print(f"{r['problem']:<30} {r['efficiency']:<15.1f}x {r['speedup']:<15.1f}x")
    print("=" * 70)


if __name__ == "__main__":
    main()
