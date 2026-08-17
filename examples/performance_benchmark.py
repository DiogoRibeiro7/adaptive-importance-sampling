"""Performance benchmark comparing original and optimized Safe-ICE."""

import time
import tracemalloc

import matplotlib.pyplot as plt
import numpy as np
import psutil

from safe_ice import OptimizedSafeICE, SafeICE
from safe_ice.problems.benchmarks import BenchmarkProblems


def measure_performance(
    ice_class, limit_state_func, dimension, N, max_iterations, **kwargs
):
    """Measure performance of a Safe-ICE implementation."""

    # Start memory tracking
    tracemalloc.start()
    process = psutil.Process()
    mem_before = process.memory_info().rss / 1024 / 1024  # MB

    # Initialize
    ice = ice_class(
        limit_state_function=limit_state_func,
        dimension=dimension,
        N=N,
        max_iterations=max_iterations,
        **kwargs,
    )

    # Run algorithm
    start_time = time.time()
    pf, results = ice.run(verbose=False)
    elapsed_time = time.time() - start_time

    # Memory usage
    mem_after = process.memory_info().rss / 1024 / 1024  # MB
    mem_used = mem_after - mem_before

    # Get memory peak
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = peak / 1024 / 1024

    return {
        "pf": pf,
        "time": elapsed_time,
        "memory_used": mem_used,
        "memory_peak": peak_mb,
        "total_samples": len(results["final_samples"]),
        "iterations": len(results["iterations"]),
        "final_K": results["final_parameters"].K,
    }


def run_benchmark_suite():
    """Run comprehensive benchmark suite."""

    print("=" * 80)
    print("SAFE-ICE PERFORMANCE BENCHMARK")
    print("=" * 80)

    problems = BenchmarkProblems()

    # Test configurations
    test_configs = [
        {
            "name": "Small 2D Problem",
            "func": problems.four_mode_series_system(),
            "dimension": 2,
            "N": 500,
            "max_iterations": 5,
        },
        {
            "name": "Medium 2D Problem",
            "func": problems.four_mode_series_system(),
            "dimension": 2,
            "N": 2000,
            "max_iterations": 10,
        },
        {
            "name": "Large 2D Problem",
            "func": problems.four_mode_series_system(),
            "dimension": 2,
            "N": 5000,
            "max_iterations": 15,
        },
        {
            "name": "10D Problem",
            "func": problems.nonlinear_oscillator(),
            "dimension": 10,
            "N": 2000,
            "max_iterations": 10,
        },
        {
            "name": "20D Problem",
            "func": problems.two_mode_opposite_directions(dimension=20),
            "dimension": 20,
            "N": 3000,
            "max_iterations": 10,
        },
    ]

    results = []

    for config in test_configs:
        print(f"\n{'=' * 80}")
        print(f"Test: {config['name']}")
        print(
            f"Dimension: {config['dimension']}, N: {config['N']}, Max iter: {config['max_iterations']}"
        )
        print("-" * 80)

        # Original SafeICE
        print("Running original SafeICE...")
        try:
            original_results = measure_performance(
                SafeICE,
                config["func"],
                config["dimension"],
                config["N"],
                config["max_iterations"],
            )
            print(f"  Time: {original_results['time']:.2f}s")
            print(f"  Memory: {original_results['memory_peak']:.1f} MB")
            print(f"  Pf: {original_results['pf']:.2e}")
        except Exception as e:
            print(f"  Error: {e}")
            original_results = None

        # Optimized SafeICE (with caching)
        print("Running optimized SafeICE (with caching)...")
        try:
            optimized_cached = measure_performance(
                OptimizedSafeICE,
                config["func"],
                config["dimension"],
                config["N"],
                config["max_iterations"],
                enable_caching=True,
                enable_parallel=False,
            )
            print(f"  Time: {optimized_cached['time']:.2f}s")
            print(f"  Memory: {optimized_cached['memory_peak']:.1f} MB")
            print(f"  Pf: {optimized_cached['pf']:.2e}")
        except Exception as e:
            print(f"  Error: {e}")
            optimized_cached = None

        # Optimized SafeICE (with caching + batching)
        print("Running optimized SafeICE (full optimization)...")
        try:
            optimized_full = measure_performance(
                OptimizedSafeICE,
                config["func"],
                config["dimension"],
                config["N"],
                config["max_iterations"],
                enable_caching=True,
                enable_parallel=False,
                batch_size=1000,
            )
            print(f"  Time: {optimized_full['time']:.2f}s")
            print(f"  Memory: {optimized_full['memory_peak']:.1f} MB")
            print(f"  Pf: {optimized_full['pf']:.2e}")
        except Exception as e:
            print(f"  Error: {e}")
            optimized_full = None

        # Calculate speedups
        if original_results and optimized_cached:
            speedup_cached = original_results["time"] / optimized_cached["time"]
            memory_ratio_cached = (
                optimized_cached["memory_peak"] / original_results["memory_peak"]
            )
            print(f"\nSpeedup (cached): {speedup_cached:.2f}x")
            print(f"Memory ratio (cached): {memory_ratio_cached:.2f}x")

        if original_results and optimized_full:
            speedup_full = original_results["time"] / optimized_full["time"]
            memory_ratio_full = (
                optimized_full["memory_peak"] / original_results["memory_peak"]
            )
            print(f"Speedup (full): {speedup_full:.2f}x")
            print(f"Memory ratio (full): {memory_ratio_full:.2f}x")

        results.append(
            {
                "config": config,
                "original": original_results,
                "optimized_cached": optimized_cached,
                "optimized_full": optimized_full,
            }
        )

    return results


def plot_performance_results(results):
    """Create performance comparison plots."""

    # Extract valid results
    valid_results = [r for r in results if r["original"] is not None]
    if not valid_results:
        print("No valid results to plot")
        return

    n_tests = len(valid_results)
    test_names = [r["config"]["name"] for r in valid_results]

    _fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Execution Time Comparison
    ax = axes[0, 0]
    x_pos = np.arange(n_tests)
    width = 0.25

    times_original = [
        r["original"]["time"] if r["original"] else 0 for r in valid_results
    ]
    times_cached = [
        r["optimized_cached"]["time"] if r["optimized_cached"] else 0
        for r in valid_results
    ]
    times_full = [
        r["optimized_full"]["time"] if r["optimized_full"] else 0 for r in valid_results
    ]

    ax.bar(
        x_pos - width, times_original, width, label="Original", color="blue", alpha=0.7
    )
    ax.bar(x_pos, times_cached, width, label="Opt. (cache)", color="green", alpha=0.7)
    ax.bar(
        x_pos + width, times_full, width, label="Opt. (full)", color="red", alpha=0.7
    )

    ax.set_xlabel("Test Configuration")
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Execution Time Comparison")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(test_names, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Memory Usage Comparison
    ax = axes[0, 1]

    mem_original = [
        r["original"]["memory_peak"] if r["original"] else 0 for r in valid_results
    ]
    mem_cached = [
        r["optimized_cached"]["memory_peak"] if r["optimized_cached"] else 0
        for r in valid_results
    ]
    mem_full = [
        r["optimized_full"]["memory_peak"] if r["optimized_full"] else 0
        for r in valid_results
    ]

    ax.bar(
        x_pos - width, mem_original, width, label="Original", color="blue", alpha=0.7
    )
    ax.bar(x_pos, mem_cached, width, label="Opt. (cache)", color="green", alpha=0.7)
    ax.bar(x_pos + width, mem_full, width, label="Opt. (full)", color="red", alpha=0.7)

    ax.set_xlabel("Test Configuration")
    ax.set_ylabel("Peak Memory (MB)")
    ax.set_title("Memory Usage Comparison")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(test_names, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Speedup Factors
    ax = axes[1, 0]

    speedups_cached = []
    speedups_full = []

    for r in valid_results:
        if r["original"] and r["optimized_cached"]:
            speedups_cached.append(
                r["original"]["time"] / r["optimized_cached"]["time"]
            )
        else:
            speedups_cached.append(0)

        if r["original"] and r["optimized_full"]:
            speedups_full.append(r["original"]["time"] / r["optimized_full"]["time"])
        else:
            speedups_full.append(0)

    ax.bar(
        x_pos - width / 2,
        speedups_cached,
        width,
        label="Cache Only",
        color="green",
        alpha=0.7,
    )
    ax.bar(
        x_pos + width / 2,
        speedups_full,
        width,
        label="Full Opt.",
        color="red",
        alpha=0.7,
    )

    ax.axhline(y=1, color="black", linestyle="--", alpha=0.5)
    ax.set_xlabel("Test Configuration")
    ax.set_ylabel("Speedup Factor")
    ax.set_title("Performance Speedup (Higher is Better)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(test_names, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Accuracy Comparison
    ax = axes[1, 1]

    pf_original = [r["original"]["pf"] if r["original"] else 0 for r in valid_results]
    pf_cached = [
        r["optimized_cached"]["pf"] if r["optimized_cached"] else 0
        for r in valid_results
    ]
    pf_full = [
        r["optimized_full"]["pf"] if r["optimized_full"] else 0 for r in valid_results
    ]

    ax.bar(x_pos - width, pf_original, width, label="Original", color="blue", alpha=0.7)
    ax.bar(x_pos, pf_cached, width, label="Opt. (cache)", color="green", alpha=0.7)
    ax.bar(x_pos + width, pf_full, width, label="Opt. (full)", color="red", alpha=0.7)

    ax.set_xlabel("Test Configuration")
    ax.set_ylabel("Failure Probability")
    ax.set_title("Accuracy Comparison")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(test_names, rotation=45, ha="right")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Print summary table
    print("\n" + "=" * 80)
    print("PERFORMANCE SUMMARY")
    print("=" * 80)
    print(f"{'Test':<25} {'Speedup (Cache)':<20} {'Speedup (Full)':<20}")
    print("-" * 80)

    for i, _r in enumerate(valid_results):
        speedup_cache = speedups_cached[i] if i < len(speedups_cached) else 0
        speedup_full = speedups_full[i] if i < len(speedups_full) else 0
        print(f"{test_names[i]:<25} {speedup_cache:<20.2f}x {speedup_full:<20.2f}x")


def main():
    """Run performance benchmark."""

    # Run benchmarks
    results = run_benchmark_suite()

    # Plot results
    plot_performance_results(results)

    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)

    # Recommendations
    print("\nRecommendations based on benchmarks:")
    print("- For small problems (N < 1000): Use original SafeICE")
    print("- For medium problems (1000 < N < 5000): Use OptimizedSafeICE with caching")
    print(
        "- For large problems (N > 5000): Use OptimizedSafeICE with full optimization"
    )
    print("- For high dimensions (d > 20): Always use OptimizedSafeICE")


if __name__ == "__main__":
    main()
