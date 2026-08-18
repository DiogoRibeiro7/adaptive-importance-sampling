"""Command-line interface for Safe-ICE."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable

from . import __version__
from .typing import LimitStateFunction


def run_demo() -> None:
    """Run a simple demonstration of the Safe-ICE algorithm."""
    from safe_ice import SafeICE
    from safe_ice.problems.benchmarks import BenchmarkProblems

    print("\n" + "=" * 60)
    print("Safe-ICE Algorithm Demonstration")
    print("=" * 60)

    # Create a simple test problem
    problems = BenchmarkProblems()
    g = problems.four_mode_series_system()

    print("\nProblem: Four-mode series system (2D)")
    print("Running Safe-ICE with:")
    print("  - Dimension: 2")
    print("  - Samples per iteration: 1000")
    print("  - Max iterations: 10")
    print("\nProgress:")

    # Initialize and run Safe-ICE
    ice = SafeICE(
        limit_state_function=g,
        dimension=2,
        N=1000,
        max_iterations=10,
        delta_target=3.0,
        delta_star=1.5,
    )

    try:
        pf, results = ice.run(verbose=True)

        print("\n" + "=" * 60)
        print("Results:")
        print(f"  Estimated failure probability: {pf:.6e}")
        print(f"  Number of samples generated: {len(results['final_samples'])}")
        print("  Reference probability: ~6.5e-5")
        print("=" * 60)

    except Exception as e:
        print(f"\nError during demonstration: {e}")
        print("Please check your installation and try again.")


def run_benchmark(
    problem_name: str | None = None, n_samples: int = 1000, max_iterations: int = 20
) -> None:
    """Run a benchmark problem."""
    from safe_ice import SafeICE
    from safe_ice.problems.benchmarks import BenchmarkProblems

    problems = BenchmarkProblems()

    # Available problems. Reference probabilities are from crude Monte Carlo
    # over 2e7 standard-normal samples; see
    # tests/test_benchmark_ground_truth.py.
    available: dict[str, tuple[Callable[[], LimitStateFunction], int, float | None]] = {
        "four-mode": (problems.four_mode_series_system, 2, 6.465e-05),
        "three-mode": (problems.three_mode_problem, 2, 3.475e-03),
        "two-mode": (problems.two_mode_opposite_directions, 2, 2.690e-03),
        "oscillator": (problems.nonlinear_oscillator, 10, 1.798e-03),
    }

    if problem_name is None or problem_name not in available:
        print("\nAvailable benchmark problems:")
        for name in available:
            print(f"  - {name}")
        if problem_name is not None:
            print(f"\nError: Unknown problem '{problem_name}'")
        return

    g_func, dim, ref_prob = available[problem_name]
    g = g_func()

    print(f"\nRunning benchmark: {problem_name}")
    print(f"  Dimension: {dim}")
    print(f"  Samples per iteration: {n_samples}")
    print(f"  Max iterations: {max_iterations}")
    if ref_prob:
        print(f"  Reference probability: {ref_prob:.2e}")

    ice = SafeICE(
        limit_state_function=g,
        dimension=dim,
        N=n_samples,
        max_iterations=max_iterations,
        delta_target=3.0,
        delta_star=1.5,
    )

    pf, _results = ice.run(verbose=True)

    print("\nResults:")
    print(f"  Estimated failure probability: {pf:.6e}")
    if ref_prob:
        rel_error = abs(pf - ref_prob) / ref_prob
        print(f"  Relative error: {rel_error:.2%}")


def analyze_results(input_file: str | None = None) -> None:  # noqa: ARG001 - placeholder for the unimplemented subcommand
    """Analyze and visualize results."""
    print("\nResult analysis functionality coming soon!")
    print("For now, please use the Python API directly:")
    print("\n  from safe_ice.analysis import AdvancedAnalysis")
    print("  analysis = AdvancedAnalysis()")
    print("  # ... load your results and analyze")


def main(args: list[str] | None = None) -> int:
    """Main entry point for the Safe-ICE CLI."""
    parser = argparse.ArgumentParser(
        prog="safe-ice",
        description="Safe Cross-Entropy-Based Importance Sampling for Rare Event Simulations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  safe-ice demo                    Run a simple demonstration
  safe-ice benchmark four-mode     Run the four-mode benchmark problem
  safe-ice benchmark --list        List available benchmark problems
  safe-ice --help                  Show this help message

For more information, visit:
  https://github.com/DiogoRibeiro7/adaptive-importance-sampling-ice
        """,
    )

    parser.add_argument(
        "--version", action="version", version=f"safe-ice {__version__}"
    )

    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Demo command
    subparsers.add_parser("demo", help="Run a simple demonstration")

    # Benchmark command
    benchmark_parser = subparsers.add_parser("benchmark", help="Run benchmark problems")
    benchmark_parser.add_argument(
        "problem",
        nargs="?",
        help="Problem name (four-mode, three-mode, two-mode, oscillator)",
    )
    benchmark_parser.add_argument(
        "--samples",
        type=int,
        default=1000,
        help="Number of samples per iteration (default: 1000)",
    )
    benchmark_parser.add_argument(
        "--iterations", type=int, default=20, help="Maximum iterations (default: 20)"
    )
    benchmark_parser.add_argument(
        "--list", action="store_true", help="List available benchmark problems"
    )

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze results")
    analyze_parser.add_argument("--input", help="Input file with results")

    # Parse arguments
    parsed_args = parser.parse_args(args)

    # Execute based on command
    if parsed_args.command == "demo":
        run_demo()
    elif parsed_args.command == "benchmark":
        if parsed_args.list:
            run_benchmark(None, parsed_args.samples, parsed_args.iterations)
        else:
            run_benchmark(
                parsed_args.problem, parsed_args.samples, parsed_args.iterations
            )
    elif parsed_args.command == "analyze":
        analyze_results(parsed_args.input)
    else:
        # No command specified, show help
        parser.print_help()
        print("\nTip: Try 'safe-ice demo' for a quick demonstration!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
