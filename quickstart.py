#!/usr/bin/env python
"""Quick start script for Safe-ICE - Run this to verify installation and see basic usage."""

import sys
import time

import numpy as np

# This script prints mathematical notation. Windows consoles still default to a
# legacy code page, where that would raise UnicodeEncodeError, so make sure
# stdout can carry it.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

try:
    from safe_ice import AdaptiveSafeICE, __version__

    print(f"✓ Safe-ICE version {__version__} successfully imported!")
except ImportError as e:
    print(f"✗ Error importing Safe-ICE: {e}")
    print("\nPlease install Safe-ICE first:")
    print("  pip install safe-ice")
    print("  # or")
    print("  pip install -e .")
    sys.exit(1)


def simple_example():
    """Run a simple 2D example."""
    print("\n" + "=" * 60)
    print("SAFE-ICE QUICK START EXAMPLE")
    print("=" * 60)

    # Define a simple limit state function
    def limit_state(u):
        """Simple sphere limit state: failure when ||u|| > 3"""
        return 3.0 - np.linalg.norm(u, axis=-1)

    print("\nProblem: Spherical limit state with radius = 3.0")
    print("Expected failure probability ≈ 0.0111")
    print("\nInitializing Adaptive Safe-ICE...")

    # Use AdaptiveSafeICE for automatic parameter tuning
    ice = AdaptiveSafeICE(limit_state_function=limit_state, dimension=2, auto_tune=True)

    print(f"  Auto-tuned N: {ice.N} samples per iteration")
    print(f"  Auto-tuned K0: {ice.K0} initial components")

    print("\nRunning algorithm...")
    start_time = time.time()

    # Run the algorithm
    pf, results = ice.run(verbose=False)

    elapsed_time = time.time() - start_time

    # Display results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Estimated failure probability: {pf:.6e}")
    print("Analytical failure probability: 1.11e-2")
    print(f"Relative error: {abs(pf - 0.0111) / 0.0111:.2%}")
    print("\nAlgorithm statistics:")
    print(f"  Total samples: {len(results['final_samples'])}")
    print(f"  Iterations: {len(results['iterations'])}")
    print(f"  Execution time: {elapsed_time:.2f} seconds")
    print(f"  Final components: {results['final_parameters'].K}")

    return pf, results


def check_optional_features():
    """Check which optional features are available."""
    print("\n" + "=" * 60)
    print("CHECKING OPTIONAL FEATURES")
    print("=" * 60)

    features = {
        "Plotly (interactive visualization)": "plotly",
        "Numba (JIT compilation)": "numba",
        "Pandas (data analysis)": "pandas",
        "Seaborn (statistical plots)": "seaborn",
        "PSUtil (memory monitoring)": "psutil",
    }

    available = []
    missing = []

    for feature, module in features.items():
        try:
            __import__(module)
            print(f"✓ {feature}")
            available.append(feature)
        except ImportError:
            print(f"○ {feature} (optional)")
            missing.append(feature)

    if missing:
        print("\nTo install optional features:")
        print("  pip install safe-ice[viz]   # Visualization tools")
        print("  pip install safe-ice[perf]  # Performance extras")
        print("  pip install safe-ice[all]   # Everything")


def show_advanced_usage():
    """Show advanced usage examples."""
    print("\n" + "=" * 60)
    print("ADVANCED USAGE EXAMPLES")
    print("=" * 60)

    print("\n1. Original SafeICE (full control):")
    print("""
from safe_ice import SafeICE

ice = SafeICE(
    limit_state_function=g,
    dimension=10,
    N=2000,
    K0=30,
    delta_target=4.0,
    max_iterations=20
)
pf, results = ice.run()
""")

    print("\n2. Optimized SafeICE (2-10x faster):")
    print("""
from safe_ice import OptimizedSafeICE

ice = OptimizedSafeICE(
    limit_state_function=g,
    dimension=10,
    N=2000,
    enable_caching=True,
    batch_size=1000
)
pf, results = ice.run()
""")

    print("\n3. Adaptive SafeICE (automatic tuning):")
    print("""
from safe_ice import AdaptiveSafeICE

ice = AdaptiveSafeICE(
    limit_state_function=g,
    dimension=10,  # Parameters auto-tuned based on dimension
    auto_tune=True,
    adaptive_schedule=True
)
pf, results = ice.run()
""")

    print("\n4. Advanced problem types:")
    print("""
from safe_ice.problems.advanced_problems import (
    TimeVariantProblem,
    SystemReliabilityProblem,
    StochasticProcessProblem
)
""")

    print("\n5. Interactive visualization (requires plotly):")
    print("""
from safe_ice.analysis.interactive_visualization import (
    InteractiveVisualizer,
    create_interactive_dashboard
)

viz = InteractiveVisualizer()
viz.plot_convergence_interactive(results)
create_interactive_dashboard(results, limit_state_func)
""")


def main():
    """Main function."""
    print("""
╔═══════════════════════════════════════════════════════════╗
║          SAFE-ICE: Safe Cross-Entropy Method              ║
║     Importance Sampling for Rare Event Simulations        ║
╚═══════════════════════════════════════════════════════════╝
""")

    # Run simple example
    try:
        _pf, _results = simple_example()
    except Exception as e:
        print(f"\n✗ Error running example: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # Check optional features
    check_optional_features()

    # Show advanced usage
    show_advanced_usage()

    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("\n1. Run more examples:")
    print("   python examples/basic_usage.py")
    print("   python examples/benchmark_comparison.py")
    print("   python examples/high_dimensional.py")
    print("   python examples/advanced_features.py")

    print("\n2. Use the CLI:")
    print("   safe-ice demo")
    print("   safe-ice benchmark four-mode")

    print("\n3. Read the documentation:")
    print("   https://safe-ice.readthedocs.io/")

    print("\n4. Explore in Jupyter:")
    print("   jupyter lab")
    print("   # Then import safe_ice")

    print("\n" + "=" * 60)
    print("Quick start completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
