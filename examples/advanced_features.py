"""Demonstration of advanced Safe-ICE features."""

import sys

import matplotlib.pyplot as plt
import numpy as np

# This script prints mathematical notation. Windows consoles still default to a
# legacy code page, where that would raise UnicodeEncodeError, so make sure
# stdout can carry it.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from safe_ice import AdaptiveSafeICE
from safe_ice.problems.advanced_problems import (
    NetworkReliabilityProblem,
    StochasticProcessProblem,
    SystemReliabilityProblem,
    TimeVariantProblem,
)

# Check for optional dependencies
try:
    from safe_ice.analysis.interactive_visualization import (
        InteractiveVisualizer,
        create_interactive_dashboard,
    )

    INTERACTIVE_AVAILABLE = True
except ImportError:
    INTERACTIVE_AVAILABLE = False
    print("Note: Install plotly for interactive visualizations")


def demo_adaptive_safe_ice():
    """Demonstrate Adaptive Safe-ICE with auto-tuning."""
    print("=" * 70)
    print("ADAPTIVE SAFE-ICE DEMONSTRATION")
    print("=" * 70)

    # Define a challenging problem
    def complex_limit_state(u):
        """Multi-modal limit state function."""
        if u.ndim == 1:
            u = u.reshape(1, -1)

        # Multiple failure modes
        g1 = 3.5 - np.linalg.norm(u[:, :2], axis=1)
        g2 = 2.5 + u[:, 0] - 0.1 * u[:, 1] ** 2
        g3 = 3.0 - np.abs(u[:, 0] + u[:, 1])

        return np.minimum(np.minimum(g1, g2), g3)

    # Test different dimensions
    for d in [2, 5, 10]:
        print(f"\n{'=' * 70}")
        print(f"Dimension: {d}")
        print("-" * 70)

        # Adaptive Safe-ICE automatically tunes parameters
        ice = AdaptiveSafeICE(
            limit_state_function=complex_limit_state,
            dimension=d,
            N=None,  # Auto-computed based on dimension
            auto_tune=True,
            adaptive_schedule=True,
        )

        print("Auto-tuned parameters:")
        print(f"  N: {ice.N}")
        print(f"  K0: {ice.K0}")
        print(f"  delta_target: {ice.delta_target}")
        print(f"  delta_star: {ice.delta_star}")

        # Run algorithm
        _pf, _results = ice.run(verbose=True)

        print("\nAdaptive metrics:")
        if ice.beta_history:
            print(f"  Average β: {np.mean(ice.beta_history):.3f}")
            print(
                f"  β range: [{min(ice.beta_history):.3f}, {max(ice.beta_history):.3f}]"
            )


def demo_time_variant_problem():
    """Demonstrate time-variant reliability analysis."""
    print("\n" + "=" * 70)
    print("TIME-VARIANT RELIABILITY PROBLEM")
    print("=" * 70)

    # Define time-variant limit state
    def g_time(u, t):
        """Degrading resistance over time."""
        resistance = 10.0 * np.exp(-0.1 * t)  # Exponential degradation
        load = 3.0 + np.linalg.norm(u[:2]) if u.ndim > 1 else 3.0 + np.linalg.norm(u)
        return resistance - load

    # Time points
    time_points = np.linspace(0, 20, 10)

    # Create time-variant problem
    tv_problem = TimeVariantProblem(limit_state_func=g_time, time_points=time_points)

    # Get series system (fails if fails at ANY time)
    g_series = tv_problem.get_series_system_limit_state()

    # Run Safe-ICE
    ice = AdaptiveSafeICE(limit_state_function=g_series, dimension=2, auto_tune=True)

    print(f"Analyzing series system over {len(time_points)} time points...")
    pf, _results = ice.run(verbose=False)
    print(f"Failure probability (series): {pf:.6e}")

    # Get parallel system (fails only if fails at ALL times)
    g_parallel = tv_problem.get_parallel_system_limit_state()

    ice_parallel = AdaptiveSafeICE(
        limit_state_function=g_parallel, dimension=2, auto_tune=True
    )

    pf_parallel, _ = ice_parallel.run(verbose=False)
    print(f"Failure probability (parallel): {pf_parallel:.6e}")

    # Cumulative damage
    def damage_func(u, t):  # noqa: ARG001 - `t` is part of the damage-model signature
        """Fatigue damage accumulation."""
        stress = (
            2.0 + 0.5 * np.linalg.norm(u[:2])
            if u.ndim > 1
            else 2.0 + 0.5 * np.linalg.norm(u)
        )
        return (stress / 10.0) ** 3  # Miner's rule

    g_damage = tv_problem.get_cumulative_damage_limit_state(
        damage_func=damage_func, threshold=1.0
    )

    ice_damage = AdaptiveSafeICE(
        limit_state_function=g_damage, dimension=2, auto_tune=True
    )

    pf_damage, _ = ice_damage.run(verbose=False)
    print(f"Failure probability (cumulative damage): {pf_damage:.6e}")


def demo_system_reliability():
    """Demonstrate system reliability analysis."""
    print("\n" + "=" * 70)
    print("SYSTEM RELIABILITY ANALYSIS")
    print("=" * 70)

    # Define component limit state functions
    def component1(u):
        return 3.0 - np.abs(u[0] if u.ndim == 1 else u[:, 0])

    def component2(u):
        return 2.5 - np.abs(u[1] if u.ndim == 1 else u[:, 1]) if len(u) > 1 else 2.5

    def component3(u):
        return 4.0 - np.linalg.norm(
            u[:2] if u.ndim == 2 else u, axis=-1 if u.ndim == 2 else None
        )

    components = [component1, component2, component3]

    # Define correlation matrix
    correlation = np.array([[1.0, 0.5, 0.3], [0.5, 1.0, 0.4], [0.3, 0.4, 1.0]])

    # Create system problem
    sys_problem = SystemReliabilityProblem(
        component_funcs=components, correlation_matrix=correlation
    )

    # Analyze different system types
    system_types = [
        ("Series", sys_problem.get_series_system()),
        ("Parallel", sys_problem.get_parallel_system()),
        ("2-out-of-3", sys_problem.get_k_out_of_n_system(k=2)),
        ("Correlated Series", sys_problem.get_correlated_system("series")),
    ]

    for name, g_system in system_types:
        ice = AdaptiveSafeICE(
            limit_state_function=g_system, dimension=2, auto_tune=True
        )

        pf, _ = ice.run(verbose=False)
        print(f"{name} system: Pf = {pf:.6e}")


def demo_stochastic_process():
    """Demonstrate stochastic process problems."""
    print("\n" + "=" * 70)
    print("STOCHASTIC PROCESS PROBLEM")
    print("=" * 70)

    # Define mean and covariance functions
    def mean_func(x):
        return 2.0 + 0.5 * x

    def cov_func(x, y):
        correlation_length = 2.0
        variance = 1.0
        return variance * np.exp(-abs(x - y) / correlation_length)

    # Discretization points
    mesh_points = np.linspace(0, 10, 20)

    # Create stochastic process problem
    sp_problem = StochasticProcessProblem(
        mean_func=mean_func, cov_func=cov_func, mesh_points=mesh_points
    )

    # Get excursion limit state
    threshold = 5.0
    g_excursion = sp_problem.get_excursion_limit_state(
        threshold=threshold, n_kl_terms=10
    )

    # Run Safe-ICE
    ice = AdaptiveSafeICE(
        limit_state_function=g_excursion,
        dimension=10,  # Using 10 KL terms
        auto_tune=True,
    )

    print(f"Analyzing excursion over threshold {threshold}...")
    print(f"Using {10} KL expansion terms")
    pf, _results = ice.run(verbose=False)
    print(f"Excursion probability: {pf:.6e}")

    # Visualize KL eigenvalues
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.semilogy(sp_problem.eigenvalues[:10], "o-")
    plt.xlabel("Mode")
    plt.ylabel("Eigenvalue")
    plt.title("KL Expansion Eigenvalues")
    plt.grid(True)

    # Visualize eigenvectors
    plt.subplot(1, 2, 2)
    for i in range(min(3, len(sp_problem.eigenvectors[0]))):
        plt.plot(mesh_points, sp_problem.eigenvectors[:, i], label=f"Mode {i + 1}")
    plt.xlabel("x")
    plt.ylabel("Eigenfunction")
    plt.title("KL Expansion Eigenfunctions")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def demo_network_reliability():
    """Demonstrate network reliability analysis."""
    print("\n" + "=" * 70)
    print("NETWORK RELIABILITY ANALYSIS")
    print("=" * 70)

    # Define a simple network (bridge network)
    #   0 --- 1
    #   |\   /|
    #   | \ / |
    #   |  2  |
    #   | / \ |
    #   |/   \|
    #   3 --- 4
    adjacency = np.array(
        [
            [0, 1, 1, 1, 0],  # Node 0 connections
            [1, 0, 1, 0, 1],  # Node 1 connections
            [1, 1, 0, 1, 1],  # Node 2 connections
            [1, 0, 1, 0, 1],  # Node 3 connections
            [0, 1, 1, 1, 0],  # Node 4 connections
        ]
    )

    # Create network problem
    net_problem = NetworkReliabilityProblem(adjacency_matrix=adjacency)

    # Get connectivity limit state (0 to 4)
    g_connectivity = net_problem.get_connectivity_limit_state(source=0, target=4)

    # Run Safe-ICE
    ice = AdaptiveSafeICE(
        limit_state_function=g_connectivity,
        dimension=7,  # Number of edges
        auto_tune=True,
    )

    print("Analyzing network connectivity (0 → 4)...")
    print(f"Network has {net_problem.n_nodes} nodes and {net_problem.n_edges} edges")
    pf, _results = ice.run(verbose=False)
    print(f"Disconnection probability: {pf:.6e}")
    print(f"Reliability: {1 - pf:.6f}")


def demo_interactive_visualization():
    """Demonstrate interactive visualization features."""
    if not INTERACTIVE_AVAILABLE:
        print("\nSkipping interactive visualization (plotly not installed)")
        return

    print("\n" + "=" * 70)
    print("INTERACTIVE VISUALIZATION")
    print("=" * 70)

    # Run a simple problem
    def g(u):
        return 3.0 - np.linalg.norm(u, axis=-1)

    ice = AdaptiveSafeICE(
        limit_state_function=g, dimension=3, N=1000, max_iterations=10
    )

    print("Running Safe-ICE for visualization...")
    _pf, results = ice.run(verbose=False)

    # Create interactive visualizer
    viz = InteractiveVisualizer()

    print("\nCreating interactive plots...")

    # 1. Interactive convergence plot
    viz.plot_convergence_interactive(results, show=False)
    print("  ✓ Convergence plot created")

    # 2. 3D sample visualization
    viz.plot_sample_evolution_3d(results, show=False)
    print("  ✓ 3D sample plot created")

    # 3. Comprehensive dashboard
    print("  Creating dashboard...")
    create_interactive_dashboard(results, g)
    print("  ✓ Dashboard created")

    print("\nInteractive features available:")
    print("  • Hover for detailed information")
    print("  • Zoom and pan with mouse")
    print("  • Click legend to toggle traces")
    print("  • Use controls to animate evolution")


def main():
    """Run all advanced feature demonstrations."""

    # 1. Adaptive Safe-ICE
    demo_adaptive_safe_ice()

    # 2. Time-variant problems
    demo_time_variant_problem()

    # 3. System reliability
    demo_system_reliability()

    # 4. Stochastic processes
    demo_stochastic_process()

    # 5. Network reliability
    demo_network_reliability()

    # 6. Interactive visualization (if available)
    demo_interactive_visualization()

    print("\n" + "=" * 70)
    print("ADVANCED FEATURES DEMONSTRATION COMPLETE")
    print("=" * 70)
    print("\nKey Features Demonstrated:")
    print("✓ Adaptive parameter tuning")
    print("✓ Time-variant reliability")
    print("✓ System reliability (series/parallel/k-out-of-n)")
    print("✓ Stochastic processes with KL expansion")
    print("✓ Network reliability")
    if INTERACTIVE_AVAILABLE:
        print("✓ Interactive visualizations")
    else:
        print("○ Interactive visualizations (install plotly)")


if __name__ == "__main__":
    main()
