"""Advanced problem types for Safe-ICE."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import numpy as np
import numpy.typing as npt
from scipy.linalg import cholesky

from ..typing import LimitStateFunction

NDArrayF = npt.NDArray[np.float64]


class TimeVariantProblem:
    """
    Time-variant reliability problems.

    Handles problems where the limit state function varies over time.
    """

    def __init__(
        self,
        limit_state_func: Callable[[NDArrayF, float], float],
        time_points: NDArrayF,
        correlation_func: Callable[[float, float], float] | None = None,
    ):
        """
        Initialize time-variant problem.

        Parameters
        ----------
        limit_state_func : callable
            Function g(u, t) where u is input and t is time.
        time_points : array_like
            Time points to evaluate.
        correlation_func : callable, optional
            Correlation function ρ(t1, t2) for time points.
        """
        self.g_base = limit_state_func
        self.time_points = np.asarray(time_points)
        self.n_time = len(self.time_points)
        self.correlation_func = correlation_func or self._default_correlation

        # Build correlation matrix if needed
        if self.correlation_func is not None:
            self.correlation_matrix = self._build_correlation_matrix()

    def _default_correlation(self, t1: float, t2: float) -> float:
        """Default exponential correlation function."""
        correlation_length = 1.0
        return float(np.exp(-abs(t1 - t2) / correlation_length))

    def _build_correlation_matrix(self) -> NDArrayF:
        """Build correlation matrix for time points."""
        n = self.n_time
        R = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                R[i, j] = self.correlation_func(
                    self.time_points[i], self.time_points[j]
                )

        # Ensure positive definiteness
        min_eig = np.min(np.linalg.eigvals(R))
        if min_eig < 1e-10:
            R += (1e-10 - min_eig) * np.eye(n)

        return R

    def get_series_system_limit_state(self) -> LimitStateFunction:
        """
        Get limit state function for series system over time.

        The system fails if it fails at ANY time point.

        Returns
        -------
        callable
            Combined limit state function.
        """

        def g_series(u: NDArrayF) -> NDArrayF:
            """Series system: min_t g(u, t)."""
            if u.ndim == 1:
                u = u.reshape(1, -1)

            n_samples = u.shape[0]
            g_values = np.zeros(n_samples)

            for i in range(n_samples):
                # Evaluate at all time points
                g_t = np.array([self.g_base(u[i], t) for t in self.time_points])
                # Series system: take minimum
                g_values[i] = np.min(g_t)

            return g_values

        return g_series

    def get_parallel_system_limit_state(self) -> LimitStateFunction:
        """
        Get limit state function for parallel system over time.

        The system fails only if it fails at ALL time points.

        Returns
        -------
        callable
            Combined limit state function.
        """

        def g_parallel(u: NDArrayF) -> NDArrayF:
            """Parallel system: max_t g(u, t)."""
            if u.ndim == 1:
                u = u.reshape(1, -1)

            n_samples = u.shape[0]
            g_values = np.zeros(n_samples)

            for i in range(n_samples):
                # Evaluate at all time points
                g_t = np.array([self.g_base(u[i], t) for t in self.time_points])
                # Parallel system: take maximum
                g_values[i] = np.max(g_t)

            return g_values

        return g_parallel

    def get_cumulative_damage_limit_state(
        self, damage_func: Callable[[NDArrayF, float], float], threshold: float = 1.0
    ) -> LimitStateFunction:
        """
        Get limit state function for cumulative damage.

        Parameters
        ----------
        damage_func : callable
            Damage accumulation function d(u, t).
        threshold : float
            Damage threshold for failure.

        Returns
        -------
        callable
            Cumulative damage limit state function.
        """

        def g_damage(u: NDArrayF) -> NDArrayF:
            """Cumulative damage: threshold - sum_t d(u, t)*dt."""
            if u.ndim == 1:
                u = u.reshape(1, -1)

            n_samples = u.shape[0]
            g_values = np.zeros(n_samples)

            # Time increments for integration
            if self.n_time > 1:
                dt = np.diff(self.time_points)
                dt = np.append(dt, dt[-1])
            else:
                dt = np.array([1.0])

            for i in range(n_samples):
                # Accumulate damage
                total_damage = 0.0
                for j, t in enumerate(self.time_points):
                    damage = damage_func(u[i], t)
                    total_damage += damage * dt[j]

                g_values[i] = threshold - total_damage

            return g_values

        return g_damage


class SystemReliabilityProblem:
    """
    System reliability problems with multiple components.

    Handles series, parallel, and k-out-of-n systems.
    """

    def __init__(
        self,
        component_funcs: list[LimitStateFunction],
        correlation_matrix: NDArrayF | None = None,
    ):
        """
        Initialize system reliability problem.

        Parameters
        ----------
        component_funcs : list of callables
            List of component limit state functions.
        correlation_matrix : array_like, optional
            Correlation matrix between components.
        """
        self.component_funcs = component_funcs
        self.n_components = len(component_funcs)

        if correlation_matrix is not None:
            self.correlation_matrix = np.asarray(correlation_matrix)
            # Validate correlation matrix
            assert self.correlation_matrix.shape == (
                self.n_components,
                self.n_components,
            )
            assert np.allclose(self.correlation_matrix, self.correlation_matrix.T)
        else:
            self.correlation_matrix = np.eye(self.n_components)

    def get_series_system(self) -> LimitStateFunction:
        """
        Get series system limit state function.

        System fails if ANY component fails.

        Returns
        -------
        callable
            Series system limit state function.
        """

        def g_series(u: NDArrayF) -> NDArrayF:
            """Series system: min_i g_i(u)."""
            if u.ndim == 1:
                u = u.reshape(1, -1)

            n_samples = u.shape[0]
            g_values = np.zeros(n_samples)

            for i in range(n_samples):
                # Evaluate all components
                g_components = np.array([func(u[i]) for func in self.component_funcs])
                # Series: minimum
                g_values[i] = np.min(g_components)

            return g_values

        return g_series

    def get_parallel_system(self) -> LimitStateFunction:
        """
        Get parallel system limit state function.

        System fails only if ALL components fail.

        Returns
        -------
        callable
            Parallel system limit state function.
        """

        def g_parallel(u: NDArrayF) -> NDArrayF:
            """Parallel system: max_i g_i(u)."""
            if u.ndim == 1:
                u = u.reshape(1, -1)

            n_samples = u.shape[0]
            g_values = np.zeros(n_samples)

            for i in range(n_samples):
                # Evaluate all components
                g_components = np.array([func(u[i]) for func in self.component_funcs])
                # Parallel: maximum
                g_values[i] = np.max(g_components)

            return g_values

        return g_parallel

    def get_k_out_of_n_system(self, k: int) -> LimitStateFunction:
        """
        Get k-out-of-n system limit state function.

        System fails if k or more components fail.

        Parameters
        ----------
        k : int
            Number of failed components for system failure.

        Returns
        -------
        callable
            k-out-of-n system limit state function.
        """

        def g_k_out_of_n(u: NDArrayF) -> NDArrayF:
            """k-out-of-n system."""
            if u.ndim == 1:
                u = u.reshape(1, -1)

            n_samples = u.shape[0]
            g_values = np.zeros(n_samples)

            for i in range(n_samples):
                # Evaluate all components
                g_components = np.array([func(u[i]) for func in self.component_funcs])

                # Sort components
                g_sorted = np.sort(g_components)

                # k-th smallest value
                g_values[i] = g_sorted[k - 1]

            return g_values

        return g_k_out_of_n

    def get_correlated_system(self, system_type: str = "series") -> LimitStateFunction:
        """
        Get system with correlated components.

        Parameters
        ----------
        system_type : str
            Type of system: "series", "parallel", or "2-out-of-n".

        Returns
        -------
        callable
            Correlated system limit state function.
        """
        # Compute Cholesky decomposition for correlation
        try:
            L = cholesky(self.correlation_matrix, lower=True)
        except np.linalg.LinAlgError:
            # If not positive definite, add small diagonal
            R_mod = self.correlation_matrix + 1e-6 * np.eye(self.n_components)
            L = cholesky(R_mod, lower=True)

        def g_correlated(u: NDArrayF) -> NDArrayF:
            """System with correlated components."""
            if u.ndim == 1:
                u = u.reshape(1, -1)

            n_samples = u.shape[0]
            d = u.shape[1]
            g_values = np.zeros(n_samples)

            for i in range(n_samples):
                # Generate correlated inputs
                if d >= self.n_components:
                    # Use first n_components dimensions
                    z = u[i, : self.n_components]
                else:
                    # Extend with zeros
                    z = np.zeros(self.n_components)
                    z[:d] = u[i]

                # Apply correlation
                u_corr = L @ z

                # Evaluate components with correlated inputs. Each component
                # gets the whole vector, as it does from the series, parallel
                # and k-out-of-n systems. This used to hand component j the
                # one-element slice u_corr[j:j+1], so any component function
                # reading past its own first entry raised IndexError -- which
                # is every function that works with the other three methods.
                g_components = np.array([func(u_corr) for func in self.component_funcs])

                # Apply system logic
                if system_type == "series":
                    g_values[i] = np.min(g_components)
                elif system_type == "parallel":
                    g_values[i] = np.max(g_components)
                elif system_type == "2-out-of-n":
                    g_sorted = np.sort(g_components)
                    g_values[i] = g_sorted[1]  # 2nd smallest
                else:
                    raise ValueError(f"Unknown system type: {system_type}")

            return g_values

        return g_correlated


class StochasticProcessProblem:
    """
    Problems involving stochastic processes.

    Handles random fields and stochastic loads.
    """

    def __init__(
        self,
        mean_func: Callable[[float], float],
        cov_func: Callable[[float, float], float],
        mesh_points: NDArrayF,
    ):
        """
        Initialize stochastic process problem.

        Parameters
        ----------
        mean_func : callable
            Mean function μ(x).
        cov_func : callable
            Covariance function C(x, y).
        mesh_points : array_like
            Discretization points.
        """
        self.mean_func = mean_func
        self.cov_func = cov_func
        self.mesh_points = np.asarray(mesh_points)
        self.n_points = len(self.mesh_points)

        # Build covariance matrix
        self.cov_matrix = self._build_covariance_matrix()

        # Compute KL expansion
        self.eigenvalues, self.eigenvectors = self._compute_kl_expansion()

    def _build_covariance_matrix(self) -> NDArrayF:
        """Build covariance matrix from covariance function."""
        C = np.zeros((self.n_points, self.n_points))

        for i in range(self.n_points):
            for j in range(self.n_points):
                C[i, j] = self.cov_func(self.mesh_points[i], self.mesh_points[j])

        return C

    def _compute_kl_expansion(
        self, n_terms: int | None = None
    ) -> tuple[NDArrayF, NDArrayF]:
        """Compute Karhunen-Loève expansion."""
        # Eigendecomposition
        eigvals, eigvecs = np.linalg.eigh(self.cov_matrix)

        # Sort in descending order
        idx = eigvals.argsort()[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        # Truncate if requested
        if n_terms is not None:
            eigvals = eigvals[:n_terms]
            eigvecs = eigvecs[:, :n_terms]

        # Remove negative eigenvalues
        positive_idx = eigvals > 1e-10
        eigvals = eigvals[positive_idx]
        eigvecs = eigvecs[:, positive_idx]

        return eigvals, eigvecs

    def generate_random_field(
        self, xi: NDArrayF, n_kl_terms: int | None = None
    ) -> NDArrayF:
        """
        Generate random field realization using KL expansion.

        Parameters
        ----------
        xi : array_like
            Standard normal variables.
        n_kl_terms : int, optional
            Number of KL terms to use.

        Returns
        -------
        array_like
            Random field values at mesh points.
        """
        # Mean field
        mean_field = np.array([self.mean_func(x) for x in self.mesh_points])

        # Determine number of terms
        if n_kl_terms is None:
            n_kl_terms = min(len(xi), len(self.eigenvalues))
        else:
            n_kl_terms = min(n_kl_terms, len(self.eigenvalues), len(xi))

        # KL expansion
        field = mean_field.copy()
        for i in range(n_kl_terms):
            field += np.sqrt(self.eigenvalues[i]) * self.eigenvectors[:, i] * xi[i]

        return field

    def get_excursion_limit_state(
        self, threshold: float, n_kl_terms: int | None = None
    ) -> LimitStateFunction:
        """
        Get limit state function for excursion over threshold.

        Parameters
        ----------
        threshold : float
            Excursion threshold.
        n_kl_terms : int, optional
            Number of KL terms.

        Returns
        -------
        callable
            Excursion limit state function.
        """

        def g_excursion(u: NDArrayF) -> NDArrayF:
            """Excursion: threshold - max(field)."""
            if u.ndim == 1:
                u = u.reshape(1, -1)

            n_samples = u.shape[0]
            g_values = np.zeros(n_samples)

            for i in range(n_samples):
                # Generate random field
                field = self.generate_random_field(u[i], n_kl_terms)

                # Check excursion
                g_values[i] = threshold - np.max(field)

            return g_values

        return g_excursion


class NetworkReliabilityProblem:
    """
    Network and graph-based reliability problems.

    Handles connectivity and flow problems.
    """

    def __init__(
        self,
        adjacency_matrix: NDArrayF,
        edge_reliability_funcs: list[LimitStateFunction] | None = None,
    ) -> None:
        """
        Initialize network reliability problem.

        Parameters
        ----------
        adjacency_matrix : array_like
            Network adjacency matrix.
        edge_reliability_funcs : list of callables, optional
            Reliability functions for each edge.
        """
        self.adj_matrix = np.asarray(adjacency_matrix)
        self.n_nodes = self.adj_matrix.shape[0]
        self.edges = self._extract_edges()
        self.n_edges = len(self.edges)

        if edge_reliability_funcs is not None:
            assert len(edge_reliability_funcs) == self.n_edges
            self.edge_funcs = edge_reliability_funcs
        else:
            # Default: all edges have same reliability
            def _edge_func(u: NDArrayF, i: int = 0) -> float:
                return float(3.0 - u[i]) if i < len(u) else 3.0

            self.edge_funcs = [partial(_edge_func, i=i) for i in range(self.n_edges)]

    def _extract_edges(self) -> list[tuple[int, int]]:
        """Extract edge list from adjacency matrix."""
        edges = []
        for i in range(self.n_nodes):
            for j in range(i + 1, self.n_nodes):
                if self.adj_matrix[i, j] != 0:
                    edges.append((i, j))
        return edges

    def get_connectivity_limit_state(
        self, source: int, target: int
    ) -> LimitStateFunction:
        """
        Get limit state function for s-t connectivity.

        Parameters
        ----------
        source : int
            Source node.
        target : int
            Target node.

        Returns
        -------
        callable
            Connectivity limit state function.
        """

        def g_connectivity(u: NDArrayF) -> NDArrayF:
            """Connectivity: 1 if connected, -1 if disconnected."""
            if u.ndim == 1:
                u = u.reshape(1, -1)

            n_samples = u.shape[0]
            g_values = np.zeros(n_samples)

            for i in range(n_samples):
                # Evaluate edge states
                edge_states = np.zeros(self.n_edges)
                for j, func in enumerate(self.edge_funcs):
                    edge_states[j] = func(u[i])

                # Build operational network
                operational = np.zeros_like(self.adj_matrix)
                for j, (node1, node2) in enumerate(self.edges):
                    if edge_states[j] > 0:  # Edge operational
                        operational[node1, node2] = 1
                        operational[node2, node1] = 1

                # Check connectivity using BFS
                connected = self._check_connectivity(operational, source, target)

                g_values[i] = 1.0 if connected else -1.0

            return g_values

        return g_connectivity

    def _check_connectivity(self, adj: NDArrayF, source: int, target: int) -> bool:
        """Check if source and target are connected."""
        visited = np.zeros(self.n_nodes, dtype=bool)
        queue = [source]
        visited[source] = True

        while queue:
            node = queue.pop(0)
            if node == target:
                return True

            for neighbor in range(self.n_nodes):
                if adj[node, neighbor] > 0 and not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)

        return False
