# safe_ice/optimization/penalized_em.py
"""Penalized EM algorithm for automatic component selection."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ..core.parameters import vMFNMParameters
from ..distributions._numeric import vmf_pdf_batch
from ..distributions.mixture import vMFNMDistribution
from ..distributions.nakagami import NakagamiDistribution

NDArrayF = npt.NDArray[np.float64]


class PenalizedEMOptimizer:
    """Penalized EM algorithm for automatic component selection."""

    def __init__(
        self, max_em_iterations: int = 100, em_tolerance: float = 1e-6
    ) -> None:
        self.max_em_iterations = int(max_em_iterations)
        self.em_tolerance = float(em_tolerance)

    def fit(
        self,
        data: NDArrayF,
        weights: NDArrayF,
        initial_params: vMFNMParameters,
        beta_init: float = 1.0,
    ) -> tuple[vMFNMParameters, int]:
        """Fit vMFNM mixture using penalized EM.

        Parameters
        ----------
        data : ndarray of shape (n, d)
            Sample data.
        weights : ndarray of shape (n,)
            Importance weights.
        initial_params : vMFNMParameters
            Starting point for the fit.
        beta_init : float
            Initial penalty parameter.

        Returns
        -------
        tuple of (vMFNMParameters, int)
            The optimised parameters and the final number of components.
        """
        _n, _d = data.shape
        params = self._copy_parameters(initial_params)
        beta: float = float(beta_init)
        K: int = int(params.K)

        # Precompute data in polar coordinates
        radii: NDArrayF = np.linalg.norm(data, axis=1).astype(np.float64, copy=False)
        directions: NDArrayF = np.zeros_like(data, dtype=np.float64)
        valid_radii = radii > 1e-12
        directions[valid_radii] = data[valid_radii] / radii[valid_radii, np.newaxis]

        # Handle zero vectors
        if np.any(~valid_radii):
            directions[~valid_radii, 0] = 1.0
            radii[~valid_radii] = 1e-12

        prev_log_likelihood: float = float("-inf")

        for _ in range(self.max_em_iterations):
            # E-step: compute responsibilities
            responsibilities: NDArrayF = self._e_step(
                data, radii, directions, params, weights
            )

            # M-step with penalization
            params, K = self._penalized_m_step(
                data, radii, directions, responsibilities, weights, params, beta
            )

            # Update penalty parameter
            beta = self._update_beta(params, beta, K)

            # Check convergence
            current_log_likelihood: float = self._weighted_log_likelihood(
                data, params, weights
            )
            if abs(current_log_likelihood - prev_log_likelihood) < self.em_tolerance:
                break
            prev_log_likelihood = current_log_likelihood

        return params, K

    # -------------------------------------------------------------------------
    # E-step
    # -------------------------------------------------------------------------
    def _e_step(
        self,
        data: NDArrayF,
        radii: NDArrayF,
        directions: NDArrayF,
        params: vMFNMParameters,
        weights: NDArrayF,  # noqa: ARG002 - applied in the M-step, not here
    ) -> NDArrayF:
        """E-step: compute posterior responsibilities.

        ``weights`` is accepted for symmetry with :meth:`_penalized_m_step`,
        which is where importance weighting is actually applied.
        """
        n = int(data.shape[0])
        K = int(params.K)

        # Component likelihoods for every sample at once: one vectorised pass
        # per component instead of a scalar pdf call per (sample, component).
        comp_like: NDArrayF = np.zeros((n, K), dtype=np.float64)
        for k in range(K):
            nak_pdf: NDArrayF = np.asarray(
                NakagamiDistribution.pdf(
                    radii, float(params.m[k]), float(params.Omega[k])
                ),
                dtype=np.float64,
            )
            vmf_pdf: NDArrayF = self._vmf_pdf_many(
                directions, params.mu[k], float(params.kappa[k])
            )
            comp_like[:, k] = float(params.pi[k]) * nak_pdf * vmf_pdf

        totals: NDArrayF = comp_like.sum(axis=1)
        degenerate = totals <= 1e-15

        responsibilities: NDArrayF = np.empty((n, K), dtype=np.float64)
        # Samples with negligible likelihood under every component fall back to
        # a uniform assignment rather than dividing by ~0.
        np.divide(
            comp_like,
            totals[:, None],
            out=responsibilities,
            where=~degenerate[:, None],
        )
        responsibilities[degenerate] = 1.0 / float(K)

        # Weighting by importance weights is done in M-step via weighted_resp
        return responsibilities

    # -------------------------------------------------------------------------
    # M-step (penalized)
    # -------------------------------------------------------------------------
    def _penalized_m_step(
        self,
        data: NDArrayF,
        radii: NDArrayF,
        directions: NDArrayF,
        responsibilities: NDArrayF,
        weights: NDArrayF,
        params: vMFNMParameters,
        beta: float,
    ) -> tuple[vMFNMParameters, int]:
        """Penalized M-step with automatic component removal."""
        _n, d = data.shape
        int(params.K)

        # Weighted responsibilities
        weighted_resp: NDArrayF = (responsibilities * weights[:, np.newaxis]).astype(
            np.float64, copy=False
        )

        # Update mixture weights with penalization
        new_pi: NDArrayF = self._update_mixture_weights_penalized(
            weighted_resp, params.pi, beta
        )

        # Remove components with negligible weights
        active_components = new_pi > 1e-4
        K_new: int = int(np.sum(active_components))

        if K_new == 0:
            K_new = 1
            active_components[0] = True

        # Extract active components
        active_indices = np.where(active_components)[0]
        new_pi = new_pi[active_components]
        new_pi = (new_pi / float(np.sum(new_pi))).astype(np.float64, copy=False)

        # Update other parameters for active components
        new_m: NDArrayF = np.zeros(K_new, dtype=np.float64)
        new_Omega: NDArrayF = np.zeros(K_new, dtype=np.float64)
        new_mu: NDArrayF = np.zeros((K_new, d), dtype=np.float64)
        new_kappa: NDArrayF = np.zeros(K_new, dtype=np.float64)

        for idx, k in enumerate(active_indices):
            resp_k: NDArrayF = weighted_resp[:, k]
            sum_resp: float = float(np.sum(resp_k))

            if sum_resp > 1e-15:
                # Update Nakagami parameters
                m_hat, Omega_hat = self._update_nakagami_parameters(radii, resp_k)
                new_m[idx] = float(m_hat)
                new_Omega[idx] = float(Omega_hat)

                # Update von Mises-Fisher parameters
                mu_hat, kappa_hat = self._update_vmf_parameters(directions, resp_k)
                new_mu[idx, :] = mu_hat
                new_kappa[idx] = float(kappa_hat)
            else:
                # Keep previous parameters
                new_m[idx] = float(params.m[k])
                new_Omega[idx] = float(params.Omega[k])
                new_mu[idx, :] = params.mu[k]
                new_kappa[idx] = float(params.kappa[k])

        new_params = vMFNMParameters(
            pi=new_pi, m=new_m, Omega=new_Omega, mu=new_mu, kappa=new_kappa
        )

        return new_params, K_new

    def _update_mixture_weights_penalized(
        self, weighted_resp: NDArrayF, old_pi: NDArrayF, beta: float
    ) -> NDArrayF:
        """Update mixture weights with cross-entropy-like penalty."""
        # Standard EM update
        total_weight: float = float(np.sum(weighted_resp))
        if total_weight <= 0.0:
            # fallback uniform
            K = int(weighted_resp.shape[1])
            return np.full(K, 1.0 / float(K), dtype=np.float64)

        pi_em: NDArrayF = (np.sum(weighted_resp, axis=0) / total_weight).astype(
            np.float64, copy=False
        )

        # Cross-entropy penalty, Equation 21:
        #   pi_k += beta * pi_k * [ ln pi_k - sum_s pi_s ln pi_s ]
        #
        # The bracket is the pointwise surprisal minus the mean surprisal, so it
        # is negative for components that carry less than average weight and
        # positive for the rest: the penalty drains small components into large
        # ones and is exactly zero when the weights are uniform.
        #
        # Note the sign. `sum_s pi_s ln pi_s` is negative and equals minus the
        # entropy. Subtracting the entropy instead, as this once did, makes the
        # bracket -2 ln K at uniform weights rather than 0, which subtracts a
        # flat ~0.3 from every weight at K=20 and drives all but the single
        # largest component to zero on the first EM step. That defeated the
        # automatic component selection this penalty exists to provide.
        safe_old = np.maximum(old_pi.astype(np.float64, copy=False), 1e-15)
        mean_log_pi: float = float(np.sum(safe_old * np.log(safe_old)))

        # Normalize factor (total_weight/total_weight == 1), kept explicit for clarity
        norm_factor: float = 1.0
        penalty_term: NDArrayF = (
            float(beta) * norm_factor * safe_old * (np.log(safe_old) - mean_log_pi)
        ).astype(np.float64, copy=False)

        new_pi: NDArrayF = (pi_em + penalty_term).astype(np.float64, copy=False)

        # Ensure non-negativity and renormalization
        new_pi = np.maximum(new_pi, 0.0)
        s = float(np.sum(new_pi))
        if s > 0.0:
            new_pi = (new_pi / s).astype(np.float64, copy=False)
        else:
            new_pi = np.full(K, 1.0 / float(K), dtype=np.float64)

        return new_pi

    def _update_nakagami_parameters(
        self, radii: NDArrayF, responsibilities: NDArrayF
    ) -> tuple[float, float]:
        """Update Nakagami parameters using method of moments."""
        sum_resp: float = float(np.sum(responsibilities))

        if sum_resp < 1e-15:
            return 1.0, 1.0

        # Weighted moments
        mean_r2: float = float(np.sum(responsibilities * (radii**2)) / sum_resp)
        mean_r4: float = float(np.sum(responsibilities * (radii**4)) / sum_resp)

        if mean_r4 <= mean_r2**2:
            return 1.0, mean_r2

        # Method-of-moments estimators
        m_est: float = float((mean_r2**2) / (mean_r4 - mean_r2**2))
        Omega_est: float = float(mean_r2)

        # Ensure valid parameters
        m_est = max(m_est, 0.5)
        Omega_est = max(Omega_est, 1e-6)

        return float(m_est), float(Omega_est)

    def _update_vmf_parameters(
        self, directions: NDArrayF, responsibilities: NDArrayF
    ) -> tuple[NDArrayF, float]:
        """Update von Mises-Fisher parameters."""
        d = int(directions.shape[1])
        sum_resp: float = float(np.sum(responsibilities))

        if sum_resp < 1e-15:
            mu = np.zeros(d, dtype=np.float64)
            mu[0] = 1.0
            return mu, 0.0

        # Weighted mean direction
        mean_direction: NDArrayF = (
            np.sum(responsibilities[:, np.newaxis] * directions, axis=0) / sum_resp
        ).astype(np.float64, copy=False)
        R: float = float(np.linalg.norm(mean_direction))

        if R < 1e-12:
            mu = np.zeros(d, dtype=np.float64)
            mu[0] = 1.0
            kappa = 0.0
        else:
            mu = (mean_direction / R).astype(np.float64, copy=False)
            kappa = self._estimate_kappa(float(R), d)

        return mu, float(kappa)

    def _estimate_kappa(self, R: float, d: int) -> float:
        """Estimate concentration parameter kappa from mean resultant length R."""
        R = float(min(max(R, 0.0), 1.0 - 1e-12))
        if R >= 1.0 - 1e-12:
            return 1_000.0  # very concentrated

        if d == 2:
            # Circular case: approximations for kappa(R)
            if R < 0.53:
                return float(2.0 * R + R**3 + 5.0 * (R**5) / 6.0)
            elif R < 0.85:
                return float(-0.4 + 1.39 * R + 0.43 / (1.0 - R))
            else:
                denom = R**3 - 4.0 * R**2 + 3.0 * R
                if abs(denom) < 1e-12:
                    return 1_000.0
                return float(1.0 / denom)
        else:
            # Higher dimensions: Banerjee et al. style approximation
            return float(R * (d - R**2) / (1.0 - R**2))

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------
    def _vmf_pdf_single(self, x: NDArrayF, mu: NDArrayF, kappa: float) -> float:
        """von Mises–Fisher PDF for a single point w.r.t. surface area measure."""
        x_arr: NDArrayF = np.asarray(x, dtype=np.float64).reshape(1, -1)
        return float(self._vmf_pdf_many(x_arr, mu, kappa)[0])

    def _vmf_pdf_many(self, X: NDArrayF, mu: NDArrayF, kappa: float) -> NDArrayF:
        """von Mises–Fisher PDF for a batch of unit rows w.r.t. surface area measure."""
        return vmf_pdf_batch(X, mu, kappa)

    def _update_beta(self, params: vMFNMParameters, beta: float, K: int) -> float:
        """Update penalty parameter beta (simple adaptive heuristic)."""
        # Use mixture spread as a guide; keep bounded in [0, 1].
        min_pi: float = float(np.min(params.pi))
        max_pi: float = float(np.max(params.pi))

        # Dimension based factor (similar spirit to paper’s dependence on d)
        d: int = int(params.mu.shape[1])
        eta: float = float(min(1.0, 0.5 ** (max(0, int(d / 2) - 1))))

        # Heuristic terms
        # Encourage spreading when weights are peaky (max_pi large, min_pi small)
        term1: float = float(
            (1.0 / float(K))
            * np.sum(
                np.exp(-eta * float(K) * np.abs(params.pi - float(np.mean(params.pi))))
            )
        )
        term2: float = float((1.0 - max_pi) / max(min_pi, 1e-15))

        new_beta: float = float(min(term1, term2))
        # Blend with previous beta to avoid oscillations
        return float(0.5 * beta + 0.5 * max(0.0, min(1.0, new_beta)))

    def _weighted_log_likelihood(
        self, data: NDArrayF, params: vMFNMParameters, weights: NDArrayF
    ) -> float:
        """Compute weighted log-likelihood."""
        dist = vMFNMDistribution(params)
        pdf_vals: NDArrayF = np.asarray(dist.pdf(data), dtype=np.float64)
        log_pdf: NDArrayF = np.log(np.maximum(pdf_vals, 1e-15)).astype(
            np.float64, copy=False
        )
        return float(np.sum(weights * log_pdf))

    def _copy_parameters(self, params: vMFNMParameters) -> vMFNMParameters:
        """Create a deep copy of parameters."""
        return vMFNMParameters(
            pi=params.pi.copy(),
            m=params.m.copy(),
            Omega=params.Omega.copy(),
            mu=params.mu.copy(),
            kappa=params.kappa.copy(),
        )
