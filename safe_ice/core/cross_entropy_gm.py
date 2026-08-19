"""Cross-entropy importance sampling with a Gaussian mixture proposal.

    Kurtz and Song, "Cross-entropy-based adaptive importance sampling using
    Gaussian mixture", Structural Safety 42:35-44, 2013

which is reference [25] of the Safe-ICE paper, and the older method that ICE was
introduced to improve on. It is here as a third point of comparison, and because
the Safe-ICE paper's case rests on two specific claims about it that are worth
being able to test rather than cite:

* it "discards most samples by relying solely on an elite subset -- diminishing
  statistical efficiency";
* "in high dimensions, its Gaussian proposals collapse onto a thin shell,
  causing numerical instability".

Both are properties of the method, so this implements it as described rather
than patching around them. See ``tests/test_cross_entropy_gm.py`` for what they
measure out at.

How it differs from ICE
-----------------------
The two methods share the idea of walking a proposal in towards the failure
region through a sequence of easier problems, and differ in how each step is
defined and fitted.

* **The intermediate level.** CE sets a hard threshold at the ``rho``-quantile
  of the sampled ``g`` and keeps the samples below it. ICE replaces that
  indicator with a smooth one, ``Phi(-g/sigma)``, so every sample contributes
  something rather than the best tenth contributing everything.
* **The family.** A Gaussian mixture on ``R^d`` here; a von Mises-Fisher
  Nakagami mixture in polar form there, which separates radius from direction
  and so does not have to represent a shell with an ellipsoid.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats

from ..typing import LimitStateFunction, NDArrayF

#: Added to the diagonal of each covariance before it is used. Fitting a
#: d-dimensional covariance from a tenth of the samples is barely determined at
#: d=50 and impossible beyond it, so without this the mixture stops being usable
#: before the method's own weakness has a chance to show.
_COVARIANCE_FLOOR = 1e-8


class CrossEntropyGaussianMixture:
    """Estimate a failure probability by cross-entropy with a Gaussian mixture.

    Parameters
    ----------
    limit_state_function:
        Function g(u) such that failure occurs when g(u) <= 0.
    dimension:
        Problem dimension.
    K:
        Number of Gaussian components, fixed for the whole run. As with
        ICE-vMFNM there is no mechanism for adapting it.
    N:
        Samples per iteration.
    rho:
        Elite fraction. The intermediate threshold is the ``rho``-quantile of
        the sampled limit-state values, so ``rho * N`` samples are kept and the
        rest are discarded. Kurtz and Song use 0.1.
    max_iterations:
        Give up after this many iterations.
    em_iterations:
        Weighted EM steps used to refit the mixture each iteration.
    random_state:
        Seed or generator, for reproducibility.
    """

    def __init__(
        self,
        limit_state_function: LimitStateFunction,
        dimension: int,
        K: int = 2,
        N: int = 1000,
        rho: float = 0.1,
        max_iterations: int = 20,
        em_iterations: int = 20,
        random_state: int | np.random.Generator | None = None,
    ) -> None:
        self.g = limit_state_function
        self.d = int(dimension)
        self.K = int(K)
        self.N = int(N)
        self.rho = float(rho)
        self.max_iterations = int(max_iterations)
        self.em_iterations = int(em_iterations)

        if not 0.0 < self.rho < 1.0:
            raise ValueError(f"rho must lie strictly between 0 and 1, got {self.rho}.")
        if int(self.rho * self.N) < 1:
            raise ValueError(
                f"rho * N must keep at least one elite sample, got {self.rho * self.N}."
            )

        if random_state is None:
            self._rng: Any = np.random.default_rng()
        elif isinstance(random_state, np.random.Generator):
            self._rng = random_state
        else:
            self._rng = np.random.default_rng(int(random_state))

        self.history: dict[str, list[float]] = {"threshold": [], "elite_fraction": []}

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    def run(self, verbose: bool = True) -> tuple[float, dict[str, Any]]:
        """Estimate the failure probability."""
        weights, means, covariances = self._initial_mixture()

        if verbose:
            print("=" * 56)
            print("Cross-entropy with a Gaussian mixture")
            print(f"  dimension {self.d}, K={self.K}, N={self.N}, rho={self.rho}")
            print("=" * 56)

        iterations: list[dict[str, float]] = []
        reached_failure_set = False

        for iteration in range(self.max_iterations):
            samples = self._sample_mixture(weights, means, covariances, self.N)
            g_values = self._evaluate(samples)

            threshold = float(np.quantile(g_values, self.rho))
            if threshold <= 0.0:
                threshold = 0.0
                reached_failure_set = True

            elite = g_values <= threshold
            n_elite = int(np.count_nonzero(elite))
            self.history["threshold"].append(threshold)
            self.history["elite_fraction"].append(n_elite / self.N)
            iterations.append(
                {
                    "iteration": float(iteration),
                    "threshold": threshold,
                    "n_elite": float(n_elite),
                    "discarded": float(self.N - n_elite),
                }
            )

            if verbose:
                print(
                    f"  iteration {iteration}: threshold {threshold:+.4g}  "
                    f"elites {n_elite}/{self.N}  discarded {self.N - n_elite}"
                )

            if reached_failure_set:
                break
            if n_elite < self.K * 2:
                if verbose:
                    print("  too few elite samples to refit; stopping")
                break

            importance = self._importance_weights(
                samples[elite], weights, means, covariances
            )
            weights, means, covariances = self._fit_mixture(
                samples[elite], importance, weights, means, covariances
            )

        # Final estimate from a fresh draw off the converged proposal, the same
        # form as every other estimator here: mean of the indicator times p/q.
        final_samples = self._sample_mixture(weights, means, covariances, self.N)
        final_g = self._evaluate(final_samples)
        ratio = self._importance_weights(final_samples, weights, means, covariances)
        pf = float(np.mean((final_g <= 0.0).astype(np.float64) * ratio))
        pf = min(max(pf, 0.0), 1.0)

        total = self.N * (len(iterations) + 1)
        results: dict[str, Any] = {
            "failure_probability": pf,
            "iterations": iterations,
            "n_iterations": len(iterations),
            "n_evaluations": total,
            "samples_discarded": float(
                sum(record["discarded"] for record in iterations)
            ),
            "final_weights": weights,
            "final_means": means,
            "final_covariances": covariances,
            "final_samples": final_samples,
            "final_g_values": final_g,
            "reached_failure_set": reached_failure_set,
            "history": self.history,
        }

        if verbose:
            print("-" * 56)
            print(f"Failure probability: {pf:.6e}")
            print(
                f"Evaluations: {total:,}   discarded as non-elite: "
                f"{int(results['samples_discarded']):,}"
            )
            if not reached_failure_set:
                print("WARNING: stopped before the threshold reached zero")

        return pf, results

    # -------------------------------------------------------------------------
    # Internals
    # -------------------------------------------------------------------------
    def _evaluate(self, samples: NDArrayF) -> NDArrayF:
        """Evaluate g on a batch, falling back to row-wise if needed."""
        try:
            values = np.asarray(self.g(samples), dtype=np.float64).reshape(-1)
            if values.shape[0] != samples.shape[0]:
                raise ValueError("limit-state output shape mismatch")
        except (ValueError, TypeError, IndexError):
            values = np.array(
                [float(np.asarray(self.g(row)).reshape(-1)[0]) for row in samples],
                dtype=np.float64,
            )
        return np.where(np.isnan(values), np.inf, values)

    def _initial_mixture(self) -> tuple[NDArrayF, NDArrayF, NDArrayF]:
        """Start from the prior: components near the origin with unit spread."""
        weights = np.full(self.K, 1.0 / self.K, dtype=np.float64)
        means = np.asarray(
            self._rng.standard_normal((self.K, self.d)), dtype=np.float64
        )
        covariances = np.array(
            [np.eye(self.d) for _ in range(self.K)], dtype=np.float64
        )
        return weights, means, covariances

    def _sample_mixture(
        self,
        weights: NDArrayF,
        means: NDArrayF,
        covariances: NDArrayF,
        n: int,
    ) -> NDArrayF:
        """Draw from the current Gaussian mixture."""
        counts = self._rng.multinomial(n, weights)
        blocks = []
        for k, count in enumerate(counts):
            if count == 0:
                continue
            blocks.append(
                self._rng.multivariate_normal(means[k], covariances[k], size=int(count))
            )
        samples = np.vstack(blocks) if blocks else np.zeros((0, self.d))
        self._rng.shuffle(samples)
        return np.asarray(samples, dtype=np.float64)

    def _mixture_log_density(
        self,
        samples: NDArrayF,
        weights: NDArrayF,
        means: NDArrayF,
        covariances: NDArrayF,
    ) -> NDArrayF:
        """Log density of the mixture at each sample."""
        components = np.empty((samples.shape[0], self.K), dtype=np.float64)
        for k in range(self.K):
            components[:, k] = np.log(max(float(weights[k]), 1e-300)) + (
                stats.multivariate_normal.logpdf(
                    samples, mean=means[k], cov=covariances[k], allow_singular=True
                )
            )
        maximum = np.max(components, axis=1, keepdims=True)
        return np.asarray(
            (
                maximum
                + np.log(np.sum(np.exp(components - maximum), axis=1, keepdims=True))
            ).reshape(-1),
            dtype=np.float64,
        )

    def _importance_weights(
        self,
        samples: NDArrayF,
        weights: NDArrayF,
        means: NDArrayF,
        covariances: NDArrayF,
    ) -> NDArrayF:
        """p(u) / q(u), with p the standard normal prior."""
        log_prior = np.asarray(
            stats.multivariate_normal.logpdf(
                samples, mean=np.zeros(self.d), cov=np.eye(self.d)
            ),
            dtype=np.float64,
        ).reshape(-1)
        log_proposal = self._mixture_log_density(samples, weights, means, covariances)
        return np.exp(np.clip(log_prior - log_proposal, -700.0, 700.0))

    def _fit_mixture(
        self,
        samples: NDArrayF,
        importance: NDArrayF,
        weights: NDArrayF,
        means: NDArrayF,
        covariances: NDArrayF,
    ) -> tuple[NDArrayF, NDArrayF, NDArrayF]:
        """Weighted EM for the Gaussian mixture, on the elite samples only."""
        n = samples.shape[0]
        weights = weights.copy()
        means = means.copy()
        covariances = covariances.copy()

        for _ in range(self.em_iterations):
            # E-step
            responsibilities = np.empty((n, self.K), dtype=np.float64)
            for k in range(self.K):
                responsibilities[:, k] = np.log(max(float(weights[k]), 1e-300)) + (
                    stats.multivariate_normal.logpdf(
                        samples, mean=means[k], cov=covariances[k], allow_singular=True
                    )
                )
            maximum = np.max(responsibilities, axis=1, keepdims=True)
            responsibilities = np.exp(responsibilities - maximum)
            totals = np.sum(responsibilities, axis=1, keepdims=True)
            responsibilities = np.divide(
                responsibilities,
                totals,
                out=np.full_like(responsibilities, 1.0 / self.K),
                where=totals > 0.0,
            )

            # M-step, weighted by the importance weights
            weighted = responsibilities * importance[:, None]
            mass = np.sum(weighted, axis=0)
            total_mass = float(np.sum(mass))
            if total_mass <= 0.0:
                break

            weights = mass / total_mass
            for k in range(self.K):
                if mass[k] <= 0.0:
                    continue
                means[k] = np.sum(weighted[:, k, None] * samples, axis=0) / mass[k]
                centred = samples - means[k]
                covariances[k] = (centred.T @ (weighted[:, k, None] * centred)) / mass[
                    k
                ] + _COVARIANCE_FLOOR * np.eye(self.d)

        return weights, means, covariances
