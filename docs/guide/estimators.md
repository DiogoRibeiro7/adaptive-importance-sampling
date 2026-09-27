# Choosing an estimator

Start with [`SafeICE`][safe_ice.SafeICE]. The other classes are either the same
algorithm with different engineering, or independent methods to check it
against.

| Class | What it is | Reach for it when |
| --- | --- | --- |
| [`SafeICE`][safe_ice.SafeICE] | The method of the paper | You want an estimate |
| [`OptimizedSafeICE`][safe_ice.OptimizedSafeICE] | The same algorithm with batched sampling | `N` is large or you repeat many runs |
| [`AdaptiveSafeICE`][safe_ice.AdaptiveSafeICE] | `OptimizedSafeICE` with its settings chosen from the dimension | A high-dimensional problem, without tuning |
| [`ICEvMFNM`][safe_ice.ICEvMFNM] | Safe-ICE minus its two additions | Measuring what those additions buy |
| [`CrossEntropyGaussianMixture`][safe_ice.CrossEntropyGaussianMixture] | The older elite-sample cross-entropy method | Reproducing the case for ICE |
| [`SubsetSimulation`][safe_ice.SubsetSimulation] | Markov chains through nested failure levels | An independent second opinion |

Every class returns `(pf, results)` from `run()` and takes a `random_state`.

## The Safe-ICE family

### SafeICE

The reference implementation, which draws its proposal one sample at a time. It
is the only class that defaults to `sigma0="auto"` (see
[Scale matters](limit-states.md#scale-matters)).

### OptimizedSafeICE

Overrides only the sampling step: component assignments are drawn in bulk and
each component's samples are generated in one call. The sigma schedule,
penalized EM, weights and final estimator are inherited, so it gives the same
estimates as `SafeICE` up to the random stream.

- `batch_size` caps how many points reach the limit-state function in one call,
  which bounds memory when \(g\) allocates per point. It defaults to
  `min(N, 10000)`.
- `enable_caching` computes each component's matched inverse-Nakagami scale once
  per iteration rather than once per sample.
- `enable_parallel` is accepted for backwards compatibility and has no effect.
- `sigma0` defaults to `1.0`, so \(g\) should be of order one.

### AdaptiveSafeICE

Chooses `N`, `K0`, `delta_target` and `delta_star` from the dimension, and
raises `max_iterations` to 30. The algorithm is inherited unchanged from
`OptimizedSafeICE`, and any of the four can still be passed explicitly.

| \(d\) | `N` | `K0` | `delta_target` | `delta_star` |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 500 | 10 | 3.0 | 1.5 |
| 5 | 1,581 | 15 | 3.5 | 1.0 |
| 10 | 4,472 | 20 | 4.0 | 1.0 |
| 20 | 9,486 | 30 | 4.5 | 0.75 |
| 50 | 25,000 | 50 | 6.0 | 0.75 |
| 100 | 50,000 | 50 | 6.0 | 0.75 |

`N` grows with \(\sqrt{d}\) and is capped at 50,000. That is a lot of limit-state
evaluations in high dimensions; pass `N` yourself if \(g\) is expensive.

## Baselines

The baselines are implemented as their papers describe, without patching around
their known weaknesses, because those weaknesses are the reason to compare
against them.

### ICEvMFNM

Improved cross-entropy importance sampling with a vMF-Nakagami mixture
(Papaioannou, Geyer and Straub, 2019). Safe-ICE is this method plus two
additions, and removing both recovers it exactly:

- \(\lambda\) is held at 1, so there is no heavy-tailed component;
- the penalty coefficient is held at 0, so the M-step is plain weighted EM and
  the number of components `K` (default 2) is fixed.

Everything else is shared with `SafeICE` through inheritance, so a comparison
between the two differs only in the parts being compared.

### CrossEntropyGaussianMixture

Cross-entropy importance sampling with a Gaussian mixture (Kurtz and Song,
2013), the method ICE was introduced to improve on. Each iteration keeps the
best `rho` fraction of samples (default 0.1) and refits a `K`-component Gaussian
mixture (default 2) to them alone. `results["samples_discarded"]` reports how
much of the sampling budget never informed the fit.

### SubsetSimulation

Not importance sampling at all (Au and Beck, 2001). It factorises the rare event
into a chain of nested, more frequent ones,

\[
P_F = \mathbb{P}(F_1) \prod_{i=2}^{m} \mathbb{P}(F_i \mid F_{i-1}),
\]

setting each threshold so that a fraction `p0` (default 0.1) of the samples
pass, and generating the next level's samples by Markov chain Monte Carlo. It
needs about \(\log P_F / \log p_0\) levels of `N` samples each.

- `N * p0` must be a whole number of seeds that divides `N`.
- `results["reached_failure_set"]` is `False` if it ran out of `max_levels`.
- The coefficient of variation it reports treats the levels as independent,
  which understates it.

## How they compare

These figures are measured in
[notebook 06](../examples.md#notebooks), which runs all four methods on the same
problems. On a gently-shaped two-dimensional problem they all find the answer;
what separates them is multiple failure modes and dimension.

**Multiple modes.** On the four-mode series system, the number of failure lobes
each method reached, across six seeds:

| Method | Lobes reached, out of 4 | Worst seed |
| --- | --- | ---: |
| Safe-ICE | 4, 4, 4, 4, 4, 4 | 0.79× |
| ICE-vMFNM | 4, 4, 1, 4, 4, 4 | 0.13× |
| CE-GM | 3, 3, 4, 3, 4, 2 | 0.17× |
| Subset simulation | 4, 4, 4, 4, 4, 4 | 0.58× |

CE-GM falls short because it fits only the best tenth of each iteration, and a
mixture fitted to elites concentrates wherever the first elites landed.
ICE-vMFNM usually finds all four lobes, but on one seed it collapsed onto a
single one: nothing holds its components apart, which is what Safe-ICE's
penalty adds. Subset simulation reaches all four because its chains walk rather
than fit, so there is no fitted object to collapse.

**Dimension.** On the sphere problem, estimate over the exact value:

| \(d\) | Exact | CE-GM | Safe-ICE |
| ---: | ---: | ---: | ---: |
| 2 | \(2.19\times10^{-3}\) | 0.72× | 1.00× |
| 10 | \(5.35\times10^{-3}\) | 0.45× | 1.00× |
| 50 | \(3.61\times10^{-3}\) | 0.01× | 1.01× |
| 100 | \(2.63\times10^{-3}\) | \(3.5\times10^{-10}\)× | 1.04× |

A Gaussian mixture on \(\mathbb{R}^d\) has to represent the thin shell where a
high-dimensional normal puts its mass with an ellipsoid, and fails. The
vMF-Nakagami mixture separates radius from direction and does not have that
problem.

## Cross-checking an answer

Two estimators that share an implementation can be wrong together.
[`SubsetSimulation`][safe_ice.SubsetSimulation] uses no proposal, mixture,
importance weights or EM step, so agreement between it and Safe-ICE is evidence
about the answer rather than about the code:

```python
from safe_ice import SafeICE, SubsetSimulation

pf_ice, _ = SafeICE(g, dimension=d, random_state=0).run(verbose=False)
pf_ss, ss = SubsetSimulation(g, dimension=d, random_state=0).run(verbose=False)

print(
    f"Safe-ICE {pf_ice:.3e}, subset simulation {pf_ss:.3e} "
    f"from {ss['n_evaluations']} evaluations"
)
```

On the flood example in [notebook 05](../examples.md#notebooks) the three
methods agree to within their spread:

| Method | \(\mathbb{P}(\text{overtopping})\) | Evaluations |
| --- | ---: | ---: |
| Crude Monte Carlo | \(5.750\times10^{-5}\) | 2,000,000 |
| Safe-ICE | \(5.391\times10^{-5}\) | 3,000 |
| Subset simulation | \(4.765\times10^{-5}\) | 10,000 |

## Where Safe-ICE struggles

- **A limit state on an awkward scale.** The smoothing only works if \(g\) is
  of order one relative to \(\sigma\). `SafeICE` handles this with
  `sigma0="auto"`; the other classes need \(g\) scaled, for example with
  [`MarginalTransform.wrap`][safe_ice.MarginalTransform.wrap].
- **A limit state with nothing to follow.** The smoothed indicator
  \(\Phi(-g/\sigma)\) guides the proposal toward failure only if \(g\) grows
  more negative as failure gets closer. A limit state that returns only
  \(\pm1\), such as a network connectivity check, gives it nothing to follow.
  Subset simulation is the better tool there.
