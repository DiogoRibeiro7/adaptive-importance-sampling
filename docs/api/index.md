# API reference

The reference is generated from the docstrings in the source tree, so it always
describes the commit the site was built from.

Everything listed here except the interactive Plotly helpers is importable from
the top-level package:

```python
from safe_ice import SafeICE, BenchmarkProblems, MarginalTransform
```

| Page | Contents |
| --- | --- |
| [Estimators](estimators.md) | [`SafeICE`][safe_ice.SafeICE] and its variants, the [`ICEvMFNM`][safe_ice.ICEvMFNM], [`CrossEntropyGaussianMixture`][safe_ice.CrossEntropyGaussianMixture] and [`SubsetSimulation`][safe_ice.SubsetSimulation] baselines, and the [`vMFNMParameters`][safe_ice.vMFNMParameters] container |
| [Distributions](distributions.md) | The von Mises-Fisher, Nakagami and inverse Nakagami building blocks, and the [`vMFNMDistribution`][safe_ice.vMFNMDistribution] mixture |
| [Penalized EM](optimization.md) | [`PenalizedEMOptimizer`][safe_ice.PenalizedEMOptimizer], the M-step that prunes mixture components |
| [Problems](problems.md) | The paper's benchmark limit states, the heat transfer PDE problem, and system, time-variant, random-field and network problems |
| [Analysis](analysis.md) | Crude Monte Carlo references, repeated-run comparisons, and Matplotlib and Plotly diagnostics |
| [Transforms](transforms.md) | [`MarginalTransform`][safe_ice.MarginalTransform], for limit states written in physical units |

## Conventions

These hold across the whole package.

**Standard normal space.** Every estimator works in `u`-space, where the inputs
are independent standard normals. A limit state stated in physical units is
mapped there with [`MarginalTransform`][safe_ice.MarginalTransform].

**Failure is `g(u) <= 0`.** A limit-state function receives an `(n, d)` array
and returns an `(n,)` array. Scalar-only functions are accepted and evaluated
row by row, which is much slower.

**`run()` returns `(pf, results)`.** `pf` is a float in `[0, 1]`; `results` is a
dictionary of diagnostics whose keys are described in
[Running and diagnosing](../guide/running.md#what-run-returns).

**`random_state` controls reproducibility.** It accepts an `int`, a
`numpy.random.Generator`, or `None`. See
[Reproducibility](../guide/running.md#reproducibility) for what `None` means for
each class.
