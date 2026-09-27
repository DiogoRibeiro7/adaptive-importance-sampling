# Running and diagnosing

## Parameters

[`SafeICE`][safe_ice.SafeICE] works with its defaults on the paper's benchmarks.
These are the settings worth knowing about:

| Parameter | Default | What it controls |
| --- | --- | --- |
| `N` | `1000` | Samples per iteration, and for the final estimate. More samples give a less noisy estimate at a proportional cost. |
| `K0` | `20` | Mixture components to start from. Penalized EM prunes the ones it does not need, so start above the number of failure modes you expect. |
| `delta_star` | `1.5` | Stopping threshold on the coefficient of variation of the stopping weights. Lower means more iterations and a proposal closer to the failure region. |
| `delta_target` | `4.0` | Target coefficient of variation used to choose the next \(\sigma\). Higher lets \(\sigma\) fall faster. |
| `max_iterations` | `20` | Hard limit on iterations. |
| `sigma0` | `"auto"` | Initial smoothing parameter. See [Scale matters](limit-states.md#scale-matters). |
| `sigma0_pilot` | `256` | Limit-state evaluations spent choosing `sigma0` when it is `"auto"`. |
| `em_max_iter` | `20` | EM iterations per refit of the mixture. |
| `lambda_max` | `0.95` | Largest share of samples drawn from the light-tailed mixture; the heavy-tailed component always keeps the rest. |
| `random_state` | `None` | Seed or generator. See [Reproducibility](#reproducibility). |

`cv_tolerance` is accepted and stored, but no part of the algorithm reads it.

## What `run()` returns

`run(initial_params=None, verbose=True)` returns `(pf, results)`. `pf` is the
estimated failure probability, clamped to \([0, 1]\). `results` holds:

| Key | Type | Meaning |
| --- | --- | --- |
| `failure_probability` | `float` | The same value as `pf` |
| `pf_unclamped` | `float` | The estimate before clamping; see [below](#estimates-outside-0-1) |
| `iterations` | `list[dict]` | One record per iteration, with `iteration`, `K`, `sigma`, `lambda`, `n_failures` and a `parameters` snapshot of the mixture |
| `final_components` | `int` | Mixture components remaining at the end |
| `final_sigma`, `final_lambda` | `float` | \(\sigma\) and \(\lambda\) of the final proposal |
| `final_cv` | `float` | Coefficient of variation of the stopping weights at the last iteration |
| `final_parameters` | [`vMFNMParameters`][safe_ice.vMFNMParameters] | The fitted mixture |
| `final_samples` | `ndarray (N, d)` | The fresh draw from the final proposal that the estimate is computed from |
| `final_weights` | `ndarray (N,)` | Importance weights \(\varphi(\mathbf{u})/q_{\text{safe}}(\mathbf{u})\) for those samples |
| `final_g_values` | `ndarray (N,)` | Limit-state values for those samples |
| `all_samples`, `all_g_values` | `ndarray` | Every iteration's samples and values, stacked. For diagnostics only; the estimate does not use them. |
| `history` | `dict` | Per-iteration lists `sigma`, `cv`, `components` and `lambda_val`, plus `pf_estimates` |
| `convergence_metrics` | `dict` | `cv_values`, `cv_threshold`, `sigma_values`, `lambda_values` and `pf_estimates`, ready to plot |

The estimate is the mean of `(final_g_values <= 0) * final_weights`, so you can
recompute it, or its standard error, from those three arrays:

```python
import numpy as np

contributions = (results["final_g_values"] <= 0) * results["final_weights"]
standard_error = contributions.std(ddof=1) / np.sqrt(contributions.size)
```

The iteration *count* is `len(results["iterations"])`.

## Checking convergence

A run stops when the coefficient of variation of the stopping weights reaches
`delta_star`, or when it runs out of iterations. The first iteration always
reports `CV=inf`, because it draws only from the heavy-tailed component to
explore, so every run takes at least two iterations.

Reaching `max_iterations` does not raise a warning. Check for it yourself:

```python
ice = SafeICE(g, dimension=d)
pf, results = ice.run(verbose=False)

if results["final_cv"] > ice.delta_star:
    print("stopped at max_iterations without converging")
```

## Estimates outside [0, 1]

The importance-sampling estimator is unbiased but unconstrained. The weights are
a ratio of densities and nothing bounds them by 1, so a single run can land
above 1. The returned `pf` is always clamped to \([0, 1]\).

On a rare-event problem this never fires, because the estimates sit orders of
magnitude below 1. When it does fire, the proposal is not covering the target
and a few samples carry the whole estimate. The run then raises a
`RuntimeWarning` and keeps the raw value in `results["pf_unclamped"]`. Treat
such a run as unconverged, not as a probability of 1.

## Reproducibility

Every estimator takes `random_state`:

```python
import numpy as np

from safe_ice import SafeICE

SafeICE(g, dimension=2, random_state=42)  # seeded
SafeICE(g, dimension=2, random_state=np.random.default_rng(0))  # your generator
SafeICE(g, dimension=2)  # NumPy's global state
```

What `None` means depends on the class:

| Class | `random_state=None` |
| --- | --- |
| [`SafeICE`][safe_ice.SafeICE] and its subclasses | NumPy's global random state, so results depend on whatever else has drawn from it in the same process |
| [`CrossEntropyGaussianMixture`][safe_ice.CrossEntropyGaussianMixture], [`SubsetSimulation`][safe_ice.SubsetSimulation] | A fresh, unseeded generator |
| [`PerformanceEvaluator.run_monte_carlo_reference`][safe_ice.PerformanceEvaluator.run_monte_carlo_reference] | NumPy's global random state |

Seed your runs when you compare results.

## Diagnostics

### Convergence plots

[`AdvancedAnalysis`][safe_ice.AdvancedAnalysis] draws the evolution of the
component count, \(\sigma\), \(\lambda\) and the CV against its threshold. Its
methods return the Matplotlib figure; pass `show=False` to skip the window, for
example in a script or a test:

```python
from safe_ice import AdvancedAnalysis

fig = AdvancedAnalysis.analyze_component_evolution(results, show=False)
fig.savefig("convergence.png")

# Two-dimensional problems only: final samples, coloured by failure.
AdvancedAnalysis.analyze_sample_distribution(results, g, show=False)
```

### Interactive plots

With the `viz` extra installed, the Plotly equivalents live in
`safe_ice.analysis.interactive_visualization`:

```python
from safe_ice.analysis.interactive_visualization import (
    InteractiveVisualizer,
    create_interactive_dashboard,
)

viz = InteractiveVisualizer()
viz.plot_convergence_interactive(results)
viz.plot_mixture_evolution(results)  # 2D runs: the fitted density, per iteration

create_interactive_dashboard(results)
```

### A crude Monte Carlo reference

When the probability is not too small, check an estimate against brute force.
[`PerformanceEvaluator`][safe_ice.PerformanceEvaluator] samples the prior
directly and returns the estimate and its standard error:

```python
from safe_ice import PerformanceEvaluator

pf_mc, se_mc = PerformanceEvaluator.run_monte_carlo_reference(
    g, dimension=2, n_samples=10_000_000, random_state=0
)
```

Crude Monte Carlo needs about \(1/(P_F\,\delta^2)\) samples for a coefficient of
variation \(\delta\), so ten million samples resolve \(10^{-5}\) to about 10%.

### Spread across runs

A single estimate is one draw of a random variable. `compare_methods` repeats
[`SafeICE`][safe_ice.SafeICE] and summarises the spread:

```python
comparison = PerformanceEvaluator.compare_methods(
    g,
    dimension=2,
    reference_pf=pf_mc,
    n_runs=10,
    safe_ice_params={"K0": 8, "N": 2000},
)
comparison["safe_ice"]  # estimates, mean, std, cv, mean_iterations, ...
```

!!! warning

    Do not put a fixed integer `random_state` in `safe_ice_params`. Every run
    would then use the same seed and produce the same estimate, and the spread
    would read as zero.

For comparisons across estimators, see
[Choosing an estimator](estimators.md#cross-checking-an-answer).
