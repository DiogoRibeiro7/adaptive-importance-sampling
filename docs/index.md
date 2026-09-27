<div class="safe-ice-hero" markdown="1">

# Failure probabilities too small for Monte Carlo

Safe-ICE estimates \(P_F = \mathbb{P}(g(\mathbf{U}) \le 0)\) for a limit-state
function \(g\) when the answer is \(10^{-5}\) or smaller. It needs thousands of
evaluations of \(g\) rather than millions, using an importance-sampling proposal
that picks its own complexity and never drops its tails.

[Get started](getting-started.md){ .md-button .md-button--primary }
[Read the guide](guide/limit-states.md){ .md-button }

</div>

<div class="safe-ice-cards" markdown="1">

<div markdown="1">

### A proposal shaped for the tails

Directions follow a von Mises-Fisher distribution and radii a Nakagami, so the
mixture can represent the thin shells that failure regions become in high
dimensions. Penalized EM prunes redundant components, so you do not choose
their number.

</div>

<div markdown="1">

### A heavy-tailed safety net

An inverse-Nakagami component keeps probability mass in the far tail for the
whole run. A failure mode the mixture has not found yet cannot then send the
importance weights out of control, which is what makes the method *safe*.

</div>

<div markdown="1">

### Checked against known answers

Estimates are tested against closed forms, crude Monte Carlo over \(2\times10^7\)
samples, and independent subset simulation. On spheres from \(d=2\) to
\(d=200\) the median estimate is within 2% of the exact probability.

</div>

</div>

## What it does

A structure, a levee or a network fails when some limit-state function of its
uncertain inputs drops to zero or below. Safe-ICE works in standard normal space
and treats \(g\) as a black box: give it a function and a dimension and it
returns the failure probability.

```python
from safe_ice import BenchmarkProblems, SafeICE

g = BenchmarkProblems.four_mode_series_system()  # reference P_F = 6.5e-5
pf, results = SafeICE(g, dimension=2, random_state=0).run(verbose=False)
```

Crude Monte Carlo would need about \(1.5\times10^6\) evaluations to estimate that
probability to 10%. Safe-ICE draws 1,000 samples per iteration and stops when
its importance weights have settled, which on this problem takes two or three
iterations. It then draws once more from the final proposal and returns the
importance-sampling estimate: about 3,000 evaluations in all. [Theory](theory.md)
explains each step.

This is an implementation of

> Gao, Z. and Karniadakis, G. (2025). *Safe Cross-Entropy-Based Importance
> Sampling for Rare Event Simulations.*
> [arXiv:2509.07160](https://arxiv.org/abs/2509.07160).

## What is in the package

| | |
| --- | --- |
| **Estimators** | [`SafeICE`][safe_ice.SafeICE], a vectorised [`OptimizedSafeICE`][safe_ice.OptimizedSafeICE], [`AdaptiveSafeICE`][safe_ice.AdaptiveSafeICE] with dimension-aware defaults, and three independent baselines to check it against: [`ICEvMFNM`][safe_ice.ICEvMFNM], [`CrossEntropyGaussianMixture`][safe_ice.CrossEntropyGaussianMixture] and [`SubsetSimulation`][safe_ice.SubsetSimulation] |
| **Problems** | The paper's benchmarks, including a nonlinear oscillator and a heat transfer PDE, plus system, time-variant, random-field and network limit states |
| **Physical units** | [`MarginalTransform`][safe_ice.MarginalTransform] maps arbitrary marginals, correlated through a Nataf transformation, into standard normal space |
| **Diagnostics** | Per-iteration records, convergence plots in Matplotlib or Plotly, and crude Monte Carlo references |
| **Notebooks** | [Six executed notebooks](examples.md#notebooks), from a first run to 95 years of real river gauge data |

!!! info "Status: alpha"

    The estimator tracks known analytical answers closely, but the API may still
    change before 1.0 and the package is not yet on PyPI.
    [Accuracy and benchmarks](guide/accuracy.md) has the numbers and the known
    limitations.
