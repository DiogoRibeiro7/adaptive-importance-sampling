# Limit-state functions

A limit-state function \(g\) is the whole description of the problem. The
estimators never look inside it: they call it on points and read the sign of
the result.

## The contract

- **Input.** An `(n, d)` NumPy array of points in standard normal space: each
  coordinate of \(\mathbf{U}\) is an independent \(\mathcal{N}(0, 1)\) variable.
- **Output.** An array of `n` values, one per row.
- **Failure** is `g(u) <= 0`. The estimate is \(\mathbb{P}(g(\mathbf{U}) \le 0)\).

Write \(g\) to work on the whole batch at once, with `axis=-1` reductions and
column indexing such as `u[:, 0]`:

```python
import numpy as np


def g(u):
    """A linear limit state at reliability index 3.5 along the diagonal."""
    return 3.5 - u.sum(axis=-1) / np.sqrt(u.shape[-1])
```

If the batch call raises, or returns the wrong number of values, the estimators
fall back to calling `g` on one `(1, d)` row at a time. That keeps scalar-only
code working, but it is much slower, so vectorise when you can.

!!! warning "NaN is treated as safe"

    A `NaN` returned by `g` is replaced with `+inf`, which counts as not failed,
    and a `RuntimeWarning` is raised. If your model can fail to evaluate, decide
    what that should mean for the structure and return a number that says so.

## Systems of components

A system with several failure modes combines their limit states. A **series**
system fails when any component fails, so take the minimum; a **parallel**
system fails only when every component fails, so take the maximum.

```python
def series(u):
    g1 = 3.0 - u[:, 0]  # fails for large u1
    g2 = 3.0 + u[:, 1]  # fails for very negative u2
    return np.minimum(g1, g2)


def parallel(u):
    return np.maximum(3.0 - u[:, 0], 3.0 + u[:, 1])
```

Each mode is a separate region of the input space, and an estimator that finds
only some of them underestimates the probability. This is the case the
vMF-Nakagami mixture and its heavy-tailed component are built for. The
[four-mode benchmark][safe_ice.BenchmarkProblems.four_mode_series_system] is a
series system of this kind, and
[`SystemReliabilityProblem`][safe_ice.SystemReliabilityProblem] builds series,
parallel, k-out-of-n and correlated systems for experiments.

## Scale matters

The method replaces the failure indicator with a smooth approximation,
\(\Phi(-g(\mathbf{u})/\sigma)\), and shrinks \(\sigma\) step by step (see
[Theory](../theory.md#improved-cross-entropy)). Only the ratio \(g/\sigma\)
matters, so the starting value \(\sigma_0\) has to suit the spread of \(g\).
A \(g\) measured in newtons, with a spread in the hundreds, would make
\(\Phi(-g/1)\) a hard indicator from the start and the smoothing would do
nothing.

[`SafeICE`][safe_ice.SafeICE] handles this by default. With `sigma0="auto"` it
spends `sigma0_pilot=256` evaluations of \(g\) on prior samples and starts from
\(\max(1, \operatorname{std}(g))\). Starting too high only costs an extra
iteration, while starting too low cannot be undone, so the rule errs upwards.

If \(g\) is expensive and you already know its scale, pass a number instead to
skip the pilot:

```python
SafeICE(g, dimension=10, sigma0=25.0)
```

!!! note

    [`OptimizedSafeICE`][safe_ice.OptimizedSafeICE] and
    [`AdaptiveSafeICE`][safe_ice.AdaptiveSafeICE] default to `sigma0=1.0`, not
    `"auto"`. With them, make sure \(g\) is of order one, for example by
    wrapping it as described below.

Dividing \(g\) by a positive constant never changes the answer, because
\(\{g \le 0\}\) and \(\{g/c \le 0\}\) are the same set.

## Problems in physical units

Real inputs are rarely standard normal. A resistance might be lognormal, a
flood peak Gumbel. [`MarginalTransform`][safe_ice.MarginalTransform] maps
between physical space and standard normal space, and `wrap()` turns a limit
state written in physical units into one the estimators accept.

```python
import numpy as np
from scipy import stats

from safe_ice import MarginalTransform, SafeICE

resistance = stats.lognorm(s=0.15, scale=200.0)  # median 200 kN
load = stats.lognorm(s=0.25, scale=80.0)  # median 80 kN
transform = MarginalTransform([resistance, load])


def capacity_margin(x):
    """x holds physical values: column 0 is resistance, column 1 is load."""
    return x[:, 0] - x[:, 1]


g = transform.wrap(capacity_margin)
pf, results = SafeICE(g, dimension=2, random_state=0).run(verbose=False)

# Both inputs are lognormal, so this problem happens to have a closed form.
beta = np.log(200.0 / 80.0) / np.hypot(0.15, 0.25)
print(f"Safe-ICE {pf:.3e}   exact {stats.norm.cdf(-beta):.3e}")
```

```text
Safe-ICE 7.562e-04   exact 8.366e-04
```

By default `wrap()` also rescales the result to a spread of order one, using a
pilot of 512 physical samples; here it divides by about 35. Pass `scale=False`
to leave \(g\) as it is, or a number to divide by that. The divisor is stored on
the wrapped function as `g.limit_state_scale`.

With independent marginals each coordinate maps on its own:

\[
u_i = \Phi^{-1}\bigl(F_i(x_i)\bigr), \qquad
x_i = F_i^{-1}\bigl(\Phi(u_i)\bigr).
\]

To inspect the result in physical units, map the samples back. For example,
these are the failed samples from the final proposal:

```python
failed = results["final_samples"][results["final_g_values"] <= 0]
transform.to_physical(failed)  # (resistance, load) pairs that failed
```

### Correlated inputs

Pass the correlation matrix of the *physical* variables:

```python
transform = MarginalTransform(
    [resistance, load],
    correlation=np.array([[1.0, 0.3], [0.3, 1.0]]),
)
transform.gaussian_correlation  # what the underlying normals need
```

This is the Nataf transformation. The marginal maps are non-linear, so they
distort correlation: the Gaussian correlation \(\rho_z\) that produces a
physical correlation \(\rho_x\) solves

\[
\rho_x = \iint
  \frac{F_i^{-1}(\Phi(z_i)) - \mu_i}{\sigma_i}\,
  \frac{F_j^{-1}(\Phi(z_j)) - \mu_j}{\sigma_j}\,
  \varphi_2(z_i, z_j; \rho_z)\, dz_i\, dz_j ,
\]

which is solved for each pair by Gauss-Hermite quadrature and a bracketed root
find. The distortion grows with skew: for two lognormals with a 30% coefficient
of variation, a physical correlation of 0.8 needs a Gaussian correlation of
0.806, and \(-0.5\) needs \(-0.535\).

`transform.sample(n)` draws physical samples with the requested marginals and
correlation, which is useful for checking the transform against data and for
crude Monte Carlo in physical space.

## Expensive limit states

Every call to \(g\) is paid for. A [`SafeICE`][safe_ice.SafeICE] run costs

| Stage | Evaluations of \(g\) |
| --- | --- |
| Scale pilot, when `sigma0="auto"` | `sigma0_pilot` (256) |
| Each iteration | `N` (1,000) |
| Final estimate | `N` |

and `MarginalTransform.wrap` adds its own `pilot` (512) when `scale=True`. The
runs in [Getting started](../getting-started.md) converge in two or three
iterations, so about 3,300 to 4,300 evaluations in all.

When one evaluation is a finite-element solve, as in
[`HeatTransferProblem`][safe_ice.HeatTransferProblem]:

- pass a numeric `sigma0` and `scale` so neither pilot runs;
- lower `N`, accepting a noisier estimate per run;
- use [`OptimizedSafeICE`][safe_ice.OptimizedSafeICE], whose `batch_size`
  bounds how many points are passed to \(g\) in one call.
