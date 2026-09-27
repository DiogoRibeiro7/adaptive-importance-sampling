# Getting started

## Install

Safe-ICE needs Python 3.11 or newer; it is tested on 3.11 to 3.14. Its runtime
dependencies are NumPy, SciPy and Matplotlib.

The package is not on PyPI yet, so install it from GitHub:

=== "pip"

    ```bash
    pip install git+https://github.com/DiogoRibeiro7/adaptive-importance-sampling.git
    ```

=== "From a checkout"

    ```bash
    git clone https://github.com/DiogoRibeiro7/adaptive-importance-sampling.git
    cd adaptive-importance-sampling
    pip install -e .
    ```

### Optional extras

| Extra | Adds | Needed for |
| --- | --- | --- |
| `viz` | plotly, seaborn, pandas | The interactive plots in `safe_ice.analysis.interactive_visualization` |
| `perf` | psutil | Memory figures in `examples/performance_benchmark.py` |
| `all` | Everything above | |

=== "pip"

    ```bash
    pip install "safe-ice[viz] @ git+https://github.com/DiogoRibeiro7/adaptive-importance-sampling.git"
    ```

=== "From a checkout"

    ```bash
    pip install -e ".[viz]"
    ```

Check that it worked:

```bash
safe-ice --version
```

## A first estimate

Start with a problem whose answer is known exactly. Take a two-dimensional
standard normal point and call it a failure when it lands more than 3 from the
origin. The squared radius is chi-square with two degrees of freedom, so the
exact probability is \(e^{-9/2} \approx 1.11\times10^{-2}\).

```python
import numpy as np
from scipy import stats

from safe_ice import SafeICE


def g(u):
    """Fails outside the radius-3 circle: g(u) <= 0 there."""
    return 3.0 - np.linalg.norm(u, axis=-1)


pf, results = SafeICE(g, dimension=2, random_state=0).run(verbose=False)

exact = stats.chi2.sf(3.0**2, df=2)
print(f"Safe-ICE {pf:.3e}   exact {exact:.3e}")
```

```text
Safe-ICE 1.123e-02   exact 1.111e-02
```

Three things to take from this:

- `g` receives a whole batch of points as an `(n, d)` array and returns `n`
  values. [Limit-state functions](guide/limit-states.md) covers the contract.
- `run()` returns the estimate and a dictionary of diagnostics.
  [Running and diagnosing](guide/running.md) describes every key.
- `random_state=0` makes the run repeatable. Without it, each run gives a
  slightly different estimate.

## A rarer event

The four-mode series system from the paper fails in four separate lobes, with a
reference probability of \(6.465\times10^{-5}\) from crude Monte Carlo over
\(2\times10^7\) samples.

```python
from safe_ice import BenchmarkProblems, SafeICE

g = BenchmarkProblems.four_mode_series_system()
pf, results = SafeICE(g, dimension=2, random_state=0).run()
```

With `verbose` left on, `run()` reports each iteration:

```text
Safe-ICE Algorithm
Problem dimension: 2
Initial components: 20
Samples per iteration: 1000
--------------------------------------------------
Iteration  1: sigma=1.000000, lambda=0.000, K=20
           CV=inf
Iteration  2: sigma=0.666612, lambda=0.250, K=9
           CV=1.4654
  Converged: CV 1.4654 <= delta_star 1.5
--------------------------------------------------
Final Results:
Failure Probability: 5.594960e-05
Total Iterations: 2
Final Components: 9
Final CV: 1.4654
```

- **sigma** is the smoothing parameter. It only falls, and a smaller value means
  the intermediate problem is closer to the real one.
- **lambda** is the share of samples drawn from the light-tailed mixture; the
  rest come from the heavy-tailed safety component. It starts at 0, so the first
  iteration explores.
- **K** is the number of mixture components. It started at 20 and penalized EM
  pruned it to 9.
- **CV** is the coefficient of variation of the stopping weights. The run stops
  once it falls to `delta_star`.

That run landed at 0.87 of the reference. Seeds 0 to 5 give between 0.79 and
1.09 of it: each run is one draw of a random estimate, so repeat with several
seeds before trusting a single number.

## From the command line

The `safe-ice` command runs the paper's benchmarks without writing any code:

```bash
safe-ice demo                     # the four-mode problem, with commentary
safe-ice benchmark --list         # four-mode, three-mode, two-mode, oscillator
safe-ice benchmark four-mode --samples 2000 --iterations 15
```

`benchmark` prints the estimate next to the reference value and the relative
error.

## Next steps

- [Limit-state functions](guide/limit-states.md): write your own, including
  problems in physical units.
- [Choosing an estimator](guide/estimators.md): the variants and baselines, and
  when to use each.
- [Examples](examples.md): notebooks and scripts, from a first run to real flood
  data.
